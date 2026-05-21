"""Evaluation: inference, metrics, baselines, and thesis figures.

# TODO: training-curves figure pending per-epoch history save in train.py.

Four functional areas (one per CLAUDE.md §4 evaluation requirement):

1. **Inference** — ``load_model_from_checkpoint`` + ``predict``. The
   checkpoint carries the train-fold standardisation params; inference
   inverts them so all returned yields are in original kg/ha units.

2. **Metrics** — ``compute_metrics`` and ``cv_summary``. CV summary
   re-runs ``spatial_kfold_split(seed=config.SEED)`` so the val folds
   reconstructed here are bit-identical to those ``train.py`` used.

3. **Baselines** — ``ndvi_mean_baseline`` (mean-NDVI OLS) is the
   defensible "is the CNN even doing anything?" comparison reported in
   the methodology chapter. ``satellite_only_baseline`` is a stub for
   the post-fusion comparison.

4. **Figures** — ``generate_figures`` writes the four artefacts that
   appear in the thesis results chapter: ``predicted_vs_actual.png``,
   ``residuals_by_aoi.png``, ``uncertainty_calibration.png``, and the
   ``cv_summary_table.csv`` summary.

The target metrics from ``config`` are RMSE < 600 kg/ha and R² > 0.55.
``cv_summary``'s aggregate row lets you read both off at a glance.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")                                   # headless rendering
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

import config
from farmers_guide.data.dataset import (
    MaizeYieldDataset,
    spatial_kfold_split,
    synthetic_labels,
)
from farmers_guide.models.satellite_cnn import SatelliteCNN

logger = logging.getLogger(__name__)

_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


# Redefined locally rather than imported from ``farmers_guide.train`` so
# that ``evaluate`` doesn't drag the trainer's heavier import graph (optimisers,
# schedulers) into notebooks that only need inference + figures.
@dataclass
class TargetStandardisation:
    mean: float
    std: float

    def invert(self, y: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        return y * self.std + self.mean


# --- 1. Inference ---------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str | None = None,
) -> SatelliteCNN:
    """Reconstitute a ``SatelliteCNN`` and attach its standardisation params.

    ``weights_only=False`` is required: PyTorch 2.6+ defaults this to
    ``True``, which refuses to unpickle the ``standardisation`` dict in
    ``train.py``'s checkpoints. The smoke test in ``train.py`` documented
    this; we match here.

    The returned model has ``model.standardisation`` set so ``predict``
    can invert back to kg/ha without the caller threading the params
    through manually.
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in ckpt or "standardisation" not in ckpt:
        raise KeyError(
            f"{checkpoint_path} is not a Farmers Guide satellite checkpoint "
            "(missing model_state_dict or standardisation)."
        )
    model = SatelliteCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.standardisation = TargetStandardisation(**ckpt["standardisation"])
    if device is not None:
        model = model.to(device)
    return model


def predict(
    model: SatelliteCNN,
    dataset: MaizeYieldDataset,
    *,
    with_uncertainty: bool = False,
    n_mc_samples: int = 30,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run ``model`` on ``dataset`` and return one row per sample.

    Returns columns ``(field_id, aoi, y_true, y_pred_kg_ha[, y_std_kg_ha])``.
    All yields are in original kg/ha — the standardisation attached at
    ``load_model_from_checkpoint`` time is inverted here.

    Under ``with_uncertainty=True`` the model's MC-dropout helper is used;
    the returned std also lives in kg/ha (std scales linearly under the
    affine standardisation transform).
    """
    if not hasattr(model, "standardisation"):
        raise AttributeError(
            "model has no .standardisation attached. Use "
            "load_model_from_checkpoint, or set model.standardisation manually."
        )
    stdz: TargetStandardisation = model.standardisation
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    field_ids: list[str] = []
    y_trues: list[np.ndarray] = []
    y_preds: list[np.ndarray] = []
    y_stds: list[np.ndarray] = []

    for batch in loader:
        patch = batch["patch"].to(device, non_blocking=True)
        y_true = batch["yield_kg_ha"]
        if with_uncertainty:
            mean_std_tensor, std_std_tensor = model.predict_with_uncertainty(
                patch, n_samples=n_mc_samples
            )
            y_pred = mean_std_tensor * stdz.std + stdz.mean
            y_std = std_std_tensor * stdz.std
            y_stds.append(y_std.cpu().numpy())
        else:
            with torch.no_grad():
                y_pred_std = model(patch)
            y_pred = y_pred_std * stdz.std + stdz.mean

        field_ids.extend(list(batch["field_id"]))
        y_trues.append(y_true.numpy())
        y_preds.append(y_pred.detach().cpu().numpy())

    df = pd.DataFrame({
        "field_id":     field_ids,
        "aoi":          dataset.aoi,
        "y_true":       np.concatenate(y_trues).astype(np.float64),
        "y_pred_kg_ha": np.concatenate(y_preds).astype(np.float64),
    })
    if with_uncertainty:
        df["y_std_kg_ha"] = np.concatenate(y_stds).astype(np.float64)
    return df


# --- 2. Metrics -----------------------------------------------------------

def _metrics_of(df: pd.DataFrame) -> dict:
    """Plain dict of {rmse_kg_ha, mae_kg_ha, r2, n} — used by both
    ``compute_metrics`` and ``cv_summary``."""
    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred_kg_ha"].to_numpy(dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse_kg_ha": rmse, "mae_kg_ha": mae, "r2": r2, "n": int(len(df))}


def compute_metrics(
    predictions_df: pd.DataFrame, by: str | None = None
) -> dict:
    """Aggregate metrics over a predictions DataFrame.

    If ``by`` is None: returns a flat ``{rmse_kg_ha, mae_kg_ha, r2, n}`` dict.
    If ``by`` is a column name (e.g. "aoi", "fold"): returns a nested dict
    keyed by group value. Group keys are coerced to str for JSON-safety.
    """
    if by is None:
        return _metrics_of(predictions_df)
    if by not in predictions_df.columns:
        raise KeyError(f"Column '{by}' not in predictions_df.")
    return {str(k): _metrics_of(g) for k, g in predictions_df.groupby(by)}


def cv_summary(
    checkpoint_dir: Path | str,
    cube_path: Path | str,
    labels: pd.DataFrame,
    n_splits: int,
    *,
    with_uncertainty: bool = False,
    buffer_size: int = config.PATCH_SIZE,
    min_valid_fraction: float = 0.5,
    device: torch.device | str | None = None,
) -> dict:
    """Re-run K-fold inference and aggregate mean ± std across folds.

    The fold construction MUST match what ``train.py`` used during training,
    or the val-fold checkpoint will be evaluated on the wrong samples.
    ``spatial_kfold_split(seed=config.SEED)`` is the contract — same call,
    same seed, same fold definitions.

    Args:
        checkpoint_dir: directory containing ``satellite_cnn_fold{k}.pt``.
        cube_path: HDF5 cube used during training.
        labels: full label table that was passed to ``train()`` originally.
        n_splits: K used during training.
    Returns:
        ``{"predictions": combined_df_with_fold_col,
           "per_fold":    [{fold, rmse_kg_ha, mae_kg_ha, r2, n}, ...],
           "aggregate":   {rmse_mean, rmse_std, mae_mean, mae_std,
                           r2_mean, r2_std}}``
    """
    checkpoint_dir = Path(checkpoint_dir)
    folds = spatial_kfold_split(labels, n_splits=n_splits, seed=config.SEED)

    per_fold: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    for k, (_train_idx, val_idx) in enumerate(folds):
        val_labels = labels.iloc[val_idx].reset_index(drop=True)
        val_ds = MaizeYieldDataset(
            cube_path, val_labels,
            buffer_size=buffer_size,
            min_valid_fraction=min_valid_fraction,
        )
        ckpt_path = checkpoint_dir / f"satellite_cnn_fold{k}.pt"
        model = load_model_from_checkpoint(ckpt_path, device=device)
        df = predict(model, val_ds, with_uncertainty=with_uncertainty)
        df["fold"] = k
        all_preds.append(df)
        per_fold.append({"fold": k, **_metrics_of(df)})
        logger.info(
            "fold %d: n=%d  rmse=%.1f  R^2=%.3f",
            k, per_fold[-1]["n"], per_fold[-1]["rmse_kg_ha"], per_fold[-1]["r2"],
        )

    predictions = pd.concat(all_preds, ignore_index=True)
    aggregate: dict[str, float] = {}
    for key in ("rmse_kg_ha", "mae_kg_ha", "r2"):
        arr = np.array([m[key] for m in per_fold], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std())
    return {"predictions": predictions, "per_fold": per_fold, "aggregate": aggregate}


# --- 3. Baselines ---------------------------------------------------------

def _ndvi_mean_features(
    dataset: MaizeYieldDataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-sample mean NDVI over the (T, H, W) patch, plus yields + field IDs."""
    ndvi_idx = config.BANDS.index("NDVI")
    xs: list[float] = []
    ys: list[float] = []
    fids: list[str] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # The dataset has already filled NaN, but nanmean is harmless and
        # keeps the helper robust if min_valid_fraction is lowered or the
        # NaN-fill policy changes upstream.
        patch = sample["patch"].numpy()
        xs.append(float(np.nanmean(patch[..., ndvi_idx])))
        ys.append(float(sample["yield_kg_ha"].item()))
        fids.append(str(sample["field_id"]))
    return np.array(xs, dtype=np.float64).reshape(-1, 1), np.array(ys, dtype=np.float64), fids


def ndvi_mean_baseline(
    train_dataset: MaizeYieldDataset,
    val_dataset: MaizeYieldDataset,
) -> pd.DataFrame:
    """OLS of yield_kg_ha ~ mean_NDVI on train, predict on val.

    This is the "did we actually need a CNN?" baseline. If the CNN can't
    beat a one-feature linear regression on a single greenness mean, the
    architecture isn't earning its parameter budget — and that's a
    methodology finding worth reporting.

    Returns:
        val-fold predictions DataFrame in the same schema as ``predict``
        (without the ``y_std_kg_ha`` column).
    """
    X_train, y_train, _ = _ndvi_mean_features(train_dataset)
    X_val, y_val, fids_val = _ndvi_mean_features(val_dataset)
    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(
            f"NDVI baseline got empty fold (train={len(X_train)}, "
            f"val={len(X_val)})."
        )
    ols = LinearRegression().fit(X_train, y_train)
    y_pred = ols.predict(X_val)
    return pd.DataFrame({
        "field_id":     fids_val,
        "aoi":          val_dataset.aoi,
        "y_true":       y_val,
        "y_pred_kg_ha": y_pred,
    })


def satellite_only_baseline(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder: satellite-only comparison for the fusion methodology.

    Until ``fusion.YieldFusion`` is trained, "satellite-only" *is* the
    current model (the trained ``SatelliteCNN`` checkpoints from
    ``train.py``). This function will become non-trivial once
    fusion training lands — at that point it loads a fusion checkpoint
    but disables / zeros the ground branch's contribution, so the
    satellite-only ablation reads the same fusion-trained features.

    Until then, call ``predict`` on a ``SatelliteCNN`` checkpoint
    directly to get the satellite-only numbers.
    """
    raise NotImplementedError(
        "satellite_only_baseline becomes meaningful only after fusion "
        "training. For now, predict() on a SatelliteCNN checkpoint IS the "
        "satellite-only result."
    )


# --- 4. Figures -----------------------------------------------------------

def generate_figures(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write the four thesis-results artefacts to ``output_dir``.

    Returns a dict mapping artefact name → written Path. Files not written
    (e.g. ``uncertainty_calibration.png`` when ``y_std_kg_ha`` is absent)
    are omitted from the dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    y_true = predictions_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = predictions_df["y_pred_kg_ha"].to_numpy(dtype=np.float64)
    metrics = _metrics_of(predictions_df)
    rmse = metrics["rmse_kg_ha"]
    r2 = metrics["r2"]

    # ---- (a) predicted_vs_actual.png -------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=28, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="1:1")
    ax.fill_between(
        [lo, hi], [lo - rmse, hi - rmse], [lo + rmse, hi + rmse],
        color="gray", alpha=0.12, label=f"±RMSE ({rmse:.0f} kg/ha)",
    )
    ax.set_xlabel("Actual yield (kg/ha)")
    ax.set_ylabel("Predicted yield (kg/ha)")
    ax.set_title(f"Predicted vs Actual — RMSE={rmse:.0f}, R²={r2:.3f}")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    pva_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(pva_path, dpi=150)
    plt.close(fig)
    written["predicted_vs_actual"] = pva_path

    # ---- (b) residuals_by_aoi.png ----------------------------------------
    if "aoi" in predictions_df.columns:
        residuals = y_pred - y_true
        df_r = predictions_df.assign(residual=residuals)
        aois = sorted(df_r["aoi"].astype(str).unique())
        data = [df_r.loc[df_r["aoi"].astype(str) == a, "residual"].to_numpy()
                for a in aois]
        fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(aois) + 2), 5))
        ax.boxplot(data, tick_labels=aois, showmeans=True)
        ax.axhline(0, color="k", linewidth=1.0, linestyle="--")
        ax.set_xlabel("AOI")
        ax.set_ylabel("Residual (kg/ha)  [pred − actual]")
        ax.set_title("Residuals by AOI")
        fig.tight_layout()
        rba_path = output_dir / "residuals_by_aoi.png"
        fig.savefig(rba_path, dpi=150)
        plt.close(fig)
        written["residuals_by_aoi"] = rba_path

    # ---- (c) uncertainty_calibration.png ---------------------------------
    if "y_std_kg_ha" in predictions_df.columns:
        abs_err = np.abs(y_pred - y_true)
        y_std = predictions_df["y_std_kg_ha"].to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_std, abs_err, alpha=0.6, s=28, edgecolor="none")
        lim = float(max(y_std.max(), abs_err.max()))
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="Ideal (y = x)")
        ax.set_xlabel("Predicted std (kg/ha)")
        ax.set_ylabel("|Residual| (kg/ha)")
        ax.set_title("Uncertainty calibration")
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        uc_path = output_dir / "uncertainty_calibration.png"
        fig.savefig(uc_path, dpi=150)
        plt.close(fig)
        written["uncertainty_calibration"] = uc_path

    # ---- (d) cv_summary_table.csv ----------------------------------------
    if "fold" in predictions_df.columns:
        rows: list[dict] = []
        for fold_val, g in predictions_df.groupby("fold"):
            rows.append({"fold": int(fold_val), **_metrics_of(g)})
        df_cv = pd.DataFrame(rows)
        for label, fn in [("mean", np.mean), ("std", np.std)]:
            agg: dict = {"fold": label}
            for col in ("rmse_kg_ha", "mae_kg_ha", "r2", "n"):
                val = float(fn(df_cv[col]))
                agg[col] = int(val) if col == "n" else val
            rows.append(agg)
        out_df = pd.DataFrame(rows)
    else:
        out_df = pd.DataFrame([{"fold": "all", **_metrics_of(predictions_df)}])
    cv_path = output_dir / "cv_summary_table.csv"
    out_df.to_csv(cv_path, index=False)
    written["cv_summary_table"] = cv_path

    logger.info("wrote %d figures to %s", len(written), output_dir)
    return written


# --- Smoke test -----------------------------------------------------------

def _build_synthetic_cube(
    path: Path,
    aoi: str = "chilanga",
    season: str = "2018_19",
    n_windows: int = 12,
    spatial: int = 128,
) -> None:
    """Inline synthetic HDF5 cube — mirrors ``train.py``'s smoke helper.

    Reflectance bands U(0.0, 0.3); index bands (NDVI/EVI/NDRE/GCI/NDWI)
    U(-0.2, 0.9). T=12 (not 3 — survives the SatelliteCNN pool schedule).
    """
    H = W = spatial
    C = config.N_BANDS
    rng = np.random.default_rng(0)
    cube = np.empty((n_windows, H, W, C), dtype=np.float32)
    cube[..., :9] = rng.uniform(0.0, 0.3, size=(n_windows, H, W, 9))
    cube[..., 9:] = rng.uniform(-0.2, 0.9, size=(n_windows, H, W, 5))

    lon_min, lat_min, lon_max, lat_max = config.AOIS[aoi]
    transform = (
        (lon_max - lon_min) / W, 0.0, lon_min,
        0.0, -(lat_max - lat_min) / H, lat_max,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("cube", data=cube)
        dates = np.array(
            [f"2018-10-{15 + 2 * i:02d}".encode("ascii") for i in range(n_windows)],
            dtype="S10",
        )
        f.create_dataset("dates", data=dates)
        f.attrs["aoi"] = aoi
        f.attrs["season"] = season
        f.attrs["bands"] = np.array(config.BANDS, dtype="S8")
        f.attrs["n_windows"] = n_windows
        f.attrs["height"] = H
        f.attrs["width"] = W
        f.attrs["n_bands"] = C
        f.attrs["crs"] = _WGS84_WKT
        f.attrs["transform"] = np.array(transform, dtype=np.float64)


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== evaluate.py smoke test ===")
    aoi, season = "chilanga", "2018_19"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cube_path = tmp / f"fg_{aoi}_{season}.h5"
        _build_synthetic_cube(cube_path, aoi=aoi, season=season)
        labels = synthetic_labels(n=40, aoi=aoi, season=season, seed=0)

        # Fresh random-weight SatelliteCNN + mock checkpoint with realistic
        # standardisation params (mean=1500, std=600 kg/ha).
        model = SatelliteCNN()
        stdz = TargetStandardisation(mean=1500.0, std=600.0)
        ckpt_path = tmp / "satellite_cnn_fold0.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "standardisation": asdict(stdz),
            "fold_index": 0,
            "best_epoch": 0,
            "best_val_rmse_kg_ha": 0.0,
            "best_val_r2": 0.0,
            "config_seed": config.SEED,
        }, ckpt_path)

        # 1. Inference --------------------------------------------------
        eval_ds = MaizeYieldDataset(cube_path, labels, min_valid_fraction=0.4)
        loaded = load_model_from_checkpoint(ckpt_path)
        assert hasattr(loaded, "standardisation"), "standardisation lost"
        preds = predict(loaded, eval_ds, with_uncertainty=True, n_mc_samples=10)
        logger.info("  predict: %d rows, columns=%s",
                    len(preds), list(preds.columns))
        assert {"field_id", "aoi", "y_true", "y_pred_kg_ha", "y_std_kg_ha"} \
            <= set(preds.columns)

        # 2. Metrics ----------------------------------------------------
        m_all = compute_metrics(preds)
        m_by = compute_metrics(preds, by="aoi")
        logger.info("  metrics overall: rmse=%.0f mae=%.0f R^2=%.3f n=%d",
                    m_all["rmse_kg_ha"], m_all["mae_kg_ha"], m_all["r2"], m_all["n"])
        logger.info("  metrics by aoi: %s", list(m_by.keys()))

        # 3. NDVI-mean baseline (split via spatial KFold for realism)
        folds = spatial_kfold_split(labels, n_splits=5, seed=config.SEED)
        tr_idx, va_idx = folds[0]
        train_ds = MaizeYieldDataset(
            cube_path, labels.iloc[tr_idx].reset_index(drop=True),
            min_valid_fraction=0.4,
        )
        val_ds = MaizeYieldDataset(
            cube_path, labels.iloc[va_idx].reset_index(drop=True),
            min_valid_fraction=0.4,
        )
        baseline = ndvi_mean_baseline(train_ds, val_ds)
        m_baseline = compute_metrics(baseline)
        logger.info("  NDVI-mean baseline: rmse=%.0f R^2=%.3f n=%d",
                    m_baseline["rmse_kg_ha"], m_baseline["r2"], m_baseline["n"])

        # 4. Figures (use the CNN predictions DF for the assertions)
        fig_dir = tmp / "figures"
        written = generate_figures(preds, fig_dir)
        expected = {
            "predicted_vs_actual": fig_dir / "predicted_vs_actual.png",
            "residuals_by_aoi":    fig_dir / "residuals_by_aoi.png",
            "uncertainty_calibration": fig_dir / "uncertainty_calibration.png",
            "cv_summary_table":    fig_dir / "cv_summary_table.csv",
        }
        for name, path in expected.items():
            assert path.exists(), f"missing figure: {path}"
            assert written[name] == path, f"unexpected return path for {name}"
            logger.info("  ✓ %s", path.name)

    logger.info("smoke test OK")


if __name__ == "__main__":
    sys.exit(_smoke_test() or 0)
