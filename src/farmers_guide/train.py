"""Training entry point — spatial K-fold CV for SatelliteCNN.

# TODO: add fusion training path once photo dataset lands (CLAUDE.md §7 item 5b).

Per fold the pipeline:
  1. ``spatial_kfold_split`` on the label table (no neighbour-leakage).
  2. Build train/val ``MaizeYieldDataset`` instances on the fold's labels.
  3. Compute (mean, std) of yield_kg_ha on the **surviving training samples
     only** — this is the leakage-prevention contract. The standardisation
     params are persisted in the checkpoint so inference can invert them
     consistently.
  4. Train with AdamW + cosine LR (config.LEARNING_RATE -> LR/100 over
     config.EPOCHS), MSE loss on standardised yields.
  5. Track per epoch: train_loss, val_loss, val_rmse_kg_ha (original units),
     val_r2 (original units).
  6. Early stop on val_loss, patience 10.
  7. Persist best-by-val_loss checkpoint to
     ``config.WEIGHTS_DIR / f"satellite_cnn_fold{k}.pt"``.

After all folds: aggregate val_rmse / val_r2 (mean ± std) and print.

Notes on augmentation:
  * Random horizontal / vertical spatial flips (axes H and W).
  * **No temporal flip.** The season has a directional time arrow
    (planting -> green-up -> senescence). Reversing it teaches the network
    a non-physical trajectory and ruins the phenology features the model
    is specifically designed to capture.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from farmers_guide import config
from farmers_guide.data.dataset import (
    MaizeYieldDataset,
    read_labels,
    spatial_kfold_split,
    synthetic_labels,
)
from farmers_guide.models.satellite_cnn import SatelliteCNN

logger = logging.getLogger(__name__)

# Minimal WGS84 WKT — used by the smoke-test cube. Avoids a rasterio import
# at cube-write time; rasterio is still required for the dataset to parse it
# at training time (see ``MaizeYieldDataset._project_to_pixels``).
_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)


@dataclass
class TargetStandardisation:
    """yield_kg_ha standardisation params. Persisted in every checkpoint."""
    mean: float
    std: float

    def standardise(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def _seed_everything(seed: int) -> None:
    """Re-seed every RNG that affects training, *per fold*.

    Called at the start of each fold so fold k is reproducible independent
    of fold k-1's stochastic ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _augment_train(patch: torch.Tensor) -> torch.Tensor:
    """Spatial-only random flips on a (B, T, H, W, C) batch.

    Flips are applied per-batch (not per-sample): a batch is either flipped
    on each axis or not. At thesis-scale this still gives the model enough
    geometric diversity across epochs while staying trivially fast.

    H is dim=2, W is dim=3. Dim=1 (T) is **never** flipped — see the module
    docstring for the physical reason.
    """
    if torch.rand(()) < 0.5:
        patch = torch.flip(patch, dims=[2])
    if torch.rand(()) < 0.5:
        patch = torch.flip(patch, dims=[3])
    return patch


def _rmse_r2(preds: np.ndarray, trues: np.ndarray) -> tuple[float, float]:
    """RMSE (same units as inputs) and R^2 = 1 - SS_res / SS_tot."""
    err = preds - trues
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    # ss_tot == 0 means all trues equal — R^2 is undefined; report nan rather
    # than a misleading 1.0 / -inf.
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def _train_one_fold(
    fold_idx: int,
    cube_path: Path,
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    buffer_size: int,
    min_valid_fraction: float,
    num_workers: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    _seed_everything(config.SEED)

    train_ds = MaizeYieldDataset(
        cube_path, train_labels,
        buffer_size=buffer_size,
        min_valid_fraction=min_valid_fraction,
    )
    val_ds = MaizeYieldDataset(
        cube_path, val_labels,
        buffer_size=buffer_size,
        min_valid_fraction=min_valid_fraction,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Fold {fold_idx}: empty after validity filter "
            f"(train={len(train_ds)}, val={len(val_ds)}). "
            "Lower min_valid_fraction or rebuild the cube."
        )

    # Standardisation: train-only, on surviving samples (those the model
    # actually sees). Computing this on the un-filtered label table would
    # use yields from samples that never reach the network.
    train_yields = np.array(
        [s.yield_kg_ha for s in train_ds.samples], dtype=np.float64
    )
    stdz = TargetStandardisation(
        mean=float(train_yields.mean()),
        std=float(train_yields.std() + 1e-8),       # guard zero-variance fold
    )
    logger.info(
        "fold %d: train=%d val=%d  yield_mean=%.1f yield_std=%.1f kg/ha",
        fold_idx, len(train_ds), len(val_ds), stdz.mean, stdz.std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SatelliteCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 100.0
    )
    loss_fn = nn.MSELoss()

    best = {
        "val_loss": float("inf"),
        "epoch": -1,
        "val_rmse_kg_ha": float("inf"),
        "val_r2": float("-inf"),
    }
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            patch = batch["patch"].to(device, non_blocking=True)
            y_true = batch["yield_kg_ha"].to(device, non_blocking=True)
            patch = _augment_train(patch)

            y_pred_std = model(patch)
            y_true_std = stdz.standardise(y_true)

            optimizer.zero_grad()
            loss = loss_fn(y_pred_std, y_true_std)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_losses: list[float] = []
        val_preds_orig: list[np.ndarray] = []
        val_trues_orig: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                patch = batch["patch"].to(device, non_blocking=True)
                y_true = batch["yield_kg_ha"].to(device, non_blocking=True)
                y_pred_std = model(patch)
                val_losses.append(
                    loss_fn(y_pred_std, stdz.standardise(y_true)).item()
                )
                y_pred_orig = stdz.invert(y_pred_std)
                val_preds_orig.append(y_pred_orig.cpu().numpy())
                val_trues_orig.append(y_true.cpu().numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        preds = np.concatenate(val_preds_orig)
        trues = np.concatenate(val_trues_orig)
        val_rmse, val_r2 = _rmse_r2(preds, trues)

        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_rmse_kg_ha": val_rmse, "val_r2": val_r2,
            "lr": optimizer.param_groups[0]["lr"],
        })
        logger.info(
            "fold %d  ep %03d  train=%.4f  val=%.4f  rmse=%.1f kg/ha  R2=%.3f",
            fold_idx, epoch, train_loss, val_loss, val_rmse, val_r2,
        )

        # ---- checkpoint + early stop on val_loss ----
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "val_rmse_kg_ha": val_rmse,
                "val_r2": val_r2,
            }
            epochs_no_improve = 0
            ckpt_path = out_dir / f"satellite_cnn_fold{fold_idx}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "standardisation": asdict(stdz),
                "fold_index": fold_idx,
                "best_epoch": best["epoch"],
                "best_val_rmse_kg_ha": best["val_rmse_kg_ha"],
                "best_val_r2": best["val_r2"],
                "config_seed": config.SEED,
                "buffer_size": buffer_size,
                "min_valid_fraction": min_valid_fraction,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "fold %d  early stop at epoch %d (no improvement for %d)",
                    fold_idx, epoch, patience,
                )
                break

    return {
        "fold_index": fold_idx,
        "best_epoch": best["epoch"],
        "best_val_rmse_kg_ha": best["val_rmse_kg_ha"],
        "best_val_r2": best["val_r2"],
        "history": history,
    }


def train(
    cube_path: Path | str,
    labels: pd.DataFrame,
    *,
    n_splits: int = 5,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = 1e-4,
    patience: int = 10,
    buffer_size: int = config.PATCH_SIZE,
    min_valid_fraction: float = 0.5,
    num_workers: int = 0,
    out_dir: Path | str | None = None,
    device: str | None = None,
) -> list[dict]:
    """Spatial K-fold cross-validated training of SatelliteCNN.

    Args:
        cube_path: HDF5 cube produced by ``hdf5_builder.build_cube``.
        labels: ground-truth label table (schema per
            ``dataset.REQUIRED_LABEL_COLUMNS``). Pre-filter to the cube's
            (aoi, season) is recommended; the dataset will re-filter as
            belt-and-braces.
        n_splits: K for ``spatial_kfold_split``.
        out_dir: where to write checkpoints. Defaults to ``config.WEIGHTS_DIR``.
    Returns:
        List of per-fold result dicts.
    """
    out_dir = Path(out_dir) if out_dir is not None else config.WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    folds = spatial_kfold_split(labels, n_splits=n_splits, seed=config.SEED)
    logger.info(
        "training %d folds on %d labels, device=%s, out=%s",
        n_splits, len(labels), dev, out_dir,
    )

    results: list[dict] = []
    for k, (train_idx, val_idx) in enumerate(folds):
        train_labels = labels.iloc[train_idx].reset_index(drop=True)
        val_labels = labels.iloc[val_idx].reset_index(drop=True)
        logger.info("=" * 60)
        logger.info("Fold %d/%d", k, n_splits - 1)
        res = _train_one_fold(
            fold_idx=k,
            cube_path=Path(cube_path),
            train_labels=train_labels,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            buffer_size=buffer_size,
            min_valid_fraction=min_valid_fraction,
            num_workers=num_workers,
            device=dev,
            out_dir=out_dir,
        )
        results.append(res)

    # ---- CV summary ----
    rmses = np.array([r["best_val_rmse_kg_ha"] for r in results])
    r2s = np.array([r["best_val_r2"] for r in results])
    logger.info("=" * 60)
    logger.info("CV summary across %d folds:", n_splits)
    logger.info(
        "  val RMSE: %.1f ± %.1f kg/ha  (target < %.0f)",
        rmses.mean(), rmses.std(), config.TARGET_RMSE_KG_HA,
    )
    logger.info(
        "  val R^2 : %.3f ± %.3f       (target > %.2f)",
        r2s.mean(), r2s.std(), config.TARGET_R2,
    )
    logger.info("  per-fold RMSE: %s",
                [f"{r:.0f}" for r in rmses])
    logger.info("  per-fold R^2 : %s",
                [f"{r:.3f}" for r in r2s])
    return results


# --- Smoke test ----------------------------------------------------------

def _build_synthetic_cube(
    path: Path,
    aoi: str = "chilanga",
    season: str = "2018_19",
    n_windows: int = 12,
    spatial: int = 128,
) -> None:
    """Write a synthetic HDF5 cube matching ``hdf5_builder``'s contract.

    Magnitudes are picked to match real Sentinel-2 composites: bands 0–8
    (reflectance) drawn from U(0.0, 0.3); bands 9–13 (NDVI/EVI/NDRE/GCI/NDWI)
    drawn from U(-0.2, 0.9).

    ``n_windows`` defaults to 12 rather than the literal "3 timesteps"
    spec note: the SatelliteCNN's three temporal MaxPool3d operations
    collapse T → ⌊T/2⌋ → ⌊T/4⌋, so T=3 → 0 after the second pool. T=12
    matches a realistic biweekly composite cadence over Oct–Jun anyway,
    and the smoke test runs in seconds on CPU.
    """
    H = W = spatial
    C = config.N_BANDS
    rng = np.random.default_rng(0)

    cube = np.empty((n_windows, H, W, C), dtype=np.float32)
    cube[..., :9] = rng.uniform(0.0, 0.3, size=(n_windows, H, W, 9))
    cube[..., 9:] = rng.uniform(-0.2, 0.9, size=(n_windows, H, W, 5))

    lon_min, lat_min, lon_max, lat_max = config.AOIS[aoi]
    pixel_size_x = (lon_max - lon_min) / W
    pixel_size_y = (lat_max - lat_min) / H
    # Affine 6-tuple in rasterio order: (a, b, c, d, e, f) where
    # x = a*col + b*row + c, y = d*col + e*row + f. North-up image.
    transform = (pixel_size_x, 0.0, lon_min, 0.0, -pixel_size_y, lat_max)

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
    logger.info("=== train.py smoke test ===")
    aoi, season = "chilanga", "2018_19"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cube_path = tmp / f"fg_{aoi}_{season}.h5"
        _build_synthetic_cube(cube_path, aoi=aoi, season=season)
        labels = synthetic_labels(n=40, aoi=aoi, season=season, seed=0)
        out_dir = tmp / "weights"

        results = train(
            cube_path=cube_path,
            labels=labels,
            n_splits=2,
            epochs=2,
            batch_size=8,
            patience=10,
            min_valid_fraction=0.4,        # small smoke cube — be lenient
            num_workers=0,
            out_dir=out_dir,
            device="cpu",
        )

        # Both checkpoints must be on disk.
        for r in results:
            ckpt = out_dir / f"satellite_cnn_fold{r['fold_index']}.pt"
            assert ckpt.exists(), f"missing checkpoint {ckpt}"
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            assert "model_state_dict" in payload
            assert "standardisation" in payload
            assert payload["fold_index"] == r["fold_index"]
            logger.info("  ✓ %s  rmse=%.0f kg/ha  R2=%.3f",
                        ckpt.name, r["best_val_rmse_kg_ha"], r["best_val_r2"])

    logger.info("smoke test OK")


# --- CLI ------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train",
        description="Spatial K-fold training of SatelliteCNN on RALS yields.",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Run the synthetic-cube smoke test and exit.")
    p.add_argument("--cube", type=Path,
                   help="Path to fg_<aoi>_<season>.h5 cube.")
    p.add_argument("--labels", type=Path,
                   help="Path to ground-truth CSV.")
    p.add_argument("--aoi", default=None,
                   help="Filter labels CSV to this AOI (optional).")
    p.add_argument("--season", default=None,
                   help="Filter labels CSV to this season (optional).")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--buffer-size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--min-valid-fraction", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override config.WEIGHTS_DIR.")
    p.add_argument("--device", default=None,
                   help="'cuda', 'cpu', or None for auto.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.smoke:
        _smoke_test()
        return 0
    if args.cube is None or args.labels is None:
        logger.error("Real training requires --cube and --labels (or use --smoke).")
        return 2

    labels = read_labels(args.labels, aoi=args.aoi, season=args.season)
    if labels.empty:
        logger.error("Label table is empty after filtering.")
        return 1

    train(
        cube_path=args.cube,
        labels=labels,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        buffer_size=args.buffer_size,
        min_valid_fraction=args.min_valid_fraction,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
