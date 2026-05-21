"""torch Dataset over HDF5 cubes joined to yield labels.

The unit of analysis is the **buffered point** described in
``docs/ground_truth_plan.md`` ("Spatial"): RALS gives a household GPS point
plus a plot area in hectares — not a field polygon — so each label produces
a (T, buffer_size, buffer_size, 14) patch centred on the projected pixel
rather than a field-polygon extraction.

Splitters are deliberately standalone functions, not methods on the dataset,
so they can run on the label table alone — before the heavy patch extraction
in ``MaizeYieldDataset.__init__``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import rasterio.transform
import rasterio.warp
import torch
from torch.utils.data import Dataset

import config

# --- Label table ----------------------------------------------------------

REQUIRED_LABEL_COLUMNS: tuple[str, ...] = (
    "aoi", "season", "field_id", "lon", "lat", "yield_kg_ha", "source",
)


def read_labels(
    csv_path: Path,
    aoi: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Read the ground-truth CSV and validate the schema.

    The CSV is the contract defined in CLAUDE.md §5: one row per (aoi,
    season, field_id) with a ``yield_kg_ha`` column and a free-text
    ``source`` attribution (e.g. "RALS2019", "CFS", "primary_collection").
    """
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} missing required columns: {sorted(missing)}"
        )
    df = df[list(REQUIRED_LABEL_COLUMNS)].copy()
    df["field_id"] = df["field_id"].astype(str)
    if aoi is not None:
        df = df[df["aoi"] == aoi]
    if season is not None:
        df = df[df["season"] == season]
    return df.reset_index(drop=True)


def synthetic_labels(
    n: int = 10,
    aoi: str = "chilanga",
    season: str = "2018_19",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic label table for smoke-testing before RALS arrives.

    Coordinates are drawn uniformly inside ``config.AOIS[aoi]`` so that, when
    projected into the cube's CRS, they actually fall on the raster.
    """
    if aoi not in config.AOIS:
        raise ValueError(f"Unknown AOI '{aoi}'. Known AOIs: {sorted(config.AOIS)}")
    lon_min, lat_min, lon_max, lat_max = config.AOIS[aoi]
    rng = np.random.default_rng(seed)
    lons = rng.uniform(lon_min, lon_max, size=n)
    lats = rng.uniform(lat_min, lat_max, size=n)
    yields = np.clip(rng.normal(2000.0, 500.0, size=n), 500.0, 5000.0)
    return pd.DataFrame({
        "aoi": [aoi] * n,
        "season": [season] * n,
        "field_id": [f"synthetic_{i:04d}" for i in range(n)],
        "lon": lons,
        "lat": lats,
        "yield_kg_ha": yields,
        "source": ["synthetic"] * n,
    })


# --- Dataset --------------------------------------------------------------

@dataclass
class _Sample:
    """One surviving label after pixel projection and validity filtering."""
    field_id: str
    row: int
    col: int
    yield_kg_ha: float
    valid_fraction: float


class MaizeYieldDataset(Dataset):
    """(T, buffer_size, buffer_size, 14) patches paired with yield (kg/ha).

    The dataset opens the HDF5 cube once at construction, reads CRS/transform
    attrs to project (lon, lat) → (row, col), then probes each candidate
    patch to compute its valid-pixel fraction. Labels whose patch falls below
    ``min_valid_fraction`` are dropped permanently. Surviving samples have
    any remaining NaN entries filled with the per-timestep / per-band spatial
    nan-mean so the downstream CNN never sees NaN.

    The HDF5 handle is opened lazily inside ``__getitem__`` so the dataset
    is picklable across DataLoader workers (h5py file handles do not
    survive ``fork``).
    """

    def __init__(
        self,
        cube_path: Path | str,
        labels: pd.DataFrame,
        buffer_size: int = config.PATCH_SIZE,
        min_valid_fraction: float = 0.5,
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        missing = set(REQUIRED_LABEL_COLUMNS) - set(labels.columns)
        if missing:
            raise ValueError(f"labels missing required columns: {sorted(missing)}")

        self.cube_path = Path(cube_path)
        self.buffer_size = int(buffer_size)
        self.min_valid_fraction = float(min_valid_fraction)
        self._h5: h5py.File | None = None

        with h5py.File(self.cube_path, "r") as f:
            cube_aoi = f.attrs["aoi"]
            cube_season = f.attrs["season"]
            if isinstance(cube_aoi, bytes):
                cube_aoi = cube_aoi.decode()
            if isinstance(cube_season, bytes):
                cube_season = cube_season.decode()
            self.aoi = str(cube_aoi)
            self.season = str(cube_season)
            self.crs_wkt = str(f.attrs["crs"])
            self.transform = rasterio.transform.Affine(*f.attrs["transform"])
            self.n_windows, self.height, self.width, self.n_bands = f["cube"].shape
            self.dates = [d.decode("ascii") for d in f["dates"][:]]
            cube_data = f["cube"]  # eagerly read patches below before close

            # Filter labels to this cube's (aoi, season). Belt-and-braces: the
            # caller may have already done this, but a mismatched join is a
            # silent class of bug we want to make impossible.
            df = labels[
                (labels["aoi"] == self.aoi) & (labels["season"] == self.season)
            ].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"No labels match cube (aoi={self.aoi}, season={self.season}). "
                    f"Caller passed {len(labels)} rows for "
                    f"{sorted(labels['aoi'].unique())} / "
                    f"{sorted(labels['season'].unique())}."
                )

            rows, cols = self._project_to_pixels(df["lon"].values, df["lat"].values)

            self.samples: list[_Sample] = []
            for i, (r, c) in enumerate(zip(rows, cols)):
                patch = self._read_patch(cube_data, int(r), int(c))
                vf = float(np.isfinite(patch).mean())
                if vf < self.min_valid_fraction:
                    continue
                self.samples.append(_Sample(
                    field_id=str(df.iloc[i]["field_id"]),
                    row=int(r),
                    col=int(c),
                    yield_kg_ha=float(df.iloc[i]["yield_kg_ha"]),
                    valid_fraction=vf,
                ))

    # -- projection ---------------------------------------------------------

    def _project_to_pixels(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """WGS84 (lon, lat) -> (row, col) in the cube's grid.

        Labels are assumed to be in EPSG:4326. Cubes are typically UTM (per
        GEE's default export CRS for southern Africa), so the reprojection
        step is required for any real cube; for a cube already in WGS84 it's
        a no-op.
        """
        xs, ys = rasterio.warp.transform(
            "EPSG:4326", self.crs_wkt, list(lons), list(lats)
        )
        rows, cols = rasterio.transform.rowcol(self.transform, xs, ys)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    # -- patch extraction ---------------------------------------------------

    def _patch_window(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Window [r0:r1, c0:c1] in cube coords; may extend beyond bounds."""
        half = self.buffer_size // 2
        r0 = row - half
        c0 = col - half
        return r0, c0, r0 + self.buffer_size, c0 + self.buffer_size

    def _read_patch(self, cube: h5py.Dataset, row: int, col: int) -> np.ndarray:
        """Extract (T, B, B, C) patch, NaN-padded for out-of-bounds regions.

        The minimum-valid-fraction filter operates on this NaN-padded patch,
        so edge labels degrade gracefully rather than silently being clipped
        to the cube boundary.
        """
        B = self.buffer_size
        T, H, W, C = cube.shape
        r0, c0, r1, c1 = self._patch_window(row, col)

        rr0, rr1 = max(0, r0), min(H, r1)
        cc0, cc1 = max(0, c0), min(W, c1)
        out = np.full((T, B, B, C), np.nan, dtype=np.float32)
        if rr0 < rr1 and cc0 < cc1:
            sub = cube[:, rr0:rr1, cc0:cc1, :]
            out[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0, :] = sub
        return out

    @staticmethod
    def _fill_nan(patch: np.ndarray) -> np.ndarray:
        """Replace NaN with per-(t, band) spatial nan-mean; 0 if all-NaN."""
        means = np.nanmean(patch, axis=(1, 2), keepdims=True)  # (T, 1, 1, C)
        means = np.where(np.isnan(means), 0.0, means)
        mask = np.isnan(patch)
        if mask.any():
            patch = np.where(mask, np.broadcast_to(means, patch.shape), patch)
        return patch

    # -- torch.Dataset interface -------------------------------------------

    def _h5_handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.cube_path, "r")
        return self._h5

    def __getstate__(self) -> dict:
        # Drop the open HDF5 handle before pickling for DataLoader workers.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cube = self._h5_handle()["cube"]
        patch = self._read_patch(cube, s.row, s.col)
        patch = self._fill_nan(patch)
        return {
            "patch": torch.from_numpy(patch),                      # (T, B, B, C)
            "yield_kg_ha": torch.tensor(s.yield_kg_ha, dtype=torch.float32),
            "field_id": s.field_id,
            "valid_fraction": torch.tensor(s.valid_fraction, dtype=torch.float32),
        }


# --- Splitters ------------------------------------------------------------
# Standalone functions on the label table — not methods on the dataset.
# This lets you split before constructing the (expensive) dataset, and lets
# you reuse the same split across multiple cubes or feature representations.

def spatial_kfold_split(
    labels: pd.DataFrame,
    n_splits: int = 5,
    seed: int = config.SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Spatial K-fold by KMeans cluster on (lon, lat) — no-leakage CV.

    Returns ``n_splits`` (train_idx, val_idx) pairs over the rows of
    ``labels``. Each fold's val set is one geographic cluster, so no field
    in val has a same-cluster neighbour in train — the leakage path called
    out in CLAUDE.md §4 and `docs/ground_truth_plan.md`.

    Uses sklearn's KMeans + GroupKFold; both are already in requirements.
    """
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GroupKFold

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(labels) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} labels, got {len(labels)}"
        )

    coords = labels[["lon", "lat"]].to_numpy()
    n_clusters = min(n_splits, len(labels))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(coords)
    groups = km.labels_

    splitter = GroupKFold(n_splits=n_splits)
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in splitter.split(coords, groups=groups)
    ]


def temporal_holdout_split(
    labels: pd.DataFrame,
    test_season: str | None = None,
    seasons_order: Iterable[str] = tuple(config.SEASONS),
) -> tuple[np.ndarray, np.ndarray]:
    """Most-recent-season hold-out: (train_idx, test_idx).

    ``test_season`` defaults to the latest season in ``seasons_order`` that
    is actually present in the label table. Per CLAUDE.md §4, the model is
    evaluated against this hold-out to defend the RMSE/R² targets.
    """
    present = [s for s in seasons_order if s in set(labels["season"])]
    if not present:
        raise ValueError(
            f"None of seasons_order={list(seasons_order)} appear in labels."
        )
    chosen = test_season if test_season is not None else present[-1]
    if chosen not in set(labels["season"]):
        raise ValueError(f"test_season='{chosen}' not present in labels.")

    is_test = (labels["season"] == chosen).to_numpy()
    test_idx = np.where(is_test)[0].astype(np.int64)
    train_idx = np.where(~is_test)[0].astype(np.int64)
    if len(train_idx) == 0:
        raise ValueError(
            f"All labels are in test_season='{chosen}'; train set is empty."
        )
    return train_idx, test_idx
