"""GeoTIFF stacks -> HDF5 tensors of shape (n_windows, H, W, 14).

Reads ``fg_<aoi>_<season>_<YYYYMMDD>.tif`` from ``config.S2_EXPORT_DIR``,
stacks the per-date 14-band composites chronologically and writes one HDF5
file per (aoi, season) into ``config.HDF5_DIR`` following the data contract
in CLAUDE.md section 5.

HDF5 layout (one file: ``fg_<aoi>_<season>.h5``):

    /cube   float32, shape (n_windows, H, W, 14)   # chronological composites
    /dates  |S10,    shape (n_windows,)            # ISO YYYY-MM-DD per window

    attrs on root:
        aoi:       str       e.g. "chilanga"
        season:    str       e.g. "2018_19"
        bands:     str[14]   exact band order from config.BANDS
        n_windows, height, width, n_bands : int
        crs:       str       CRS as WKT (geospatial provenance)
        transform: float64[6] affine transform of the source GeoTIFFs
        source_files: str[n_windows] basenames of input GeoTIFFs (audit)

Run as a module from the repo root::

    python -m farmers_guide.data.hdf5_builder --aoi chilanga --season 2018_19
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio

import config

logger = logging.getLogger(__name__)

# fg_<aoi>_<season>_<YYYYMMDD>.tif  — season carries an underscore (e.g. 2018_19)
_FILENAME_RE = re.compile(
    r"^fg_(?P<aoi>[a-z_]+?)_(?P<season>\d{4}_\d{2})_(?P<date>\d{8})\.tif$"
)


def _discover_geotiffs(aoi: str, season: str, export_dir: Path) -> list[tuple[datetime, Path]]:
    """Return [(date, path), ...] for files matching (aoi, season), sorted by date.

    Filters by exact regex match on the contracted filename so that an AOI
    prefix collision (e.g. ``chongwe`` vs ``chongwe_east``) cannot leak
    files across cubes.
    """
    if not export_dir.is_dir():
        raise FileNotFoundError(f"S2 export directory not found: {export_dir}")

    matches: list[tuple[datetime, Path]] = []
    for path in export_dir.iterdir():
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        if m.group("aoi") != aoi or m.group("season") != season:
            continue
        try:
            date = datetime.strptime(m.group("date"), "%Y%m%d")
        except ValueError:
            logger.warning("Skipping file with unparseable date: %s", path.name)
            continue
        matches.append((date, path))

    matches.sort(key=lambda x: x[0])

    dates_seen = [d for d, _ in matches]
    if len(set(dates_seen)) != len(dates_seen):
        dupes = sorted({d.isoformat() for d in dates_seen if dates_seen.count(d) > 1})
        raise ValueError(
            f"Duplicate composite dates for (aoi={aoi}, season={season}): {dupes}. "
            "Re-export to a clean directory or de-duplicate before building."
        )
    return matches


def _read_window(path: Path, expected_bands: int) -> tuple[np.ndarray, dict]:
    """Read one GeoTIFF as (H, W, C) float32 and return (array, metadata).

    Metadata captures CRS/transform/shape so callers can verify cross-file
    consistency before stacking.
    """
    with rasterio.open(path) as src:
        if src.count != expected_bands:
            raise ValueError(
                f"{path.name}: expected {expected_bands} bands, got {src.count}"
            )
        arr = src.read()  # (C, H, W)
        meta = {
            "height": src.height,
            "width": src.width,
            "crs": src.crs.to_wkt() if src.crs else "",
            "transform": tuple(src.transform)[:6],
        }
    arr = np.moveaxis(arr, 0, -1).astype(np.float32, copy=False)  # (H, W, C)
    return arr, meta


def build_cube(
    aoi: str,
    season: str,
    export_dir: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Build one HDF5 cube for ``(aoi, season)`` and return the output path.

    Args:
        aoi: AOI key, must be in ``config.AOIS``.
        season: season key, must be in ``config.SEASONS``.
        export_dir: source GeoTIFF directory; defaults to ``config.S2_EXPORT_DIR``.
        output_dir: HDF5 destination; defaults to ``config.HDF5_DIR``.
        overwrite: if False and output file exists, raise ``FileExistsError``.

    Returns:
        Path to the written HDF5 file.

    The output dataset ``/cube`` has shape ``(n_windows, H, W, 14)`` and
    dtype float32. Composites are written in ascending chronological order;
    the ``/dates`` vector aligns 1:1 with the first axis.
    """
    if aoi not in config.AOIS:
        raise ValueError(f"Unknown AOI '{aoi}'. Known AOIs: {sorted(config.AOIS)}")
    if season not in config.SEASONS:
        raise ValueError(f"Unknown season '{season}'. Known seasons: {config.SEASONS}")

    export_dir = Path(export_dir) if export_dir is not None else config.S2_EXPORT_DIR
    output_dir = Path(output_dir) if output_dir is not None else config.HDF5_DIR

    pairs = _discover_geotiffs(aoi, season, export_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No GeoTIFFs matched fg_{aoi}_{season}_*.tif in {export_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fg_{aoi}_{season}.h5"
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass overwrite=True (or --overwrite) to replace."
        )

    n_windows = len(pairs)
    logger.info(
        "Building cube for aoi=%s season=%s from %d GeoTIFFs", aoi, season, n_windows
    )

    # Probe the first raster to fix shape and write the dataset incrementally;
    # this keeps peak memory at one composite rather than the full T-stack.
    first_arr, ref_meta = _read_window(pairs[0][1], config.N_BANDS)
    h, w = ref_meta["height"], ref_meta["width"]

    tmp_path = out_path.with_suffix(".h5.tmp")
    try:
        with h5py.File(tmp_path, "w") as f:
            cube = f.create_dataset(
                "cube",
                shape=(n_windows, h, w, config.N_BANDS),
                dtype=np.float32,
                chunks=(1, min(h, 256), min(w, 256), config.N_BANDS),
                compression="gzip",
                compression_opts=4,
            )
            dates_ds = f.create_dataset(
                "dates", shape=(n_windows,), dtype="S10"
            )
            files_ds = f.create_dataset(
                "source_files",
                shape=(n_windows,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            cube[0] = first_arr
            dates_ds[0] = pairs[0][0].strftime("%Y-%m-%d").encode("ascii")
            files_ds[0] = pairs[0][1].name

            for i, (date, path) in enumerate(pairs[1:], start=1):
                arr, meta = _read_window(path, config.N_BANDS)
                if (meta["height"], meta["width"]) != (h, w):
                    raise ValueError(
                        f"{path.name}: shape {(meta['height'], meta['width'])} "
                        f"differs from reference {(h, w)}; cannot stack."
                    )
                if meta["crs"] != ref_meta["crs"]:
                    raise ValueError(
                        f"{path.name}: CRS differs from reference; cannot stack."
                    )
                cube[i] = arr
                dates_ds[i] = date.strftime("%Y-%m-%d").encode("ascii")
                files_ds[i] = path.name
                logger.debug("  [%d/%d] %s", i + 1, n_windows, path.name)

            f.attrs["aoi"] = aoi
            f.attrs["season"] = season
            f.attrs["bands"] = np.array(config.BANDS, dtype="S8")
            f.attrs["n_windows"] = n_windows
            f.attrs["height"] = h
            f.attrs["width"] = w
            f.attrs["n_bands"] = config.N_BANDS
            f.attrs["crs"] = ref_meta["crs"]
            f.attrs["transform"] = np.array(ref_meta["transform"], dtype=np.float64)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(out_path)
    logger.info(
        "Wrote %s  (n_windows=%d, H=%d, W=%d, C=%d)",
        out_path, n_windows, h, w, config.N_BANDS,
    )
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hdf5_builder",
        description="Stack Sentinel-2 GeoTIFF composites into one HDF5 cube per (aoi, season).",
    )
    p.add_argument("--aoi", required=True, choices=sorted(config.AOIS))
    p.add_argument("--season", required=True, choices=config.SEASONS)
    p.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Override config.S2_EXPORT_DIR (useful for local mirrors of Drive).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override config.HDF5_DIR.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cube file at the output path.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        build_cube(
            aoi=args.aoi,
            season=args.season,
            export_dir=args.export_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
