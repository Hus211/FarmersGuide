#!/usr/bin/env python3
"""
Farmers Guide — Sentinel-2 batch export driver
================================================

Programmatic counterpart to farmers_guide_s2_export.js. Builds the same
season x AOI export tasks but auto-starts them, supports resume (skip
already-running/completed tasks), and provides task monitoring.

Setup
-----
    pip install earthengine-api click tqdm
    earthengine authenticate            # one-time browser auth

Typical use
-----------
    # 1. Validate one combo, no-op (no tasks started)
    python fg_export.py queue --seasons 2024_25 --aois chilanga --dry-run

    # 2. Fire the validation export
    python fg_export.py queue --seasons 2024_25 --aois chilanga

    # 3. Once that one finishes cleanly, fire the rest of the matrix
    python fg_export.py queue \\
        --seasons 2022_23,2023_24,2024_25 \\
        --aois chilanga,kafue,chongwe

    # 4. Watch progress (Ctrl-C to detach; tasks keep running)
    python fg_export.py monitor

    # 5. Snapshot current state without polling
    python fg_export.py status

Idempotent — re-running the same queue command will skip any task whose
description matches one already READY, RUNNING, or COMPLETED.
"""

import logging
import time

import click
import ee
from tqdm import tqdm


# ============================================================================
# CONFIG — must match farmers_guide_s2_export.js exactly
# ============================================================================

EXPORT_DRIVE_FOLDER = "farmers_guide_s2"
CLEAR_THRESHOLD = 0.60
CS_BAND = "cs"
SCALE_M = 10
CRS = "EPSG:32735"
MAX_PIXELS = int(1e10)

BANDS_OUT = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12",
    "NDVI", "EVI", "NDRE", "GCI", "NDWI",
]

LUSAKA_PROVINCE_BBOX = [27.50, -16.50, 29.50, -15.00]

AOIS = {
    "chilanga":         [28.15, -15.65, 28.40, -15.45],
    "kafue":            [28.05, -15.95, 28.35, -15.60],
    "chongwe":          [28.50, -15.55, 29.05, -15.10],
    "lusaka_province":  LUSAKA_PROVINCE_BBOX,
}

SEASONS = {
    # Supervised training season — pairs with RALS 2019 labels (2018/19).
    "2018_19": ("2018-10-15", "2019-06-15"),
    # Inference / weak-label seasons.
    "2022_23": ("2022-10-15", "2023-06-15"),
    "2023_24": ("2023-10-15", "2024-06-15"),
    "2024_25": ("2024-10-15", "2025-06-15"),
}

POLL_INTERVAL_S = 30
TASK_PREFIX = "fg_"


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fg-export")


# ============================================================================
# GEE pipeline (mirrors the JS exactly)
# ============================================================================

def init_ee(project: str | None = None) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def aoi_geom(name: str) -> ee.Geometry:
    return ee.Geometry.Rectangle(AOIS[name])


def _mask_clouds(img: ee.Image) -> ee.Image:
    return img.updateMask(img.select(CS_BAND).gte(CLEAR_THRESHOLD))


def _add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")
    evi = img.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    gci = img.expression(
        "(NIR / GREEN) - 1",
        {"NIR": img.select("B8"), "GREEN": img.select("B3")},
    ).rename("GCI")
    return img.addBands([ndvi, ndre, ndwi, evi, gci])


def build_season_collection(season_key: str) -> ee.ImageCollection:
    start, end = SEASONS[season_key]
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    province = ee.Geometry.Rectangle(LUSAKA_PROVINCE_BBOX)
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(province)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .linkCollection(cs_plus, [CS_BAND])
        .map(_mask_clouds)
        .map(_add_indices)
    )


def ten_day_composites(coll: ee.ImageCollection, season_key: str) -> ee.ImageCollection:
    start, end = SEASONS[season_key]
    start_d = ee.Date(start)
    end_d = ee.Date(end)
    n_windows = end_d.difference(start_d, "day").divide(10).ceil()
    indices = ee.List.sequence(0, n_windows.subtract(1))

    def _make_window(i):
        i = ee.Number(i)
        win_start = start_d.advance(i.multiply(10), "day")
        win_end = win_start.advance(10, "day")
        composite = coll.filterDate(win_start, win_end).median()
        return (
            composite
            .set("system:time_start", win_start.millis())
            .set("window_index", i)
            .set("season", season_key)
        )

    return ee.ImageCollection.fromImages(indices.map(_make_window))


# ============================================================================
# Task management
# ============================================================================

def existing_task_names(states: tuple = ("READY", "RUNNING", "COMPLETED")) -> set:
    """Return descriptions of tasks already in any of the given states."""
    names = set()
    for t in ee.batch.Task.list():
        s = t.status()
        if s.get("state") in states:
            desc = s.get("description") or ""
            if desc.startswith(TASK_PREFIX):
                names.add(desc)
    return names


def queue_combo(
    season_key: str,
    aoi_key: str,
    dry_run: bool,
    skip_existing: bool,
) -> list[str]:
    """Build and (unless dry_run) start tasks for one season × AOI combo."""
    aoi = aoi_geom(aoi_key)
    coll = build_season_collection(season_key)
    composites = ten_day_composites(coll, season_key)

    # Batch-fetch all window date strings in ONE server call
    date_strs = (
        composites
        .aggregate_array("system:time_start")
        .map(lambda ms: ee.Date(ms).format("YYYYMMdd"))
        .getInfo()
    )
    n = len(date_strs)
    img_list = composites.toList(n)
    existing = existing_task_names() if skip_existing else set()

    started: list[str] = []
    skipped = 0

    desc = f"{aoi_key}/{season_key}"
    for i in tqdm(range(n), desc=desc, ncols=80):
        date_str = date_strs[i]
        name = f"{TASK_PREFIX}{aoi_key}_{season_key}_{date_str}"

        if name in existing:
            skipped += 1
            continue

        img = ee.Image(img_list.get(i)).select(BANDS_OUT).clip(aoi).toFloat()
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=name,
            folder=EXPORT_DRIVE_FOLDER,
            fileNamePrefix=name,
            region=aoi,
            scale=SCALE_M,
            crs=CRS,
            maxPixels=MAX_PIXELS,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        if not dry_run:
            task.start()
        started.append(name)

    verb = "built" if dry_run else "started"
    log.info(f"  {desc}: {len(started)} {verb}, {skipped} skipped (existing)")
    return started


def monitor_tasks(prefix: str = TASK_PREFIX, poll_s: int = POLL_INTERVAL_S) -> None:
    """Poll until no fg_ task is in a non-terminal state."""
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}
    while True:
        ours = []
        for t in ee.batch.Task.list():
            s = t.status()
            if (s.get("description") or "").startswith(prefix):
                ours.append(s)

        if not ours:
            log.info("No matching tasks.")
            return

        counts: dict[str, int] = {}
        for s in ours:
            counts[s.get("state", "UNKNOWN")] = counts.get(s.get("state", "UNKNOWN"), 0) + 1

        active = sum(v for k, v in counts.items() if k not in terminal)
        log.info(f"  {dict(sorted(counts.items()))}  active={active}")

        if active == 0:
            log.info("All tasks terminal.")
            return
        time.sleep(poll_s)


# ============================================================================
# CLI
# ============================================================================

@click.group()
@click.option("--project", default=None, help="GEE Cloud project ID (recommended)")
@click.pass_context
def cli(ctx, project):
    """Farmers Guide — Sentinel-2 batch export driver."""
    init_ee(project)
    ctx.ensure_object(dict)


@cli.command()
@click.option(
    "--seasons", default="2022_23,2023_24,2024_25",
    help="Comma-separated season keys",
)
@click.option(
    "--aois", default="chilanga,kafue,chongwe",
    help="Comma-separated AOI keys",
)
@click.option("--dry-run", is_flag=True, help="Build tasks without starting them")
@click.option(
    "--skip-existing/--no-skip-existing", default=True,
    help="Skip tasks already READY/RUNNING/COMPLETED",
)
def queue(seasons, aois, dry_run, skip_existing):
    """Build and start export tasks for season × AOI combinations."""
    season_keys = [s.strip() for s in seasons.split(",") if s.strip()]
    aoi_keys = [a.strip() for a in aois.split(",") if a.strip()]

    bad_s = [s for s in season_keys if s not in SEASONS]
    bad_a = [a for a in aoi_keys if a not in AOIS]
    if bad_s:
        raise click.UsageError(f"Unknown seasons: {bad_s}. Valid: {list(SEASONS)}")
    if bad_a:
        raise click.UsageError(f"Unknown AOIs: {bad_a}. Valid: {list(AOIS)}")

    log.info(f"Seasons: {season_keys}")
    log.info(f"AOIs:    {aoi_keys}")
    log.info(f"Dry run: {dry_run}  |  Skip existing: {skip_existing}")

    total = 0
    for s in season_keys:
        for a in aoi_keys:
            total += len(queue_combo(s, a, dry_run, skip_existing))

    verb = "built" if dry_run else "started"
    log.info(f"\nTotal tasks {verb}: {total}")


@cli.command()
@click.option("--prefix", default=TASK_PREFIX)
@click.option("--poll", default=POLL_INTERVAL_S, help="Poll interval seconds")
def monitor(prefix, poll):
    """Watch active tasks until all are terminal."""
    monitor_tasks(prefix, poll)


@cli.command()
@click.option("--prefix", default=TASK_PREFIX)
@click.option("--show", default=5, help="How many task names to show per state")
def status(prefix, show):
    """One-shot snapshot of task states matching prefix."""
    by_state: dict[str, list[str]] = {}
    for t in ee.batch.Task.list():
        s = t.status()
        desc = s.get("description") or ""
        if desc.startswith(prefix):
            by_state.setdefault(s.get("state", "UNKNOWN"), []).append(desc)

    if not by_state:
        log.info(f"No tasks found with prefix '{prefix}'.")
        return

    for state, names in sorted(by_state.items()):
        log.info(f"{state:<12} {len(names)}")
        for n in sorted(names)[:show]:
            log.info(f"   {n}")
        if len(names) > show:
            log.info(f"   ... and {len(names) - show} more")


@cli.command()
@click.option("--prefix", default=TASK_PREFIX)
@click.option(
    "--state", default="READY",
    type=click.Choice(["READY", "RUNNING", "FAILED"]),
    help="Which state to cancel",
)
@click.confirmation_option(prompt="This cancels live tasks. Proceed?")
def cancel(prefix, state):
    """Cancel tasks matching prefix and state."""
    cancelled = 0
    for t in ee.batch.Task.list():
        s = t.status()
        desc = s.get("description") or ""
        if desc.startswith(prefix) and s.get("state") == state:
            t.cancel()
            cancelled += 1
    log.info(f"Cancelled {cancelled} tasks (state={state})")


if __name__ == "__main__":
    cli(obj={})
