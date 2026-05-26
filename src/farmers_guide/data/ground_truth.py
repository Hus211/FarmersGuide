"""Build the canonical maize-yield label CSV from RALS 2019 microdata.

Anticipatory scaffold — RALS 2019 microdata has not arrived yet (see
``docs/ground_truth_plan.md`` for the access timeline). The ETL structure
is invariant to exact column names; this file owns the schema-mapping
constants at the top so adapting to the real data is a one-block edit.

Output CSV contract (CLAUDE.md §5):
    ``aoi, season, field_id, lon, lat, yield_kg_ha, source``
The single ``source`` value is ``"RALS_2019"``.

Unit of analysis: **buffered point at the household / EA centroid**, not a
field polygon. RALS publishes an EA-centroid GPS with a privacy offset,
which is what the downstream ``MaizeYieldDataset`` consumes via
``_project_to_pixels`` to extract a buffered patch from the Sentinel-2
cube. See ``docs/ground_truth_plan.md`` "Spatial" for the reasoning.

Run:
    python -m farmers_guide.data.ground_truth \\
        --section2 .../section2.dta \\
        --section3 .../section3.dta \\
        --geovars  .../geovars.dta \\
        --season   2018_19
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from farmers_guide import config

logger = logging.getLogger(__name__)


# ============================================================================
# RALS 2019 EXPECTED SCHEMA — VERIFY AGAINST ACTUAL DATA ON RECEIPT
# Based on RALS 2015 documentation (IHSN catalog); RALS 2019 panel
# continuation should match but confirm before running on real data.
# ============================================================================
COL_HHID = "hhid"                  # household identifier (all files)
COL_PLOT_AREA_HA = "area_ha"       # plot area in hectares (Section 2)
COL_PLOT_CROP_CODE = "plot_crop_code"  # primary crop per plot (Section 2) — ADDED
COL_CROP_CODE = "crop_code"        # crop identifier (Section 3)
MAIZE_CROP_CODE = 11               # numeric code for maize per RALS conventions
COL_PRODUCTION_KG = "prod_kg"      # maize production in kg (Section 3)
COL_LAT = "ea_lat"                 # EA centroid latitude (geovariables file)
COL_LON = "ea_lon"                 # EA centroid longitude


# ============================================================================
# Output / filter constants — not in the user-spec list above because they
# describe THIS file's output, not the upstream RALS schema.
# ============================================================================
SOURCE_TAG = "RALS_2019"
SUB_AOIS = ["chilanga", "kafue", "chongwe"]
OUTPUT_COLUMNS = ["aoi", "season", "field_id", "lon", "lat", "yield_kg_ha", "source"]

# Zambian smallholder maize is typically 1500–3000 kg/ha rain-fed.
# Values outside this band are almost always reporting errors (zeros for
# total crop loss, or unit confusion between bags and kg).
MIN_YIELD_KG_HA = 100.0
MAX_YIELD_KG_HA = 8000.0


# ============================================================================
# I/O — auto-detect file format from extension
# ============================================================================

def _load_any(path: Path | str) -> pd.DataFrame:
    """Read .dta (IAPRI's default), .csv, or .parquet by file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(p, convert_categoricals=False)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(
        f"Unsupported extension '{suffix}' for {p}. "
        "Expected .dta, .csv, or .parquet."
    )


# ============================================================================
# AOI assignment
# ============================================================================

def _assign_aoi(lon: float, lat: float) -> str | None:
    """First-match AOI bbox assignment over the three sub-AOIs.

    The Chilanga and Kafue AOIs overlap by ~0.05° on one corner. Iteration
    order ``[chilanga, kafue, chongwe]`` gives Chilanga priority in that
    overlap — defensible because Chilanga is the smaller, denser district
    and a point that's in both is more characteristic of Chilanga's
    sub-Lusaka peri-urban farming pattern than Kafue's broader basin.
    """
    for name in SUB_AOIS:
        lon_min, lat_min, lon_max, lat_max = config.AOIS[name]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name
    return None


# ============================================================================
# Core ETL (operates on DataFrames, not paths — testable via _smoke_pipeline)
# ============================================================================

def _build_from_frames(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    geovars: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """ETL pipeline core, in the order specified in CLAUDE.md §5 / spec:

      1. (Loading done by caller.)
      2. Filter to maize-cultivating households.
      3. Aggregate maize production + maize plot area per household; derive
         yield_kg_ha as the ratio.
      4. Drop households with implausible yields (sanity bounds).
      5. Join geovariables to get (lon, lat).
      6. Assign AOI via bbox; drop households outside chilanga/kafue/chongwe.
      7. Emit the canonical (aoi, season, field_id, lon, lat,
         yield_kg_ha, source) DataFrame.
    """
    for df, name, required in [
        (s2, "section2", [COL_HHID, COL_PLOT_AREA_HA, COL_PLOT_CROP_CODE]),
        (s3, "section3", [COL_HHID, COL_CROP_CODE, COL_PRODUCTION_KG]),
        (geovars, "geovars", [COL_HHID, COL_LON, COL_LAT]),
    ]:
        missing = set(required) - set(df.columns)
        if missing:
            raise KeyError(
                f"{name}: missing required columns {sorted(missing)}. "
                "Update the schema constants at the top of ground_truth.py "
                "if the RALS file uses different names."
            )

    # 2 + 3a. Maize production aggregated per household.
    maize_prod = (
        s3[s3[COL_CROP_CODE] == MAIZE_CROP_CODE]
        .groupby(COL_HHID, as_index=False)[COL_PRODUCTION_KG].sum()
        .rename(columns={COL_PRODUCTION_KG: "maize_prod_kg"})
    )
    # 2 + 3b. Maize plot area aggregated per household.
    # Only plots whose primary crop is maize count toward maize area. This
    # is the safest reading; multi-crop attribution (crop_a/b/c) can be
    # added later by listing all crop columns in COL_PLOT_CROP_CODE.
    maize_area = (
        s2[s2[COL_PLOT_CROP_CODE] == MAIZE_CROP_CODE]
        .groupby(COL_HHID, as_index=False)[COL_PLOT_AREA_HA].sum()
        .rename(columns={COL_PLOT_AREA_HA: "maize_area_ha"})
    )

    # Inner join: keep only households reporting BOTH maize production and
    # at least one maize plot. A household with one without the other is a
    # data-quality red flag, not a usable yield row.
    df = maize_prod.merge(maize_area, on=COL_HHID, how="inner")
    n_after_join = len(df)
    logger.info(
        "maize households: prod=%d  area=%d  joined=%d",
        len(maize_prod), len(maize_area), n_after_join,
    )

    # 3c. Derive yield.
    # Guard against div-by-zero — a maize plot with reported area 0 is a
    # measurement error, not a tiny field.
    df = df[df["maize_area_ha"] > 0].copy()
    df["yield_kg_ha"] = df["maize_prod_kg"] / df["maize_area_ha"]

    # 4. Sanity-bound filter.
    n_before_bounds = len(df)
    in_bounds = (df["yield_kg_ha"] >= MIN_YIELD_KG_HA) & (df["yield_kg_ha"] <= MAX_YIELD_KG_HA)
    n_dropped = int((~in_bounds).sum())
    if n_dropped:
        logger.info(
            "sanity-bounds: dropped %d / %d rows outside [%.0f, %.0f] kg/ha",
            n_dropped, n_before_bounds, MIN_YIELD_KG_HA, MAX_YIELD_KG_HA,
        )
    df = df[in_bounds].copy()

    # 5. Join geovariables (lon, lat) — EA-centroid with privacy offset.
    df = df.merge(
        geovars[[COL_HHID, COL_LON, COL_LAT]],
        on=COL_HHID, how="left",
    )
    n_missing_geo = int(df[[COL_LON, COL_LAT]].isna().any(axis=1).sum())
    if n_missing_geo:
        logger.info("dropping %d households with no geovariables", n_missing_geo)
        df = df.dropna(subset=[COL_LON, COL_LAT])

    # 6. AOI assignment via bbox. Vectorised would be nicer but the AOI
    # count is fixed at 3 and the row count is in the hundreds; the per-
    # row apply is fine and trivially auditable.
    df["aoi"] = df.apply(
        lambda r: _assign_aoi(float(r[COL_LON]), float(r[COL_LAT])),
        axis=1,
    )
    n_outside = int(df["aoi"].isna().sum())
    if n_outside:
        logger.info(
            "dropping %d households outside %s bboxes",
            n_outside, SUB_AOIS,
        )
        df = df.dropna(subset=["aoi"])

    # 7. Final shape.
    out = pd.DataFrame({
        "aoi":          df["aoi"].astype(str).values,
        "season":       season,
        "field_id":     df[COL_HHID].astype(str).values,
        "lon":          df[COL_LON].astype(np.float64).values,
        "lat":          df[COL_LAT].astype(np.float64).values,
        "yield_kg_ha":  df["yield_kg_ha"].astype(np.float64).values,
        "source":       SOURCE_TAG,
    })
    logger.info(
        "final: %d rows  (chilanga=%d kafue=%d chongwe=%d)",
        len(out),
        int((out["aoi"] == "chilanga").sum()),
        int((out["aoi"] == "kafue").sum()),
        int((out["aoi"] == "chongwe").sum()),
    )
    return out


# ============================================================================
# Public API
# ============================================================================

def build_ground_truth(
    section2_path: Path | str,
    section3_path: Path | str,
    geovars_path: Path | str,
    season: str = config.SUPERVISED_SEASON,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build the canonical maize-yield label CSV from RALS microdata.

    Args:
        section2_path: Section 2 file (Farm Land and Use, plot-level).
        section3_path: Section 3 file (Crop Sales / Production, hh × crop).
        geovars_path: geovariables file (EA-centroid coordinates).
        season: agricultural season key; defaults to the configured
            supervised season (2018_19). Stored in every output row.
        output_path: where to write the CSV. Defaults to
            ``config.GROUND_TRUTH_CSV``. Pass an explicit Path to
            override (e.g. for sandboxed test runs).

    Returns:
        The canonical DataFrame. Also written to ``output_path``.
    """
    s2 = _load_any(section2_path)
    s3 = _load_any(section3_path)
    geovars = _load_any(geovars_path)
    df = _build_from_frames(s2, s3, geovars, season)
    validate_labels(df)
    out = Path(output_path) if output_path is not None else config.GROUND_TRUTH_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("wrote %d rows to %s", len(df), out)
    return df


def validate_labels(df: pd.DataFrame) -> None:
    """Schema and range validation. Raises ``ValueError`` on any violation.

    Errors are accumulated and reported together — one validation call
    surfaces all problems with the data rather than one at a time.
    """
    errors: list[str] = []

    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        # Skip further checks: downstream column references would themselves
        # raise KeyError and obscure the real problem.
        raise ValueError(
            f"validate_labels: missing required columns {sorted(missing)}"
        )

    if len(df) == 0:
        raise ValueError("validate_labels: zero rows in label table")

    bad_y = df[(df["yield_kg_ha"] < MIN_YIELD_KG_HA) |
               (df["yield_kg_ha"] > MAX_YIELD_KG_HA)]
    if len(bad_y):
        errors.append(
            f"{len(bad_y)} rows have yield_kg_ha outside "
            f"[{MIN_YIELD_KG_HA}, {MAX_YIELD_KG_HA}]"
        )

    bad_aoi = df[~df["aoi"].isin(SUB_AOIS)]
    if len(bad_aoi):
        errors.append(
            f"{len(bad_aoi)} rows have aoi outside {SUB_AOIS} "
            f"(got: {sorted(bad_aoi['aoi'].unique().tolist())})"
        )

    bad_source = df[df["source"] != SOURCE_TAG]
    if len(bad_source):
        errors.append(
            f"{len(bad_source)} rows have source != '{SOURCE_TAG}'"
        )

    # Lusaka Province envelope as sanity check on lon/lat.
    lon_min, lat_min, lon_max, lat_max = config.AOIS["lusaka_province"]
    bad_geo = df[
        (df["lon"] < lon_min) | (df["lon"] > lon_max) |
        (df["lat"] < lat_min) | (df["lat"] > lat_max)
    ]
    if len(bad_geo):
        errors.append(
            f"{len(bad_geo)} rows have lon/lat outside Lusaka Province envelope "
            f"({lon_min}..{lon_max}, {lat_min}..{lat_max})"
        )

    non_str_fid = df[~df["field_id"].map(lambda v: isinstance(v, str))]
    if len(non_str_fid):
        errors.append(f"{len(non_str_fid)} rows have non-string field_id")

    if errors:
        raise ValueError(
            "validate_labels failed:\n  - " + "\n  - ".join(errors)
        )


# ============================================================================
# Smoke test — synthesise three in-memory DataFrames, end-to-end ETL
# ============================================================================

def _smoke_pipeline(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    geovars: pd.DataFrame,
    season: str = config.SUPERVISED_SEASON,
) -> pd.DataFrame:
    """In-memory ETL — same core as ``build_ground_truth`` but no file I/O.

    Lets the smoke test exercise the full pipeline without writing
    anything to disk; ``build_ground_truth``'s only extra responsibility
    is file loading and CSV writing.
    """
    df = _build_from_frames(s2, s3, geovars, season)
    validate_labels(df)
    return df


def _synth_smoke_data(
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Synthesise plausible RALS-shaped DataFrames for the smoke test.

    Distributes ~17 households across each of chilanga / kafue / chongwe,
    1–2 plots per household, ~80% maize plots. Yields drawn from
    N(2000, 500) kg/ha clipped to [800, 4000] — realistic for Zambian
    smallholders. Two outlier households are injected to exercise the
    sanity-bounds filter (one < MIN_YIELD, one > MAX_YIELD).

    Returns (section2_df, section3_df, geovars_df, audit) where ``audit``
    carries the truth-table the smoke test asserts against.
    """
    rng = np.random.default_rng(seed)
    s2_rows: list[dict] = []
    s3_rows: list[dict] = []
    geo_rows: list[dict] = []
    audit: dict = {"by_aoi": {a: [] for a in SUB_AOIS}, "outliers": []}

    next_hhid = 1
    for aoi in SUB_AOIS:
        lon_min, lat_min, lon_max, lat_max = config.AOIS[aoi]
        # Pull the bbox in a bit so we don't sample on the overlap seam
        # between chilanga and kafue — the smoke test's spot-check
        # assumes the aoi assignment is unambiguous.
        margin = 0.02
        for _ in range(17):
            hhid = next_hhid
            next_hhid += 1
            lon = float(rng.uniform(lon_min + margin, lon_max - margin))
            lat = float(rng.uniform(lat_min + margin, lat_max - margin))
            geo_rows.append({COL_HHID: hhid, COL_LON: lon, COL_LAT: lat})
            audit["by_aoi"][aoi].append(hhid)

            n_plots = int(rng.integers(1, 3))
            maize_area = 0.0
            for _ in range(n_plots):
                area = float(rng.uniform(0.5, 3.0))
                is_maize = bool(rng.random() < 0.8)
                code = MAIZE_CROP_CODE if is_maize else 99
                s2_rows.append({
                    COL_HHID: hhid,
                    COL_PLOT_AREA_HA: area,
                    COL_PLOT_CROP_CODE: code,
                })
                if is_maize:
                    maize_area += area
            if maize_area > 0:
                yld = float(np.clip(rng.normal(2000.0, 500.0), 800.0, 4000.0))
                s3_rows.append({
                    COL_HHID: hhid,
                    COL_CROP_CODE: MAIZE_CROP_CODE,
                    COL_PRODUCTION_KG: yld * maize_area,
                })

    # Outliers: one absurdly high, one absurdly low — both should be
    # filtered by the sanity bound.
    for tag, target_yield in [("too_high", 12_000.0), ("too_low", 40.0)]:
        hhid = next_hhid
        next_hhid += 1
        aoi = "chilanga"
        lon_min, lat_min, lon_max, lat_max = config.AOIS[aoi]
        lon = float((lon_min + lon_max) / 2)
        lat = float((lat_min + lat_max) / 2)
        geo_rows.append({COL_HHID: hhid, COL_LON: lon, COL_LAT: lat})
        s2_rows.append({
            COL_HHID: hhid, COL_PLOT_AREA_HA: 1.0,
            COL_PLOT_CROP_CODE: MAIZE_CROP_CODE,
        })
        s3_rows.append({
            COL_HHID: hhid, COL_CROP_CODE: MAIZE_CROP_CODE,
            COL_PRODUCTION_KG: target_yield * 1.0,
        })
        audit["outliers"].append({"hhid": hhid, "tag": tag, "yield": target_yield})

    return (
        pd.DataFrame(s2_rows),
        pd.DataFrame(s3_rows),
        pd.DataFrame(geo_rows),
        audit,
    )


def _smoke_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=== ground_truth.py smoke test ===")

    s2, s3, geovars, audit = _synth_smoke_data(seed=0)
    logger.info(
        "synth shapes: section2=%s  section3=%s  geovars=%s",
        s2.shape, s3.shape, geovars.shape,
    )

    df = _smoke_pipeline(s2, s3, geovars)
    logger.info("output: %d rows, columns=%s", len(df), list(df.columns))

    # ---- 1. Schema -------------------------------------------------------
    assert list(df.columns) == OUTPUT_COLUMNS, f"unexpected columns: {list(df.columns)}"

    # ---- 2. All sub-AOIs represented ------------------------------------
    seen_aois = set(df["aoi"].unique())
    assert seen_aois <= set(SUB_AOIS), f"unknown aoi: {seen_aois - set(SUB_AOIS)}"
    assert seen_aois == set(SUB_AOIS), \
        f"missing aoi in output: {set(SUB_AOIS) - seen_aois}"

    # ---- 3. Bbox spot-check on a chongwe household ----------------------
    # Chongwe doesn't overlap with chilanga/kafue, so the assignment is
    # unambiguous — pick the first synthesised chongwe household.
    chongwe_hhid = audit["by_aoi"]["chongwe"][0]
    row = df[df["field_id"] == str(chongwe_hhid)]
    assert len(row) == 1, f"chongwe spot-check hhid={chongwe_hhid} missing"
    assert row.iloc[0]["aoi"] == "chongwe", \
        f"chongwe hhid={chongwe_hhid} got aoi={row.iloc[0]['aoi']}"
    logger.info(
        "  ✓ bbox spot-check: hhid=%s -> aoi=chongwe (lon=%.3f lat=%.3f)",
        chongwe_hhid, row.iloc[0]["lon"], row.iloc[0]["lat"],
    )

    # ---- 4. Sanity-bound filter actually fired --------------------------
    for outlier in audit["outliers"]:
        assert str(outlier["hhid"]) not in df["field_id"].values, \
            f"outlier hhid={outlier['hhid']} ({outlier['tag']}) leaked through"
    logger.info("  ✓ sanity bounds dropped both outlier households")

    # ---- 5. validate_labels passes on a clean output -------------------
    validate_labels(df)
    logger.info("  ✓ validate_labels passed")

    # ---- 6. validate_labels raises on a corrupted copy ------------------
    corrupted = df.copy()
    corrupted.loc[corrupted.index[0], "yield_kg_ha"] = 99_999.0
    try:
        validate_labels(corrupted)
    except ValueError as exc:
        logger.info("  ✓ validate_labels correctly rejected corrupted row: %s",
                    str(exc).splitlines()[0])
    else:
        raise AssertionError("validate_labels did not raise on corrupted yield")

    logger.info("smoke test OK")


# ============================================================================
# CLI
# ============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ground_truth",
        description="Build the canonical maize-yield label CSV from RALS microdata.",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Run the in-memory smoke test and exit.")
    p.add_argument("--section2", type=Path,
                   help="RALS Section 2 file (.dta/.csv/.parquet).")
    p.add_argument("--section3", type=Path,
                   help="RALS Section 3 file (.dta/.csv/.parquet).")
    p.add_argument("--geovars", type=Path,
                   help="RALS geovariables file (.dta/.csv/.parquet).")
    p.add_argument("--season", default=config.SUPERVISED_SEASON,
                   help="Season key written into every output row.")
    p.add_argument("--output", type=Path, default=None,
                   help="Override config.GROUND_TRUTH_CSV.")
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
    if not (args.section2 and args.section3 and args.geovars):
        logger.error(
            "Real run requires --section2, --section3, --geovars (or use --smoke)."
        )
        return 2
    try:
        build_ground_truth(
            section2_path=args.section2,
            section3_path=args.section3,
            geovars_path=args.geovars,
            season=args.season,
            output_path=args.output,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
