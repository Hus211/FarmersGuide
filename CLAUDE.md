# CLAUDE.md — Farmers Guide operating brief

You are working on **Farmers Guide**, an MSc thesis project: AI-powered maize
yield prediction for smallholder farms in Lusaka Province, Zambia.

Read this file fully before acting. It defines scope, workflow, and hard
boundaries. When in doubt, re-read the SCOPE section — staying in scope matters
more than completeness.

---

## 1. SCOPE — read this first, every session

This is **thesis-grade research code**, NOT a production system. Production
comes later as a separate phase. That means:

**DO build:**
- Reproducible data pipeline (GeoTIFF → HDF5 tensors)
- The three models (satellite CNN, ground CNN, fusion) as importable modules
- Training + evaluation code that runs in Colab and produces defensible metrics
- Thesis figures and result tables

**DO NOT build (deferred to production phase — do not touch unless asked):**
- No FastAPI / REST API / web server
- No PostgreSQL / PostGIS / any database (data = HDF5 + CSV files on disk/Drive)
- No Docker / containers
- No Railway / Vercel / cloud deployment
- No CI/CD pipelines / GitHub Actions
- No web frontend / React / Next.js
- No authentication / user management
- No observability / monitoring / logging infrastructure beyond plain logging

If a task seems to require any "DO NOT build" item, STOP and ask the user first.
Default to the simplest thing that produces a defensible thesis result.

---

## 2. THE THREE-ENVIRONMENT WORKFLOW

One repo is the source of truth. Three environments are execution surfaces.
Google Drive is the shared data lake.

1. **Local + Claude Code (you, here):** write/edit all code in `src/`, run the
   GEE Python wrapper, run CPU-side validation and evaluation, generate thesis
   figures, manage git. NO model training here (no GPU).

2. **Google Earth Engine:** the `gee/fg_export.py` wrapper (run from local
   terminal) exports Sentinel-2 GeoTIFFs to a Google Drive folder
   (`farmers_guide_s2`). Already built and working.

3. **Google Colab:** mounts Google Drive, runs
   `pip install git+<repo-url>` to get THIS EXACT codebase, trains models on
   GPU, writes weights + metrics back to Drive. The training notebook is a
   thin shell — all real logic lives in `src/` so Colab and local never drift.

**The consistency rule:** all model/data logic lives in `src/farmers_guide/`
as importable modules. Notebooks and the GEE wrapper only orchestrate. Never
put core logic in a notebook cell — it can't be version-controlled or reused
across environments.

**The loop:** edit code locally → commit + push → Colab re-installs from git →
trains → writes to Drive → pull results locally → evaluate → repeat.

---

## 3. REPO STRUCTURE

```
farmers-guide/
├── CLAUDE.md                 # this file
├── README.md
├── pyproject.toml            # makes repo pip-installable (Colab parity)
├── requirements.txt
├── config.py                 # ALL paths, AOIs, seasons, hyperparams here
├── gee/
│   ├── farmers_guide_s2_export.js
│   └── fg_export.py          # Sentinel-2 batch export wrapper
├── src/farmers_guide/
│   ├── __init__.py
│   ├── data/
│   │   ├── hdf5_builder.py    # GeoTIFF stacks → HDF5 (T,H,W,C) tensors
│   │   ├── dataset.py         # torch Dataset / DataLoader
│   │   └── ground_truth.py    # yield label join (IAPRI/CFS/MoA)
│   ├── models/
│   │   ├── satellite_cnn.py   # temporal CNN, yield + uncertainty head
│   │   ├── ground_cnn.py      # MobileNetV2 transfer, health score
│   │   └── fusion.py          # decision-level weighted fusion
│   ├── train.py               # training loop (called by Colab notebook)
│   └── evaluate.py            # spatial/temporal CV, baselines, metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train.ipynb         # Colab: pip install repo, mount Drive, train
│   └── 03_evaluate_figures.ipynb
└── reports/figures/           # thesis-ready plots
```

---

## 4. ARCHITECTURE (the models)

- **Satellite CNN:** input is a temporal stack of Sentinel-2 composites,
  shape `(T, H, W, C)` where C = 14 bands (9 reflectance + NDVI, EVI, NDRE,
  GCI, NDWI). 3D-conv or ConvLSTM head. Output: maize yield (kg/ha) + an
  uncertainty estimate (MC dropout or quantile heads).
- **Ground CNN:** MobileNetV2, transfer-learned, on smallholder field photos.
  Output: growth-stage / canopy-health embedding + score.
- **Fusion:** decision-level weighted combination of the satellite yield
  estimate and the photo-derived health adjustment. Weights tunable per
  district.
- **Validation:** spatial K-fold (no neighbouring-field leakage) + temporal
  hold-out (most recent season as test). Always report against baselines:
  NDVI-mean OLS, satellite-only, photo-only.
- **Targets:** field-level RMSE < 600 kg/ha, R² > 0.55 on the temporal
  hold-out.

---

## 5. DATA CONTRACTS

- **AOIs:** Lusaka Province + sub-AOIs chilanga, kafue, chongwe. Bounding
  boxes are defined in `config.py` and `gee/fg_export.py` — keep them
  identical across both. If you change one, change both.
- **Seasons:** `2022_23`, `2023_24`, `2024_25` (Zambia maize, Oct–Jun window).
- **GeoTIFF naming:** `fg_<aoi>_<season>_<YYYYMMDD>.tif` in Drive folder
  `farmers_guide_s2`. Band order is fixed: B2,B3,B4,B5,B6,B7,B8,B11,B12,
  NDVI,EVI,NDRE,GCI,NDWI.
- **Ground truth:** a single CSV keyed by `(aoi, season, field_id)` with a
  `yield_kg_ha` column and source attribution. Lives in `data/ground_truth/`.
- **HDF5 cubes:** built by `hdf5_builder.py`, one file per (aoi, season),
  datasets shaped `(n_windows, H, W, 14)` plus a `dates` vector.

---

## 6. CONVENTIONS

- Python 3.11+, PyTorch for models, rasterio for GeoTIFF I/O, h5py for cubes.
- Every config value (paths, hyperparams, AOIs, seasons) goes in `config.py`.
  No magic numbers or hardcoded paths anywhere in `src/`.
- Functions over notebooks. Notebooks call functions; they don't define logic.
- Type hints on all function signatures. Docstrings state shapes for tensors.
- Keep dependencies minimal — every added package is friction in Colab.
- Commit messages: imperative mood, scoped, < 72 chars.
- When unsure whether something is in thesis scope, ask before building.

---

## 7. CURRENT STATE

- GEE export pipeline (`gee/`) is built and validated. Chilanga 2024/25
  exports confirmed producing valid 14-band GeoTIFFs.
- Next priorities (in order): (1) `hdf5_builder.py`, (2) `ground_truth.py`
  label acquisition + join, (3) `dataset.py`, (4) the three model modules,
  (5) `train.py` + the Colab notebook, (6) `evaluate.py` + figures.

Update this section as milestones complete.
