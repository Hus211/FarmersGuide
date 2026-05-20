# Farmers Guide

Maize yield prediction for Zambian smallholder farms — a season forecast in
kg/ha from Sentinel-2 imagery and a single phone photo, no sensors required.

---

## The problem

Smallholder maize farmers in Zambia make planting, input, and selling
decisions with no field-level yield forecast. The tools that exist either
require buying sensor hardware or are trained at a continental scale too
coarse to be useful on a two-hectare plot.

Farmers Guide closes that gap. Given a field location and one ground photo,
it produces a maize yield estimate in kilograms per hectare with an
uncertainty band — using only satellite data and a phone, trained on local
field data rather than generic pan-African averages.

This repository is an MSc thesis project (research code, not production — see
[Scope](#scope)). The longer trajectory is a web application that turns the
yield number into a financial decision: buyer matching, credit signals,
parametric-insurance triggers.

---

## Approach

A dual-branch model with decision-level fusion:

- **Satellite branch** — a temporal CNN reads a season of Sentinel-2
  composites over the field: 9 reflectance bands plus NDVI, EVI, NDRE, GCI,
  NDWI (14 channels), stacked across ~24 ten-day windows from land
  preparation through harvest.
- **Ground branch** — a MobileNetV2 transfer model reads a phone photo of
  the crop for growth stage and canopy health.
- **Fusion** — a decision-level weighted combination of the satellite yield
  estimate and the photo-derived health adjustment, with weights tunable per
  district.

Evaluation uses spatial K-fold (no neighbouring-field leakage) and a temporal
hold-out (the most recent season as test), reported against three baselines:
NDVI-mean regression, satellite-only, and photo-only. Thesis success
criteria: field-level RMSE below 600 kg/ha and R² above 0.55 on the temporal
hold-out.

---

## Repository structure

```
farmers-guide/
├── CLAUDE.md                 # operating brief — scope, workflow, contracts
├── config.py                 # all paths, AOIs, seasons, hyperparameters
├── gee/
│   ├── farmers_guide_s2_export.js   # Earth Engine editor script
│   └── fg_export.py                 # Sentinel-2 batch export wrapper
├── src/farmers_guide/
│   ├── data/
│   │   ├── hdf5_builder.py    # GeoTIFF stacks → HDF5 (T,H,W,14) tensors
│   │   ├── dataset.py         # torch Dataset + CV split logic
│   │   └── ground_truth.py    # yield label join (IAPRI / CFS / FAOSTAT)
│   ├── models/
│   │   ├── satellite_cnn.py   # temporal CNN, yield + uncertainty head
│   │   ├── ground_cnn.py      # MobileNetV2 transfer, health score
│   │   └── fusion.py          # decision-level weighted fusion
│   ├── train.py               # training entry point (called by Colab)
│   └── evaluate.py            # CV, baselines, metrics, figures
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train.ipynb         # Colab: pip install repo, mount Drive, train
│   └── 03_evaluate_figures.ipynb
└── reports/figures/           # thesis-ready plots
```

---

## The three-environment workflow

One repo is the source of truth. Three environments are execution surfaces;
Google Drive is the shared data lake.

| Environment | Role |
|---|---|
| **Local + Claude Code** | Write all code, run the GEE wrapper, run evaluation and figures, manage git. No training (no GPU). |
| **Google Earth Engine** | `gee/fg_export.py` exports Sentinel-2 GeoTIFFs to a Drive folder. |
| **Google Colab** | `pip install git+<this repo>` to get identical code, mount Drive, train on GPU, write weights back to Drive. |

The consistency rule: all model and data logic lives in `src/farmers_guide/`
as importable modules. Notebooks and the GEE wrapper only orchestrate — they
never define logic. This is what keeps Colab and local from drifting.

The loop: edit code locally → commit and push → Colab re-installs from git →
trains → writes to Drive → pull results locally → evaluate → repeat.

---

## Setup

### Local

```bash
git clone <this repo>
cd farmers-guide
python3 -m venv .venv && source .venv/bin/activate   # or use conda base
pip install -r requirements.txt
```

### Earth Engine export

```bash
earthengine authenticate
python gee/fg_export.py queue --seasons 2024_25 --aois chilanga --dry-run
python gee/fg_export.py queue --seasons 2022_23,2023_24,2024_25 \
    --aois chilanga,kafue,chongwe
python gee/fg_export.py monitor
```

### Colab training

Open `notebooks/02_train.ipynb` in Colab. It mounts Drive and runs
`pip install git+<this repo>` so it trains on exactly the code in `main`.

---

## Data

| Item | Value |
|---|---|
| Crop | Maize |
| Region | Lusaka Province, Zambia |
| Sub-AOIs | Chilanga, Kafue, Chongwe |
| Seasons | 2022/23, 2023/24, 2024/25 (Oct–Jun window) |
| Source | Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`) |
| Cloud masking | Cloud Score+ (`cs` band, threshold 0.60) |
| Bands | B2,B3,B4,B5,B6,B7,B8,B11,B12, NDVI, EVI, NDRE, GCI, NDWI |
| Composite | 10-day median, EPSG:32735, 10 m |
| Ground truth | IAPRI RALS, MoA Crop Forecasting Survey, FAOSTAT Zambia |

GeoTIFF naming is fixed: `fg_<aoi>_<season>_<YYYYMMDD>.tif`. All constants
live in `config.py` — nothing is hardcoded elsewhere.

---

## Status

Data export pipeline built and validated (Chilanga 2024/25 confirmed
producing valid 14-band GeoTIFFs). Next, in order:

1. `hdf5_builder.py` — GeoTIFF stacks to HDF5 cubes
2. `ground_truth.py` — yield label acquisition and join
3. `dataset.py` — Dataset with spatial/temporal splits
4. The three model modules
5. `train.py` and the Colab training notebook
6. `evaluate.py`, baselines, thesis figures

---

## Scope

This is thesis-grade research code. It deliberately does **not** include a
REST API, database, containers, cloud deployment, CI/CD, or a web frontend —
those belong to a later production phase. The deliverable here is a
reproducible pipeline, defensible models, and thesis results. See `CLAUDE.md`
for the full scope boundary and conventions.
