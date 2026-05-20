#!/usr/bin/env bash
# ============================================================================
# Farmers Guide — repo bootstrap
# Run once, from inside your empty farmers-guide folder:
#     bash bootstrap_farmers_guide.sh
#
# Creates the thesis-grade repo skeleton, stub modules, config, packaging,
# and initializes git. Idempotent-ish: won't overwrite files that exist.
# ============================================================================
set -euo pipefail

echo "→ Scaffolding Farmers Guide repo in $(pwd)"

# ---- Directories ----------------------------------------------------------
mkdir -p gee
mkdir -p docs
mkdir -p src/farmers_guide/data
mkdir -p src/farmers_guide/models
mkdir -p notebooks
mkdir -p data/raw data/interim data/ground_truth
mkdir -p reports/figures

# ---- Helper: create file only if absent -----------------------------------
mk() { if [ ! -f "$1" ]; then cat > "$1"; echo "  created $1"; else echo "  skipped $1 (exists)"; fi }

# ---- Package markers ------------------------------------------------------
mk src/farmers_guide/__init__.py <<'EOF'
__version__ = "0.1.0"
EOF
mk src/farmers_guide/data/__init__.py <<'EOF'
EOF
mk src/farmers_guide/models/__init__.py <<'EOF'
EOF

# ---- config.py ------------------------------------------------------------
mk config.py <<'EOF'
"""Single source of truth for paths, AOIs, seasons, hyperparameters.
No hardcoded values anywhere else in the codebase — import from here.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# On Colab, DRIVE_ROOT is /content/drive/MyDrive ; locally, point at a synced
# Drive folder or a local mirror.
DRIVE_ROOT = Path("/content/drive/MyDrive")
S2_EXPORT_DIR = DRIVE_ROOT / "farmers_guide_s2"        # GEE GeoTIFF output
HDF5_DIR = DRIVE_ROOT / "farmers_guide_cubes"           # built HDF5 cubes
GROUND_TRUTH_CSV = Path("data/ground_truth/yields.csv") # label table
WEIGHTS_DIR = DRIVE_ROOT / "farmers_guide_weights"      # trained model output

# --- AOIs (must match gee/fg_export.py exactly) ---------------------------
AOIS = {
    "chilanga":        [28.15, -15.65, 28.40, -15.45],
    "kafue":           [28.05, -15.95, 28.35, -15.60],
    "chongwe":         [28.50, -15.55, 29.05, -15.10],
    "lusaka_province": [27.50, -16.50, 29.50, -15.00],
}

# --- Seasons (Zambia maize, Oct–Jun) --------------------------------------
SEASONS = ["2022_23", "2023_24", "2024_25"]

# --- Band order in exported GeoTIFFs (fixed) ------------------------------
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12",
         "NDVI", "EVI", "NDRE", "GCI", "NDWI"]
N_BANDS = len(BANDS)

# --- Model / training hyperparameters -------------------------------------
PATCH_SIZE = 64          # H, W of training patches
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
SEED = 42

# --- Targets (thesis success criteria) ------------------------------------
TARGET_RMSE_KG_HA = 600.0
TARGET_R2 = 0.55
EOF

# ---- Stub modules ---------------------------------------------------------
mk src/farmers_guide/data/hdf5_builder.py <<'EOF'
"""GeoTIFF stacks -> HDF5 tensors of shape (n_windows, H, W, 14).

TODO: implement. Reads fg_<aoi>_<season>_<YYYYMMDD>.tif from S2_EXPORT_DIR,
stacks by date, writes one HDF5 per (aoi, season) into HDF5_DIR with a
`dates` vector. See CLAUDE.md section 5 for the data contract.
"""
EOF
mk src/farmers_guide/data/ground_truth.py <<'EOF'
"""Build/join the yield label table keyed by (aoi, season, field_id).

TODO: implement. Sources: IAPRI RALS microdata, MoA Crop Forecasting
Survey, FAOSTAT Zambia. Output: data/ground_truth/yields.csv.
"""
EOF
mk src/farmers_guide/data/dataset.py <<'EOF'
"""torch Dataset / DataLoader over HDF5 cubes joined to yield labels.

TODO: implement. Spatial K-fold + temporal hold-out split logic lives here.
"""
EOF
mk src/farmers_guide/models/satellite_cnn.py <<'EOF'
"""Temporal satellite CNN. Input (T,H,W,14) -> yield kg/ha + uncertainty.

TODO: implement (3D-conv or ConvLSTM head, MC-dropout or quantile output).
"""
EOF
mk src/farmers_guide/models/ground_cnn.py <<'EOF'
"""MobileNetV2 transfer model on field photos -> health score / embedding.

TODO: implement.
"""
EOF
mk src/farmers_guide/models/fusion.py <<'EOF'
"""Decision-level weighted fusion of satellite + ground branches.

TODO: implement. Per-district tunable weights.
"""
EOF
mk src/farmers_guide/train.py <<'EOF'
"""Training entry point. Called by notebooks/02_train.ipynb on Colab.

TODO: implement. Reads config.py, builds dataset, trains, writes weights
+ metrics to WEIGHTS_DIR.
"""
EOF
mk src/farmers_guide/evaluate.py <<'EOF'
"""Evaluation: spatial/temporal CV, baselines (NDVI-OLS, sat-only,
photo-only), metric tables, thesis figures into reports/figures/.

TODO: implement.
"""
EOF

# ---- Packaging ------------------------------------------------------------
mk pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "farmers-guide"
version = "0.1.0"
description = "Maize yield prediction for smallholder farms, Lusaka Province"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "rasterio",
    "h5py",
    "scikit-learn",
    "matplotlib",
]

[tool.setuptools.packages.find]
where = ["src"]
EOF

mk requirements.txt <<'EOF'
torch
torchvision
numpy
pandas
rasterio
h5py
scikit-learn
matplotlib
# GEE wrapper only (not needed on Colab for training)
earthengine-api
click
tqdm
EOF

mk .gitignore <<'EOF'
__pycache__/
*.py[cod]
.venv/
.ipynb_checkpoints/
data/raw/
data/interim/
*.h5
*.tif
.DS_Store
EOF

mk README.md <<'EOF'
# Farmers Guide

MSc thesis: AI-powered maize yield prediction for smallholder farms in
Lusaka Province, Zambia.

See `CLAUDE.md` for architecture, workflow, and scope. Thesis-grade research
code — not production. Three environments (local, Earth Engine, Colab) share
this one repo + Google Drive.

## Quick start
- Data export: `python gee/fg_export.py queue --seasons 2024_25 --aois chilanga`
- Training: open `notebooks/02_train.ipynb` in Colab
EOF

# ---- Move GEE files in if they're sitting alongside ----------------------
for f in fg_export.py farmers_guide_s2_export.js; do
  if [ -f "$f" ]; then mv "$f" gee/ && echo "  moved $f -> gee/"; fi
  if [ -f "$HOME/Downloads/$f" ]; then cp "$HOME/Downloads/$f" gee/ && echo "  copied ~/Downloads/$f -> gee/"; fi
done

# ---- Move docs in if sitting alongside -----------------------------------
for f in ground_truth_plan.md; do
  if [ -f "$f" ]; then mv "$f" docs/ && echo "  moved $f -> docs/"; fi
done

# ---- Git ------------------------------------------------------------------
if [ ! -d .git ]; then
  git init -q
  git add -A
  if git commit -q -m "Scaffold thesis-grade repo structure" 2>/dev/null; then
    echo "  git initialised + first commit"
  else
    echo "  git initialised — first commit skipped (set git user.name/user.email, then: git commit -m 'init')"
  fi
else
  echo "  git already initialised (skipped)"
fi

echo ""
echo "✓ Done. Structure built. Verify:"
echo "  - CLAUDE.md, config.py, README.md at root"
echo "  - gee/fg_export.py, gee/farmers_guide_s2_export.js"
echo "  - docs/ground_truth_plan.md"
echo ""
echo "Next:"
echo "  git add -A && git commit -m 'Initial repo' && git push"
echo "  claude        # launches Claude Code, auto-reads CLAUDE.md"
