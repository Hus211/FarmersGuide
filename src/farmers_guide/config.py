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
# SUPERVISED_SEASON is the one with RALS 2019 ground-truth labels (the
# 2018/19 agricultural year). The model is TRAINED on this season only.
# INFERENCE_SEASONS have no plot-level labels — used for the operational
# demo and optional CFS district-level weak supervision. See
# docs/ground_truth_plan.md for the reasoning.
SUPERVISED_SEASON = "2018_19"
INFERENCE_SEASONS = ["2022_23", "2023_24", "2024_25"]
SEASONS = [SUPERVISED_SEASON, *INFERENCE_SEASONS]

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
