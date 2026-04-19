"""
Common configuration for dataset analysis scripts.

This module defines shared paths and constants for all dataset analysis
scripts in the scripts/ directory.

All scripts should be executed from the project root:
    cd /path/to/TomatoProject_v2
    python scripts/dataset_*.py
"""

import os

# ============================================================
# Project Root Detection
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ============================================================
# Data Paths
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

WATER_FILE = os.path.join(DATA_DIR, "water_stress", "Water_Stress.npz")
IRON_FILE = os.path.join(DATA_DIR, "iron_stress", "Iron_Stress.npz")

# ============================================================
# Output Paths
# ============================================================
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Specific output subdirectories for different analysis types
DATASET_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "dataset_analysis")
TEMPORAL_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "dataset_analysis")
PCA_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "dataset_analysis")
CLASSIFICATION_DIR = os.path.join(RESULTS_DIR, "dataset_analysis")

# ============================================================
# Constants
# ============================================================
LABEL_MAP = {
    0: "Control",
    1: "Early Stress",
    2: "Late Stress",
}

PLANT_COLORS = {
    "P0": "#1f77b4",   # blue
    "P1": "#9467bd",   # purple
    "P3": "#ff7f0e",   # orange
}

DAYS_PER_CLASS = {
    0: list(range(1, 7)),    # Control: days 1-6
    1: list(range(7, 19)),   # Early: days 7-18
    2: list(range(19, 31)),  # Late: days 19-30
}

# ============================================================
# Utility Functions
# ============================================================

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[OK] Created output directory: {output_dir}")

def build_frequency_axis(n=200, fmin=100, fmax=10e6):
    """Build logarithmic frequency axis for bioimpedance (100 Hz - 10 MHz)."""
    import numpy as np
    return np.logspace(np.log10(fmin), np.log10(fmax), n)
