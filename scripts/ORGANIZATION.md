# 🎯 Scripts Organization

**Data**: 7 Aprile 2026  
**Project**: TomatoProject_v2  
**Status**: ✅ Fully Standardized

---

## 📋 Summary

### ✅ What Was Done

**1. Flat Structure** (NO SUBDIRECTORIES)
```
scripts/
├── __init__.py
├── README.md                     ← Detailed documentation
├── ORGANIZATION.md               ← This file
├── dataset_config.py             ← Shared config (imports this!)
└── dataset_*.py (8 scripts)      ← All analysis scripts
```

**2. Unified Naming**
- All scripts start with `dataset_` prefix
- Names describe what they do
- Examples:
  - `dataset_plant_explorer.py` → 3D/PCA exploration
  - `dataset_temporal_inspector.py` → Structure analysis
  - `dataset_svm_classifier.py` → Classification

**3. Script Renames**
| Old Name | New Name | Reason |
|----------|----------|--------|
| plant_explorer_bioimpedance.py | dataset_plant_explorer.py | Unified prefix |
| dataset_SVM_classifier.py | dataset_svm_classifier.py | Consistent casing |
| dataset_plotly_reading_viewer.py | dataset_reading_viewer.py | Shorter name |

**4. Shared Configuration**
- Created `dataset_config.py` for all common paths and constants
- All scripts import from this ONE file
- Paths automatically resolved from file location

---

## 📂 File List (8 Scripts)

```
scripts/
├── dataset_plant_explorer.py         │ 3D/PCA visualization
├── dataset_temporal_inspector.py     │ Temporal structure
├── dataset_temporal_analysis.py      │ Signal progression
├── dataset_pca_temporal.py           │ PCA over time
├── dataset_pca_comparative.py        │ Comparative PCA
├── dataset_spectrum_explorer.py      │ Frequency spectrum
├── dataset_reading_viewer.py         │ Reading index view
└── dataset_svm_classifier.py         │ SVM classification
```

---

## 🔧 Shared Configuration: dataset_config.py

All scripts should use this module for paths and constants:

```python
# Import in your script:
from dataset_config import (
    DATA_DIR,
    DATASET_ANALYSIS_DIR,
    LABEL_MAP,
    PLANT_COLORS,
    ensure_output_dir
)
```

**What it provides**:
- Auto-resolved PROJECT_ROOT (works from any directory)
- DATA_DIR → ./data
- DATASET_ANALYSIS_DIR → ./results/dataset_analysis
- Common constants (LABEL_MAP, PLANT_COLORS, DAYS_PER_CLASS)
- Utility functions (ensure_output_dir, build_frequency_axis)

---

## 🚀 Output Paths

All scripts save to `results/dataset_analysis/`:
```
results/dataset_analysis/
├── raw3d_*.html
├── pca_*.html
├── temporal_*.html
├── spectrum_*.html
├── cm_*.html (confusion matrix)
└── summary_*.html (reports)
```

---

## ✅ Design Goals

- [x] **Flat structure**: No subdirectories in scripts/
- [x] **Unified naming**: All scripts start with `dataset_`
- [x] **Descriptive names**: Names clearly indicate purpose
- [x] **Shared config**: `dataset_config.py` for common paths
- [x] **Auto path resolution**: Works from any directory
- [x] **Consistent output**: All results in `results/dataset_analysis/`
- [x] **Documented**: README.md with all 8 scripts

---

## 📖 Documentation

- **README.md**: Detailed guide for each script and usage
- **dataset_config.py**: Inline documentation for imports
- This file: Overview of organization

---

## 🎯 Adding New Scripts

Pattern to follow:

```python
# scripts/dataset_my_new_script.py
"""Description of what this script does."""

import os
import numpy as np
import plotly.graph_objects as go

# Import shared config
from dataset_config import (
    DATA_DIR,
    DATASET_ANALYSIS_DIR,
    LABEL_MAP,
    ensure_output_dir
)

# Set output directory
OUTPUT_DIR = DATASET_ANALYSIS_DIR

def main():
    ensure_output_dir(OUTPUT_DIR)
    # ... your code ...

if __name__ == "__main__":
    main()
```

Then run from root:
```bash
python scripts/dataset_my_new_script.py
```

---

## 🔄 Workflow

1. **Run from root directory always**:
   ```bash
   cd /Users/harin/Developer/Amadeus/TomatoProject_v2
   python scripts/dataset_*.py
   ```

2. **Paths are auto-resolved** via `dataset_config.py`

3. **Output goes to** `./results/dataset_analysis/`

4. **No manual path management needed**

---

**Last Updated**: 7 Aprile 2026  
**Scripts**: 8 analysis scripts  
**Naming**: `dataset_*` prefix ✅  
**Config**: Shared `dataset_config.py` ✅
