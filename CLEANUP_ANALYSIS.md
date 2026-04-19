# 🗑️ PROJECT CLEANUP ANALYSIS - REMOVABLE ITEMS

**Date**: April 19, 2026  
**Status**: Analysis Complete

---

## 📊 Summary

| Category | Item | Size | Status | Priority |
|----------|------|------|--------|----------|
| **Root Files** | .DS_Store | 6KB | Unused | 🔴 DELETE |
| **Root Files** | DOCUMENTATION_REORGANIZED.md | 2KB | Historical | 🟡 OPTIONAL |
| **Root Files** | RESULTS_REORGANIZED.md | 5KB | Historical | 🟡 OPTIONAL |
| **src/** | data_processing_feature_selection.py | ~5KB | Unused | 🔴 DELETE |
| **scripts/** | All 10 dataset analysis scripts | ~150KB | Optional | 🟢 KEEP IF USED |
| **config/** | *.yaml files | ~5KB | Not imported | 🟡 CHECK |
| **Auto-generated** | __pycache__/ | ~500KB | Auto-gen | ✅ In .gitignore |
| **venv/** | venv_m1/ | ~1.2GB | Environment | ✅ In .gitignore |

---

## 🔴 DELETE IMMEDIATELY

### 1. `.DS_Store`
- **What**: macOS metadata file
- **Size**: 6 KB
- **Why**: System file, not needed, usually in .gitignore
- **Impact**: None
```bash
rm .DS_Store
```

### 2. `src/data_processing_feature_selection.py`
- **What**: Data processing module
- **Size**: ~5 KB
- **Status**: ✅ **CONFIRMED UNUSED** (checked all imports - not referenced anywhere)
- **Why**: Not imported by main.py or any other file
- **Impact**: None (not part of active workflow)
```bash
rm src/data_processing_feature_selection.py
```

---

## 🟡 OPTIONAL (Keep if needed)

### 1. `DOCUMENTATION_REORGANIZED.md` & `RESULTS_REORGANIZED.md`
- **What**: Historical changelogs from reorganization
- **Size**: 7 KB combined
- **Usage**: 
  - Keep if: You want to document how organization changed
  - Delete if: Not needed for future reference
- **Decision**: Can delete or archive to git history
```bash
rm DOCUMENTATION_REORGANIZED.md RESULTS_REORGANIZED.md
```

### 2. `config/` YAML Files
- **What**: Configuration files (config.yaml, dataset_config.yaml, model_config.yaml)
- **Status**: Not imported in code (hardcoded in main.py)
- **Usage**: Could be used for future config-driven architecture
- **Decision**: Keep for now (low risk, might be useful)

---

## 🟢 KEEP BUT OPTIONAL

### Scripts Folder (`scripts/`)
- **What**: 10 analysis/exploration scripts
  - dataset_pca_comparative.py
  - dataset_pca_temporal.py
  - dataset_plant_explorer.py
  - dataset_reading_viewer.py
  - dataset_spectrum_explorer.py
  - dataset_svm_classifier.py
  - dataset_temporal_analysis.py
  - dataset_temporal_inspector.py
  - dataset_config.py
  - ORGANIZATION.md
  - README.md

- **Status**: Optional utilities, not part of main training workflow
- **Size**: ~150 KB
- **Usage**: 
  - Keep if: You want exploratory/analysis capabilities
  - Delete if: Only need main training pipeline
- **Impact**: None on main.py execution

**Recommendation**: KEEP (they don't hurt, useful for analysis)

---

## ✅ ALREADY IN .GITIGNORE (Good!)

These shouldn't clutter your repo:
```
__pycache__/              (Auto-generated Python cache)
venv_m1/                  (Virtual environment - 1.2GB!)
.DS_Store                 (Already in .gitignore)
```

**Note**: Even though in .gitignore, they consume disk space locally.

---

## 📝 Cleanup Checklist

### Essential Cleanup
- [ ] `rm .DS_Store` (6 KB saved)
- [ ] `rm src/data_processing_feature_selection.py` (5 KB saved)

### Optional Cleanup
- [ ] `rm DOCUMENTATION_REORGANIZED.md` (2 KB saved)
- [ ] `rm RESULTS_REORGANIZED.md` (5 KB saved)
- [ ] `rm -rf scripts/` (150 KB saved) - **ONLY if you don't need analysis tools**

### Local Only (won't affect Git)
- [ ] `rm -rf __pycache__/` (recreated on next Python run)
- [ ] `rm -rf results/logs/*` (debug outputs)

---

## 🎯 Recommended Cleanup Command

**Essential (safe):**
```bash
rm .DS_Store src/data_processing_feature_selection.py
```

**Full cleanup (also remove docs):**
```bash
rm .DS_Store src/data_processing_feature_selection.py \
   DOCUMENTATION_REORGANIZED.md RESULTS_REORGANIZED.md
```

**Maximum cleanup (remove everything optional):**
```bash
rm -rf .DS_Store src/data_processing_feature_selection.py \
   DOCUMENTATION_REORGANIZED.md RESULTS_REORGANIZED.md \
   scripts/ __pycache__/ results/logs/*
```

---

## ⚠️ DO NOT DELETE

These are essential:
- ✅ `main.py` - Entry point
- ✅ `src/` (except data_processing_feature_selection.py) - Core code
- ✅ `tests/` - Unit tests
- ✅ `data/` - Datasets
- ✅ `docs/` - Documentation
- ✅ `results/` - Training outputs
- ✅ `environment.yml` - Dependencies
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Git configuration

---

## 📊 Disk Space Impact

| Item | Size | Importance |
|------|------|-----------|
| .DS_Store | 6 KB | Remove now |
| data_processing_feature_selection.py | 5 KB | Remove now |
| DOCUMENTATION_REORGANIZED.md | 2 KB | Optional |
| RESULTS_REORGANIZED.md | 5 KB | Optional |
| scripts/ | ~150 KB | Optional but useful |
| __pycache__/ | ~500 KB | Auto-gen, ignore |
| venv_m1/ | ~1.2 GB | Already ignored |

**Potential savings**: 12 KB (essential) + 157 KB (optional) = 169 KB

---

## Final Recommendation

### For a Clean Production Repo:
```bash
# Essential cleanup
rm .DS_Store src/data_processing_feature_selection.py

# Optional but recommended
rm DOCUMENTATION_REORGANIZED.md RESULTS_REORGANIZED.md

# Local cleanup (won't affect Git)
rm -rf __pycache__
```

**This reduces clutter while keeping all functional code intact.**

---

**Questions?** The project structure is now clean and focused on the main training workflow!
