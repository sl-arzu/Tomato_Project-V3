# 📊 Results Folder Organization (FLAT - Max 1 Level Deep)

This folder contains all outputs from training, analysis, and visualization scripts.
Structure **simplified** to max 1 level deep - all related outputs grouped together by execution.

---

## 🎯 Where to Find What?

### I just ran `python main.py` — where are my results?

```
✅ Saved model:           results/models/water_lif_ep10_hid12.pt
✅ Training curves:       results/training/water_lif_eprop_ep10_hid12_training.pdf
✅ Confusion matrix:      results/training/water_lif_eprop_ep10_hid12_confusion.pdf
✅ Weight evolution:      results/training/water_lif_eprop_ep10_hid12_weights_*.pdf
✅ Spike raster (KEY!):   results/training/water_lif_eprop_ep10_hid12_raster.pdf
```

**Start here:** Open `results/training/` to see all outputs from one training run together.

---

## 📁 Folder Breakdown (FLAT Structure)

| Folder | Contains | When Generated | Used For |
|--------|----------|---------------|----|  
| **models/** | Trained `.pt` files | After each training run | Model inference/deployment |
| **training/** | All training outputs (FLAT) | After each training run | **PRIMARY**: Metrics, confusion, weights, rasters all together |
| **spike_analysis/** | Spike statistics & raster plots | After each training run | **Diagnostic tool** - spike propagation analysis |
| **dataset/** | Interactive HTML visualizations | Run dataset scripts | Dataset exploration & visualization |
| **encoding/** | Encoder comparisons & test reports | Run encoding tests | Validate/compare encoders |
| **logs/** | Training logs & debug output | Long-running tasks | Troubleshooting |

---

## 🔍 Key Diagnostic Tools

### 1. **Raster Plot** (Most Important!)
**File**: `training/*_raster.pdf` (now in training/ folder with other outputs)

Shows spike timing across network layers:
- **Encoding Input layer**: Should see ~20-25% spike density
- **Hidden layer**: Should see ~10-15% spike density  
- **Output layer**: Should see ~1-5% spike density

**What to look for**:
- ✅ **Good**: Varied spike patterns, no periodicity, cascade through layers
- ❌ **Bad**: Regular periodic spikes (encoder problem), all zeros (threshold too high), all ones (threshold too low)

### 2. **Training Curves**
**File**: `training/*_training.pdf` (all in training/ folder)

- Loss should decrease (not oscillate)
- Accuracy should increase or stabilize
- If not improving: lower learning rate or increase epochs

### 3. **Confusion Matrix**
**File**: `training/*_confusion.pdf` (all in training/ folder)

- Diagonal = correct predictions (good!)
- Off-diagonal = misclassifications (which classes confuse the network?)

### 4. **Weight Evolution**
**File**: `training/*_weights*.pdf` (all in training/ folder)

- Mean weights should stabilize (not explode/vanish)
- Individual weights should show learning dynamics

---

## 📋 Standard Naming Pattern

Output files follow this naming convention:

```
{stress_type}_{encoding_type}_{algorithm}_ep{epochs}_hid{hidden_neurons}_{plot_type}.pdf
```

**Example**: `water_lif_eprop_ep10_hid12_training.pdf`
- `water` = Stress type (water or iron)
- `lif` = Encoder (lif, rate, or rate_hz)
- `eprop` = Algorithm (eprop or bptt)
- `ep10` = 10 training epochs
- `hid12` = 12 hidden neurons
- `training` = Plot type (training, confusion, weights, raster)

---

## 🚀 Quick Start with Results

### 1. Check if training worked
```bash
open results/training/water_lif_eprop_ep10_hid12_training.pdf
```
→ Loss decreasing? Accuracy increasing? ✅

### 2. Diagnose spike propagation
```bash
open results/training/water_lif_eprop_ep10_hid12_raster.pdf
```
→ Spikes flowing through all layers? ✅

### 3. Evaluate per-class accuracy
```bash
open results/training/water_lif_eprop_ep10_hid12_confusion.pdf
```
→ Diagonal bright? Classes well-separated? ✅

### 4. Load model for inference
```python
import torch
model = torch.load("results/models/water_lif_ep10_hid12.pt")
```

---

## 🧹 Cleanup

Old/temporary results can be safely deleted:
```bash
rm -rf results/logs/*           # Debug output
rm -rf results/spike_analysis/statistics/*  # Intermediate analysis
```

Keep everything else for reproducibility!

---

## 🔗 Related Documentation

- **Understanding parameters**: [`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md)
- **Project structure**: [`docs/PROJECT_STRUCTURE.md`](../docs/PROJECT_STRUCTURE.md)
- **Full API reference**: [`docs/API.md`](../docs/API.md)

---

**Last Updated**: April 19, 2026
