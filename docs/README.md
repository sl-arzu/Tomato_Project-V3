# 📚 TomatoProject Documentation

**Quick Navigation**:
- 🚀 [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
- 📁 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project organization
- ⚙️ [CONFIGURATION.md](CONFIGURATION.md) - Parameters & tuning
- 🔌 [API.md](API.md) - Core classes & methods

---

## 📖 Overview

**TomatoProject v2** - Classification of plant stress (water/iron) using **Spiking Neural Networks (SNN)** and bioimpedance measurements.

**Tech Stack**: PyTorch 2.0+, NumPy, Scikit-learn, Python 3.12 (M1 Mac optimized)

---

## 🗂️ Documentation Structure

| File | Purpose |
|------|---------|
| **QUICK_START.md** | 5-minute setup + common tasks |
| **PROJECT_STRUCTURE.md** | Folder organization & file purposes |
| **CONFIGURATION.md** | Encoding & SNN parameters explained |
| **API.md** | PlantDataManager, TemporalEncoder, SNNTrainer |

---

## 📂 Folder Organization

```
TomatoProject_Rigore/
├── main.py                 ← Entry point (training orchestrator)
├── src/                    ← Core modules (15 files, flat structure)
├── tests/                  ← Unit tests
├── config/                 ← YAML configuration files
├── data/                   ← Datasets (water_stress, iron_stress)
├── results/                ← Models & figures (auto-generated)
└── docs/                   ← THIS DOCUMENTATION
```

---

## 🎯 Common Tasks

### Run Training
```bash
cd /Users/harin/Developer/Amadeus/TomatoProject_Rigore
source venv_m1/bin/activate
python main.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Load & Explore Data
```python
from src.data_processing_manager import PlantDataManager

manager = PlantDataManager(stress_type="water")
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz"
)
```

### Adjust Parameters
Edit parameters in `main.py` (lines 30-85):
- **Encoding**: `GAIN_LIF`, `NOISE_STD`, `TAU_MEM`
- **Recurrent**: `HIDDEN_NEURONS`, `THRESHOLD`, `EPOCHS`

See [CONFIGURATION.md](CONFIGURATION.md) for details.

---

## 🔑 Key Concepts

### Temporal Encoding
Converts continuous bioimpedance signals → spike sequences using:
- **LIF Encoder**: Biologically-inspired (PyTorch, GPU-accelerated)
- **Rate Encoder**: Simple frequency-to-spike mapping

### Spiking Neural Network (SNN)
3-layer architecture:
```
Input (6 features) → LIF Hidden Layer (12 neurons) → Output (3 classes)
```

### E-prop Algorithm
Learning algorithm that computes eligibility traces for SNNs with surrogate gradients.

---

## ⚠️ Important Notes

1. **Data Leakage Prevention**: Normalization statistics computed from train set ONLY
2. **Device Auto-Detection**: Code automatically uses MPS (M1) > CUDA > CPU
3. **Separate Parameters**: Encoder (`TAU_REF_ENCODER=3.5ms`) ≠ Recurrent (`TAU_REF=2.5ms`)

---

## 📊 Results Location

- **Models**: `results/models/water_lif_ep10_hid12.pt`
- **Raster Plots**: `results/figures/water_lif_eprop_ep10_hid12_raster.pdf`
- **Training Curves**: `results/figures/water_lif_eprop_ep10_hid12_training.pdf`

---

**Last Updated**: 19 April 2026 | Status: ✅ Production Ready
