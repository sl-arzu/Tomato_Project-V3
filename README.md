# 🍅 TomatoProject v2

**Classification of plant stress (water/iron deficiency) using Spiking Neural Networks & bioimpedance measurements.**

---

## 🚀 Quick Start

```bash
# 1. Setup
source venv_m1/bin/activate

# 2. Run Training
python main.py

# 3. Check Results
open results/training/water_lif_eprop_ep10_hid12_raster.pdf
```

**Full guide**: → [`docs/QUICK_START.md`](docs/QUICK_START.md)

---

## 📚 Documentation

All documentation is in [`docs/`](docs/) folder:

| File | Purpose |
|------|---------|
| [`docs/README.md`](docs/README.md) | **Start here** - Overview & navigation |
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | 5-minute setup guide |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | File organization & data flow |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Parameter tuning guide |
| [`docs/API.md`](docs/API.md) | Class reference & examples |

**Concise documentation**: 5 files, 930 lines. No duplicates. ✓

---

## 📁 Project Structure

```
TomatoProject_Rigore/
├── main.py                    ← Entry point (all parameters in lines 30-85)
├── src/                       ← Core modules (15 files)
├── tests/                     ← Unit tests
├── config/                    ← YAML configuration
├── data/                      ← Datasets
├── results/                   ← Auto-generated models & figures
└── docs/                      ← Documentation (THIS)
```

**Full details**: → [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

---

## ⚙️ Configure & Run

Edit `main.py` lines 30-85 to adjust parameters:

```python
ENCODING_TYPE = "lif"         # Encoder type
NB_STEPS = 50                 # Temporal window (ms)
GAIN_LIF = 0.35               # Input amplification
HIDDEN_NEURONS = 12           # Hidden layer size
EPOCHS = 10                   # Training iterations
LEARNING_RATE = 0.003         # Optimization rate
```

**Parameter explanations**: → [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)

---

## 🧠 Architecture

```
Bioimpedance Input (400 features)
  ↓ Feature Selection (6 optimal)
  ↓ Temporal Encoding (LIF → spikes)
  ↓ SNN Hidden Layer (12 neurons, LIF)
  ↓ Output Classification (3 classes)
Result: Plant stress classification
```

**Detailed flow**: → [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

---

## 📊 Results

After training, check these in `results/` (flat structure - all related outputs together):

- **Raster plots** (neural spike activity across layers)
  ```
  results/training/water_lif_eprop_ep10_hid12_raster.pdf
  ```

- **Training metrics** (loss & accuracy curves)
  ```
  results/training/water_lif_eprop_ep10_hid12_training.pdf
  ```

- **Confusion matrix** (per-class accuracy)
  ```
  results/training/water_lif_eprop_ep10_hid12_confusion.pdf
  ```

- **Weight evolution** (learning dynamics)
  ```
  results/training/water_lif_eprop_ep10_hid12_weights_*.pdf
  ```

- **Saved model** (for inference)
  ```
  results/models/water_lif_ep10_hid12.pt
  ```

**Full guide**: → [`results/README.md`](results/README.md)

---

## 🔧 Common Tasks

### Load Data Manually
```python
from src.data_processing_manager import PlantDataManager
from torch.utils.data import DataLoader

manager = PlantDataManager(stress_type="water")
ds_train, ds_test, _ = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz"
)
loader = DataLoader(ds_train, batch_size=24)
```

### Use Different Encoder
```python
from src.temp_enco_dispatcher import TemporalEncoder

encoder = TemporalEncoder(encoding_type="rate")  # or "lif", "rate_hz"
```

### Debug Spike Activity
```python
# Check spike distribution
import matplotlib.pyplot as plt
spikes = encoder.encode(X)  # (n, 50, 6)
plt.imshow(spikes[0].mean(axis=0))  # Average across timesteps
```

**More examples**: → [`docs/API.md`](docs/API.md)

---

## 🐛 Troubleshooting

### Output Layer has 0 spikes?
→ Increase `HIDDEN_NEURONS` or reduce `THRESHOLD`  
See: [`docs/QUICK_START.md`](docs/QUICK_START.md) "Debug Common Issues"

### Training not converging?
→ Reduce `LEARNING_RATE` or increase `EPOCHS`  
See: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) "Convergence Issues"

### Need to add new features?
→ Check file locations in [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

---

## 📈 Performance

Current configuration achieves:
- **Spike propagation**: Encoding 20%, Hidden 11%, Output 1%
- **Training accuracy**: ~38% (random baseline 33%)
- **Training time**: ~10-20 min per run (10 epochs)
- **GPU**: M1 Mac (MPS acceleration, ~10x faster than CPU)

---

## 🔑 Key Concepts

**Temporal Encoding**: Continuous signals → binary spike sequences using LIF neurons

**Spiking Neural Network**: 3-layer network with LIF dynamics and E-prop learning

**E-prop Algorithm**: Eligibility propagation - learning rule for SNNs with surrogate gradients

**Raster Plot**: Visualization of spike times across neurons (key diagnostic tool)

---

## 📞 For AI Assistants

This is a well-documented SNN project. To work on it:

1. **Understand structure**: Read [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
2. **Find code**: Check file locations in PROJECT_STRUCTURE
3. **Use API**: Reference [`docs/API.md`](docs/API.md) for class usage
4. **Configure**: Edit parameters in `main.py` (lines 30-85)
5. **Diagnose**: Check raster plots in `results/figures/`

**No need to read historical analysis** - problems already solved!

---

## 📝 License & Status

- **Status**: ✅ Production ready
- **Last Updated**: 19 April 2026
- **Python**: 3.12.12 (M1 Mac optimized)
- **Framework**: PyTorch 2.0+

---

**👉 Start here**: [`docs/README.md`](docs/README.md)

DA VERSIONARE (git add):
  ✓ src/
  ✓ scripts/
  ✓ tests/
  ✓ docs/
  ✓ config/
  ✓ *.md, *.txt, *.yml, *.ini

DA ESCLUDERE (.gitignore):
  ✗ venv_m1/
  ✗ results/ (tranne struttura dir)
  ✗ __pycache__/
  ✗ *.pt (modelli grandi)
  ✗ *.npz (dataset)
```

---

## 🎯 Prossimi Step

1. **Verificare setup**: `python main.py --help`
2. **Esplorare dataset**: `python scripts/dataset_analysis/plant_explorer_bioimpedance.py`
3. **Leggere doc**: REFACTORING_COMPLETE.md, SCRIPTS_ORGANIZATION.md
4. **Eseguire training**: `python main.py`
5. **Analizzare risultati**: Vedi `results/`

---

**Progetto**: TomatoProject_v2  
**Autore**: Shanti Leonardo Arzu  
**Data**: Aprile 2026  
**Status**: ✅ Riorganizzato e funzionante
