# 🚀 Quick Start — 5 Minuti

## Setup

```bash
cd /Users/harin/Developer/Amadeus/Tomato_Project-V3
source venv_m1/bin/activate
python main.py
```

Output: modelli in `results/models/`, grafici in `results/training/`

---

## Verificare Risultati

**Raster plot** (diagnostica principale):
```bash
open results/training/water_lif_eprop_ep100_hid300_raster.pdf
```

Cercare:
- ✅ Input: 20-30% spike density
- ✅ Hidden: 10-15% spike density
- ✅ Output: >1% spike density

**Curve di training**:
```bash
open results/training/water_lif_eprop_ep100_hid300_training.pdf
```

Loss deve diminuire, accuracy deve aumentare.

---

## Modificare Parametri

Edita `main.py` linee 30-85. I parametri principali:

### Input (Encoding)
```python
NB_STEPS = 150           # Durata finestra (ms)
GAIN_LIF = 0.35          # Amplificazione segnale
NOISE_STD = 1.0          # Caos (evita spike periodici)
TAU_MEM = 18.0           # Decadimento membrana
```

### Rete (Hidden Layer)
```python
HIDDEN_NEURONS = 300     # Capacità rete
THRESHOLD = 0.80         # Soglia spike
EPOCHS = 100             # Iterazioni training
LEARNING_RATE = 0.003    # Passo gradiente
```

Vedi [CONFIGURATION.md](CONFIGURATION.md) per dettagli su ogni parametro.

---

## Problemi Comuni

| Problema | Soluzione |
|----------|-----------|
| Output layer silenzioso (0% spike) | ↑ `HIDDEN_NEURONS` o ↓ `THRESHOLD` |
| Hidden layer troppo pochi spike | ↑ `GAIN_LIF` |
| Spike troppo periodici | ↑ `NOISE_STD` |
| Loss non diminuisce | ↓ `LEARNING_RATE`, ↑ `EPOCHS` |

---

## Usare i Dati

```python
from src.data_processing_manager import PlantDataManager
from torch.utils.data import DataLoader

manager = PlantDataManager(stress_type="water")
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz", test_size=0.3
)

loader = DataLoader(ds_train, batch_size=24, shuffle=True)
for X_spikes, y_labels in loader:
    print(X_spikes.shape)  # (24, 150, 6)
    break
```

---

## Test

```bash
pytest tests/ -v
```

---

📖 Prossimo: [CONFIGURATION.md](CONFIGURATION.md) per tuning avanzato
