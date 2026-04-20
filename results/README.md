# 📊 Results (Output Training)

Tutti gli output di training sono in questa cartella.

---

## 📁 Cartelle

| Cartella | Contenuto | Uso |
|----------|-----------|-----|
| **models/** | Modelli .pt allenati | Load per inference |
| **training/** | Tutte le figure (training, confusion, raster, pesi) | **PRINCIPALE** - verificare qualità training |
| **spike_analysis/** | Analisi spike e statistiche | Diagnostica spike |
| **dataset/** | Visualizzazioni HTML esplorative | Esplorare dati |
| **logs/** | Debug output | Troubleshooting |

---

## 🎯 Subito Dopo `python main.py`

```
results/models/water_lif_ep100_hid300.pt        ← Modello salvato
results/training/water_lif_eprop_ep100_hid300_training.pdf    ← Loss + accuracy
results/training/water_lif_eprop_ep100_hid300_confusion.pdf   ← Confusion matrix
results/training/water_lif_eprop_ep100_hid300_raster.pdf      ← Spike raster (KEY!)
```

---

## 🔍 Tre Cose da Controllare

### 1. Raster Plot
```bash
open results/training/water_lif_eprop_ep100_hid300_raster.pdf
```

Verificare spike density:
- Input layer: 20-30% ✅
- Hidden layer: 10-15% ✅
- Output layer: >1% ✅

### 2. Training Curves
```bash
open results/training/water_lif_eprop_ep100_hid300_training.pdf
```

- Loss deve **diminuire** ✅
- Accuracy deve **aumentare** ✅

### 3. Confusion Matrix
```bash
open results/training/water_lif_eprop_ep100_hid300_confusion.pdf
```

- Diagonale luminosa = predizioni corrette ✅

---

## 🎯 Nomenclatura File

```
{stress}_{encoding}_{algo}_ep{epochs}_hid{neurons}_{tipo}.pdf
```

Esempio: `water_lif_eprop_ep100_hid300_training.pdf`

---

📖 Vedi [../docs/CONFIGURATION.md](../docs/CONFIGURATION.md) per fix se non funziona
