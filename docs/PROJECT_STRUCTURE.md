# 📁 Struttura del Progetto

```
Tomato_Project-V3/
├── main.py                    ← Punto di ingresso
├── environment.yml
├── requirements.txt
│
├── src/                       ← Codice principale
│   ├── data_processing_manager.py              Load + split + encode
│   ├── data_processing_plant_feature_selector.py   Selezione features
│   ├── temp_enco_dispatcher.py                 Router encoder (LIF/Rate)
│   ├── temp_enco_lif_population.py            LIF encoder
│   ├── temp_enco_rate.py                      Rate encoder
│   ├── snn_layer_model.py                     Layer SNN
│   ├── learning_trainer.py                    Training loop
│   ├── learning_evaluator.py                  Predictions + metrics
│   ├── learning_bptt.py                       BPTT algorithm
│   ├── learning_eprop.py                      E-prop algorithm
│   ├── snn_gradient_surrogate.py              Spike + surrogate
│   └── plot_visualizations_new.py             Grafici
│
├── data/                      ← Dataset
│   ├── water_stress/Water_Stress.npz
│   └── iron_stress/Iron_Stress.npz
│
├── results/                   ← Output (auto-generato)
│   ├── models/                Modelli .pt
│   ├── training/              Output training
│   ├── spike_analysis/        Analisi spike
│   ├── dataset/               Esplorazione
│   ├── encoding/              Test encoder
│   └── logs/                  Debug
│
├── tests/                     ← Test unitari
│   ├── test_encoding_with_real_data.py
│   └── test_temp_enco_models.py
│
├── config/                    ← File YAML
│   ├── config.yaml
│   ├── dataset_config.yaml
│   └── model_config.yaml
│
└── docs/                      ← Documentazione
    ├── README.md
    ├── QUICK_START.md
    ├── CONFIGURATION.md
    ├── API.md
    └── PROJECT_STRUCTURE.md
```

---

## Core Modules (src/)

### Data Processing
| File | Purpose |
|------|---------|
| `data_processing_manager.py` | Load .npz → Split train/test → Apply encoding |
| `data_processing_feature_selection.py` | Select 6 optimal features per stress type |

**Data Flow**:
```
.npz (400 features) → Select (6 features) → Normalize → Encode → (50, 6) spikes
```

### Temporal Encoding
| File | Purpose |
|------|---------|
| `temp_enco_dispatcher.py` | Route to LIF/Rate based on `encoding_type` |
| `temp_enco_lif_population.py` | **LIF encoder** (biologically-inspired, GPU) |
| `temp_enco_rate.py` | Rate/Poisson encoder (simple baseline) |

**Encoding Process**:
```
Input signal (6 features, z-score normalized)
        ↓
LIFEncoder: Leaky integrate-and-fire dynamics
        ↓
Spike sequences (50 timesteps × 6 features)
```

---

## 🔄 Flusso Dati

```
.npz (400 features, 1000 campioni)
  ↓
Selezione Features (6 features z-score normalized)
  ↓
Temporal Encoding LIF/Rate (150 timestep)
  ↓
Spike Sequences (1000, 150, 6)
  ↓
Train/Test Split (70/30)
  ↓
SNN Training (E-prop o BPTT)
```

---

## 📊 Output Generati

Dopo `python main.py`:

### `results/models/`
```
water_lif_ep100_hid300.pt    ← Modello allenato
```

### `results/training/` (tutti insieme)
```
water_lif_eprop_ep100_hid300_training.pdf     ← Loss + accuracy
water_lif_eprop_ep100_hid300_confusion.pdf    ← Confusion matrix
water_lif_eprop_ep100_hid300_raster.pdf       ← Spike raster (KEY!)
water_lif_eprop_ep100_hid300_weights_mean.pdf ← Evoluzione pesi
```

---

## 🎯 Nominazione Output

```
{stress}_{encoding}_{algo}_ep{epochs}_hid{neurons}_{tipo}.pdf
```

**Esempio**: `water_lif_eprop_ep100_hid300_raster.pdf`
- `water` = Tipo stress
- `lif` = Encoder
- `eprop` = Algoritmo
- `ep100` = 100 epoche
- `hid300` = 300 neuroni hidden

---

**Aggiornato**: 20 Aprile 2026
water_stress_pca_temporal.html                    ← PCA colored by day
water_stress_spectrum_explorer.html               ← Impedance spectrum visualization
...
```
- All HTML files from dataset analysis scripts
- No subdirectories: open directly from results/dataset/

### `results/encoding/` (FLAT - no subdirectories)
**Temporal encoder validation** - comparisons & test reports
```
lif_vs_rate_comparison.pdf                   ← LIF vs Rate encoder side-by-side
encoding_test_report.txt                     ← Test metrics from test_encoding_with_real_data.py
lif_population_density_analysis.pdf          ← Spike density analysis
```

### `results/logs/`
**Debug logs and training traces**
- Training logs (if enabled)
- Debug output from long-running tasks
- Temporary cache files

---

## File Naming Convention

### Output File Naming

### Prefixes in src/
| Prefix | Meaning | Examples |
|--------|---------|----------|
| `data_processing_*` | Data loading & preprocessing | manager, feature_selection |
| `temp_enco_*` | Temporal encoding | lif, rate, dispatcher |
| `snn_*` | SNN architecture | layer_model, gradient_surrogate |
| `learning_*` | Training & evaluation | trainer, evaluator, eprop |
| `plot_*` | Visualization | visualizations_new |

### Output File Naming
Format: `{stress_type}_{encoding}_{algorithm}_ep{epochs}_hid{hidden}.{ext}`

Example: `water_lif_eprop_ep10_hid12_raster.pdf`
- `water` = dataset type
- `lif` = encoder
- `eprop` = algorithm
- `ep10` = 10 epochs
- `hid12` = 12 hidden neurons

---

## Key Files to Know

1. **main.py** - Start here (orchestrator)
2. **src/data_processing_manager.py** - Understanding data loading
3. **src/temp_enco_lif_population.py** - How spikes are generated
4. **src/learning_trainer.py** - How training works
5. **results/spike_analysis/raster_plots/** - Primary diagnostic plots
6. **results/README.md** - Output folder guide

---

**Next**: [CONFIGURATION.md](CONFIGURATION.md) - Understand what each parameter does
