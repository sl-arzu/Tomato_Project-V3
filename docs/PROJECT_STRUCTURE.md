# 📁 PROJECT STRUCTURE

## Directory Tree

```
TomatoProject_Rigore/
│
├── 📄 main.py                      Entry point - training orchestrator
├── 📄 environment.yml              Conda environment
├── 📄 requirements.txt             Python dependencies
│
├── 📁 src/                         ← CORE CODE (15 files)
│   ├── data_processing_manager.py           Load & split datasets
│   ├── data_processing_feature_selection.py Select optimal frequencies
│   ├── temp_enco_dispatcher.py              Encoder selector (LIF/Rate)
│   ├── temp_enco_lif_population.py          LIF encoder (main)
│   ├── temp_enco_rate.py                    Rate encoder
│   ├── snn_layer_model.py                   Network layers (RecurrentLayer, FeedforwardLayer)
│   ├── learning_trainer.py                  Training loop (SNNTrainer)
│   ├── learning_evaluator.py                Prediction & metrics (ModelEvaluator)
│   ├── learning_bptt.py                     BPTT algorithm
│   ├── learning_eprop.py                    E-prop algorithm
│   ├── snn_gradient_surrogate.py            Surrogate gradient
│   ├── plot_visualizations_new.py           Plotting functions
│   └── [other utilities]
│
├── 📁 tests/                       Unit tests
│   ├── test_encoding_with_real_data.py
│   └── test_temp_enco_models.py
│
├── 📁 config/                      Configuration files
│   ├── config.yaml
│   ├── dataset_config.yaml
│   └── model_config.yaml
│
├── 📁 data/                        Datasets
│   ├── water_stress/Water_Stress.npz
│   └── iron_stress/Iron_Stress.npz
│
├── 📁 results/                     Auto-generated outputs ← OUTPUT ORGANIZATION (FLAT: max 1 level)
│   ├── models/                     Saved SNN models (.pt files)
│   ├── training/                   All training results (metrics, confusion, weights, rasters)
│   ├── spike_analysis/             Neural activity analysis (raster plots & statistics)
│   ├── dataset/                    Dataset exploration & analysis
│   ├── encoding/                   Encoder validation & comparisons
│   └── logs/                       Training logs & debug output
│
└── 📁 docs/                        ← YOU ARE HERE
    ├── README.md                   Documentation index
    ├── QUICK_START.md              5-minute guide
    ├── PROJECT_STRUCTURE.md        (this file)
    ├── CONFIGURATION.md            Parameters explained
    ├── API.md                       Class reference
    └── [model_docs/, dataset_docs/]
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

### SNN Architecture
| File | Purpose |
|------|---------|
| `snn_layer_model.py` | `RecurrentLayer`, `FeedforwardLayer` definitions |
| `snn_gradient_surrogate.py` | Spike function + surrogate gradient |

### Learning & Evaluation
| File | Purpose |
|------|---------|
| `learning_eprop.py` | E-prop algorithm (eligibility propagation) |
| `learning_bptt.py` | BPTT algorithm (backprop through time) |
| `learning_trainer.py` | **SNNTrainer** - main training loop |
| `learning_evaluator.py` | **ModelEvaluator** - predictions & metrics |

### Visualization
| File | Purpose |
|------|---------|
| `plot_visualizations_new.py` | Raster plots, confusion matrix, weight evolution |

---

## Data Format

### Input (NPZ files)
```
Water_Stress.npz or Iron_Stress.npz
  ├── X: (n_samples, 400)           Raw bioimpedance measurements
  ├── y: (n_samples,)               Labels: 0=Control, 1=Early, 2=Late
  └── plant_ids: (n_samples,)       Plant identifiers
```

### After Feature Selection
```
X_selected: (n_samples, 6)
  └── 3 frequencies × 2 (Real + Imaginary)
```

### After Encoding
```
X_encoded: (n_samples, NB_STEPS, 6)
  └── Binary spike sequences (0 or 1)
```

---

## Parameter Files

### main.py (Lines 30-85)
Central configuration location with organized sections:
- **PARAMETRI ESPERIMENTO** (dataset, split)
- **PARAMETRI DI TEMPORAL ENCODING** (encoder params)
- **IPERPARAMETRI ENCODER LIF** (encoder tuning)
- **IPERPARAMETRI RETE RICORRENTE** (network tuning)
- **IPERPARAMETRI TRAINING** (learning params)

All parameters are in one place for easy modification.

---

## Output Generation

### During Training
- `water_lif_eprop_ep10_hid12_training.pdf` - Loss & accuracy curves
- `water_lif_eprop_ep10_hid12_confusion.pdf` - Confusion matrix
- `water_lif_eprop_ep10_hid12_weights_mean.pdf` - Weight evolution

### After Training
- `water_lif_eprop_ep10_hid12_raster.pdf` - Spike raster plots (key diagnostic!)
- `water_lif_ep10_hid12.pt` - Saved model weights

---

## Results Folder Structure 📊 (FLAT - Max 1 Level Deep)

The `results/` folder is **simplified** to max 1 level deep, grouping related outputs from the same execution together:

### `results/models/`
**Saved SNN models** (PyTorch .pt files)
```
water_lif_ep10_hid12.pt          ← Model trained on water_stress, 10 epochs
iron_rate_ep5_hid8.pt            ← Model on iron_stress with Rate encoder
```
- One model file per unique configuration
- Load with: `torch.load("results/models/model_name.pt")`

### `results/training/` (FLAT - no subdirectories)
**All training outputs in one level**: metrics, confusion matrices, weight evolution, raster plots
```
water_lif_eprop_ep10_hid12_training.pdf      ← Loss & accuracy per epoch
water_lif_eprop_ep10_hid12_confusion.pdf     ← Confusion matrix (rows=true, cols=predicted)
water_lif_eprop_ep10_hid12_weights_mean.pdf  ← Average weight evolution
water_lif_eprop_ep10_hid12_weights_indiv.pdf ← Individual weight samples (10-15 per layer)
water_lif_eprop_ep10_hid12_raster.pdf        ← Spike raster (Encoding → Hidden → Output)
```

**Why flat?** All files from one training run have the same prefix (`water_lif_eprop_ep10_hid12_*`), making them easy to:
- Find together
- Organize by experiment
- Delete as a batch (same config)

### `results/spike_analysis/` (FLAT - no subdirectories)
**Neural activity analysis** - spike statistics and raster plots
```
water_lif_eprop_ep10_hid12_raster.pdf        ← KEY DIAGNOSTIC: spike timing per layer
water_lif_eprop_ep10_hid12_spike_stats.txt   ← Spike rates, firing patterns, ISI distributions
```

**Key diagnostic in raster plot:**
- No spikes = problem with thresholds/gains
- Too many spikes = signals too strong
- Periodic patterns = refractory period misconfigured
- Chaotic patterns = normal, expected

### `results/dataset/` (FLAT - no subdirectories)
**Dataset exploration & analysis** - explorers & visualizations
```
pca3d_water_stress_normalizzato_zscore.html       ← Interactive 3D PCA plot
pca3d_water_stress_non_normalizzato.html          ← PCA without normalization
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
