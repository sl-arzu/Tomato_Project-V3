# 🔌 API REFERENCE - Core Classes

## PlantDataManager

**Location**: `src/data_processing_manager.py`

Load, normalize, split, and encode plant bioimpedance data.

### Initialization
```python
from src.data_processing_manager import PlantDataManager

manager = PlantDataManager(
    stress_type="water",  # "water" or "iron"
    encoding_params={
        "encoding_type": "lif",
        "nb_steps": 50,
        "gain_lif": 0.35,
        "noise_std": 1.0,
        # ... more params
    }
)
```

### Main Methods

#### `prepare_dataset_standard_split()`
```python
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    file_path="data/water_stress/Water_Stress.npz",
    test_size=0.3,
    val_size=0.0
)

# Returns:
# ds_train: TensorDataset with (spike_sequences, labels)
# ds_test: TensorDataset with (spike_sequences, labels)
# metadata: dict with {'nb_inputs': 6, 'nb_outputs': 3, ...}
```

#### `prepare_dataset_leave_one_plant_split()`
```python
ds_train, ds_test, metadata = manager.prepare_dataset_leave_one_plant_split(
    file_path="data/water_stress/Water_Stress.npz",
    leave_plant="P3",  # Test on this plant only
    val_size=0.0
)
```

### Data Flow
```
Raw NPZ (1000, 400) 
  ↓ Feature Selection
Selected (1000, 6)
  ↓ Normalization
Normalized (1000, 6)
  ↓ Temporal Encoding
Spikes (1000, 50, 6)
  ↓ Train/Test Split
Datasets ready for DataLoader
```

---

## TemporalEncoder

**Location**: `src/temp_enco_dispatcher.py`

Convert continuous signals → spike sequences (dispatcher for LIF/Rate encoders).

### Initialization
```python
from src.temp_enco_dispatcher import TemporalEncoder

encoder = TemporalEncoder(
    encoding_type="lif",           # "lif", "rate", "rate_hz"
    nb_steps=50,
    dt=1.0,
    gain_lif=0.35,
    noise_std=1.0,
    tau_mem=18.0,
    tau_syn=12.0,
    threshold=0.80,
    input_shift=3.8,
    tau_ref=3.5,
    population_size=1,
    seed=None  # None for stochastic, 42 for reproducible
)
```

### Main Method

#### `encode()`
```python
import numpy as np

X = np.random.randn(10, 6)  # 10 samples, 6 features (z-score normalized)

spikes = encoder.encode(X)
# Output: (10, 50, 6)
#   - 10 samples
#   - 50 timesteps
#   - 6 features
#   - Values: 0.0 (no spike) or 1.0 (spike)
```

### Supported Encoders

| Type | Speed | Biological | Deterministic |
|------|-------|------------|---------------|
| `"lif"` | ⭐⭐⭐ (GPU) | ✅ | ✅ |
| `"rate"` | ⭐⭐ (CPU) | ✅ | ❌ (Poisson) |
| `"rate_hz"` | ⭐⭐⭐ | ~ | ✅ |

**Recommendation**: Use `"lif"` for production.

---

## SNNTrainer

**Location**: `src/learning_trainer.py`

Training loop for Spiking Neural Networks with E-prop or BPTT.

### Initialization
```python
from src.learning_trainer import SNNTrainer

trainer = SNNTrainer(
    layers=(w_in, w_out, w_rec),  # Weight matrices
    device=torch.device("mps"),
    nb_outputs=3,
    tau_mem_ms=18.0,
    max_time_ms=50.0,
    lr=0.003,
    algorithm="eprop"  # or "bptt"
)
```

### Main Method

#### `fit()`
```python
history, weight_history = trainer.fit(
    train_loader=train_loader,
    epochs=10,
    run_snn_kwargs={
        "decay": decay_factors,
        "nb_steps": 50,
        "nb_hidden": 12,
        "nb_outputs": 3,
        "device": device,
        "threshold": 0.80,
        # ... more
    },
    test_loader=test_loader
)

# Returns:
# history: dict with 'loss', 'train_acc', 'test_acc' per epoch
# weight_history: list of weight snapshots per epoch
```

### History Output
```python
print(history['loss'])        # [0.829, 0.590, 0.531, ...]
print(history['train_acc'])   # [0.32, 0.35, 0.37, ...]
print(history['test_acc'])    # [0.30, 0.34, 0.36, ...]
```

---

## ModelEvaluator

**Location**: `src/learning_evaluator.py`

Evaluate trained model, extract predictions & neural activity.

### Initialization
```python
from src.learning_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    layers=(w_in, w_out, w_rec),
    device=torch.device("mps"),
    nb_outputs=3
)
```

### Main Method

#### `collect_predictions_and_activity()`
```python
y_true, y_pred, spk_input, spk_hidden, spk_out = evaluator.collect_predictions_and_activity(
    test_loader=test_loader,
    run_snn_kwargs=run_snn_kwargs
)

# Returns:
# y_true: (n_samples,) true labels
# y_pred: (n_samples,) predicted labels
# spk_input: list of input spike arrays
# spk_hidden: list of hidden spike arrays
# spk_out: list of output spike arrays
```

### Access Neural Activity
```python
# Get first sample's spike raster
spikes_input = spk_input[0][0]    # (50, 6) - input layer
spikes_hidden = spk_hidden[0][0]  # (50, 12) - hidden layer
spikes_out = spk_out[0][0]        # (50, 3) - output layer

print(f"Input spike rate: {spikes_input.sum() / spikes_input.size:.1%}")
print(f"Hidden spike rate: {spikes_hidden.sum() / spikes_hidden.size:.1%}")
```

---

## PlantFeatureSelector

**Location**: `src/data_processing_feature_selection.py`

Select optimal frequencies & normalize data.

### Initialization
```python
from src.data_processing_feature_selection import PlantFeatureSelector

selector = PlantFeatureSelector(stress_type="water")
```

### Main Methods

#### `select_features()`
```python
X_raw = np.random.randn(1000, 400)

X_selected = selector.select_features(X_raw)
# Input: (1000, 400)
# Output: (1000, 6)
#
# Selects: [Real@f0, Imag@f0, Real@f1, Imag@f1, Real@f2, Imag@f2]
# where f0, f1, f2 are optimal frequencies for stress type
```

#### `normalize_features()`
```python
X_train_norm, X_test_norm = selector.normalize_features(X_train, X_test)

# Uses train statistics ONLY (prevents data leakage)
# Computes: mean = X_train.mean(), std = X_train.std()
# Applies: (X - mean) / std for all data
```

---

## LIFEncoder (Advanced)

**Location**: `src/temp_enco_lif_population.py`

Direct usage of LIF encoder (bypassing dispatcher).

### Initialization
```python
from src.temp_enco_lif_population import LIFEncoder

encoder = LIFEncoder(
    nb_steps=50,
    dt=1.0,
    tau_syn=12.0,
    tau_mem=18.0,
    tau_ref=3.5,
    threshold=0.80,
    gain=0.35,
    input_shift=3.8,
    noise_std=1.0,
    seed=None,
    population_size=1
)
```

### Main Method

#### `forward()`
```python
X = np.random.randn(10, 6)

spikes = encoder(X)
# Output: (10, 50, 6) spike tensor on GPU (if available)
```

---

## Decay Factors Helper

**Location**: `src/snn_layer_model.py`

Pre-compute exponential decay factors for efficiency.

```python
from src.snn_layer_model import compute_decay_factors

decay_factors = compute_decay_factors(
    dt=1.0,
    tau_mem=18.0,
    tau_mem_rec=35.0,
    tau_syn=12.0,
    tau_trace=80.0,
    tau_trace_out=105.0
)

# Returns: dict with pre-computed α_mem, α_syn, etc.
# Used in run_snn_kwargs
```

---

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from src.data_processing_manager import PlantDataManager
from src.snn_layer_model import RecurrentLayer, FeedforwardLayer, compute_decay_factors
from src.learning_trainer import SNNTrainer
from src.learning_evaluator import ModelEvaluator

# 1. Load data
manager = PlantDataManager(stress_type="water", encoding_params={
    "encoding_type": "lif", "nb_steps": 50, "gain_lif": 0.35
})
ds_train, ds_test, meta = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz"
)
train_loader = DataLoader(ds_train, batch_size=24, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=24, shuffle=False)

# 2. Create model
device = torch.device("mps")
w_in, w_rec = RecurrentLayer.create_layer(6, 12, device=device)
w_out = FeedforwardLayer.create_layer(12, 3, device=device)
decay_factors = compute_decay_factors(dt=1.0, tau_mem=18.0, tau_mem_rec=35.0)

# 3. Train
trainer = SNNTrainer((w_in, w_out, w_rec), device, 3, lr=0.003, algorithm="eprop")
history, _ = trainer.fit(
    train_loader, epochs=10,
    run_snn_kwargs={
        "decay": decay_factors, "nb_steps": 50, "nb_hidden": 12, 
        "nb_outputs": 3, "device": device, "threshold": 0.80
    },
    test_loader=test_loader
)

# 4. Evaluate
evaluator = ModelEvaluator((w_in, w_out, w_rec), device, 3)
y_true, y_pred, spk_in, spk_hid, spk_out = evaluator.collect_predictions_and_activity(
    test_loader, run_snn_kwargs
)

print(f"Final accuracy: {(y_true == y_pred).mean():.1%}")
```

---

**Questions?** → See [README.md](README.md) for documentation index
