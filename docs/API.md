# 🔌 API - Classi Principali

## PlantDataManager

**File**: `src/data_processing_manager.py`

Carica, normalizza, divide e applica temporal encoding.

```python
from src.data_processing_manager import PlantDataManager
from torch.utils.data import DataLoader

manager = PlantDataManager(stress_type="water", encoding_params={...})

# Split standard (70/30)
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    file_path="data/water_stress/Water_Stress.npz"
)

# Leave-One-Plant-Out (testa su pianta specifica)
ds_train, ds_test, metadata = manager.prepare_dataset_leave_one_plant_split(
    file_path="data/water_stress/Water_Stress.npz", 
    leave_plant="P3"
)

train_loader = DataLoader(ds_train, batch_size=24, shuffle=True)
```

**Return**: `(dataset_train, dataset_test, metadata)` con spike sequences e labels

---

## TemporalEncoder

**File**: `src/temp_enco_dispatcher.py`

Converte segnali continui → spike sequences.

```python
from src.temp_enco_dispatcher import TemporalEncoder

encoder = TemporalEncoder(
    encoding_type="lif",  # o "rate", "rate_hz"
    nb_steps=150,
    gain_lif=0.35,
    noise_std=1.0,
    # ... altri parametri
)

X = np.random.randn(10, 6)  # 10 campioni, 6 features
spikes = encoder.encode(X)  # Output: (10, 150, 6)
```

Usa sempre **LIF** per migliore qualità biologica (su GPU).

---

## SNNTrainer

**File**: `src/learning_trainer.py`

Training loop per SNN con E-prop o BPTT.

```python
from src.learning_trainer import SNNTrainer

trainer = SNNTrainer(
    layers=(w_in, w_out, w_rec),
    nb_outputs=3,
    algorithm="eprop",
    learning_rate=0.003
)

history, weight_history = trainer.fit(
    train_loader=train_loader,
    epochs=100,
    test_loader=test_loader
)

print(history['loss'])      # Loss per epoch
print(history['train_acc']) # Accuracy train
print(history['test_acc'])  # Accuracy test
```

---

## ModelEvaluator

**File**: `src/learning_evaluator.py`

Estrae predizioni e attività neurale.

```python
from src.learning_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    layers=(w_in, w_out, w_rec),
    nb_outputs=3,
    device=device
)

y_true, y_pred, spk_input, spk_hidden, spk_out = evaluator.collect_predictions_and_activity(
    test_loader=test_loader,
    run_snn_kwargs={...}
)

# Accedi spike:
spikes_input = spk_input[0][0]    # (150, 6) input layer
spikes_hidden = spk_hidden[0][0]  # (150, 300) hidden layer
```

---

## PlantFeatureSelector

**File**: `src/data_processing_plant_feature_selector.py`

Seleziona 6 features ottimali per tipo di stress.

```python
from src.data_processing_plant_feature_selector import PlantFeatureSelector

selector = PlantFeatureSelector(stress_type="water")

X_selected = selector.select_features(X_raw)  # (1000, 400) → (1000, 6)
X_train_norm, X_test_norm = selector.normalize_features(X_train, X_test)
```

Usa solo statistiche di training (previene data leakage).

---

## Utility

### compute_decay_factors()

**File**: `src/snn_layer_model.py`

Pre-calcola fattori di decadimento esponenziale.

```python
from src.snn_layer_model import compute_decay_factors

decay = compute_decay_factors(
    dt=1.0,
    tau_mem=18.0,
    tau_mem_rec=35.0,
    tau_syn=12.0,
    tau_trace=80.0,
    tau_trace_out=105.0
)

# Usa in run_snn_kwargs
```

---

## Flusso Completo

```python
# 1. Load data
manager = PlantDataManager(stress_type="water", encoding_params={...})
ds_train, ds_test, meta = manager.prepare_dataset_standard_split("data/water_stress/Water_Stress.npz")

# 2. Create loaders
train_loader = DataLoader(ds_train, batch_size=24, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=24, shuffle=False)

# 3. Initialize model
w_in = FeedforwardLayer.create_layer(6, 300, 0.90, device)
w_out = FeedforwardLayer.create_layer(300, 3, 0.90, device)
w_rec, _ = RecurrentLayer.create_layer(300, 300, 0.90, 0.40, device)

# 4. Train
trainer = SNNTrainer(layers=(w_in, w_out, w_rec), nb_outputs=3, algorithm="eprop")
history, weights = trainer.fit(train_loader, epochs=100, test_loader=test_loader)

# 5. Evaluate
evaluator = ModelEvaluator(layers=(w_in, w_out, w_rec), nb_outputs=3, device=device)
y_true, y_pred, _, _, _ = evaluator.collect_predictions_and_activity(test_loader, {...})
```

---

Vedi [QUICK_START.md](QUICK_START.md) per esempi rapidi
