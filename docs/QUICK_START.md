# 🚀 QUICK START - 5 Minutes

## Setup

```bash
cd /Users/harin/Developer/Amadeus/TomatoProject_Rigore
source venv_m1/bin/activate
python --version  # Verify 3.12.12
```

---

## Run Training

```bash
python main.py
```

**Output**: 
- Models → `results/models/`
- Plots → `results/figures/`
- Duration: ~10-20 minutes (10 epochs)

---

## Load & Use Data

```python
from src.data_processing_manager import PlantDataManager
from torch.utils.data import DataLoader

# Load dataset
manager = PlantDataManager(stress_type="water")
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz",
    test_size=0.3
)

# Create dataloader
train_loader = DataLoader(ds_train, batch_size=24, shuffle=True)

# Check data
for X_spikes, y_labels in train_loader:
    print(f"Spikes shape: {X_spikes.shape}")  # (24, 50, 6)
    print(f"Labels: {y_labels}")              # Class indices
    break
```

---

## Common Parameters to Modify

Edit `main.py` lines 30-85:

### 🔴 Encoding (Input Layer)
```python
GAIN_LIF = 0.35              # ↑ Higher = stronger input signal
NOISE_STD = 1.0              # ↑ Higher = more chaos (breaks periodicity)
TAU_MEM = 18.0               # ↓ Lower = faster response
TAU_REF_ENCODER = 3.5        # ↑ Higher = more refractory period
```

### 🟡 Recurrent Network (Hidden Layer)
```python
HIDDEN_NEURONS = 12          # ↑ More capacity
THRESHOLD = 0.80             # ↓ Lower = easier spiking
TAU_MEM_REC = 35.0           # ↓ Lower = faster dynamics
W_OUT_SCALE = 0.90           # ↑ Stronger output weights
```

### 🟢 Training
```python
EPOCHS = 10                  # ↑ More training
LEARNING_RATE = 0.003        # ↓ Slower learning (more stable)
```

**See [CONFIGURATION.md](CONFIGURATION.md) for detailed effects.**

---

## Monitor Results

### Check Spike Activity
```bash
# Open the raster plot
open results/figures/water_lif_eprop_ep10_hid12_raster.pdf
```

Look for:
- **Encoding Input**: 20-30% spike density
- **Hidden Layer**: 10-15% spike density
- **Output Layer**: >1% spike density

### Check Accuracy
```bash
grep "Train\|Test" results/figures/*.pdf  # Or check console output
```

---

## Debug Common Issues

### Issue: Output Layer has 0% spikes
**Solution**: Increase `HIDDEN_NEURONS` or reduce `THRESHOLD`

### Issue: Hidden Layer has too few spikes
**Solution**: Increase `GAIN_LIF` or reduce `TAU_MEM_REC`

### Issue: Training not converging
**Solution**: 
- Reduce `LEARNING_RATE` to 0.001
- Increase `EPOCHS` to 20
- Check that data is properly normalized

### Issue: Spikes too periodic
**Solution**: Increase `NOISE_STD` and `TAU_REF_ENCODER`

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Next Steps

1. ✅ Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) to understand file organization
2. ✅ Check [CONFIGURATION.md](CONFIGURATION.md) for parameter effects
3. ✅ Explore [API.md](API.md) for advanced usage

---

**Need Help?** → See [README.md](README.md) for documentation index
