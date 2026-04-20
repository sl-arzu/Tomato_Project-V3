# Dataset - Bioimpedance Measurements

Misurazioni di bioimpedenza per piante di pomodoro sotto diversi stress.

---

## 📊 File Dati

### water_stress/Water_Stress.npz
Bioimpedenza da piante sotto stress idrico.
- Campioni: 1000
- Features: 400 frequenze
- Label: 0=Control, 1=Early stress, 2=Late stress

### iron_stress/Iron_Stress.npz
Bioimpedenza da piante sotto stress da ferro.
- Campioni: 1000
- Features: 400 frequenze
- Label: 0=Control, 1=Early stress, 2=Late stress

---

## 📥 Uso

```python
from src.data_processing_manager import PlantDataManager

manager = PlantDataManager(stress_type="water")
ds_train, ds_test, metadata = manager.prepare_dataset_standard_split(
    "data/water_stress/Water_Stress.npz"
)
```

Il manager automaticamente:
1. Seleziona 6 features ottimali
2. Normalizza con statistica train
3. Applica temporal encoding
4. Split train/test (70/30)

---

🚀 Vedi [../docs/QUICK_START.md](../docs/QUICK_START.md) per primi passi
