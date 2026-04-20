# 📚 Documentazione TomatoProject V3

**Navigazione**:
- 🚀 [QUICK_START.md](QUICK_START.md) — Inizia in 5 minuti
- 📁 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — Organizzazione progetto
- ⚙️ [CONFIGURATION.md](CONFIGURATION.md) — Parametri e tuning
- 🔌 [API.md](API.md) — Classi e metodi principali

---

## 📖 Cosa Fa il Progetto

Classifica lo stress idrico e da ferro in piante di pomodoro usando **Spiking Neural Networks (SNN)** su misurazioni di bioimpedenza.

**Stack**: PyTorch 2.0+ | Python 3.12 | M1 Mac ottimizzato

---

## 🚀 Primo Avvio

```bash
cd /Users/harin/Developer/Amadeus/Tomato_Project-V3
source venv_m1/bin/activate
python main.py
```

Risultati in `results/` (modelli e grafici).

---

## 📁 Struttura Cartelle

```
Tomato_Project-V3/
├── main.py              ← Punto di ingresso
├── src/                 ← Codice principale (13 moduli)
├── data/                ← Dataset (water_stress, iron_stress)
├── results/             ← Output (auto-generato)
├── tests/               ← Unit test
├── config/              ← File YAML
└── docs/                ← Questa documentazione
```

---

## 🎯 Task Comuni

### Adattare Parametri
Modifica `main.py` linee 30-85:
- **Encoding**: `GAIN_LIF`, `NOISE_STD`, `TAU_MEM`
- **Rete**: `HIDDEN_NEURONS`, `THRESHOLD`, `EPOCHS`

Vedi [CONFIGURATION.md](CONFIGURATION.md) per effetti.

### Controllare i Risultati
```bash
open results/training/water_lif_eprop_ep100_hid300_raster.pdf
```

Verificare:
- Input layer: 20-30% spike density ✓
- Hidden layer: 10-15% spike density ✓
- Output layer: >1% spike density ✓

### Eseguire Test
```bash
pytest tests/ -v
```

---

## 🔑 Concetti Chiave

| Componente | Funzione |
|-----------|----------|
| **Temporal Encoding** | Bioimpedance continua → spike discreti (LIF o Rate) |
| **SNN** | Input (6) → Hidden (300) → Output (3) |
| **E-prop** | Algoritmo di training con eligibility traces |
| **Results** | Modelli `.pt` + PDF con grafici |

---

## ⚠️ Note Importanti

1. Normalizzazione usa solo statistica train (no data leakage)
2. Device auto: MPS (M1) > CUDA > CPU
3. Parametri encoder ≠ parametri rete (`TAU_REF_ENCODER` vs `TAU_REF`)

---

**Aggiornato**: 20 Aprile 2026
