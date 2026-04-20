# ⚙️ Parametri e Tuning

Tutti i parametri in `main.py` linee 30-85. Modifica prima di lanciare `python main.py`.

---

## 📥 Encoding (Conversione segnale → spike)

| Parametro | Valore | Effetto |
|-----------|--------|--------|
| `ENCODING_TYPE` | "rate" | "lif" = biologico (GPU) / "rate" = semplice |
| `NB_STEPS` | 150 | Durata finestra temporale (ms) |
| `DT` | 1.0 | Timestep - lasciare 1.0 |
| `GAIN_LIF` | 0.35 | Amplificazione segnale: ↑ = più spike |
| `NOISE_STD` | 1.0 | Caos (evita periodicità): ↑ = meno regolare |
| `TAU_MEM` | 18.0 | Decadimento membrana encoder (ms) |
| `TAU_SYN` | 12.0 | Decadimento sinaptico (ms) |
| `TAU_REF_ENCODER` | 3.5 | Periodo refrattario encoder (ms) |

**Problemi e soluzioni**:
- **Pochi spike?** → ↑ `GAIN_LIF` (0.35→0.40) o ↓ `TAU_MEM` (18→16)
- **Spike troppo periodici?** → ↑ `NOISE_STD` (1.0→1.2) + ↑ `TAU_REF_ENCODER` (3.5→4.0)
- **Nessun spike?** → ↑ `GAIN_LIF` e ↑ `NB_STEPS`

---

## 🧠 Rete Neurale (Hidden Layer)

| Parametro | Valore | Effetto |
|-----------|--------|--------|
| `HIDDEN_NEURONS` | 300 | Neuroni hidden: ↑ = più capacità (lento) |
| `THRESHOLD` | 0.80 | Soglia spike: ↓ = spike più facili |
| `TAU_MEM_REC` | 35.0 | Decadimento membrana hidden (ms) |
| `TAU_REF` | 2.5 | Periodo refrattario hidden (ms) |
| `W_IN_SCALE` | 0.90 | Peso input→hidden |
| `W_REC_SCALE` | 0.40 | Peso ricorrente |
| `W_OUT_SCALE` | 0.90 | Peso hidden→output |

**Problemi e soluzioni**:
- **Output layer silenzioso (0% spike)?** → ↑ `HIDDEN_NEURONS` + ↓ `THRESHOLD` (0.80→0.70)
- **Hidden layer troppo silenzioso?** → ↓ `TAU_MEM_REC` (35→25) + ↓ `THRESHOLD`
- **Segnali deboli?** → ↑ scale weights

---

## 🎓 Training

| Parametro | Valore | Effetto |
|-----------|--------|--------|
| `ALGORITHM` | "eprop" | "eprop" (migliore) o "bptt" |
| `EPOCHS` | 100 | Iterazioni training: ↑ = convergenza |
| `BATCH_SIZE` | 24 | Campioni per step - mantieni 24 |
| `LEARNING_RATE` | 0.003 | Passo gradiente: ↓ = stabile ma lento |
| `GAMMA` | 0.3 | Damping surrogate gradient - mantieni 0.3 |

**Problemi e soluzioni**:
- **Loss non diminuisce?** → ↓ `LEARNING_RATE` (0.003→0.001) + ↑ `EPOCHS`
- **Loss oscilla?** → ↓ `LEARNING_RATE`
- **Convergenza lenta?** → ↑ `EPOCHS` + ↓ `LEARNING_RATE`

---

## 🔍 Verificare la Salute dei Spike

Apri `results/training/*_raster.pdf` e cerca:

| Layer | Target | Problema | Soluzione |
|-------|--------|----------|-----------|
| **Input** | 20-30% | Nessuno spike | ↑ `GAIN_LIF` |
| **Input** | 20-30% | Troppo periodico | ↑ `NOISE_STD` |
| **Hidden** | 10-15% | Silenzioso | ↓ `THRESHOLD` |
| **Output** | 1-5% | Nessuno spike | ↑ `HIDDEN_NEURONS` |

---

## 🎯 Profili Rapidi

**Conservative** (lento, stabile):
```python
GAIN_LIF=0.30, NOISE_STD=0.8, HIDDEN_NEURONS=8, THRESHOLD=0.90, EPOCHS=20, LEARNING_RATE=0.001
```

**Aggressive** (veloce, caos):
```python
GAIN_LIF=0.40, NOISE_STD=1.2, HIDDEN_NEURONS=20, THRESHOLD=0.70, EPOCHS=15, LEARNING_RATE=0.002
```

**Default** (attuale):
```python
GAIN_LIF=0.35, NOISE_STD=1.0, HIDDEN_NEURONS=300, THRESHOLD=0.80, EPOCHS=100, LEARNING_RATE=0.003
```

---

## ⚠️ Errori Comuni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| Spike troppo periodici | Solo ↑ GAIN senza ↑ NOISE | Modifica ENTRAMBI |
| Confusione tra parametri | `TAU_REF` vs `TAU_REF_ENCODER` sono DIVERSI | Encoder ha REF_ENCODER |
| Non converge | Troppi parametri cambiati insieme | Cambia 1-2 alla volta |

---

## 🔬 Workflow Debugging

1. **Check raster plot** → Spike distribuiti bene?
2. **Check training loss** → Diminuisce?
3. **Check test accuracy** → >33% (sopra random)?
4. **Ripeti** finché soddisfatto

---

📖 Vedi [QUICK_START.md](QUICK_START.md) per primi passi
