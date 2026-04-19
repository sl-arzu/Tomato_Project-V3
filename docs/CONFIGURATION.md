# ⚙️ CONFIGURATION - Parameters & Tuning

## Parameter Groups

All parameters are in **main.py** (lines 30-85). Edit them before running training.

---

## 📥 ENCODING PARAMETERS

Located in main.py lines 30-43. Control how continuous signals → spike sequences.

### Global Encoding

| Parameter | Value | Effect | Tuning |
|-----------|-------|--------|--------|
| `ENCODING_TYPE` | "lif" | Encoder choice (lif/rate) | Use "lif" (better) |
| `NB_STEPS` | 50 | Temporal window (ms) | ↑ For longer sequences |
| `DT` | 1.0 | Timestep duration (ms) | Keep 1.0 |
| `GAIN_LIF` | 0.35 | Input amplification | ↑ = Stronger spikes |
| `GAIN_RATE` | 10.0 | Rate encoder gain | Not used if ENCODING_TYPE="lif" |
| `INPUT_SHIFT` | 3.8 | Input offset | Keep 3.8 |
| `POPULATION_SIZE` | 1 | Neurons per feature | Keep 1 |

**When to change**:
- **Too few spikes?** → Increase `GAIN_LIF` (0.35 → 0.40)
- **Too many periodic spikes?** → Increase `NOISE_STD`
- **Signals dying?** → Increase `GAIN_LIF` or `NB_STEPS`

---

### LIF Encoder Parameters (Lines 45-48)

| Parameter | Value | Effect | Biology |
|-----------|-------|--------|---------|
| `TAU_MEM` | 18.0 ms | Membrane decay time | How fast neuron "forgets" input |
| `TAU_SYN` | 12.0 ms | Synaptic decay time | Synaptic integration speed |
| `TAU_REF_ENCODER` | 3.5 ms | Refractory period | Minimum time between spikes |
| `NOISE_STD` | 1.0 | Gaussian noise level | Prevents periodic patterns |

**Tuning Guide**:

```
Problem: Spikes too periodic/regular
Solution: Increase TAU_REF_ENCODER (3.5 → 4.5) 
          + Increase NOISE_STD (1.0 → 1.2)
Result: More chaotic spike patterns

Problem: Not enough spikes
Solution: Decrease TAU_MEM (18 → 16)
          + Decrease TAU_REF_ENCODER (3.5 → 3.0)
Result: Neurons spike more easily

Problem: Spikes lost (signals die)
Solution: Increase GAIN_LIF (0.35 → 0.40)
          + Increase TAU_MEM (18 → 20)
Result: Stronger, more stable signals
```

---

## 🧠 RECURRENT NETWORK PARAMETERS

Located in main.py lines 50-79. Control hidden layer dynamics & learning.

### Architecture

| Parameter | Value | Effect | Tuning |
|-----------|-------|--------|--------|
| `HIDDEN_NEURONS` | 12 | Hidden layer size | ↑ = More capacity (but slower) |

### Neuron Dynamics

| Parameter | Value | Effect | Biology |
|-----------|-------|--------|---------|
| `TAU_MEM_REC` | 35.0 ms | Recurrent membrane decay | How fast hidden neurons integrate |
| `TAU_REF` | 2.5 ms | Refractory period (hidden) | **Note: Different from TAU_REF_ENCODER!** |
| `THRESHOLD` | 0.80 | Spike firing threshold | ↓ = Neurons spike more easily |
| `TAU_TRACE` | 80.0 ms | E-prop gradient trace | Keep 60-100 |
| `TAU_TRACE_OUT` | 105.0 ms | Output gradient trace | Keep 80-120 |

### Weight Scales

| Parameter | Value | Effect |
|-----------|-------|--------|
| `W_IN_SCALE` | 0.90 | Input → Hidden strength |
| `W_REC_SCALE` | 0.40 | Recurrent connections |
| `W_OUT_SCALE` | 0.90 | Hidden → Output strength |

All initialized as N(0, scale²). Increase if signals too weak.

**Tuning Guide**:

```
Problem: Output layer has 0% spikes
Solution: Increase HIDDEN_NEURONS (12 → 16)
          + Lower THRESHOLD (0.80 → 0.75)
          + Increase W_OUT_SCALE (0.90 → 0.95)
Result: Stronger output layer activation

Problem: Accuracy not improving
Solution: Increase EPOCHS (5 → 10 → 20)
          + Reduce LEARNING_RATE (0.003 → 0.001)
Result: More training iterations, finer updates

Problem: Hidden layer too silent (<5% spike)
Solution: Lower TAU_MEM_REC (35 → 25)
          + Lower THRESHOLD (0.80 → 0.70)
Result: Neurons fire more easily
```

---

## 🎓 TRAINING PARAMETERS

Located in main.py lines 81-85.

| Parameter | Value | Purpose | Tuning |
|-----------|-------|---------|--------|
| `ALGORITHM` | "eprop" | Learning algorithm | Use "eprop" (better) |
| `EPOCHS` | 10 | Training iterations | ↑ = Better convergence (slower) |
| `BATCH_SIZE` | 24 | Samples per update | Keep 24 |
| `LEARNING_RATE` | 0.003 | Gradient step size | ↓ = More stable, slower |
| `GAMMA` | 0.3 | Surrogate gradient damping | Keep 0.3 |

**Convergence Issues**?

```
If loss not decreasing:
  1. Increase EPOCHS (10 → 20)
  2. Reduce LEARNING_RATE (0.003 → 0.001)
  3. Check raster plot - are there enough spikes?

If loss oscillating:
  1. Reduce LEARNING_RATE (0.003 → 0.001)
  2. Increase BATCH_SIZE (24 → 32) for averaging
```

---

## 📊 Monitoring Spike Health

Check **raster plots** in `results/figures/water_lif_eprop_ep10_hid12_raster.pdf`:

```
Encoding Input Layer
  Goal: 20-30% spike density
  Regular pattern? → Increase NOISE_STD
  Empty? → Increase GAIN_LIF

Hidden Layer
  Goal: 10-15% spike density
  Empty? → Lower THRESHOLD or TAU_MEM_REC
  Saturated? → Increase THRESHOLD

Output Layer
  Goal: 1-5% spike density (starts low, improves with training)
  Empty? → Increase W_OUT_SCALE or lower THRESHOLD
```

---

## 🎯 Pre-configured Profiles

### Profile A: Conservative (Stable Learning)
```python
GAIN_LIF = 0.30
NOISE_STD = 0.8
TAU_MEM = 20.0
HIDDEN_NEURONS = 8
THRESHOLD = 0.90
EPOCHS = 20
LEARNING_RATE = 0.001
```

### Profile B: Aggressive (High Spike Activity)
```python
GAIN_LIF = 0.40
NOISE_STD = 1.2
TAU_MEM = 15.0
HIDDEN_NEURONS = 16
THRESHOLD = 0.70
EPOCHS = 15
LEARNING_RATE = 0.002
```

### Profile C: Default (Current - Balanced)
```python
GAIN_LIF = 0.35
NOISE_STD = 1.0
TAU_MEM = 18.0
HIDDEN_NEURONS = 12
THRESHOLD = 0.80
EPOCHS = 10
LEARNING_RATE = 0.003
```

---

## ⚠️ Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Changing only GAIN without NOISE | Signals too periodic | Adjust both together |
| Using `TAU_REF` for encoder | Wrong refractory period | Use `TAU_REF_ENCODER` instead |
| Not normalizing with train stats | Data leakage | Already handled by code ✓ |
| Adjusting too many parameters | Can't identify what works | Change 1-2 at a time |

---

## 🔍 Debugging Workflow

```
1. Check raster plot (spike distribution healthy?)
   └─→ If no: Adjust GAIN_LIF, NOISE_STD, THRESHOLD

2. Check training loss (decreasing?)
   └─→ If no: Adjust EPOCHS, LEARNING_RATE, BATCH_SIZE

3. Check test accuracy (>33% = above random?)
   └─→ If no: More hidden neurons, longer NB_STEPS, more data

4. Repeat until satisfied
```

---

**Next**: [API.md](API.md) - Understand the main classes
