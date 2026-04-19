# Scripts Directory

Repository di script per l'analisi preliminare del dataset e elaborazione dati.

## 📁 Struttura (FLAT - no subdirectories)

```
scripts/
├── __init__.py
├── README.md (questo file)
├── ORGANIZATION.md           ← Dettagli riorganizzazione
├── dataset_config.py         ← Configurazione condivisa (importare)
└── dataset_*.py              ← 8 script di analisi
```

---

## 🔧 Configurazione Condivisa

**`dataset_config.py`**: Modulo condiviso per path e costanti

Contiene:
- Path automatici per data/ e results/
- Costanti: LABEL_MAP, PLANT_COLORS, DAYS_PER_CLASS
- Utility: ensure_output_dir(), build_frequency_axis()

**Utilizzo nei tuoi script**:
```python
from dataset_config import DATA_DIR, DATASET_ANALYSIS_DIR, LABEL_MAP
```

---

## 📊 Dataset Analysis Scripts

Tutti gli script devono essere **eseguiti da root**:

```bash
cd /Users/harin/Developer/Amadeus/TomatoProject_v2
python scripts/dataset_NOME.py
```

### 1. **dataset_plant_explorer.py**

**Scopo**: Analisi interattiva del dataset di bioimpedenza PRIMA dell'addestramento

**Features**:
- Caricamento dataset da file .npz
- Filtraggio per frequenze, classi e piante
- Rimozione selettiva di campioni
- Standardizzazione z-score
- Plot 3D interattivo
- PCA analysis (normalizzata e non)

**Output**: `results/dataset/` (file HTML)

**Configurazione** (modifica nel file):
```python
STRESS_TYPE = "water"           # or "iron"
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 199
CLASSI_DA_INCLUDERE = [0, 1, 2]
```

---

### 2. **dataset_temporal_inspector.py**

**Scopo**: Esplorare struttura temporale del dataset

**Features**:
- Analisi della struttura (campioni, piante, classi)
- Ordine dei campioni nel file
- Ricostruzione day_ids
- Verifica coerenza
- Trend del segnale nel tempo

**Output**: 
- File `day_ids_water.npy` o `day_ids_iron.npy` (in root)
- Print di statistiche e debug

**Configurazione**:
```python
STRESS_TYPE = "water"  # or "iron"
```

**Note**: Genera `day_ids_*.npy` necessari per altri script!

---

### 3. **dataset_temporal_analysis.py**

**Scopo**: Analisi temporale del segnale con tracciamento giornaliero

**Features**:
- Plot traiettoria temporale (Z_reale vs giorno)
- Bande di sfondo per classi
- Medie giornaliere con deviazione standard
- Separato per pianta e parte (reale/immaginaria)

**Output**: `results/dataset/` (file HTML)

**Prerequisito**: Genera prima `day_ids_*.npy` con `dataset_temporal_inspector.py`

**Configurazione**:
```python
STRESS_TYPE = "water"
FREQ_IDX = 0                # quale frequenza visualizzare
GIORNI_MIN = 1
GIORNI_MAX = 30
PIANTE = ["P0", "P1", "P3"]
```

---

### 4. **dataset_pca_temporal.py**

**Scopo**: PCA colorata per giorno (mostra se stress è continuo/discreto)

**Features**:
- PCA su dataset completo
- Punti colorati per giorno
- Plot 2D o 3D interattivo
- Visualizza progressione temporale nello spazio PCA

**Output**: `results/dataset/` (file HTML)

**Configurazione**:
```python
STRESS_TYPE = "water"
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 2
N_COMPONENTS = 2  # (2 per 2D, 3 per 3D)
PIANTE = ["P0", "P1", "P3"]
CLASSI = [0, 1, 2]
STANDARDIZE = True
```

---

### 5. **dataset_pca_comparative.py**

**Scopo**: PCA comparativa con overlay di piante escluse

**Logica**:
- PCA fittata su un **gruppo di riferimento** (es. P0+P1)
- Altre piante proiettate come **overlay** (es. P3)
- Mostra differenze dinamiche tra piante

**Output**: `results/dataset/` (file HTML)

**Configurazione**:
```python
REFERENCE_GROUPS = [
    ["P0"],           # PCA su P0, overlay P1+P3
    ["P0", "P1"],     # PCA su P0+P1, overlay P3
]
ALL_PLANTS = ["P0", "P1", "P3"]
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 2
```

---

### 6. **dataset_spectrum_explorer.py**

**Scopo**: Visualizzazione dello spettro di bioimpedenza

**Plots**:
- 2D: Z_reale vs Frequenza
- 2D: Z_immaginaria vs Frequenza
- 3D: Frequenza + Z_reale + Z_immaginaria

**Output**: `results/dataset/` (file HTML con subplot)

**Prerequisito**: `day_ids_*.npy` (da dataset_temporal_inspector.py)

**Configurazione**:
```python
STRESS_TYPE = "water"
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 19
PIANTE = ["P0"]
CLASSI = [0, 1, 2]
MODALITA = "media"  # "media" o "singole"
NORMALIZZA = False
```

---

### 7. **dataset_reading_viewer.py**

**Scopo**: Visualizzare dataset con numero di lettura locale come asse X

**Features**:
- Plot Real vs lettura locale
- Plot Imag vs lettura locale
- Uno per ogni frequenza selezionata
- Tracce separate per pianta e classe

**Output**: `results/dataset/` (file HTML per ogni frequenza)

**Configurazione**:
```python
STRESS_TYPE = "water"
WATER_FREQ_IDXS = [0, 1, 2]
IRON_FREQ_IDXS = [197, 198, 199]
```

---

### 8. **dataset_svm_classifier.py**

**Scopo**: Classificazione SVM su features di bioimpedenza

**Pipeline**:
1. Carica dataset
2. Filtra e normalizza (Z-score)
3. [Opzionale] PCA
4. Addestra SVM (linear + RBF)
5. Valuta e visualizza

**Output**: `results/dataset/` (confusion matrix, scatter, hyperplanes, summary)

**Modalità**:
- **LOPO** (Leave-One-Plant-Out): Plant in LEAVE_PLANT solo nel test
- **Random**: Split casuale

**Configurazione**:
```python
STRESS_TYPE = "water"
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 2
PIANTE = ["P0", "P1", "P3"]
CLASSI = [0, 1, 2]
SPLIT_MODE = "lopo"  # o "random"
LEAVE_PLANT = "P3"
RUN_PCA_SVM = True
N_PCA_COMPONENTS = 3
SVM_KERNELS = ["linear", "rbf"]
```

---

## 🎯 Workflow Consigliato

### Step 1: Inspect Temporal Structure
```bash
python scripts/dataset_temporal_inspector.py
# Genera: day_ids_water.npy o day_ids_iron.npy
```

### Step 2: Explore Dataset Visually
```bash
python scripts/dataset_plant_explorer.py
# Output: 3D scatter + PCA visualization
```

### Step 3: Analyze Temporal Progression
```bash
python scripts/dataset_temporal_analysis.py
# Output: Signal trajectory over days
```

### Step 4: PCA Analysis
```bash
python scripts/dataset_pca_temporal.py
python scripts/dataset_pca_comparative.py
# Output: PCA projections with temporal coloring
```

### Step 5: Spectral Analysis
```bash
python scripts/dataset_spectrum_explorer.py
python scripts/dataset_reading_viewer.py
# Output: Impedance spectrum visualization
```

### Step 6: Classification
```bash
python scripts/dataset_svm_classifier.py
# Output: Confusion matrix, decision boundaries, metrics
```

---

## 📝 Come aggiungere nuovi script

1. **Nome**: Inizia con `dataset_` per uniformità
2. **Location**: Direttamente in `scripts/` (FLAT)
3. **Imports**: Usa `dataset_config.py` per i path
   ```python
   from dataset_config import DATA_DIR, DATASET_ANALYSIS_DIR
   ```
4. **Esecuzione**: Sempre da root
   ```bash
   python scripts/dataset_mio_script.py
   ```

---

## 🔗 Variabili d'Ambiente

Se esegui gli script da directory diverse, assicurati che `dataset_config.py` trovi la root correttamente:

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)               # TomatoProject_v2/
```

Questo funziona indipendentemente da dove esegui lo script.

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'dataset_config'"
→ Verifica import: `from dataset_config import ...`

### "FileNotFoundError: data/water_stress/Water_Stress.npz"
→ Assicurati di eseguire da root: `cd TomatoProject_v2 && python scripts/...`

### "day_ids_water.npy not found"
→ Esegui prima `dataset_temporal_inspector.py` per generare il file

### Path dei risultati sbagliati
→ Tutti gli script usano ora `dataset_config.py` per path automatici. Se serve personalizzare, modifica lì.

---

**Last Updated**: 7 Aprile 2026  
**Max Scripts**: 8 dataset analysis scripts (flat structure)  
**Common Prefix**: `dataset_` ✅  
**Shared Config**: `dataset_config.py` ✅

- **Nome script**: `{categoria}_{azione}_{oggetto}.py`
- **Output**: Sempre in `results/{categoria}/`
- **Input**: Sempre da `data/` (non modificato)
- **Logging**: Print con `[CATEGORIA]` prefix

---

## 🔄 Workflow Consigliato

1. **Esplorare il dataset**:
   ```bash
   python scripts/dataset_analysis/plant_explorer_bioimpedance.py
   ```
   → Visualizza i dati in forma 3D e PCA

2. **Analizzare le frequenze**:
   - Modifica `FREQ_IDX_MIN` e `FREQ_IDX_MAX` per zoomare su specifiche frequenze
   - Riesegui per confrontare

3. **Rimuovere campioni anomali** (opzionale):
   - Modifica `REMOVE_RULES_RAW` e `REMOVE_RULES_PCA`
   - Riesegui per vedere l'effetto

4. **Eseguire il training**:
   ```bash
   python main.py --stress water --epochs 50
   ```

---

## 📚 Riferimenti

Per dettagli sulla struttura del progetto, vedi: [REFACTORING_COMPLETE.md](../../REFACTORING_COMPLETE.md)

Per la documentazione della rete neurale, vedi: [docs/](../../docs/)
