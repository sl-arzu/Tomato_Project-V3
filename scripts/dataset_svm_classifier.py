# dataset_svm_classifier.py  
# ============================================================
# SVM CLASSIFIER per bioimpedenza piante
# ============================================================
#
# Classifica misure di bioimpedenza (parte reale + immaginaria)
# usando un Support Vector Machine (SVM).
#
# PIPELINE:
#   1. Carica il dataset (.npz) con le misure di impedenza
#   2. Filtra per piante e classi scelte (PIANTE, CLASSI)
#   3. Estrae le feature: seleziona frequenze e appiattisce a 2D
#   4. Divide in train/test (LOPO o random split)
#   5. Normalizza con Z-score (StandardScaler su train)
#   6. [Opzionale] Riduce con PCA (N_PCA_COMPONENTS)
#   7. Addestra SVM con i kernel scelti (SVM_KERNELS)
#   8. Valuta: accuracy, classification report, silhouette
#   9. Salva grafici HTML: confusion matrix, boundary/scatter 3D,
#      piani di separazione (solo linear + PCA 3D), riepilogo
#
# OUTPUTS:
#   temporal_plots/
#     cm_*.html          → confusion matrix per ogni run
#     boundary_*.html    → decision boundary 2D (se N_PCA=2)
#     scatter3d_*.html   → scatter PCA 3D (se N_PCA=3)
#     hyperplanes_*.html → piani SVM lineare 3D (se N_PCA=3)
#     summary_*.html     → riepilogo accuracy tutti i run
#
# NOTA CLASSI:
#   Se usi un sottoinsieme di classi (es. CLASSI=[0,2]),
#   le etichette vengono rimappate a indici contigui {0,1}
#   per compatibilità con sklearn. I plot mostrano i nomi
#   originali ("Control", "Late Stress") in modo automatico.
#
# REQUISITI:
#   pip install numpy scikit-learn plotly
#   Dataset: data/Water_Stress.npz  oppure  data/Iron_Stress.npz
# ============================================================

import os, sys
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              silhouette_score, accuracy_score)


# ==============================================================
# CONFIGURAZIONE — modifica solo questa sezione
# ==============================================================

# --------------------------------------------------------------
# DATASET
# --------------------------------------------------------------

DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

# Quale dataset usare: "water" oppure "iron"
STRESS_TYPE = "water"


# --------------------------------------------------------------
# SELEZIONE FREQUENZE
# --------------------------------------------------------------
# Il dataset ha 200 frequenze in scala log da 100 Hz a 10 MHz.
# Indici validi: 0 (= 100 Hz) ... 199 (= 10 MHz)
#
# Esempi:
#   0 - 2    → banda molto bassa  (~100-130 Hz)
#   0 - 49   → primo quarto dello spettro
#   0 - 199  → spettro completo

FREQ_IDX_MIN = 0     # indice frequenza minima inclusa
FREQ_IDX_MAX = 2     # indice frequenza massima inclusa

# Quale parte del segnale di impedenza usare:
#   "both" → parte reale + immaginaria (6 features con FREQ 0-2)
#   "real" → solo parte reale
#   "imag" → solo parte immaginaria
IMPEDANCE_COMPONENTS = "both"


# --------------------------------------------------------------
# SELEZIONE PIANTE E CLASSI
# --------------------------------------------------------------
# Piante disponibili nel dataset: "P0", "P1", "P3"
# Puoi escludere una pianta togliendo il suo ID dalla lista.
# Esempio: ["P0", "P1"] esclude P3 da training e test.
PIANTE = ["P0", "P1", "P3"]

# Classi da classificare:
#   0 = Control      (giorni  1 - 6,  nessuno stress)
#   1 = Early Stress (giorni  7 - 18, stress iniziale)
#   2 = Late Stress  (giorni 19 - 30, stress avanzato)
#
# Classificazione 3 classi (default):
#   CLASSI = [0, 1, 2]
#
# Classificazione binaria (esempi):
#   CLASSI = [0, 1]   → Control vs Early Stress
#   CLASSI = [0, 2]   → Control vs Late Stress
#   CLASSI = [1, 2]   → Early vs Late Stress
#
# IMPORTANTE: le etichette vengono rimappate internamente
# a indici contigui (0, 1, ...) — non serve modificare
# altro nel codice quando usi un sottoinsieme.
CLASSI = [0, 1, 2]


# --------------------------------------------------------------
# SPLIT TRAIN / TEST
# --------------------------------------------------------------
# Due modalità disponibili:
#
#   "lopo" (Leave-One-Plant-Out) → la pianta in LEAVE_PLANT
#           viene usata SOLO per il test. Le altre piante
#           vanno tutte nel training. Simula la generalizzazione
#           a una pianta mai vista.
#
#   "random" → divisione casuale stratificata su tutti i campioni.
#              Usato per baseline rapida, ma mescola piante in
#              train e test (valutazione più ottimistica).
#
SPLIT_MODE  = "lopo"

# [solo per SPLIT_MODE = "lopo"]
# Pianta da usare come test set. Deve essere presente in PIANTE.
LEAVE_PLANT = "P3"

# [solo per SPLIT_MODE = "random"]
# Percentuale di campioni da riservare al test (es. 0.30 = 30%)
TEST_SIZE = 0.30


# --------------------------------------------------------------
# MODALITA' DI CLASSIFICAZIONE
# --------------------------------------------------------------
# Puoi eseguire SVM su features raw (normalizzate) e/o su
# componenti PCA. Entrambe le modalità sono indipendenti.

# True → addestra SVM direttamente sulle features normalizzate
RUN_RAW_SVM = True

# True → riduce prima con PCA, poi addestra SVM sulle componenti
RUN_PCA_SVM = True


# --------------------------------------------------------------
# PARAMETRI SVM
# --------------------------------------------------------------
# Lista di kernel da testare. Ogni kernel produce un run separato.
#   "linear" → confine lineare, interpretabile (supporta hyperplanes 3D)
#   "rbf"    → kernel gaussiano, più flessibile, spesso più accurato
#   "poly"   → kernel polinomiale (aggiungibile se utile)
SVM_KERNELS = ["linear", "rbf"]

# C: parametro di regolarizzazione.
#   Valori bassi (0.01–1)  → margine largo, più errori tollerati (underfitting)
#   Valori alti  (10–100)  → margine stretto, meno errori (rischio overfitting)
SVM_C = 1.0

# gamma: usato solo da kernel "rbf" e "poly".
#   "scale" → 1 / (n_features * X.var())  [default consigliato]
#   "auto"  → 1 / n_features
#   float   → valore fisso (es. 0.01)
SVM_GAMMA = "scale"


# --------------------------------------------------------------
# PCA
# --------------------------------------------------------------
# Numero di componenti principali da usare (se RUN_PCA_SVM = True).
#
#   2 → abilita plot "Decision Boundary 2D" (boundary_*.html)
#       Visualizzazione più leggibile, ma meno varianza catturata
#
#   3 → abilita "Scatter 3D" e "Piani di separazione 3D"
#       (scatter3d_*.html, hyperplanes_*.html)
#       Più informativo, solo per kernel linear si vedono i piani
#
# Nota: se N_PCA_COMPONENTS = 3 e kernel = "rbf", viene prodotto
# solo lo scatter 3D (i piani non sono definibili per kernel non-lineare).
N_PCA_COMPONENTS = 3


# --------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------
# Cartella dove vengono salvati tutti i file HTML dei grafici.
OUTPUT_DIR = "temporal_plots"

# True  → apre ogni grafico nel browser subito dopo la generazione
# False → salva solo i file HTML senza aprire nulla
#          (consigliato per esecuzione da terminale su macOS/server)
SHOW_IN_BROWSER = False

# ==============================================================
# COSTANTI
# ==============================================================

LABEL_MAP    = {0: "Control", 1: "Early Stress", 2: "Late Stress"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"}
RANDOM_STATE = 42

# dataset_SVM_classifier.py
# ============================================================
# SVM CLASSIFIER per bioimpedenza piante
# Versione semplificata:
# - testa internamente tutti i kernel
# - stampa solo Migliore RAW e Migliore PCA
# - RAW: report + confusion matrix + hyperplanes SEMPRE
# - PCA: report + confusion matrix + un solo scatter PCA + hyperplanes SEMPRE
# - niente plot_summary
# - niente ranking completo
#
# Nota hyperplanes:
# Gli hyperplanes vengono SEMPRE mostrati.
# Se il miglior modello non e' lineare, il codice costruisce
# un proxy lineare 3D nel medesimo spazio di visualizzazione
# del modello vincitore, cosi' il plot non viene mai saltato.
# ============================================================

import os
import sys
import numpy as np
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier


# ==============================================================
# CONFIGURAZIONE
# ==============================================================

# --------------------------------------------------------------
# DATASET
# --------------------------------------------------------------
DATA_DIR = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE = os.path.join(DATA_DIR, "Iron_Stress.npz")

# "water" oppure "iron"
STRESS_TYPE = "water"

# --------------------------------------------------------------
# SELEZIONE FREQUENZE
# --------------------------------------------------------------
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 2

# "both" | "real" | "imag"
IMPEDANCE_COMPONENTS = "both"

# --------------------------------------------------------------
# SELEZIONE PIANTE E CLASSI
# --------------------------------------------------------------
PIANTE = ["P0", "P1", "P3"]

# 0 = Control
# 1 = Early Stress
# 2 = Late Stress
CLASSI = [0, 1, 2]

# --------------------------------------------------------------
# SPLIT TRAIN / TEST
# --------------------------------------------------------------
# "lopo" | "random"
SPLIT_MODE = "lopo"
LEAVE_PLANT = "P3"
TEST_SIZE = 0.30

# --------------------------------------------------------------
# MODALITA' DI CLASSIFICAZIONE
# --------------------------------------------------------------
RUN_RAW_SVM = True
RUN_PCA_SVM = True

# --------------------------------------------------------------
# PARAMETRI SVM
# --------------------------------------------------------------
SVM_KERNELS = ["linear", "rbf"]
SVM_C = 1.0
SVM_GAMMA = "scale"

# --------------------------------------------------------------
# PCA
# --------------------------------------------------------------
N_PCA_COMPONENTS = 3

# --------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------
OUTPUT_DIR = "temporal_plots"
SHOW_IN_BROWSER = False

# --------------------------------------------------------------
# COSTANTI
# --------------------------------------------------------------
LABEL_MAP = {0: "Control", 1: "Early Stress", 2: "Late Stress"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"}
RANDOM_STATE = 42


# ==============================================================
# UTILITY
# ==============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    print(f" [HTML] {path}")
    if SHOW_IN_BROWSER:
        fig.show()


def build_frequency_axis(n=200, fmin=100, fmax=10e6):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


def _slugify(s):
    for ch in [" ", "'", "/", "[", "]", ",", ":", "(", ")", "="]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.lower().strip("_")


def _colors_for_classes(cls_list):
    return {c: CLASS_COLORS.get(c, "#95a5a6") for c in cls_list}


def _pad_to_3d(X):
    """
    Restituisce sempre una matrice (n_samples, 3).
    - Se X ha >=3 colonne, usa le prime 3
    - Se X ha 2 colonne, aggiunge una colonna zero
    - Se X ha 1 colonna, aggiunge due colonne zero
    """
    n, d = X.shape
    if d >= 3:
        return X[:, :3]
    if d == 2:
        return np.column_stack([X, np.zeros(n)])
    if d == 1:
        return np.column_stack([X, np.zeros(n), np.zeros(n)])
    return np.zeros((n, 3), dtype=float)


def _raw_axis_titles(n_features):
    if n_features >= 3:
        return ["RAW feat 1", "RAW feat 2", "RAW feat 3"]
    if n_features == 2:
        return ["RAW feat 1", "RAW feat 2", "pad 0"]
    return ["RAW feat 1", "pad 0", "pad 0"]


def _pca_axis_titles(explained, n_comp_available):
    titles = []
    for i in range(min(3, n_comp_available)):
        titles.append(f"PC{i+1} ({explained[i]:.1f}%)")
    while len(titles) < 3:
        titles.append("pad 0")
    return titles


# ==============================================================
# VALIDAZIONE
# ==============================================================

def validate_config(plant_ids_all, y_all):
    avail_p = set(np.unique(plant_ids_all).tolist())
    avail_c = set(np.unique(y_all).tolist())

    for p in PIANTE:
        if p not in avail_p:
            sys.exit(f"[ERRORE] Pianta '{p}' non trovata. Disponibili: {sorted(avail_p)}")

    for c in CLASSI:
        if c not in avail_c:
            sys.exit(f"[ERRORE] Classe {c} non trovata. Disponibili: {sorted(avail_c)}")

    if len(CLASSI) < 2:
        sys.exit("[ERRORE] CLASSI deve contenere almeno 2 classi.")

    if SPLIT_MODE not in {"lopo", "random"}:
        sys.exit("[ERRORE] SPLIT_MODE deve essere 'lopo' oppure 'random'.")

    if SPLIT_MODE == "lopo":
        if LEAVE_PLANT not in PIANTE:
            sys.exit(f"[ERRORE] LEAVE_PLANT='{LEAVE_PLANT}' non e' in PIANTE={PIANTE}.")
        if len([p for p in PIANTE if p != LEAVE_PLANT]) == 0:
            sys.exit("[ERRORE] Nessuna pianta rimasta per il training dopo LOPO.")

    if IMPEDANCE_COMPONENTS not in {"both", "real", "imag"}:
        sys.exit("[ERRORE] IMPEDANCE_COMPONENTS deve essere 'both', 'real' o 'imag'.")

    if FREQ_IDX_MIN < 0 or FREQ_IDX_MAX > 199 or FREQ_IDX_MIN > FREQ_IDX_MAX:
        sys.exit("[ERRORE] Intervallo frequenze non valido.")

    if not RUN_RAW_SVM and not RUN_PCA_SVM:
        sys.exit("[ERRORE] Attiva almeno una tra RUN_RAW_SVM e RUN_PCA_SVM.")

    print(f"[CONFIG OK] piante={PIANTE} | classi={CLASSI} | split={SPLIT_MODE}")


# ==============================================================
# STEP 1 — CARICAMENTO
# ==============================================================

def load_data():
    if STRESS_TYPE == "water":
        path, label = WATER_FILE, "Water Stress"
    elif STRESS_TYPE == "iron":
        path, label = IRON_FILE, "Iron Stress"
    else:
        sys.exit("[ERRORE] STRESS_TYPE deve essere 'water' o 'iron'.")

    if not os.path.exists(path):
        sys.exit(f"[ERRORE] File non trovato: {path}")

    data = np.load(path, allow_pickle=True)
    X = data["X"].reshape(-1, 200, 2)
    y = data["y"].astype(int)
    plant_ids = data["plant_ids"].astype(str)

    print(
        f"[LOAD] {label}: {X.shape[0]} campioni | "
        f"piante={np.unique(plant_ids).tolist()} | "
        f"classi={dict(zip(*np.unique(y, return_counts=True)))}"
    )

    return X, y, plant_ids, label


# ==============================================================
# STEP 2 — FILTRO PIANTE + CLASSI
# ==============================================================

def apply_filters(X, y, plant_ids):
    mask = np.isin(plant_ids, PIANTE) & np.isin(y, CLASSI)

    X_f = X[mask]
    y_f = y[mask]
    ids_f = plant_ids[mask]

    if X_f.shape[0] == 0:
        sys.exit("[ERRORE] Nessun campione dopo il filtro. Controlla PIANTE e CLASSI.")

    classi_sorted = sorted(CLASSI)
    remap = {old: new for new, old in enumerate(classi_sorted)}
    y_remapped = np.array([remap[c] for c in y_f], dtype=int)
    label_map = {new: LABEL_MAP[old] for old, new in remap.items()}

    print(f"[FILTER] {X_f.shape[0]} campioni (piante={PIANTE}, classi={CLASSI})")
    for new_c, name in sorted(label_map.items()):
        print(f" indice {new_c} = '{name}': {np.sum(y_remapped == new_c)} campioni")

    return X_f, y_remapped, ids_f, label_map


# ==============================================================
# STEP 3 — FEATURES
# ==============================================================

def build_feature_matrix(X_f):
    freq_axis = build_frequency_axis()
    f0, f1 = freq_axis[FREQ_IDX_MIN], freq_axis[FREQ_IDX_MAX]

    X_freq = X_f[:, FREQ_IDX_MIN:FREQ_IDX_MAX + 1, :]

    if IMPEDANCE_COMPONENTS == "real":
        X_sel = X_freq[:, :, [0]]
    elif IMPEDANCE_COMPONENTS == "imag":
        X_sel = X_freq[:, :, [1]]
    else:
        X_sel = X_freq

    X_flat = X_sel.reshape(X_sel.shape[0], -1)

    n_comp_str = "2 comp" if IMPEDANCE_COMPONENTS == "both" else "1 comp"
    print(
        f"[FEATURES] freq indici {FREQ_IDX_MIN}-{FREQ_IDX_MAX} "
        f"({f0:.0f}-{f1:.0f} Hz) × {n_comp_str} = {X_flat.shape[1]} features"
    )

    return X_flat


# ==============================================================
# STEP 4 — SPLIT
# ==============================================================

def split_data(X_flat, y, plant_ids):
    if SPLIT_MODE == "lopo":
        mask_train = plant_ids != LEAVE_PLANT
        mask_test = plant_ids == LEAVE_PLANT

        X_tr, y_tr = X_flat[mask_train], y[mask_train]
        X_te, y_te = X_flat[mask_test], y[mask_test]

        train_plants = sorted(set(plant_ids[mask_train].tolist()))
        split_info = f"LOPO test={LEAVE_PLANT} train={train_plants}"

        if X_te.shape[0] == 0:
            sys.exit(f"[ERRORE] Nessun campione di test per '{LEAVE_PLANT}'.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_flat,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )
        split_info = f"Random train={1-TEST_SIZE:.0%} test={TEST_SIZE:.0%}"

    print(f"[SPLIT] {split_info}")
    print(f" Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")

    all_cls = sorted(np.unique(y).tolist())
    for name, y_sp in [("train", y_tr), ("test", y_te)]:
        missing = set(all_cls) - set(np.unique(y_sp).tolist())
        if missing:
            print(f" [WARN] Classi {missing} assenti nel {name} set.")

    return X_tr, X_te, y_tr, y_te, split_info


# ==============================================================
# STEP 5 — NORMALIZZAZIONE
# ==============================================================

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)
    return X_train_n, X_test_n, scaler


# ==============================================================
# STEP 6 — PCA
# ==============================================================

def apply_pca(X_train_n, X_test_n):
    n_comp = min(N_PCA_COMPONENTS, X_train_n.shape[0], X_train_n.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)

    X_tr_pca = pca.fit_transform(X_train_n)
    X_te_pca = pca.transform(X_test_n)
    explained = pca.explained_variance_ratio_ * 100

    print(
        f"[PCA] {n_comp} componenti | "
        f"varianza: {[f'{v:.1f}%' for v in explained]} | "
        f"totale={explained.sum():.1f}%"
    )

    return X_tr_pca, X_te_pca, explained, pca


# ==============================================================
# CORE — FIT SVM
# ==============================================================

def fit_svm(X_tr, X_te, y_tr, y_te, kernel, mode):
    clf = SVC(
        kernel=kernel,
        C=SVM_C,
        gamma=SVM_GAMMA,
        decision_function_shape="ovr",
        random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred) * 100

    return {
        "mode": mode,
        "kernel": kernel,
        "clf": clf,
        "y_pred": y_pred,
        "acc": acc
    }


def select_best(results):
    return max(results, key=lambda r: r["acc"]) if results else None


def print_model_report(title, result, y_true, label_map):
    cls_list = sorted(label_map.keys())

    print(f"\n=== {title} ===")
    print(f"Kernel   : {result['kernel'].upper()}")
    print(f"Accuracy : {result['acc']:.2f}%")
    print(
        classification_report(
            y_true,
            result["y_pred"],
            labels=cls_list,
            target_names=[label_map[c] for c in cls_list],
            digits=3,
            zero_division=0
        )
    )


# ==============================================================
# PLOT — CONFUSION MATRIX
# ==============================================================

def plot_confusion_matrix(y_true, y_pred, title_tag, label_map):
    cls_list = sorted(label_map.keys())
    classes = [label_map[c] for c in cls_list]
    n_cls = len(cls_list)

    cm = confusion_matrix(y_true, y_pred, labels=cls_list, normalize="true")
    cell_text = [[f"{cm[r, c] * 100:.0f}%" for c in range(n_cls)] for r in range(n_cls)]
    acc = accuracy_score(y_true, y_pred) * 100

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale="YlGnBu",
        zmin=0,
        zmax=1,
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=15),
        hovertemplate="Reale: %{y}<br>Predetto: %{x}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Confusion Matrix — {title_tag} (Acc: {acc:.1f}%)",
        template="plotly_white",
        height=480,
        xaxis_title="Predetto",
        yaxis_title="Reale",
        yaxis=dict(autorange="reversed")
    )

    save_fig(fig, f"cm_{_slugify(title_tag)}.html")


# ==============================================================
# PLOT — SCATTER PCA
# ==============================================================

def plot_pca_scatter(X_tr, y_tr, X_te, y_te, explained, title_tag, label_map):
    cls_list = sorted(label_map.keys())
    col = _colors_for_classes(cls_list)

    X_tr_3d = _pad_to_3d(X_tr)
    X_te_3d = _pad_to_3d(X_te)
    axis_titles = _pca_axis_titles(explained, X_tr.shape[1])

    fig = go.Figure()

    for c in cls_list:
        m = y_tr == c
        if np.any(m):
            fig.add_trace(go.Scatter3d(
                x=X_tr_3d[m, 0],
                y=X_tr_3d[m, 1],
                z=X_tr_3d[m, 2],
                mode="markers",
                name=f"Train {label_map[c]}",
                marker=dict(color=col[c], size=2, opacity=0.25)
            ))

    for c in cls_list:
        m = y_te == c
        if np.any(m):
            fig.add_trace(go.Scatter3d(
                x=X_te_3d[m, 0],
                y=X_te_3d[m, 1],
                z=X_te_3d[m, 2],
                mode="markers",
                name=f"Test {label_map[c]}",
                marker=dict(color=col[c], size=5, opacity=0.95, line=dict(color="black", width=1))
            ))

    fig.update_layout(
        title=f"Scatter PCA — {title_tag}",
        template="plotly_white",
        height=760,
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2]
        ),
        legend=dict(font=dict(size=11)),
        margin=dict(l=0, r=0, b=0, t=60)
    )

    save_fig(fig, f"scatter_pca_{_slugify(title_tag)}.html")


# ==============================================================
# HYPERPLANES — PROXY SEMPRE PRESENTE
# ==============================================================

def build_visual_proxy_classifier(best_result, X_tr_model, X_te_model, X_tr_vis3, X_te_vis3, y_tr, y_te):
    """
    Costruisce SEMPRE un classificatore lineare 3D per i plot.

    Strategia:
    - Se il miglior modello e' gia' linear, il proxy viene addestrato
      sulle etichette predette dal miglior modello nel suo stesso split.
    - Se il miglior modello e' non lineare, stesso approccio:
      proietta in 3D e approssima il vincitore con un proxy lineare 3D.
    - Se il vincitore collassa su una sola classe, fallback sulle etichette reali.
    """
    X_all_model = np.vstack([X_tr_model, X_te_model])
    X_all_vis3 = np.vstack([X_tr_vis3, X_te_vis3])

    y_hat_all = best_result["clf"].predict(X_all_model)
    if len(np.unique(y_hat_all)) < 2:
        y_fit = np.concatenate([y_tr, y_te])
        source = "fallback_true_labels"
    else:
        y_fit = y_hat_all
        source = f"proxy_from_best_{best_result['kernel']}"

    proxy = OneVsRestClassifier(
        LinearSVC(
            C=SVM_C,
            random_state=RANDOM_STATE,
            dual=False,
            max_iter=20000
        )
    )
    proxy.fit(X_all_vis3, y_fit)

    return proxy, source


def extract_ovr_params(proxy_clf):
    classes = proxy_clf.classes_
    estimators = proxy_clf.estimators_

    W = []
    b = []
    for est in estimators:
        W.append(est.coef_.ravel())
        b.append(float(est.intercept_.ravel()[0]))

    W = np.vstack(W)
    b = np.array(b)

    return classes, W, b


def plot_hyperplanes_always(best_result, X_tr_model, X_te_model, y_tr, y_te, X_tr_vis3, X_te_vis3, axis_titles, title_tag, label_map):
    cls_list = sorted(label_map.keys())
    col = _colors_for_classes(cls_list)

    proxy_clf, proxy_source = build_visual_proxy_classifier(
        best_result=best_result,
        X_tr_model=X_tr_model,
        X_te_model=X_te_model,
        X_tr_vis3=X_tr_vis3,
        X_te_vis3=X_te_vis3,
        y_tr=y_tr,
        y_te=y_te
    )

    proxy_classes, W, b = extract_ovr_params(proxy_clf)
    class_to_row = {c: i for i, c in enumerate(proxy_classes)}

    X_all3 = np.vstack([X_tr_vis3, X_te_vis3])

    pad = 0.30
    x_rng = np.linspace(X_all3[:, 0].min() - pad, X_all3[:, 0].max() + pad, 35)
    y_rng = np.linspace(X_all3[:, 1].min() - pad, X_all3[:, 1].max() + pad, 35)
    Xg, Yg = np.meshgrid(x_rng, y_rng)
    z_clip = (X_all3[:, 2].min() - pad, X_all3[:, 2].max() + pad)

    plane_palette = [
        ("rgba(46,204,113,0.35)", "#27ae60"),
        ("rgba(52,152,219,0.35)", "#2980b9"),
        ("rgba(231,76,60,0.35)", "#c0392b"),
        ("rgba(155,89,182,0.35)", "#8e44ad"),
        ("rgba(241,196,15,0.35)", "#f39c12"),
        ("rgba(26,188,156,0.35)", "#16a085"),
    ]

    fig = go.Figure()
    plane_idx = 0

    pairs_done = 0
    for ii, ci in enumerate(cls_list):
        for jj, cj in enumerate(cls_list):
            if jj <= ii:
                continue
            if ci not in class_to_row or cj not in class_to_row:
                continue

            wi = W[class_to_row[ci]]
            wj = W[class_to_row[cj]]
            bi = b[class_to_row[ci]]
            bj = b[class_to_row[cj]]

            w_diff = wi - wj
            b_diff = bi - bj

            if abs(w_diff[2]) < 1e-9:
                w_diff = w_diff.copy()
                w_diff[2] = 1e-9

            Zg = (-b_diff - w_diff[0] * Xg - w_diff[1] * Yg) / w_diff[2]
            Zg_clip = np.clip(Zg, *z_clip)
            in_range = (Zg >= z_clip[0]) & (Zg <= z_clip[1])

            surf_c, _ = plane_palette[plane_idx % len(plane_palette)]
            plane_idx += 1
            pairs_done += 1

            fig.add_trace(go.Surface(
                x=Xg,
                y=Yg,
                z=np.where(in_range, Zg_clip, np.nan),
                name=f"Piano: {label_map[ci]} vs {label_map[cj]}",
                opacity=0.38,
                colorscale=[[0, surf_c], [1, surf_c]],
                showscale=False,
                hovertemplate=(
                    f"Piano: {label_map[ci]} vs {label_map[cj]}"
                    "<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
                )
            ))

    if pairs_done == 0:
        # Fallback estremo: un piano orizzontale nel centro del cloud
        z0 = float(np.mean(X_all3[:, 2]))
        fig.add_trace(go.Surface(
            x=Xg,
            y=Yg,
            z=np.full_like(Xg, z0),
            name="Piano fallback",
            opacity=0.20,
            colorscale=[[0, "rgba(127,140,141,0.35)"], [1, "rgba(127,140,141,0.35)"]],
            showscale=False
        ))

    for c in cls_list:
        m = y_tr == c
        if np.any(m):
            fig.add_trace(go.Scatter3d(
                x=X_tr_vis3[m, 0],
                y=X_tr_vis3[m, 1],
                z=X_tr_vis3[m, 2],
                mode="markers",
                name=f"Train {label_map[c]}",
                marker=dict(color=col[c], size=2, opacity=0.25)
            ))

    ok_mask = best_result["y_pred"] == y_te
    err_mask = ~ok_mask

    for c in cls_list:
        m_ok = (y_te == c) & ok_mask
        m_err = (y_te == c) & err_mask

        if np.any(m_ok):
            fig.add_trace(go.Scatter3d(
                x=X_te_vis3[m_ok, 0],
                y=X_te_vis3[m_ok, 1],
                z=X_te_vis3[m_ok, 2],
                mode="markers",
                name=f"Test {label_map[c]} OK",
                marker=dict(color=col[c], size=5, opacity=0.95, line=dict(color="black", width=1))
            ))

        if np.any(m_err):
            fig.add_trace(go.Scatter3d(
                x=X_te_vis3[m_err, 0],
                y=X_te_vis3[m_err, 1],
                z=X_te_vis3[m_err, 2],
                mode="markers",
                name=f"Test {label_map[c]} ERR",
                marker=dict(color="black", size=7, symbol="x", line=dict(color=col[c], width=2))
            ))

    fig.update_layout(
        title=(
            f"Hyperplanes — {title_tag} | "
            f"best={best_result['kernel'].upper()} | "
            f"best acc={best_result['acc']:.1f}% | "
            f"{proxy_source}"
        ),
        template="plotly_white",
        height=850,
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8))
        ),
        legend=dict(font=dict(size=10), itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=70)
    )

    save_fig(fig, f"hyperplanes_{_slugify(title_tag)}.html")


# ==============================================================
# OUTPUT FINALI
# ==============================================================

def render_best_raw(best_raw, X_tr_raw, X_te_raw, y_tr, y_te, label_map, label, split_info):
    print_model_report("Migliore RAW", best_raw, y_te, label_map)

    cm_title = f"{label} | Migliore RAW | {best_raw['kernel'].upper()} | {split_info}"
    plot_confusion_matrix(y_te, best_raw["y_pred"], cm_title, label_map)

    X_tr_raw_3d = _pad_to_3d(X_tr_raw)
    X_te_raw_3d = _pad_to_3d(X_te_raw)

    hp_title = f"{label} | Hyperplanes RAW | {split_info}"
    plot_hyperplanes_always(
        best_result=best_raw,
        X_tr_model=X_tr_raw,
        X_te_model=X_te_raw,
        y_tr=y_tr,
        y_te=y_te,
        X_tr_vis3=X_tr_raw_3d,
        X_te_vis3=X_te_raw_3d,
        axis_titles=_raw_axis_titles(X_tr_raw.shape[1]),
        title_tag=hp_title,
        label_map=label_map
    )


def render_best_pca(best_pca, X_tr_pca, X_te_pca, y_tr, y_te, explained, label_map, label, split_info):
    print_model_report("Migliore PCA", best_pca, y_te, label_map)

    cm_title = f"{label} | Migliore PCA | {best_pca['kernel'].upper()} | {split_info}"
    plot_confusion_matrix(y_te, best_pca["y_pred"], cm_title, label_map)

    scatter_title = f"{label} | PCA | {split_info}"
    plot_pca_scatter(X_tr_pca, y_tr, X_te_pca, y_te, explained, scatter_title, label_map)

    X_tr_pca_3d = _pad_to_3d(X_tr_pca)
    X_te_pca_3d = _pad_to_3d(X_te_pca)

    hp_title = f"{label} | Hyperplanes PCA | {split_info}"
    plot_hyperplanes_always(
        best_result=best_pca,
        X_tr_model=X_tr_pca,
        X_te_model=X_te_pca,
        y_tr=y_tr,
        y_te=y_te,
        X_tr_vis3=X_tr_pca_3d,
        X_te_vis3=X_te_pca_3d,
        axis_titles=_pca_axis_titles(explained, X_tr_pca.shape[1]),
        title_tag=hp_title,
        label_map=label_map
    )


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("SVM CLASSIFIER — bioimpedenza piante")
    print("=" * 60)
    print(f" Dataset : {STRESS_TYPE}")
    print(f" Piante : {PIANTE}")
    print(f" Classi : {CLASSI} ({[LABEL_MAP[c] for c in CLASSI]})")
    print(f" Frequenze : {FREQ_IDX_MIN}-{FREQ_IDX_MAX}")
    print(f" Impedenza : {IMPEDANCE_COMPONENTS}")
    print(f" Split : {SPLIT_MODE}" + (f" leave={LEAVE_PLANT}" if SPLIT_MODE == "lopo" else ""))
    print(f" Kernels : {SVM_KERNELS}")
    print(f" PCA componenti : {N_PCA_COMPONENTS}")
    print(f" Grafici in : ./{OUTPUT_DIR}/")
    print("=" * 60)

    ensure_output_dir()

    X, y, plant_ids, label = load_data()
    validate_config(plant_ids, y)

    X_f, y_f, ids_f, label_map = apply_filters(X, y, plant_ids)
    X_flat = build_feature_matrix(X_f)
    X_tr, X_te, y_tr, y_te, split_info = split_data(X_flat, y_f, ids_f)
    X_tr_n, X_te_n, _ = normalize(X_tr, X_te)

    print(f"[NORM] range train: [{X_tr_n.min():.2f}, {X_tr_n.max():.2f}]")

    raw_results = []
    pca_results = []

    X_tr_pca = None
    X_te_pca = None
    explained = None

    if RUN_RAW_SVM:
        for kernel in SVM_KERNELS:
            raw_results.append(
                fit_svm(X_tr_n, X_te_n, y_tr, y_te, kernel=kernel, mode="RAW")
            )

    if RUN_PCA_SVM:
        X_tr_pca, X_te_pca, explained, _ = apply_pca(X_tr_n, X_te_n)
        for kernel in SVM_KERNELS:
            pca_results.append(
                fit_svm(X_tr_pca, X_te_pca, y_tr, y_te, kernel=kernel, mode="PCA")
            )

    best_raw = select_best(raw_results)
    best_pca = select_best(pca_results)

    if best_raw is not None:
        render_best_raw(
            best_raw=best_raw,
            X_tr_raw=X_tr_n,
            X_te_raw=X_te_n,
            y_tr=y_tr,
            y_te=y_te,
            label_map=label_map,
            label=label,
            split_info=split_info
        )

    if best_pca is not None:
        render_best_pca(
            best_pca=best_pca,
            X_tr_pca=X_tr_pca,
            X_te_pca=X_te_pca,
            y_tr=y_tr,
            y_te=y_te,
            explained=explained,
            label_map=label_map,
            label=label,
            split_info=split_info
        )

    print(f"\nGrafici salvati in: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()