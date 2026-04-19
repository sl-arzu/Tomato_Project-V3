# pca_temporal.py
# ============================================================
# PCA del dataset di bioimpedenza colorata per giorno.
# Mostra se lo stress è un processo continuo o discreto.
#
# Richiede:
#   - data/Water_Stress.npz  oppure  data/Iron_Stress.npz
#   - day_ids_water.npy      oppure  day_ids_iron.npy
#     (generati da dataset_temporal_inspector.py)
# ============================================================

import os
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

STRESS_TYPE   = "water"
FREQ_IDX_MIN  = 0
FREQ_IDX_MAX  = 2
N_COMPONENTS  = 2

# Se True, standardizza i dati prima di PCA (media=0, varianza=1) altriti inserire False per usare i dati originali.
STANDARDIZE   = True

# Piante da includere (es. ["P0", "P1"] oppure ["P0", "P3"] oppure ["P1", "P3"])
PIANTE        = ["P0", "P1", "P3"]

# Classi da includere (0=Control, 1=Early, 2=Late)
CLASSI        = [0, 1, 2]

# se voglio visualizzare solo alcuni giorni specifici, metti qui la lista (es. [1, 6, 7, 18, 19, 30])
GIORNI_DA_VISUALIZZARE = None
OUTPUT_DIR    = "temporal_plots"
SAVE_HTML     = True

# ==============================================================
# COSTANTI
# ==============================================================

TOTAL_FREQ  = 200
LABEL_MAP   = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

CLASS_BOUNDARIES = {
    "Control":      (1,  6),
    "Early Stress": (7,  18),
    "Late Stress":  (19, 30),
}

# ==============================================================
# UTILITY
# ==============================================================

def ensure_output_dir():
    if SAVE_HTML and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Cartella creata: {OUTPUT_DIR}/")


def save_fig(fig, filename):
    if SAVE_HTML:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(path)
        print(f"[SALVATO] {path}")


def build_frequency_axis(n=200, fmin=100, fmax=10e6):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)

# ==============================================================
# Caricamento dati
# ==============================================================

def load_data(stress_type):
    if stress_type == "water":
        file_path    = WATER_FILE
        day_ids_path = "day_ids_water.npy"
        label        = "Water Stress"
    elif stress_type == "iron":
        file_path    = IRON_FILE
        day_ids_path = "day_ids_iron.npy"
        label        = "Iron Stress"
    else:
        print(f"[ERRORE] STRESS_TYPE deve essere 'water' o 'iron'.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"[ERRORE] Dataset non trovato: {file_path}")
        sys.exit(1)
    if not os.path.exists(day_ids_path):
        print(f"[ERRORE] day_ids non trovato: {day_ids_path}")
        print(f"         Esegui prima dataset_temporal_inspector.py")
        sys.exit(1)

    data      = np.load(file_path, allow_pickle=True)
    X         = data["X"].reshape(-1, 200, 2)
    y         = data["y"]
    plant_ids = data["plant_ids"]
    day_ids   = np.load(day_ids_path)
    freq_axis = build_frequency_axis()

    print(f"\n[LOAD] Dataset      : {label}")
    print(f"       Campioni     : {X.shape[0]}")
    print(f"       Shape X      : {X.shape}")
    print(f"       Giorni range : {day_ids.min()} – {day_ids.max()}")
    print(f"       Classi       : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"       Piante       : {np.unique(plant_ids)}")

    return X, y, plant_ids, day_ids, freq_axis, label

# ==============================================================
# Prepara per PCA
# ==============================================================

def prepare_pca_input(X, y, plant_ids, day_ids,
                      freq_min, freq_max,
                      classi, piante,
                      giorni=None,
                      standardize=True):
    mask = np.ones(len(y), dtype=bool)

    if classi:
        mask &= np.isin(y, classi)
    if piante:
        mask &= np.isin(plant_ids, piante)
    if giorni is not None:
        mask &= np.isin(day_ids, giorni)

    X_f       = X[mask]
    y_f       = y[mask]
    ids_f     = plant_ids[mask]
    day_ids_f = day_ids[mask]

    freq_slice = slice(freq_min, freq_max + 1)
    X_f        = X_f[:, freq_slice, :]
    freq_sel   = build_frequency_axis()[freq_slice]

    n_freq = X_f.shape[1]
    print(f"\n[PREPARE] Campioni dopo filtro  : {X_f.shape[0]}")
    print(f"          Frequenze usate        : {n_freq}  "
          f"({freq_sel[0]:.1f} – {freq_sel[-1]:.1f} Hz)")
    print(f"          Features totali per PCA: {n_freq * 2}")
    print(f"          Piante incluse nella PCA: {np.unique(ids_f)}")
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       Conferma esplicita che TUTTE le piante selezionate
    #       entrano nel fit_transform insieme — nessuna separazione.

    n_campioni = X_f.shape[0]
    X_flat     = X_f.reshape(n_campioni, -1)

    print(f"          Shape dopo flatten     : {X_flat.shape}")

    if standardize:
        scaler  = StandardScaler()
        X_ready = scaler.fit_transform(X_flat)
        print(f"          Standardizzazione      : applicata")
        print(f"          Range dopo standard    : "
              f"[{X_ready.min():.2f}, {X_ready.max():.2f}]")
    else:
        X_ready = X_flat
        print(f"          Standardizzazione      : NON applicata")

    return X_ready, y_f, ids_f, day_ids_f

# ==============================================================
# PCA
# ==============================================================

def run_pca(X_ready, n_components):
    n_comp_max = min(n_components, X_ready.shape[0], X_ready.shape[1])

    if n_comp_max < n_components:
        print(f"[PCA] Attenzione: richieste {n_components} componenti "
              f"ma il massimo è {n_comp_max}.")

    pca   = PCA(n_components=n_comp_max)
    X_pca = pca.fit_transform(X_ready)
    #       ^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       fit_transform su TUTTI i campioni di tutte le piante
    #       insieme: la PCA è una sola, condivisa.

    explained = pca.explained_variance_ratio_ * 100

    print(f"\n[PCA] Componenti calcolate    : {n_comp_max}")
    print(f"      Varianza per componente  : "
          f"{[f'{v:.1f}%' for v in explained]}")
    print(f"      Varianza totale catturata: {explained.sum():.1f}%")
    print(f"      Shape output X_pca       : {X_pca.shape}")

    return X_pca, explained, pca

# ==============================================================
# PLOT
# ==============================================================

def plot_pca_by_day(X_pca, y_f, ids_f, day_ids_f, explained,
                    title="", standardized=True):
    n_comp     = X_pca.shape[1]
    norm_label = "standardizzato" if standardized else "non_standardizzato"

    # ▼▼▼ MAPPE SIMBOLI — una per 3D, una per 2D ▼▼▼
    symbol_map_3d = {"P0": "circle",   "P1": "square",   "P3": "diamond"}
    symbol_map_2d = {"P0": "circle",   "P1": "square",   "P3": "diamond"}
    # In go.Scatter (2D) i simboli validi sono: "circle", "square",
    # "diamond", "cross", "x", "triangle-up", ecc.
    # NON si possono usare nomi arbitrari come "P0", "P1".

    print(f"\n[PLOT PCA BY DAY] {title} | {norm_label} | {n_comp} componenti")

    fig = go.Figure()

    piante_ordinate = sorted(np.unique(ids_f))
    prima_pianta    = piante_ordinate[0]

    for plant in piante_ordinate:
        mask_p = ids_f == plant

        if not np.any(mask_p):
            continue

        hover_texts = [
            f"Giorno: {day_ids_f[i]}<br>"
            f"Classe: {LABEL_MAP[y_f[i]]}<br>"
            f"Pianta: {plant}"
            for i in np.where(mask_p)[0]
        ]

        # Colorbar con tickvals mostrata solo per il primo trace
        colorbar_cfg = dict(
            title     = "Giorno",
            thickness = 15,
            len       = 0.7,
            tickvals  = [1, 6, 7, 18, 19, 30],
            ticktext  = [
                "1 (inizio Control)",
                "6 (fine Control)",
                "7 (inizio Early)",
                "18 (fine Early)",
                "19 (inizio Late)",
                "30 (fine Late)",
            ],
        )
        show_scale = (plant == prima_pianta)

        # ----------------------------------------------------------
        if n_comp >= 3:
        # ----------------------------------------------------------
            fig.add_trace(go.Scatter3d(
                x    = X_pca[mask_p, 0],
                y    = X_pca[mask_p, 1],
                z    = X_pca[mask_p, 2],
                mode = "markers",
                name = plant,
                marker = dict(
                    size       = 4,
                    symbol     = symbol_map_3d.get(plant, "circle"),
                    opacity    = 0.85,
                    color      = day_ids_f[mask_p],
                    colorscale = "Turbo",
                    cmin       = 1,
                    cmax       = 30,
                    showscale  = show_scale,
                    colorbar   = colorbar_cfg,
                ),
                text          = hover_texts,
                hovertemplate = "%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
            ))

        # ----------------------------------------------------------
        else:  # n_comp == 2
        # ----------------------------------------------------------
            fig.add_trace(go.Scatter(
                x    = X_pca[mask_p, 0],
                y    = X_pca[mask_p, 1],
                mode = "markers",
                name = plant,
                marker = dict(
                    size       = 7,
                    # ▼ FIX: usa la mappa, non la stringa "P0"/"P1"/"P3"
                    symbol     = symbol_map_2d.get(plant, "circle"),
                    opacity    = 0.85,
                    color      = day_ids_f[mask_p],
                    colorscale = "Turbo",
                    cmin       = 1,
                    cmax       = 30,
                    showscale  = show_scale,
                    colorbar   = colorbar_cfg,
                ),
                text          = hover_texts,
                hovertemplate = "%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            ))

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------
    if n_comp >= 3:
        fig.update_layout(
            title    = f"PCA per Giorno — {title} — {norm_label}",
            template = "plotly_white",
            height   = 800,
            scene    = dict(
                xaxis = dict(title=f"PC1 ({explained[0]:.1f}%)"),
                yaxis = dict(title=f"PC2 ({explained[1]:.1f}%)"),
                zaxis = dict(title=f"PC3 ({explained[2]:.1f}%)"),
            ),
            legend = dict(title="Pianta", x=0.0, y=0.5),
            margin = dict(l=0, r=0, b=0, t=50),
        )
    else:
        fig.update_layout(
            title       = f"PCA per Giorno — {title} — {norm_label}",
            template    = "plotly_white",
            height      = 650,
            xaxis_title = f"PC1 ({explained[0]:.1f}%)",
            yaxis_title = f"PC2 ({explained[1]:.1f}%)",
            legend      = dict(title="Pianta"),
        )

    fig.show()
    save_fig(fig, f"pca_by_day_{title.replace(' ','_').lower()}_{norm_label}.html")

# ==============================================================
# MAIN
# ==============================================================

def main():
    print("="*60)
    print("PCA TEMPORALE")
    print("="*60)

    ensure_output_dir()

    X, y, plant_ids, day_ids, freq_axis, label = load_data(STRESS_TYPE)

    X_ready, y_f, ids_f, day_ids_f = prepare_pca_input(
        X, y, plant_ids, day_ids,
        freq_min    = FREQ_IDX_MIN,
        freq_max    = FREQ_IDX_MAX,
        classi      = CLASSI,
        piante      = PIANTE,
        giorni      = GIORNI_DA_VISUALIZZARE,
        standardize = STANDARDIZE,
    )

    X_pca, explained, pca = run_pca(X_ready, N_COMPONENTS)

    plot_pca_by_day(
        X_pca, y_f, ids_f, day_ids_f, explained,
        title        = label,
        standardized = STANDARDIZE,
    )

    print("\n" + "="*60)
    print("COMPLETATO")
    print("="*60)


if __name__ == "__main__":
    main()