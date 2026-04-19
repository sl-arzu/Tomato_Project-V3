# plant_analysis_explorer.py
# ================================================================
# Analisi interattiva essenziale dei dataset di bioimpedenza:
# - filtri per frequenze, classi e piante
# - rimozione selettiva di porzioni del dataset per pianta/classe
# - standardizzazione z-score
# - plot 3D dei dati grezzi:
#       asse X = frequenza
#       asse Y = parte reale
#       asse Z = parte immaginaria
#   con un punto per ogni lettura
# - PCA con versione non normalizzata e normalizzata
#
# Nessun grafico di medie, nessun LDA, nessun plot extra.
# ================================================================

import os
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# CONFIGURAZIONE — modifica solo questa sezione
# ==============================================================================
DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "water_stress", "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "iron_stress", "Iron_Stress.npz")

FREQ_IDX_MIN        = 0
FREQ_IDX_MAX        = 199
CLASSI_DA_INCLUDERE = [0, 1, 2]          # 0=Control, 1=Early, 2=Late
PIANTE_DA_INCLUDERE = ["P0", "P1", "P3"]
STRESS_TYPE         = "water"            # "water" / "iron" / "both"
N_COMPONENTS_PCA    = 3
SAVE_HTML           = True
OUTPUT_DIR          = "../results/dataset"

# ------------------------------------------------------------------------------
# RIMOZIONE PORZIONI DATASET
# ------------------------------------------------------------------------------
# Struttura:
# {
#   "P0": {0: [(0, 9)], 1: [], 2: [(5, 12)]},
#   "P1": {0: [],       1: [], 2: []},
#   "P3": {0: [],       1: [], 2: []},
# }
#
# Ogni tupla (start, end) usa indici LOCALI al gruppo (pianta, classe),
# dopo l'applicazione dei filtri base.
# Gli estremi sono inclusivi.
#
# Esempio:
# "P0": {0: [(0, 9)]}
# -> rimuove i primi 10 campioni della pianta P0 in classe Control.
# ------------------------------------------------------------------------------

REMOVE_RULES_RAW = {
    "P0": {0: [], 1: [], 2: []},
    "P1": {0: [], 1: [], 2: []},
    "P3": {0: [], 1: [], 2: []},
}

REMOVE_RULES_PCA = {
    "P0": {0: [], 1: [], 2: []},
    "P1": {0: [], 1: [], 2: []},
    "P3": {0: [], 1: [], 2: []},
}


# ==============================================================================
# COSTANTI VISUALI
# ==============================================================================
LABEL_MAP  = {0: "Control", 1: "Early Stress", 2: "Late Stress"}
COLOR_MAP  = {0: "#1FD068", 1: "#f39c12", 2: "#e74c3c"}
SYMBOL_MAP = {"P0": "circle", "P1": "square", "P3": "diamond"}


# ==============================================================================
# UTILITY
# ==============================================================================
def ensure_output_dir():
    if SAVE_HTML and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_fig(fig, filename):
    if SAVE_HTML:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(path)
        print(f"[SALVATO] {path}")


def safe_slug(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def build_frequency_axis(n_points=200, f_min=100, f_max=10e6):
    return np.logspace(np.log10(f_min), np.log10(f_max), n_points)


def validate_frequency_range(freq_min, freq_max, total_points=200):
    if freq_min < 0 or freq_max >= total_points or freq_min > freq_max:
        print(f"[ERRORE] Range frequenze non valido: {freq_min}–{freq_max}")
        sys.exit(1)


# ==============================================================================
# LOAD DATASET
# ==============================================================================
def load_dataset(file_path, label="dataset"):
    if not os.path.exists(file_path):
        print(f"[ERRORE] File non trovato: {file_path}")
        sys.exit(1)

    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    plant_ids = data["plant_ids"]

    print(f"\n[OK] Caricato {label}:")
    print(f"     Campioni : {X.shape[0]}")
    print(f"     Features : {X.shape[1]}")
    print(f"     Classi   : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"     Piante   : {np.unique(plant_ids)}")

    return X, y, plant_ids


# ==============================================================================
# FILTRI BASE
# ==============================================================================
def apply_filters(X, y, plant_ids,
                  freq_min=0, freq_max=199,
                  classes=None, plants=None):
    validate_frequency_range(freq_min, freq_max, total_points=200)

    sample_mask = np.ones(len(y), dtype=bool)

    if classes is not None:
        sample_mask &= np.isin(y, classes)

    if plants is not None:
        sample_mask &= np.isin(plant_ids, plants)

    X_f   = X[sample_mask]
    y_f   = y[sample_mask]
    ids_f = plant_ids[sample_mask]

    X_shaped = X_f.reshape(-1, 200, 2)

    freq_slice = slice(freq_min, freq_max + 1)
    X_selected = X_shaped[:, freq_slice, :]
    freq_axis  = build_frequency_axis()[freq_slice]

    print(f"\n[FILTRO] Campioni selezionati  : {X_selected.shape[0]}")
    print(f"         Frequenze selezionate  : {X_selected.shape[1]} (indici {freq_min}–{freq_max})")
    print(f"         Range Hz               : {freq_axis[0]:.1f} – {freq_axis[-1]:.1f}")
    print(f"         Shape finale X         : {X_selected.shape}")

    return X_selected, y_f, ids_f, freq_axis


# ==============================================================================
# RIMOZIONE SELETTIVA PORZIONI DATASET
# ==============================================================================
def remove_dataset_portions(X_selected, y_f, ids_f, remove_rules=None, stage_name=""):
    """
    Rimuove campioni in base a regole per pianta/classe.

    remove_rules esempio:
    {
        "P0": {0: [(0, 9)], 1: [], 2: [(5, 7)]},
        "P1": {0: [], 1: [], 2: []},
    }

    Gli indici sono LOCALI al sottogruppo (pianta, classe) e inclusivi.
    """
    if not remove_rules:
        print(f"[RIMOZIONE] Nessuna rimozione applicata per {stage_name}.")
        return X_selected, y_f, ids_f

    keep_mask = np.ones(len(y_f), dtype=bool)
    total_removed = 0

    print(f"\n[RIMOZIONE] Applicazione regole per {stage_name}")

    for plant, class_map in remove_rules.items():
        for class_id, ranges in class_map.items():
            if not ranges:
                continue

            group_idx = np.where((ids_f == plant) & (y_f == class_id))[0]

            if len(group_idx) == 0:
                print(f"  - Gruppo assente: pianta={plant}, classe={class_id}")
                continue

            removed_here = []

            for start, end in ranges:
                if end < start:
                    start, end = end, start

                start = max(0, start)
                end   = min(len(group_idx) - 1, end)

                if start > end:
                    continue

                global_to_remove = group_idx[start:end + 1]
                keep_mask[global_to_remove] = False
                removed_here.extend(global_to_remove.tolist())

            if removed_here:
                total_removed += len(removed_here)
                print(
                    f"  - Pianta={plant}, Classe={LABEL_MAP.get(class_id, class_id)}: "
                    f"rimossi {len(removed_here)} campioni"
                )

    X_out   = X_selected[keep_mask]
    y_out   = y_f[keep_mask]
    ids_out = ids_f[keep_mask]

    print(f"[RIMOZIONE] Totale campioni rimossi: {total_removed}")
    print(f"[RIMOZIONE] Campioni rimanenti    : {X_out.shape[0]}")

    if X_out.shape[0] == 0:
        print("[ERRORE] Tutti i campioni sono stati rimossi.")
        sys.exit(1)

    return X_out, y_out, ids_out


# ==============================================================================
# STANDARDIZZAZIONE Z-SCORE
# ==============================================================================
def zscore_standardize(X_selected):
    """
    Applica standardizzazione z-score feature-wise:
    z = (x - media) / deviazione_standard

    Input:
        X_selected -> shape (n_samples, n_freqs, 2)

    Output:
        X_z        -> stessa shape di input
        scaler     -> StandardScaler fitted
    """
    n_samples = X_selected.shape[0]
    X_flat = X_selected.reshape(n_samples, -1)

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_flat).reshape(X_selected.shape)

    return X_z, scaler


# ==============================================================================
# PREPARAZIONE DATI PER PLOT 3D RAW
# ==============================================================================
def build_raw_3d_trace_data(X_selected, y_f, ids_f, freq_axis):
    """
    Converte:
        X_selected -> (n_samples, n_freqs, 2)
    in punti 3D:
        x = frequenza
        y = parte reale
        z = parte immaginaria
    con un punto per ogni lettura.
    """
    n_samples, n_freqs, _ = X_selected.shape
    sample_ids = np.arange(n_samples)

    traces = []

    for class_id in sorted(np.unique(y_f)):
        for plant in sorted(np.unique(ids_f)):
            mask = (y_f == class_id) & (ids_f == plant)
            if not np.any(mask):
                continue

            X_group = X_selected[mask]
            sample_group = sample_ids[mask]

            x_vals = np.tile(freq_axis, X_group.shape[0])
            y_vals = X_group[:, :, 0].reshape(-1)
            z_vals = X_group[:, :, 1].reshape(-1)

            sample_rep = np.repeat(sample_group, n_freqs)
            freq_idx_rep = np.tile(np.arange(n_freqs), X_group.shape[0])

            customdata = np.empty((len(x_vals), 4), dtype=object)
            customdata[:, 0] = sample_rep
            customdata[:, 1] = plant
            customdata[:, 2] = LABEL_MAP[class_id]
            customdata[:, 3] = freq_idx_rep

            traces.append({
                "class_id": class_id,
                "plant": plant,
                "x": x_vals,
                "y": y_vals,
                "z": z_vals,
                "customdata": customdata,
            })

    return traces


# ==============================================================================
# PLOT DATI GREZZI 3D
# ==============================================================================
def plot_raw_3d(X_selected, y_f, ids_f, freq_axis, title="", normalized=False):
    """
    Plot 3D:
      X = frequenza
      Y = parte reale
      Z = parte immaginaria

    Un punto per ogni lettura.
    """
    norm_label = "normalizzato_zscore" if normalized else "non_normalizzato"
    print(f"\n[PLOT RAW 3D] {title} | {norm_label}")

    if X_selected.shape[0] == 0:
        print("[SKIP] Nessun campione disponibile per il plot raw 3D.")
        return

    traces_data = build_raw_3d_trace_data(X_selected, y_f, ids_f, freq_axis)
    fig = go.Figure()

    for td in traces_data:
        class_id = td["class_id"]
        plant = td["plant"]

        fig.add_trace(go.Scatter3d(
            x=td["x"],
            y=td["y"],
            z=td["z"],
            mode="markers",
            name=f"{LABEL_MAP[class_id]} | {plant}",
            marker=dict(
                size=2.5,
                color=COLOR_MAP[class_id],
                symbol=SYMBOL_MAP.get(plant, "circle"),
                opacity=0.75,
            ),
            customdata=td["customdata"],
            hovertemplate=(
                "Classe: %{customdata[2]}<br>"
                "Pianta: %{customdata[1]}<br>"
                "Campione: %{customdata[0]}<br>"
                "Indice freq locale: %{customdata[3]}<br>"
                "Frequenza: %{x:.4e} Hz<br>"
                "Reale: %{y:.6f}<br>"
                "Immaginaria: %{z:.6f}<extra></extra>"
            )
        ))

    y_title = "Parte reale (z-score)" if normalized else "Parte reale"
    z_title = "Parte immaginaria (z-score)" if normalized else "Parte immaginaria"

    fig.update_layout(
        title=f"Dati grezzi 3D — {title} — {norm_label}",
        template="plotly_white",
        height=800,
        scene=dict(
            xaxis=dict(title="Frequenza (Hz)", type="log"),
            yaxis=dict(title=y_title),
            zaxis=dict(title=z_title),
        ),
        legend=dict(title="Classe | Pianta"),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig.show()
    save_fig(fig, f"raw3d_{safe_slug(title)}_{norm_label}.html")


# ==============================================================================
# PCA
# ==============================================================================
def flatten_for_pca(X_selected):
    return X_selected.reshape(X_selected.shape[0], -1)


def run_pca(X_selected, n_components=3, already_normalized=False):
    X_flat = flatten_for_pca(X_selected)

    max_components = min(X_flat.shape[0], X_flat.shape[1], n_components)
    if max_components < 1:
        raise ValueError("Numero di componenti PCA non valido.")

    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X_flat)

    explained = pca.explained_variance_ratio_ * 100
    return X_pca, explained, pca


def plot_pca(X_selected, y_f, ids_f, title="", normalized=False, n_components=3):
    norm_label = "normalizzato_zscore" if normalized else "non_normalizzato"
    print(f"\n[PCA] {title} | {norm_label}")

    if X_selected.shape[0] < 2:
        print("[SKIP] Troppi pochi campioni per PCA.")
        return

    X_pca, explained, _ = run_pca(
        X_selected=X_selected,
        n_components=n_components,
        already_normalized=normalized
    )

    n_comp = X_pca.shape[1]

    if n_comp >= 3:
        fig = go.Figure()

        for class_id in sorted(np.unique(y_f)):
            for plant in sorted(np.unique(ids_f)):
                mask = (y_f == class_id) & (ids_f == plant)
                if not np.any(mask):
                    continue

                fig.add_trace(go.Scatter3d(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    z=X_pca[mask, 2],
                    mode="markers",
                    name=f"{LABEL_MAP[class_id]} | {plant}",
                    marker=dict(
                        size=5,
                        color=COLOR_MAP[class_id],
                        symbol=SYMBOL_MAP.get(plant, "circle"),
                        opacity=0.8,
                    ),
                    customdata=np.column_stack([ids_f[mask], y_f[mask]]),
                    hovertemplate=(
                        "Pianta: %{customdata[0]}<br>"
                        "Classe: %{customdata[1]}<br>"
                        "PC1: %{x:.6f}<br>"
                        "PC2: %{y:.6f}<br>"
                        "PC3: %{z:.6f}<extra></extra>"
                    )
                ))

        fig.update_layout(
            title=f"PCA 3D — {title} — {norm_label}",
            template="plotly_white",
            height=800,
            scene=dict(
                xaxis=dict(title=f"PC1 ({explained[0]:.2f}%)"),
                yaxis=dict(title=f"PC2 ({explained[1]:.2f}%)"),
                zaxis=dict(title=f"PC3 ({explained[2]:.2f}%)"),
            ),
            legend=dict(title="Classe | Pianta"),
            margin=dict(l=0, r=0, b=0, t=60),
        )

        fig.show()
        save_fig(fig, f"pca3d_{safe_slug(title)}_{norm_label}.html")

    elif n_comp == 2:
        fig = go.Figure()

        for class_id in sorted(np.unique(y_f)):
            for plant in sorted(np.unique(ids_f)):
                mask = (y_f == class_id) & (ids_f == plant)
                if not np.any(mask):
                    continue

                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode="markers",
                    name=f"{LABEL_MAP[class_id]} | {plant}",
                    marker=dict(
                        size=8,
                        color=COLOR_MAP[class_id],
                        symbol=SYMBOL_MAP.get(plant, "circle"),
                        opacity=0.8,
                    ),
                    customdata=np.column_stack([ids_f[mask], y_f[mask]]),
                    hovertemplate=(
                        "Pianta: %{customdata[0]}<br>"
                        "Classe: %{customdata[1]}<br>"
                        "PC1: %{x:.6f}<br>"
                        "PC2: %{y:.6f}<extra></extra>"
                    )
                ))

        fig.update_layout(
            title=f"PCA 2D — {title} — {norm_label}",
            template="plotly_white",
            height=650,
            xaxis_title=f"PC1 ({explained[0]:.2f}%)",
            yaxis_title=f"PC2 ({explained[1]:.2f}%)",
            legend=dict(title="Classe | Pianta"),
        )

        fig.show()
        save_fig(fig, f"pca2d_{safe_slug(title)}_{norm_label}.html")

    else:
        fig = go.Figure()

        for class_id in sorted(np.unique(y_f)):
            mask = (y_f == class_id)
            if not np.any(mask):
                continue

            fig.add_trace(go.Scatter(
                x=np.arange(np.sum(mask)),
                y=X_pca[mask, 0],
                mode="markers",
                name=LABEL_MAP[class_id],
                marker=dict(color=COLOR_MAP[class_id], size=8, opacity=0.8),
            ))

        fig.update_layout(
            title=f"PCA 1D — {title} — {norm_label}",
            template="plotly_white",
            height=500,
            xaxis_title="Indice campione",
            yaxis_title=f"PC1 ({explained[0]:.2f}%)",
            legend=dict(title="Classe"),
        )

        fig.show()
        save_fig(fig, f"pca1d_{safe_slug(title)}_{norm_label}.html")


# ==============================================================================
# ANALISI SINGOLO DATASET
# ==============================================================================
def analyze_dataset(file_path, stress_label):
    print("\n" + "=" * 72)
    print(f"ANALISI DATASET: {stress_label}")
    print("=" * 72)

    X, y, plant_ids = load_dataset(file_path, label=stress_label)

    X_sel, y_f, ids_f, freq_axis = apply_filters(
        X, y, plant_ids,
        freq_min=FREQ_IDX_MIN,
        freq_max=FREQ_IDX_MAX,
        classes=CLASSI_DA_INCLUDERE if CLASSI_DA_INCLUDERE else None,
        plants=PIANTE_DA_INCLUDERE if PIANTE_DA_INCLUDERE else None,
    )

    # ----------------------------------------------------------
    # RAW 3D - NON NORMALIZZATO
    # RAW 3D - NORMALIZZATO
    # con regole di rimozione dedicate al raw
    # ----------------------------------------------------------
    X_raw, y_raw, ids_raw = remove_dataset_portions(
        X_sel, y_f, ids_f,
        remove_rules=REMOVE_RULES_RAW,
        stage_name=f"{stress_label} | RAW"
    )

    plot_raw_3d(
        X_selected=X_raw,
        y_f=y_raw,
        ids_f=ids_raw,
        freq_axis=freq_axis,
        title=stress_label,
        normalized=False
    )

    X_raw_z, _ = zscore_standardize(X_raw)

    plot_raw_3d(
        X_selected=X_raw_z,
        y_f=y_raw,
        ids_f=ids_raw,
        freq_axis=freq_axis,
        title=stress_label,
        normalized=True
    )

    # ----------------------------------------------------------
    # PCA - NON NORMALIZZATA
    # PCA - NORMALIZZATA
    # con regole di rimozione dedicate alla PCA
    # ----------------------------------------------------------
    X_pca_base, y_pca, ids_pca = remove_dataset_portions(
        X_sel, y_f, ids_f,
        remove_rules=REMOVE_RULES_PCA,
        stage_name=f"{stress_label} | PCA"
    )

    plot_pca(
        X_selected=X_pca_base,
        y_f=y_pca,
        ids_f=ids_pca,
        title=stress_label,
        normalized=False,
        n_components=N_COMPONENTS_PCA
    )

    X_pca_z, _ = zscore_standardize(X_pca_base)

    plot_pca(
        X_selected=X_pca_z,
        y_f=y_pca,
        ids_f=ids_pca,
        title=stress_label,
        normalized=True,
        n_components=N_COMPONENTS_PCA
    )


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 72)
    print("PLANT BIOIMPEDANCE — RAW 3D + PCA")
    print("=" * 72)

    ensure_output_dir()

    if STRESS_TYPE == "water":
        analyze_dataset(WATER_FILE, "Water Stress")

    elif STRESS_TYPE == "iron":
        analyze_dataset(IRON_FILE, "Iron Stress")

    elif STRESS_TYPE == "both":
        analyze_dataset(WATER_FILE, "Water Stress")
        analyze_dataset(IRON_FILE, "Iron Stress")

    else:
        print(f"[ERRORE] STRESS_TYPE deve essere 'water', 'iron' o 'both'.")
        print(f"         Valore trovato: {STRESS_TYPE}")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("ANALISI COMPLETATA")
    if SAVE_HTML:
        print(f"Grafici salvati in: ./{OUTPUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
