# dataset_plotly_reading_viewer.py
# ============================================================
# Obiettivo:
# Visualizzare tutto il dataset con Plotly usando come asse X
# il numero di lettura locale dentro ogni gruppo (pianta, classe),
# evitando per ora la ricostruzione artificiale dei day_ids.
#
# Output:
# - un file HTML per ogni frequenza selezionata
# - due subplot verticali:
#     1) parte reale
#     2) parte immaginaria
# - tracce separate per pianta e classe
#
# Dataset supportati:
# - Water Stress  -> frequenze [0, 1, 2]
# - Iron Stress   -> frequenze [198, 199]  # come richiesto
# ============================================================

import os
import sys
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

STRESS_TYPE = "water"   # "water" oppure "iron"

# Water -> primi 3 indici
WATER_FREQ_IDXS = [0, 1, 2]

# Iron -> ultimi 2 indici, come richiesto da te
# Se vuoi anche l'ultimo terzo indice, puoi cambiare in [197, 198, 199]
IRON_FREQ_IDXS = [197, 198, 199]

OUTPUT_DIR = "./plotly_reading_viewer_output"
SHOW_FIGURES = True

PLANT_ORDER = ["P0", "P1", "P3"]
LABEL_MAP = {
    0: "Control",
    1: "Early Stress",
    2: "Late Stress",
}

PLANT_COLORS = {
    "P0": "#1f77b4",   # blu
    "P1": "#9467bd",   # viola
    "P3": "#ff7f0e",   # arancio
}

CLASS_DASH = {
    0: "solid",
    1: "dot",
    2: "dash",
}

CLASS_SYMBOL = {
    0: "circle",
    1: "square",
    2: "diamond",
}


# ============================================================
# UTILS
# ============================================================
def approx_freq_label(freq_idx: int) -> str:
    """
    Etichette note dalla documentazione:
    0   -> 100 Hz
    10  -> ~170 Hz
    100 -> ~31 kHz
    190 -> ~4.7 MHz
    199 -> 10 MHz

    Per gli altri indici mostriamo semplicemente freq_idx.
    """
    known = {
        0: "100 Hz",
        1: "vicino a 100 Hz",
        2: "basse frequenze",
        10: "~170 Hz",
        100: "~31 kHz",
        190: "~4.7 MHz",
        197: "alta frequenza",
        198: "alta frequenza",
        199: "10 MHz",
    }
    return known.get(freq_idx, f"freq_idx={freq_idx}")


def get_dataset_config():
    if STRESS_TYPE == "water":
        return WATER_FILE, WATER_FREQ_IDXS, "Water Stress"
    elif STRESS_TYPE == "iron":
        return IRON_FILE, IRON_FREQ_IDXS, "Iron Stress"
    else:
        raise ValueError("STRESS_TYPE deve essere 'water' oppure 'iron'")


def load_dataset(file_path):
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]              # (n, 400)
    y = data["y"]              # (n,)
    plant_ids = data["plant_ids"]  # (n,)
    X3d = X.reshape(-1, 200, 2)    # (n, 200, 2) -> [freq_idx, real/imag]
    return X, X3d, y, plant_ids


def make_customdata(global_idx, local_idx, plant, cls, freq_idx):
    n = len(global_idx)
    custom = np.empty((n, 5), dtype=object)
    custom[:, 0] = global_idx
    custom[:, 1] = local_idx
    custom[:, 2] = plant
    custom[:, 3] = LABEL_MAP[cls]
    custom[:, 4] = freq_idx
    return custom


# ============================================================
# PLOT
# ============================================================
def build_plot_for_frequency(X3d, y, plant_ids, freq_idx, dataset_label):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            f"Parte reale — freq_idx={freq_idx} ({approx_freq_label(freq_idx)})",
            f"Parte immaginaria — freq_idx={freq_idx} ({approx_freq_label(freq_idx)})"
        )
    )

    for plant in PLANT_ORDER:
        for cls in sorted(np.unique(y)):
            mask = (plant_ids == plant) & (y == cls)
            idx_global = np.where(mask)[0]

            if len(idx_global) == 0:
                continue

            local_idx = np.arange(1, len(idx_global) + 1)

            y_real = X3d[idx_global, freq_idx, 0]
            y_imag = X3d[idx_global, freq_idx, 1]

            trace_name = f"{plant} | {LABEL_MAP[cls]}"
            customdata = make_customdata(idx_global, local_idx, plant, cls, freq_idx)

            common_kwargs = dict(
                mode="lines+markers",
                name=trace_name,
                legendgroup=trace_name,
                line=dict(
                    color=PLANT_COLORS[plant],
                    dash=CLASS_DASH[cls],
                    width=2.0,
                ),
                marker=dict(
                    color=PLANT_COLORS[plant],
                    symbol=CLASS_SYMBOL[cls],
                    size=5,
                    opacity=0.85,
                ),
                customdata=customdata,
                hovertemplate=(
                    "Pianta: %{customdata[2]}<br>"
                    "Classe: %{customdata[3]}<br>"
                    "freq_idx: %{customdata[4]}<br>"
                    "Indice lettura locale: %{customdata[1]}<br>"
                    "Indice globale dataset: %{customdata[0]}<br>"
                    "Valore: %{y:.4f}<extra></extra>"
                )
            )

            fig.add_trace(
                go.Scattergl(
                    x=local_idx,
                    y=y_real,
                    showlegend=True,
                    **common_kwargs
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scattergl(
                    x=local_idx,
                    y=y_imag,
                    showlegend=False,
                    **common_kwargs
                ),
                row=2,
                col=1
            )

    fig.update_layout(
        title=(
            f"{dataset_label} — Visualizzazione per numero di lettura locale "
            f"(freq_idx={freq_idx}, {approx_freq_label(freq_idx)})"
        ),
        template="plotly_white",
        height=950,
        width=1500,
        hovermode="closest",
        legend_title="Pianta | Classe",
    )

    fig.update_xaxes(
        title_text="Numero lettura locale nel gruppo (pianta, classe)",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Numero lettura locale nel gruppo (pianta, classe)",
        row=2, col=1
    )

    fig.update_yaxes(title_text="Z_reale", row=1, col=1)
    fig.update_yaxes(title_text="Z_immaginaria", row=2, col=1)

    return fig


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_path, freq_idxs, dataset_label = get_dataset_config()

    print("=" * 70)
    print("DATASET PLOTLY READING VIEWER")
    print("=" * 70)
    print(f"[CONFIG] Dataset      : {dataset_label}")
    print(f"[CONFIG] File         : {file_path}")
    print(f"[CONFIG] Frequenze    : {freq_idxs}")
    print(f"[CONFIG] Output dir   : {OUTPUT_DIR}")
    print(f"[CONFIG] Show figures : {SHOW_FIGURES}")

    if not os.path.exists(file_path):
        print(f"[ERRORE] File non trovato: {file_path}")
        sys.exit(1)

    X, X3d, y, plant_ids = load_dataset(file_path)

    print("\n[DATA]")
    print(f"  X shape         : {X.shape}")
    print(f"  X reshaped      : {X3d.shape}")
    print(f"  y shape         : {y.shape}")
    print(f"  plant_ids shape : {plant_ids.shape}")
    print(f"  Classi uniche   : {np.unique(y)}")
    print(f"  Piante uniche   : {np.unique(plant_ids)}")

    for freq_idx in freq_idxs:
        fig = build_plot_for_frequency(X3d, y, plant_ids, freq_idx, dataset_label)

        out_html = os.path.join(
            OUTPUT_DIR,
            f"{STRESS_TYPE}_freqidx_{freq_idx:03d}_reading_index_view.html"
        )
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"[SALVATO] {out_html}")

        if SHOW_FIGURES:
            fig.show()

    print("\nFatto.")
    print("Hai un file HTML per ogni frequenza selezionata.")


if __name__ == "__main__":
    main()