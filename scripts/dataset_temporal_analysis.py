# dataset_temporal_analysis.py
# ============================================================
# Analisi temporale del dataset di bioimpedenza.
# Richiede day_ids_water.npy generato da dataset_temporal_inspector.py
# ============================================================

import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

OUTPUT_DIR = "temporal_plots"
SAVE_HTML  = True

LABEL_MAP = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

# Colori per classe (sfondo/zone)
CLASS_COLOR = {0: "rgba(31,208,104,0.10)",   # verde trasparente
               1: "rgba(243,156,18,0.10)",    # arancio trasparente
               2: "rgba(231,76,60,0.10)"}     # rosso trasparente

# Colori per pianta (linee del segnale)
PLANT_COLOR = {"P0": "#1a6ef5",   # blu
               "P1": "#9b59b6",   # viola
               "P3": "#e67e22"}   # arancio scuro

# Giorni reali per classe
DAYS_PER_CLASS = {
    0: list(range(1,  7)),
    1: list(range(7, 19)),
    2: list(range(19, 31)),
}

# ==============================================================
# CONFIGURAZIONE ANALISI TEMPORALE
# ==============================================================



STRESS_TYPE = "water"   # "water" oppure "iron"
# Range di giorni da visualizzare (1–30, estremi inclusi)
# Esempi:
#   Tutto       → GIORNI_MIN=1,  GIORNI_MAX=30
#   Solo Early  → GIORNI_MIN=7,  GIORNI_MAX=18
#   Transizione → GIORNI_MIN=5,  GIORNI_MAX=22
GIORNI_MIN = 1
GIORNI_MAX = 30

# Quale frequenza guardare (indice 0–199)
# 0   = 100 Hz    (zona Water Stress)
# 199 = 10 MHz    (zona Iron Stress)
FREQ_IDX = 0
# Piante da includere
PIANTE = ["P0", "P1", "P3"]



# ==============================================================
# UTILITY
# ==============================================================
def ensure_output_dir():
    if SAVE_HTML and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def save_fig(fig, filename):
    if SAVE_HTML:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(path)
        print(f"[SALVATO] {path}")

def build_frequency_axis(n=200, fmin=100, fmax=10e6):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


# ==============================================================
# CARICA DATI + DAY_IDS
# ==============================================================
def load_data_with_days(file_path, day_ids_path, label="dataset"):
    """
    Carica X, y, plant_ids dal .npz e day_ids dal .npy.
    Restituisce anche X in forma 3D (n, 200, 2).
    """
    if not os.path.exists(file_path):
        print(f"[ERRORE] File non trovato: {file_path}")
        sys.exit(1)

    if not os.path.exists(day_ids_path):
        print(f"[ERRORE] File day_ids non trovato: {day_ids_path}")
        print("         Esegui prima dataset_temporal_inspector.py")
        sys.exit(1)

    data      = np.load(file_path, allow_pickle=True)
    X         = data["X"].reshape(-1, 200, 2)  # forma 3D subito
    y         = data["y"]
    plant_ids = data["plant_ids"]
    day_ids   = np.load(day_ids_path)

    freq_axis = build_frequency_axis()

    print(f"[OK] Caricato {label}: {X.shape[0]} campioni, day_ids range {day_ids.min()}–{day_ids.max()}")

    return X, y, plant_ids, day_ids, freq_axis


# ==============================================================
# PLOT 1 — Traiettoria temporale del segnale
# ==============================================================
def plot_temporal_signal(X, y, plant_ids, day_ids, freq_axis,
                         freq_idx=0, giorni_min=1, giorni_max=30,
                         piante=None, title="Water Stress"):
    """
    Traccia Z_reale e Z_immaginaria vs giorno per ogni pianta.

    Per ogni giorno calcola la MEDIA dei campioni di quel giorno
    (perché ci sono ~18–37 misurazioni per giorno, vogliamo la tendenza).

    Aggiunge bande colorate di sfondo per distinguere le classi.

    Asse X = giorno (1→30)
    Asse Y = valore medio di impedenza a freq_idx
    """
    if piante is None:
        piante = ["P0", "P1", "P3"]

    freq_hz = freq_axis[freq_idx]
    print(f"\n[PLOT TEMPORALE] Frequenza: {freq_hz:.1f} Hz (indice {freq_idx})")
    print(f"                 Giorni: {giorni_min}–{giorni_max}")

    giorni_range = list(range(giorni_min, giorni_max + 1))

    # Due subplot: Real (sopra) e Imag (sotto)
    # shared_xaxes=True → zoom orizzontale sincronizzato
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"Z Reale @ {freq_hz:.1f} Hz",
            f"Z Immaginaria @ {freq_hz:.1f} Hz"
        ],
        vertical_spacing=0.10
    )

    # ----------------------------------------------------------
    # Bande di sfondo per le classi
    # Aggiunte PRIMA delle linee dati così stanno sotto
    # ----------------------------------------------------------
    # Ogni banda è un rettangolo colorato che copre i giorni della classe
    class_intervals = [
        (0, 1,  6,  "Control"),
        (1, 7,  18, "Early Stress"),
        (2, 19, 30, "Late Stress"),
    ]

    for class_id, g_start, g_end, nome in class_intervals:
        # Limita la banda al range selezionato
        g_start_clip = max(g_start, giorni_min)
        g_end_clip   = min(g_end,   giorni_max)
        if g_start_clip > g_end_clip:
            continue   # questa classe è fuori dal range selezionato

        for row in [1, 2]:
            fig.add_vrect(
                x0=g_start_clip - 0.5,
                x1=g_end_clip   + 0.5,
                fillcolor=CLASS_COLOR[class_id],
                layer="below",         # dietro le linee dati
                line_width=0,
                annotation_text=nome if row == 1 else "",
                annotation_position="top left",
                row=row, col=1
            )

    # ----------------------------------------------------------
    # Linee del segnale per ogni pianta
    # ----------------------------------------------------------
    for plant in piante:
        medie_real = []
        medie_imag = []
        std_real   = []
        std_imag   = []
        giorni_ok  = []

        for giorno in giorni_range:
            # Maschera: campioni di questa pianta in questo giorno
            mask = (plant_ids == plant) & (day_ids == giorno)

            if not np.any(mask):
                continue

            # Media e std dei campioni di questo giorno
            # X[mask, freq_idx, 0] → tutti i campioni di questo giorno,
            #                         alla frequenza freq_idx,
            #                         componente reale (0)
            vals_real = X[mask, freq_idx, 0]
            vals_imag = X[mask, freq_idx, 1]

            medie_real.append(vals_real.mean())
            medie_imag.append(vals_imag.mean())
            std_real.append(vals_real.std())
            std_imag.append(vals_imag.std())
            giorni_ok.append(giorno)

        giorni_ok  = np.array(giorni_ok)
        medie_real = np.array(medie_real)
        medie_imag = np.array(medie_imag)
        std_real   = np.array(std_real)
        std_imag   = np.array(std_imag)
        colore     = PLANT_COLOR[plant]

        for row, (medie, std) in enumerate(
            [(medie_real, std_real), (medie_imag, std_imag)], start=1
        ):
            # Banda ±1σ (quanto variano i campioni nello stesso giorno)
            # Costruita come area chiusa: avanti con +std, indietro con -std
            fig.add_trace(go.Scatter(
                x    = np.concatenate([giorni_ok, giorni_ok[::-1]]),
                y    = np.concatenate([medie + std, (medie - std)[::-1]]),
                fill = "toself",
                fillcolor = colore.replace("#", "rgba(").replace(
                    "rgba(", "rgba("
                ) if False else f"rgba({int(colore[1:3],16)},"
                                f"{int(colore[3:5],16)},"
                                f"{int(colore[5:7],16)},0.12)",
                line       = dict(color="rgba(0,0,0,0)"),
                showlegend = False,
                hoverinfo  = "skip",
                legendgroup= plant,
            ), row=row, col=1)

            # Linea media
            fig.add_trace(go.Scatter(
                x           = giorni_ok,
                y           = medie,
                mode        = "lines+markers",
                name        = plant,
                showlegend  = (row == 1),   # legenda solo nel subplot 1
                line        = dict(color=colore, width=2),
                marker      = dict(size=5),
                legendgroup = plant,
                hovertemplate = (
                    f"<b>{plant}</b><br>"
                    "Giorno: %{x}<br>"
                    "Valore medio: %{y:.1f}<br>"
                    f"<extra></extra>"
                )
            ), row=row, col=1)

    # Linea verticale al giorno 7 (inizio stress)
    for row in [1, 2]:
        if 7 >= giorni_min and 7 <= giorni_max:
            fig.add_vline(
                x=6.5, line_dash="dash",
                line_color="gray", line_width=1,
                row=row, col=1
            )
        if 19 >= giorni_min and 19 <= giorni_max:
            fig.add_vline(
                x=18.5, line_dash="dash",
                line_color="gray", line_width=1,
                row=row, col=1
            )

    fig.update_xaxes(title_text="Giorno", row=2, col=1)
    fig.update_xaxes(
        tickvals=giorni_range,
        ticktext=[str(g) for g in giorni_range],
        row=2, col=1
    )
    fig.update_yaxes(title_text="Z_reale (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Z_imag (Ω)",  row=2, col=1)

    fig.update_layout(
        title    = f"Traiettoria Temporale — {title} @ {freq_hz:.1f} Hz",
        height   = 700,
        template = "plotly_white",
        legend   = dict(title="Pianta"),
        hovermode= "x unified",
    )

    fig.show()
    save_fig(fig, f"temporal_signal_freq{freq_idx}_{title.replace(' ','_').lower()}.html")


# ==============================================================
# MAIN — per ora solo PASSO 1
# ==============================================================
def main():
    print("="*60)
    print("DATASET TEMPORAL ANALYSIS")
    print("="*60)

    ensure_output_dir()

    # Scegli file e day_ids in base a STRESS_TYPE
    if STRESS_TYPE == "water":
        file_path    = WATER_FILE
        day_ids_path = "day_ids_water.npy"
        label        = "Water Stress"

    elif STRESS_TYPE == "iron":
        file_path    = IRON_FILE
        day_ids_path = "day_ids_iron.npy"
        label        = "Iron Stress"
        # NOTA: devi prima eseguire dataset_temporal_inspector.py
        # puntando a IRON_FILE per generare day_ids_iron.npy

    else:
        print(f"[ERRORE] STRESS_TYPE deve essere 'water' o 'iron'.")
        print(f"         Valore trovato: '{STRESS_TYPE}'")
        sys.exit(1)

    X, y, plant_ids, day_ids, freq_axis = load_data_with_days(
        file_path    = file_path,
        day_ids_path = day_ids_path,
        label        = label
    )

    plot_temporal_signal(
        X, y, plant_ids, day_ids, freq_axis,
        freq_idx   = FREQ_IDX,
        giorni_min = GIORNI_MIN,
        giorni_max = GIORNI_MAX,
        piante     = PIANTE,
        title      = label
    )

if __name__ == "__main__":
    main()