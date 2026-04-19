# spectrum_explorer.py
# ============================================================
# Visualizzazione dello spettro di bioimpedenza con:
#   - Plot 2D: Z_reale vs Frequenza
#   - Plot 2D: Z_immaginaria vs Frequenza
#   - Plot 3D: Frequenza + Z_reale + Z_immaginaria
#
# Opzioni di configurazione:
#   - Scelta dataset (water / iron)
#   - Filtro per giorni, classi, piante
#   - Media per giorno  oppure  tutte le misurazioni singole
#   - Dati normalizzati (z-score) oppure grezzi
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
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

STRESS_TYPE = "water"      # "water" oppure "iron"

# Range di frequenze da visualizzare (indici 0–199)
# 0   = 100 Hz  |  199 = 10 MHz
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 19

# Piante da includere
PIANTE = ["P0"]

# Classi da includere (0=Control, 1=Early, 2=Late)
CLASSI = [0, 1, 2]

# Giorni da visualizzare
# None              → tutti i giorni
# [1, 7, 19]        → solo quei giorni specifici
# list(range(7,13)) → giorni 7,8,9,10,11,12
GIORNI_DA_VISUALIZZARE = [1, 6, 7, 18, 19, 30]

# Modalità di visualizzazione delle misurazioni
# "media"    → una linea per giorno (media di tutte le misurazioni del giorno)
# "singole"  → un punto per ogni misurazione individuale
MODALITA = "media"       # "media" oppure "singole"

# Normalizzazione z-score
# True  → standardizza (media=0, std=1) prima di plottare
# False → valori grezzi in Ohm
NORMALIZZA = False

# Dove salvare i grafici HTML
OUTPUT_DIR = "temporal_plots"
SAVE_HTML  = True

# ==============================================================
# COSTANTI — non modificare
# ==============================================================

TOTAL_FREQ = 200
LABEL_MAP  = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

# Colore per giorno: scala Turbo (blu=inizio, rosso=fine)
# Viene usata sia per la colorbar che per assegnare il colore
# a ogni linea/punto in base al giorno
COLORSCALE = "Turbo"

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
    """
    Genera l'asse delle frequenze in scala logaritmica.
    200 punti equidistanti in log tra 100 Hz e 10 MHz.
    Risultato: [100, 110, 121, ..., 9.1e6, 10e6]
    """
    return np.logspace(np.log10(fmin), np.log10(fmax), n)

def giorno_to_color_turbo(giorno, gmin=1, gmax=30):
    """
    Converte un numero di giorno (1–30) in un colore RGB
    interpolando la colorscale Turbo di Plotly.

    Turbo è definita da una lista di colori a posizioni fisse (0.0–1.0).
    Normalizziamo il giorno in [0,1] e poi interponiamo i canali RGB.

    Questo ci permette di assegnare un colore preciso a ogni giorno
    anche quando usiamo trace separati (uno per giorno),
    invece di affidarci alla colorscale automatica di Plotly
    (che funziona solo quando tutti i punti sono nello stesso trace).

    Restituisce una stringa "rgb(R, G, B)".
    """
    # Tabella colori Turbo (posizione 0–1 → [R, G, B] in 0–255)
    # Fonte: Plotly built-in Turbo colorscale campionata a 10 punti
    turbo_stops = [
        (0.00, (48,  18,  59)),
        (0.11, (65, 100, 210)),
        (0.22, (30, 168, 228)),
        (0.33, (53, 217, 145)),
        (0.44, (132, 240,  57)),
        (0.55, (219, 220,  40)),
        (0.66, (253, 149,  28)),
        (0.77, (234,  72,  14)),
        (0.88, (180,  20,   4)),
        (1.00, (122,   4,   3)),
    ]

    # Normalizza il giorno in [0, 1]
    t = (giorno - gmin) / max(gmax - gmin, 1)
    t = max(0.0, min(1.0, t))

    # Trova i due stop tra cui interpolare
    for idx in range(len(turbo_stops) - 1):
        t0, c0 = turbo_stops[idx]
        t1, c1 = turbo_stops[idx + 1]
        if t0 <= t <= t1:
            # Interpolazione lineare tra c0 e c1
            alpha = (t - t0) / (t1 - t0)
            r = int(c0[0] + alpha * (c1[0] - c0[0]))
            g = int(c0[1] + alpha * (c1[1] - c0[1]))
            b = int(c0[2] + alpha * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"

    # Fallback: ultimo colore
    return f"rgb({turbo_stops[-1][1][0]},{turbo_stops[-1][1][1]},{turbo_stops[-1][1][2]})"


# ==============================================================
# CARICAMENTO DATI
# ==============================================================

def load_data(stress_type):
    """
    Carica X (2016,200,2), y, plant_ids, day_ids, freq_axis.
    Identico a pca_temporal.py — stesso formato dati.
    """
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

    print(f"\n[LOAD] {label}: {X.shape[0]} campioni, "
          f"giorni {day_ids.min()}–{day_ids.max()}")

    return X, y, plant_ids, day_ids, freq_axis, label


# ==============================================================
# FILTRO + NORMALIZZAZIONE
# ==============================================================

def apply_filters(X, y, plant_ids, day_ids, freq_axis,
                  freq_min, freq_max, classi, piante, giorni):
    """
    Applica tutti i filtri configurati e ritaglia le frequenze.

    Restituisce:
        X_f       → (n_filtrati, n_freq_sel, 2)
        y_f       → (n_filtrati,)
        ids_f     → (n_filtrati,)
        day_ids_f → (n_filtrati,)
        freq_sel  → (n_freq_sel,)  frequenze in Hz selezionate
    """
    # Maschera campioni
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

    # Ritaglia frequenze
    freq_slice = slice(freq_min, freq_max + 1)
    X_f        = X_f[:, freq_slice, :]
    freq_sel   = freq_axis[freq_slice]

    print(f"\n[FILTRO] Campioni : {X_f.shape[0]}")
    print(f"         Frequenze: {X_f.shape[1]}  "
          f"({freq_sel[0]:.1f} – {freq_sel[-1]:.1f} Hz)")
    print(f"         Giorni   : {sorted(np.unique(day_ids_f))}")

    return X_f, y_f, ids_f, day_ids_f, freq_sel


def apply_zscore(X_f):
    """
    Standardizzazione z-score su dati 3D (n, n_freq, 2).

    Appiattisce → standardizza → ridà la forma originale.
    Ogni feature (colonna) diventa media=0, std=1.

    Perché: le impedenze a bassa frequenza sono ~10.000 Ω,
    quelle ad alta frequenza ~10 Ω. Senza normalizzare,
    le frequenze basse dominerebbero visivamente anche se
    non sono più informative.
    """
    forma  = X_f.shape
    X_flat = X_f.reshape(forma[0], -1)

    scaler  = StandardScaler()
    X_znorm = scaler.fit_transform(X_flat).reshape(forma)

    print(f"[ZSCORE] Range prima : [{X_f.min():.1f}, {X_f.max():.1f}]")
    print(f"         Range dopo  : [{X_znorm.min():.2f}, {X_znorm.max():.2f}]")

    return X_znorm


# ==============================================================
# COSTRUZIONE DATI PER I PLOT
# ==============================================================

def build_plot_data(X_f, y_f, ids_f, day_ids_f, freq_sel, modalita):
    """
    Prepara i dati da passare alle funzioni di plot.

    Modalità "media":
        Per ogni combinazione (giorno, pianta) calcola la media
        di tutte le misurazioni. Risultato: una curva per (giorno, pianta).
        Utile per vedere il trend pulito senza rumore.

    Modalità "singole":
        Restituisce ogni misurazione individuale.
        Utile per vedere la variabilità dentro ogni giorno.

    In entrambi i casi restituisce una lista di dizionari,
    ognuno con le chiavi:
        "giorno"  → int
        "pianta"  → str
        "classe"  → int
        "real"    → array 1D (n_freq,)   [media o singola misurazione]
        "imag"    → array 1D (n_freq,)
        "n"       → int  numero di misurazioni usate (1 se singole)
    """
    records = []
    giorni_unici = sorted(np.unique(day_ids_f))
    piante_uniche = sorted(np.unique(ids_f))

    for giorno in giorni_unici:
        for pianta in piante_uniche:
            mask = (day_ids_f == giorno) & (ids_f == pianta)
            if not np.any(mask):
                continue

            X_gruppo = X_f[mask]         # (n_mis, n_freq, 2)
            y_gruppo = y_f[mask]
            classe   = int(y_gruppo[0])  # stessa classe per tutto il gruppo

            if modalita == "media":
                # np.mean(axis=0) → media lungo i campioni
                # risultato: (n_freq, 2) → un valore medio per frequenza
                records.append({
                    "giorno" : giorno,
                    "pianta" : pianta,
                    "classe" : classe,
                    "real"   : X_gruppo[:, :, 0].mean(axis=0),
                    "imag"   : X_gruppo[:, :, 1].mean(axis=0),
                    "n"      : X_gruppo.shape[0],
                })

            else:  # "singole"
                # Un record per ogni misurazione individuale
                for i in range(X_gruppo.shape[0]):
                    records.append({
                        "giorno" : giorno,
                        "pianta" : pianta,
                        "classe" : classe,
                        "real"   : X_gruppo[i, :, 0],
                        "imag"   : X_gruppo[i, :, 1],
                        "n"      : 1,
                    })

    print(f"\n[BUILD DATA] Modalità : {modalita}")
    print(f"             Record   : {len(records)}  "
          f"({'curve medie' if modalita == 'media' else 'misurazioni singole'})")

    return records


# ==============================================================
# PLOT 1 — 2D: Real + Imag vs Frequenza (subplot affiancati)
# ==============================================================

def plot_2d_spectra(records, freq_sel, title="", normalizzato=False,
                    modalita="media"):
    """
    Due subplot verticali:
      - Sopra:  Z_reale    vs Frequenza
      - Sotto:  Z_imag     vs Frequenza

    Asse X in scala logaritmica (le frequenze sono log-spaced).

    Ogni traccia è colorata per giorno (scala Turbo).
    Il simbolo del nome nella legenda indica la pianta.

    In modalità "media": linee continue, una per (giorno, pianta).
    In modalità "singole": markers, uno per misurazione.
    """
    norm_tag = "zscore" if normalizzato else "raw"
    y_label  = "Z (z-score)" if normalizzato else "Z (Ω)"
    mode     = "lines" if modalita == "media" else "markers"

    print(f"\n[PLOT 2D] {title} | {norm_tag} | {modalita}")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,        # stesso asse X per entrambi i subplot
        subplot_titles=["Z Reale vs Frequenza",
                        "Z Immaginaria vs Frequenza"],
        vertical_spacing=0.10,
    )

    # Tieni traccia di quali giorni hanno già la legenda mostrata
    # per evitare duplicati (ogni giorno appare una sola volta)
    giorni_in_legenda = set()

    for rec in records:
        giorno = rec["giorno"]
        pianta = rec["pianta"]
        colore = giorno_to_color_turbo(giorno, gmin=1, gmax=30)

        # Il nome nella legenda include giorno e pianta
        nome        = f"G{giorno} | {pianta}"
        show_legend = nome not in giorni_in_legenda
        giorni_in_legenda.add(nome)

        marker_cfg = dict(size=3, color=colore, opacity=0.7)
        line_cfg   = dict(color=colore, width=1.5)

        hover = (f"Giorno: {giorno}<br>"
                 f"Pianta: {pianta}<br>"
                 f"Classe: {LABEL_MAP[rec['classe']]}<br>"
                 f"Misurazioni usate: {rec['n']}")

        for row, valori in enumerate([rec["real"], rec["imag"]], start=1):
            fig.add_trace(go.Scatter(
                x             = freq_sel,
                y             = valori,
                mode          = mode,
                name          = nome,
                showlegend    = show_legend and (row == 1),
                # showlegend solo nel subplot 1 per non duplicare la legenda
                legendgroup   = nome,
                # legendgroup collega i due trace dello stesso (giorno,pianta)
                # così cliccando sulla legenda si nascondono entrambi
                line          = line_cfg  if modalita == "media"   else None,
                marker        = marker_cfg if modalita == "singole" else
                                dict(size=0),
                # in modalità "linee" il marker è invisibile (size=0)
                hovertemplate = hover + "<br>Freq: %{x:.2e} Hz<br>"
                                "Valore: %{y:.4f}<extra></extra>",
            ), row=row, col=1)

    # Asse X logaritmico: le frequenze vanno da 100 a 10M Hz
    # In scala lineare tutto si ammucchierebbe a destra
    fig.update_xaxes(type="log", title_text="Frequenza (Hz)", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text=f"Z Reale {y_label}", row=1, col=1)
    fig.update_yaxes(title_text=f"Z Imag {y_label}",  row=2, col=1)

    fig.update_layout(
        title    = f"Spettro 2D — {title} — {norm_tag} — {modalita}",
        height   = 750,
        template = "plotly_white",
        legend   = dict(title="Giorno | Pianta",
                        font=dict(size=10)),
        hovermode= "x unified",
    )

    fig.show()
    save_fig(fig, f"spectrum2d_{title.replace(' ','_').lower()}"
                  f"_{norm_tag}_{modalita}.html")


# ==============================================================
# PLOT 2 — 3D: Frequenza + Z_reale + Z_immaginaria
# ==============================================================

def plot_3d_spectrum(records, freq_sel, title="", normalizzato=False,
                     modalita="media"):
    """
    Scatter 3D dove ogni punto è (frequenza, Z_reale, Z_imag).

    Asse X = Frequenza (Hz, scala log)
    Asse Y = Z_reale
    Asse Z = Z_immaginaria

    In modalità "media": per ogni (giorno, pianta) una curva 3D
    In modalità "singole": tutti i punti individuali

    Il colore di ogni punto/curva indica il giorno.

    Questo grafico è l'equivalente del Nyquist plot tradizionale
    (Z_real vs Z_imag) ma con la frequenza come terza dimensione,
    così puoi vedere come cambia il punto del piano complesso
    al variare della frequenza.
    """
    norm_tag = "zscore" if normalizzato else "raw"
    print(f"\n[PLOT 3D] {title} | {norm_tag} | {modalita}")

    fig = go.Figure()

    # In 3D, scala log sull'asse X non è supportata nativamente
    # da Plotly Scatter3d. Soluzione: usiamo log10(frequenza)
    # come coordinata X e aggiustiamo le etichette dei tick.
    log_freq = np.log10(freq_sel)

    # Tick labels per l'asse X (frequenze leggibili)
    tick_vals_hz  = [100, 1e3, 1e4, 1e5, 1e6, 1e7]
    tick_vals_log = [np.log10(f) for f in tick_vals_hz if
                     freq_sel[0] <= f <= freq_sel[-1]]
    tick_text     = [f"{f:.0e} Hz" for f in tick_vals_hz if
                     freq_sel[0] <= f <= freq_sel[-1]]

    giorni_in_legenda = set()

    for rec in records:
        giorno = rec["giorno"]
        pianta = rec["pianta"]
        colore = giorno_to_color_turbo(giorno, gmin=1, gmax=30)
        nome   = f"G{giorno} | {pianta}"
        show_l = nome not in giorni_in_legenda
        giorni_in_legenda.add(nome)

        n_punti = len(log_freq)

        # Costruisce il testo hover per ogni punto della curva
        hover_texts = [
            f"Giorno: {giorno}<br>"
            f"Pianta: {pianta}<br>"
            f"Classe: {LABEL_MAP[rec['classe']]}<br>"
            f"Freq: {freq_sel[k]:.2e} Hz<br>"
            f"Reale: {rec['real'][k]:.4f}<br>"
            f"Imag: {rec['imag'][k]:.4f}"
            for k in range(n_punti)
        ]

        mode_3d = "lines" if modalita == "media" else "markers"

        fig.add_trace(go.Scatter3d(
            x    = log_freq,
            y    = rec["real"],
            z    = rec["imag"],
            mode = mode_3d,
            name = nome,
            showlegend  = show_l,
            legendgroup = nome,
            line   = dict(color=colore, width=3)   if modalita == "media"
                     else None,
            marker = dict(size=2, color=colore, opacity=0.7)
                     if modalita == "singole"
                     else dict(size=0),
            text          = hover_texts,
            hovertemplate = "%{text}<extra></extra>",
        ))

    y_ax = "Z Reale (z-score)" if normalizzato else "Z Reale (Ω)"
    z_ax = "Z Imag (z-score)"  if normalizzato else "Z Imag (Ω)"

    fig.update_layout(
        title    = f"Spettro 3D — {title} — {norm_tag} — {modalita}",
        template = "plotly_white",
        height   = 850,
        scene    = dict(
            xaxis = dict(
                title    = "Frequenza (Hz)  [log10]",
                tickvals = tick_vals_log,
                ticktext = tick_text,
            ),
            yaxis = dict(title=y_ax),
            zaxis = dict(title=z_ax),
        ),
        legend = dict(title="Giorno | Pianta", font=dict(size=10)),
        margin = dict(l=0, r=0, b=0, t=50),
    )

    fig.show()
    save_fig(fig, f"spectrum3d_{title.replace(' ','_').lower()}"
                  f"_{norm_tag}_{modalita}.html")


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("="*60)
    print("SPECTRUM EXPLORER")
    print("="*60)
    print(f"  Dataset    : {STRESS_TYPE}")
    print(f"  Modalità   : {MODALITA}")
    print(f"  Normalizza : {NORMALIZZA}")
    print(f"  Giorni     : {GIORNI_DA_VISUALIZZARE if GIORNI_DA_VISUALIZZARE else 'tutti'}")
    print("="*60)

    ensure_output_dir()

    # 1. Carica dati grezzi
    X, y, plant_ids, day_ids, freq_axis, label = load_data(STRESS_TYPE)

    # 2. Applica filtri (campioni + frequenze)
    X_f, y_f, ids_f, day_ids_f, freq_sel = apply_filters(
        X, y, plant_ids, day_ids, freq_axis,
        freq_min = FREQ_IDX_MIN,
        freq_max = FREQ_IDX_MAX,
        classi   = CLASSI,
        piante   = PIANTE,
        giorni   = GIORNI_DA_VISUALIZZARE,
    )

    # 3. Normalizzazione opzionale
    # Nota: la normalizzazione avviene DOPO il filtro,
    # così media e std sono calcolate solo sui dati che stai guardando
    if NORMALIZZA:
        X_plot = apply_zscore(X_f)
    else:
        X_plot = X_f

    # 4. Costruisce i dati per i plot (media o singole)
    records = build_plot_data(X_plot, y_f, ids_f, day_ids_f,
                               freq_sel, MODALITA)

    # 5. Plot 2D (Real + Imag vs Frequenza)
    plot_2d_spectra(
        records, freq_sel,
        title       = label,
        normalizzato= NORMALIZZA,
        modalita    = MODALITA,
    )

    # 6. Plot 3D (Frequenza + Real + Imag)
    plot_3d_spectrum(
        records, freq_sel,
        title       = label,
        normalizzato= NORMALIZZA,
        modalita    = MODALITA,
    )

    print("\n" + "="*60)
    print("COMPLETATO")
    if SAVE_HTML:
        print(f"Grafici in: ./{OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()