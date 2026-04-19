# dataset_pca_comparative.py  
# ============================================================
# Analisi comparativa: PCA fittata su un gruppo di piante,
# poi le piante escluse vengono proiettate come overlay.
#
# Obiettivo:
#   REFERENCE_GROUPS sostituisce REFERENCE_PLANTS.
#   Ogni gruppo puo' contenere UNA o PIU' piante.
#
# Esempi:
#   [["P0", "P1"]]  → PCA su P0+P1 mescolate, overlay: P3
#   [["P0"]]        → PCA solo su P0, overlay: P1 e P3
#   [["P0","P1"], ["P0","P3"], ["P1","P3"]]
#                   → 3 plot separati, uno per coppia
#
# Logica di overlay:
#   Tutte le piante in ALL_PLANTS che NON sono nel gruppo
#   di riferimento vengono proiettate come overlay.
#
# Interpretazione:
#   Punti sovrapposti per stesso giorno → piante coerenti
#   Punti lontani                       → dinamica diversa
#   Stesso pattern ma traslato          → offset (calibrazione?)
#
# Richiede:
#   - data/Water_Stress.npz  oppure  data/Iron_Stress.npz
#   - day_ids_water.npy      oppure  day_ids_iron.npy
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

STRESS_TYPE = "water"    # "water" | "iron"

# Frequenze da usare per la PCA (indici 0-199)
FREQ_IDX_MIN = 0
FREQ_IDX_MAX = 2

# Quante componenti PCA calcolare
# 2 -> plot 2D  |  3 -> plot 3D ruotabile
N_COMPONENTS = 2

# Standardizzazione Z-score prima della PCA?
# True  -> consigliato (equalizza scale tra frequenze)
# False -> dati grezzi
STANDARDIZE = True

# Componenti dell'impedenza da usare: "both" | "real" | "imag"
PCA_COMPONENTS = "both"

# Classi da includere (0=Control, 1=Early, 2=Late)
CLASSI = [0, 1, 2]

# Giorni da visualizzare (None = tutti)
GIORNI_DA_VISUALIZZARE = None

# ------------------------------------------------------------------
# GRUPPI DI RIFERIMENTO  ← configurazione principale
# ------------------------------------------------------------------
# Lista di liste. Ogni sotto-lista definisce UN run:
#   - Le piante nella lista FITTANO la PCA (gruppo di riferimento)
#   - Le piante di ALL_PLANTS non presenti nella lista
#     vengono proiettate come OVERLAY
#
# Per ogni gruppo viene prodotto un plot separato.
#
# ESEMPI:
#   [["P0", "P1"]]              → 1 plot: ref=P0+P1, overlay=P3
#   [["P0"]]                    → 1 plot: ref=P0, overlay=P1+P3
#   [["P0","P1"], ["P0","P3"], ["P1","P3"]]
#                               → 3 plot, uno per ogni coppia
#   [["P0","P1","P3"]]          → 1 plot: tutti nel ref, nessun overlay
#   [["P0"],["P1"],["P3"]]      → 3 plot: ogni pianta come ref singola
# REFERENCE_GROUPS = [["P0", "P1"]]   # overlay solo P3
# REFERENCE_GROUPS = [["P0"],["P1"],["P3"]]   # classico, uno per volta
# REFERENCE_GROUPS = [["P0","P1"],["P0","P3"],["P1","P3"]]  # tutte le coppie


REFERENCE_GROUPS = [
    ["P0"]   # PCA fittata su P0+P1 mescolate -> overlay: P3
]

# Tutte le piante disponibili nel dataset
ALL_PLANTS = ["P0", "P1", "P3"]

# ------------------------------------------------------------------
# OPZIONI VISUALIZZAZIONE
# ------------------------------------------------------------------

# True  -> un punto per (pianta, giorno) = media delle misurazioni
#          piu' pulito, traiettorie chiare
# False -> tutti i campioni individuali (piu' denso)
SHOW_DAILY_MEANS = False

# True  -> calcola e stampa la distanza euclidea overlay vs ref per giorno
COMPUTE_DISTANCES = True

# Dove salvare i grafici HTML
OUTPUT_DIR = "temporal_plots"
SAVE_HTML  = True

# ==============================================================
# COSTANTI
# ==============================================================

LABEL_MAP = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

SYMBOL_MAP_2D = {"P0": "circle",  "P1": "square",  "P3": "diamond"}
SYMBOL_MAP_3D = {"P0": "circle",  "P1": "square",  "P3": "diamond"}

# Colori fissi per pianta (usati per il bordo dei simboli overlay)
PLANT_COLORS = {
    "P0": "rgba(31,119,180,",    # blu
    "P1": "rgba(148,103,189,",   # viola
    "P3": "rgba(214,137,16,",    # arancio
}

# ==============================================================
# UTILITY
# ==============================================================

def ensure_output_dir():
    if SAVE_HTML:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename):
    if SAVE_HTML:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(path)
        print(f"  [HTML] {path}")

def build_frequency_axis(n=200, fmin=100, fmax=10e6):
    return np.logspace(np.log10(fmin), np.log10(fmax), n)

def _label_group(plants):
    """Restituisce una stringa leggibile per un gruppo di piante."""
    return "+".join(sorted(plants))

# ==============================================================
# CARICAMENTO DATI
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
        print("         Esegui prima dataset_temporal_inspector.py")
        sys.exit(1)

    data      = np.load(file_path, allow_pickle=True)
    X         = data["X"].reshape(-1, 200, 2)
    y         = data["y"]
    plant_ids = data["plant_ids"].astype(str)
    day_ids   = np.load(day_ids_path)

    print(f"\n[LOAD] {label}: {X.shape[0]} campioni, "
          f"piante={np.unique(plant_ids)}, "
          f"giorni {day_ids.min()}-{day_ids.max()}")

    return X, y, plant_ids, day_ids, label

# ==============================================================
# PREPARAZIONE FEATURES
# ==============================================================

def extract_features(X, y, plant_ids, day_ids,
                     plant_filter, freq_min, freq_max,
                     classi, giorni, pca_components):
    """
    Filtra per pianta/classe/giorno e appiattisce a 2D.

    plant_filter: lista di piante da includere (es. ["P0", "P1"])
    """
    mask = np.isin(plant_ids, plant_filter)
    if classi:
        mask &= np.isin(y, classi)
    if giorni is not None:
        mask &= np.isin(day_ids, giorni)

    X_f = X[mask][:, freq_min:freq_max + 1, :]

    if pca_components == "real":
        X_sel = X_f[:, :, 0]
    elif pca_components == "imag":
        X_sel = X_f[:, :, 1]
    else:   # "both"
        X_sel = X_f

    X_flat = X_sel.reshape(X_sel.shape[0], -1)
    return X_flat, y[mask], plant_ids[mask], day_ids[mask]

# ==============================================================
# MEDIE GIORNALIERE
# ==============================================================

def compute_daily_means(X_pca, y_f, ids_f, day_ids_f):
    """
    Per ogni (pianta, giorno) calcola il centroide nello spazio PCA.

    Returns:
        Lista di dict {plant, day, classe, mean (array), n}
    """
    records = []
    for plant in np.unique(ids_f):
        for day in np.unique(day_ids_f):
            mask = (ids_f == plant) & (day_ids_f == day)
            if not np.any(mask):
                continue
            records.append({
                "plant"  : str(plant),
                "day"    : int(day),
                "classe" : int(y_f[mask][0]),
                "mean"   : X_pca[mask].mean(axis=0),
                "n"      : int(np.sum(mask)),
            })
    return records

def samples_to_records(X_pca, y_f, ids_f, day_ids_f):
    """
    Converte campioni individuali in lista di record
    con lo stesso formato di compute_daily_means.
    """
    return [
        {
            "plant"  : str(ids_f[i]),
            "day"    : int(day_ids_f[i]),
            "classe" : int(y_f[i]),
            "mean"   : X_pca[i],
            "n"      : 1,
        }
        for i in range(len(y_f))
    ]

# ==============================================================
# DISTANZE
# ==============================================================

def compute_overlay_distances(ref_records, overlay_records,
                               ref_group_label):
    """
    Per ogni giorno, calcola la distanza euclidea tra il centroide
    del gruppo di riferimento e il centroide di ogni pianta overlay.

    Centroide del riferimento:
      Media dei punti PCA di TUTTE le piante del gruppo ref in quel giorno.
      Questo e' il "punto rappresentativo" del gruppo per quel giorno.

    Returns:
        dist_by_day: dict {giorno: {pianta_overlay: distanza}}
    """
    # Aggrega tutti i record del riferimento per giorno
    ref_by_day = {}
    for r in ref_records:
        ref_by_day.setdefault(r["day"], []).append(r["mean"])

    # Aggrega per (pianta overlay, giorno)
    ovl_by_plant_day = {}
    for r in overlay_records:
        key = (r["plant"], r["day"])
        ovl_by_plant_day.setdefault(key, []).append(r["mean"])

    giorni      = sorted(ref_by_day.keys())
    ovl_plants  = sorted({r["plant"] for r in overlay_records})

    dist_by_day = {}
    for day in giorni:
        ref_centroid = np.mean(ref_by_day[day], axis=0)
        dist_by_day[day] = {}
        for op in ovl_plants:
            key = (op, day)
            if key in ovl_by_plant_day:
                ovl_centroid = np.mean(ovl_by_plant_day[key], axis=0)
                dist_by_day[day][op] = float(np.linalg.norm(
                    ovl_centroid - ref_centroid
                ))

    return dist_by_day, ovl_plants

# ==============================================================
# PLOT: DISTANZE PER GIORNO
# ==============================================================

def plot_distances(dist_by_day, overlay_plants, ref_group_label,
                   label, norm_label, dim_label):
    """
    Linea: distanza euclidea (overlay vs centroide riferimento) per giorno.
    Bande di sfondo indicano le classi.
    """
    giorni = sorted(dist_by_day.keys())

    ov_colors = {"P0": "#1f77b4", "P1": "#9467bd", "P3": "#d68910"}

    fig = go.Figure()

    # Bande di sfondo classi
    for g0, g1, col in [(1,6,"rgba(31,208,104,0.10)"),
                         (7,18,"rgba(243,156,18,0.10)"),
                         (19,30,"rgba(231,76,60,0.10)")]:
        g0c = max(g0, min(giorni))
        g1c = min(g1, max(giorni))
        if g0c <= g1c:
            fig.add_vrect(x0=g0c-0.5, x1=g1c+0.5,
                          fillcolor=col, line_width=0, layer="below")

    for op in overlay_plants:
        dists = [dist_by_day[g].get(op, float("nan")) for g in giorni]
        fig.add_trace(go.Scatter(
            x=giorni, y=dists,
            mode="lines+markers",
            name=f"dist({op} vs {ref_group_label})",
            line=dict(color=ov_colors.get(op, "gray"), width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"Overlay: {op} vs Ref: {ref_group_label}<br>"
                "Giorno: %{x}<br>"
                "Distanza PCA: %{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title    = (f"Distanza PCA — Ref: {ref_group_label} — "
                    f"{label} — {norm_label} — {dim_label}"),
        template = "plotly_white",
        height   = 450,
        xaxis    = dict(title="Giorno", tickvals=list(range(1, 31))),
        yaxis    = dict(title="Distanza euclidea nello spazio PCA"),
        legend   = dict(title="Overlay vs Riferimento"),
        hovermode= "x unified",
    )

    save_fig(fig, f"pca_dist_ref{ref_group_label.replace('+','_')}"
                  f"_{label.replace(' ','_').lower()}_{norm_label}_{dim_label}.html")

# ==============================================================
# PLOT: PCA COMPARATIVA (2D o 3D)
# ==============================================================

def plot_pca_comparative(ref_group, overlay_plants,
                         ref_records, overlay_records_by_plant,
                         explained, label, norm_label):
    """
    Scatter 2D o 3D con:
      COLORE  = giorno (scala Turbo, uguale per tutti)
      SIMBOLO = pianta (circle=P0 / square=P1 / diamond=P3)
      PIENO   = pianta nel gruppo di RIFERIMENTO
      APERTO  = pianta OVERLAY

    Questo ti permette di vedere, per ogni giorno (stesso colore),
    se i punti del riferimento e dell'overlay si sovrappongono
    (dinamica coerente) o divergono (dinamica diversa).
    """
    n_comp         = ref_records[0]["mean"].shape[0]
    is_3d          = (n_comp >= 3)
    dim_label      = "3D" if is_3d else "2D"
    ref_group_label = _label_group(ref_group)

    print(f"\n[PLOT {dim_label}] Ref={ref_group_label} | "
          f"overlay={overlay_plants}")

    fig          = go.Figure()
    show_colorbar = True

    colorbar_cfg = dict(
        title     = "Giorno",
        thickness = 15,
        len       = 0.75,
        tickvals  = [1, 6, 7, 18, 19, 30],
        ticktext  = ["1 Control", "6 Control",
                     "7 Early",   "18 Early",
                     "19 Late",   "30 Late"],
    )

    def _make_marker(plant, is_overlay, days, show_cb, is_3d):
        """
        Costruisce il dict marker per go.Scatter o go.Scatter3d.

        is_overlay = True  -> simbolo APERTO + bordo colorato della pianta
        is_overlay = False -> simbolo PIENO   + bordo sottile
        """
        base_sym = SYMBOL_MAP_3D.get(plant, "circle") if is_3d \
                   else SYMBOL_MAP_2D.get(plant, "circle")
        sym      = (base_sym + "-open") if is_overlay else base_sym
        pc       = PLANT_COLORS.get(plant, "rgba(100,100,100,")

        mk = dict(
            size       = (6 if is_3d else 11) if is_overlay
                         else (4 if is_3d else 7),
            symbol     = sym,
            color      = days,
            colorscale = "Turbo",
            cmin=1, cmax=30,
            showscale  = show_cb,
            opacity    = 0.95 if is_overlay else 0.70,
            line       = (dict(color=pc + "0.9)", width=2)
                          if is_overlay
                          else dict(color="rgba(0,0,0,0.2)", width=0.5)),
        )
        if show_cb:
            mk["colorbar"] = colorbar_cfg
        return mk

    def _hover_texts(records, is_overlay, ref_group_label):
        role = "OVERLAY" if is_overlay else f"REF ({ref_group_label})"
        return [
            f"{role}<br>"
            f"Pianta: {r['plant']}<br>"
            f"Giorno: {r['day']}<br>"
            f"Classe: {LABEL_MAP[r['classe']]}<br>"
            f"({'media' if SHOW_DAILY_MEANS else 'campione'}, n={r['n']})"
            for r in records
        ]

    def _add_traces(plant, records, is_overlay):
        nonlocal show_colorbar

        days  = [r["day"]      for r in records]
        pc1   = [r["mean"][0]  for r in records]
        pc2   = [r["mean"][1]  for r in records]
        name  = (f"{plant} (Ovl.)" if is_overlay
                 else f"{plant} (Rif.:{ref_group_label})")
        hover = _hover_texts(records, is_overlay, ref_group_label)
        mk    = _make_marker(plant, is_overlay, days,
                             show_colorbar, is_3d)

        if is_3d:
            pc3 = [r["mean"][2] for r in records]
            fig.add_trace(go.Scatter3d(
                x=pc1, y=pc2, z=pc3,
                mode="markers", name=name,
                marker=mk, text=hover,
                hovertemplate=(
                    "%{text}<br>"
                    "PC1:%{x:.3f} PC2:%{y:.3f} PC3:%{z:.3f}"
                    "<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=pc1, y=pc2,
                mode="markers", name=name,
                marker=mk, text=hover,
                hovertemplate=(
                    "%{text}<br>"
                    "PC1:%{x:.3f} PC2:%{y:.3f}"
                    "<extra></extra>"
                ),
            ))

        show_colorbar = False   # la colorbar appare solo sul primo trace

    # ---- piante nel gruppo di RIFERIMENTO ----
    # Raggruppa i ref_records per pianta e aggiungi un trace per ognuna
    ref_by_plant = {}
    for r in ref_records:
        ref_by_plant.setdefault(r["plant"], []).append(r)

    for plant in sorted(ref_by_plant.keys()):
        _add_traces(plant, ref_by_plant[plant], is_overlay=False)

    # ---- piante OVERLAY ----
    for op in overlay_plants:
        if op in overlay_records_by_plant:
            _add_traces(op, overlay_records_by_plant[op], is_overlay=True)

    # ---- layout ----
    subtitle = (
        f"PCA fittata su: {ref_group_label}  |  "
        f"Proiettati: {', '.join(overlay_plants) if overlay_plants else 'nessuno'}  |  "
        f"{'Medie giornaliere' if SHOW_DAILY_MEANS else 'Tutti i campioni'}"
    )
    main_title = (
        f"PCA Comparativa — Ref: {ref_group_label} — "
        f"{label} — {norm_label} — {dim_label}"
    )

    if is_3d:
        pc3_lbl = (f"PC3 ({explained[2]:.1f}%)"
                   if len(explained) >= 3 else "PC3")
        fig.update_layout(
            title    = f"{main_title}<br><sup>{subtitle}</sup>",
            template = "plotly_white",
            height   = 850,
            scene    = dict(
                xaxis=dict(title=f"PC1 ({explained[0]:.1f}%)"),
                yaxis=dict(title=f"PC2 ({explained[1]:.1f}%)"),
                zaxis=dict(title=pc3_lbl),
            ),
            legend = dict(title="Pianta (ruolo)",
                          font=dict(size=11), itemsizing="constant"),
            margin = dict(l=0, r=0, b=0, t=90),
        )
    else:
        fig.update_layout(
            title       = f"{main_title}<br><sup>{subtitle}</sup>",
            template    = "plotly_white",
            height      = 700,
            xaxis_title = f"PC1 ({explained[0]:.1f}%)",
            yaxis_title = f"PC2 ({explained[1]:.1f}%)",
            legend      = dict(title="Pianta (ruolo)",
                               font=dict(size=11), itemsizing="constant"),
        )

    tag = (f"ref{ref_group_label.replace('+','_')}"
           f"_{label.replace(' ','_').lower()}_{norm_label}_{dim_label}")
    save_fig(fig, f"pca_comparative_{tag}.html")
    return dim_label

# ==============================================================
# RUN PER UN SINGOLO GRUPPO DI RIFERIMENTO
# ==============================================================

def run_one_group(ref_group, X, y, plant_ids, day_ids,
                  label, norm_label):
    """
    Esegue l'analisi completa per un gruppo di riferimento:
      1. Estrae e unisce le features di TUTTE le piante del gruppo
      2. Fitta scaler e PCA sul gruppo combinato
      3. Proietta ogni pianta overlay (esclusa dal gruppo)
      4. Calcola medie giornaliere (opzionale)
      5. Calcola distanze overlay vs centroide ref (opzionale)
      6. Produce i plot
    """
    ref_group_label = _label_group(ref_group)
    overlay_plants  = sorted([p for p in ALL_PLANTS if p not in ref_group])

    print(f"\n{'='*60}")
    print(f"RIFERIMENTO: {ref_group_label}  |  OVERLAY: {overlay_plants}")
    print(f"{'='*60}")

    # ── 1. Estrai features del gruppo di riferimento (tutte le piante insieme)
    X_ref, y_ref, ids_ref, day_ref = extract_features(
        X, y, plant_ids, day_ids,
        plant_filter  = ref_group,   # lista con 1 o piu' piante
        freq_min      = FREQ_IDX_MIN,
        freq_max      = FREQ_IDX_MAX,
        classi        = CLASSI,
        giorni        = GIORNI_DA_VISUALIZZARE,
        pca_components= PCA_COMPONENTS,
    )
    print(f"[REF] {ref_group_label}: {X_ref.shape[0]} campioni, "
          f"{X_ref.shape[1]} features "
          f"(piante={sorted(np.unique(ids_ref))})")

    # ── 2. Standardizzazione (fit SOLO sul riferimento)
    if STANDARDIZE:
        scaler  = StandardScaler()
        X_ref_s = scaler.fit_transform(X_ref)
    else:
        X_ref_s = X_ref
        scaler  = None

    # ── 3. PCA (fit SOLO sul riferimento)
    n_comp_max = min(N_COMPONENTS, X_ref_s.shape[0], X_ref_s.shape[1])
    pca        = PCA(n_components=n_comp_max)
    X_ref_pca  = pca.fit_transform(X_ref_s)
    explained  = pca.explained_variance_ratio_ * 100

    print(f"[PCA] {n_comp_max} componenti | "
          f"varianza: {[f'{v:.1f}%' for v in explained]}  "
          f"tot={explained.sum():.1f}%")

    # ── 4. Records per il riferimento
    if SHOW_DAILY_MEANS:
        ref_records = compute_daily_means(X_ref_pca, y_ref, ids_ref, day_ref)
    else:
        ref_records = samples_to_records(X_ref_pca, y_ref, ids_ref, day_ref)

    print(f"[REF] {len(ref_records)} record "
          f"({'medie' if SHOW_DAILY_MEANS else 'campioni'})")

    # ── 5. Overlay: estrai, standardizza, proietta
    overlay_records_by_plant = {}
    all_overlay_records      = []

    for op in overlay_plants:
        X_ov, y_ov, ids_ov, day_ov = extract_features(
            X, y, plant_ids, day_ids,
            plant_filter  = [op],
            freq_min      = FREQ_IDX_MIN,
            freq_max      = FREQ_IDX_MAX,
            classi        = CLASSI,
            giorni        = GIORNI_DA_VISUALIZZARE,
            pca_components= PCA_COMPONENTS,
        )

        # Applica lo scaler fittato sul riferimento
        X_ov_s   = (scaler.transform(X_ov)
                    if STANDARDIZE and scaler is not None
                    else X_ov)

        # Proietta nello spazio PCA del riferimento
        X_ov_pca = pca.transform(X_ov_s)

        if SHOW_DAILY_MEANS:
            recs = compute_daily_means(X_ov_pca, y_ov, ids_ov, day_ov)
        else:
            recs = samples_to_records(X_ov_pca, y_ov, ids_ov, day_ov)

        overlay_records_by_plant[op] = recs
        all_overlay_records.extend(recs)
        print(f"[OVL] {op}: {X_ov.shape[0]} campioni proiettati "
              f"-> {len(recs)} record")

    # ── 6. Plot PCA
    dim_label = plot_pca_comparative(
        ref_group, overlay_plants,
        ref_records, overlay_records_by_plant,
        explained, label, norm_label,
    )

    # ── 7. Distanze (opzionale)
    if COMPUTE_DISTANCES and all_overlay_records:
        dist_by_day, ovl_plants = compute_overlay_distances(
            ref_records, all_overlay_records, ref_group_label
        )

        # Stampa tabella
        print(f"\n  Distanze euclidee per giorno "
              f"(overlay vs centroide {ref_group_label}):")
        header = (f"  {'Giorno':>8} | {'Classe':>12} | " +
                  " | ".join(f"{p:>8}" for p in ovl_plants))
        print(header)
        print("  " + "-" * len(header))

        for day in sorted(dist_by_day.keys()):
            cl = ("Control" if day <= 6
                  else "Early" if day <= 18
                  else "Late")
            row = f"  {day:>8} | {cl:>12} | "
            row += " | ".join(
                f"{dist_by_day[day].get(p, float('nan')):>8.4f}"
                for p in ovl_plants
            )
            print(row)

        plot_distances(dist_by_day, ovl_plants, ref_group_label,
                       label, norm_label, dim_label)

# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("PCA COMPARATIVA  — v2 con gruppi di riferimento")
    print("=" * 60)
    print(f"  Dataset         : {STRESS_TYPE}")
    print(f"  REFERENCE_GROUPS: {REFERENCE_GROUPS}")
    print(f"  N_COMPONENTS    : {N_COMPONENTS}")
    print(f"  PCA_COMPONENTS  : {PCA_COMPONENTS}")
    print(f"  STANDARDIZE     : {STANDARDIZE}")
    print(f"  SHOW_DAILY_MEANS: {SHOW_DAILY_MEANS}")
    print(f"  COMPUTE_DISTANCES:{COMPUTE_DISTANCES}")
    print("=" * 60)

    ensure_output_dir()

    X, y, plant_ids, day_ids, label = load_data(STRESS_TYPE)
    norm_label = "std" if STANDARDIZE else "raw"

    # Valida i gruppi prima di partire
    available = set(np.unique(plant_ids).tolist())
    for grp in REFERENCE_GROUPS:
        missing = [p for p in grp if p not in available]
        if missing:
            print(f"[WARN] Piante {missing} non trovate nel dataset "
                  f"(gruppo {grp}) — skip.")
            continue
        run_one_group(grp, X, y, plant_ids, day_ids, label, norm_label)

    print("\n" + "=" * 60)
    print("COMPLETATO")
    if SAVE_HTML:
        print(f"Grafici salvati in: ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()