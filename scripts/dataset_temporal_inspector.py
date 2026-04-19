# dataset_temporal_inspector.py
# ============================================================
# Obiettivo: capire la struttura temporale del dataset e
# ricostruire i day_ids da assegnare a ogni campione.
# ============================================================

import os
import sys
import numpy as np

DATA_DIR   = "./data"
WATER_FILE = os.path.join(DATA_DIR, "Water_Stress.npz")
IRON_FILE  = os.path.join(DATA_DIR, "Iron_Stress.npz")

STRESS_TYPE = "iron"   # "water" oppure "iron"

LABEL_MAP = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

# Giorni reali per classe (dal protocollo sperimentale)
DAYS_PER_CLASS = {
    0: list(range(1,  7)),   # Control   → giorni  1 – 6
    1: list(range(7, 19)),   # Early     → giorni  7 – 18
    2: list(range(19, 31)),  # Late      → giorni 19 – 30
}


# ==============================================================
# STEP 1 — Struttura base
# ==============================================================
def step1_struttura_base(X, y, plant_ids):
    print("\n" + "="*60)
    print("STEP 1 — Struttura base del dataset")
    print("="*60)

    print(f"\nShape X        : {X.shape}")
    print(f"Shape y        : {y.shape}")
    print(f"Shape plant_ids: {plant_ids.shape}")

    print(f"\nClassi uniche  : {np.unique(y)}")
    print(f"Piante uniche  : {np.unique(plant_ids)}")

    print("\nCampioni per classe:")
    for c in sorted(np.unique(y)):
        print(f"  Classe {c} ({LABEL_MAP[c]}): {np.sum(y == c)}")

    print("\nCampioni per pianta:")
    for p in sorted(np.unique(plant_ids)):
        print(f"  Pianta {p}: {np.sum(plant_ids == p)}")

    print("\nCampioni per (pianta, classe):")
    for p in sorted(np.unique(plant_ids)):
        for c in sorted(np.unique(y)):
            n = np.sum((plant_ids == p) & (y == c))
            giorni = DAYS_PER_CLASS[c]
            print(f"  {p} | {LABEL_MAP[c]:12s} : {n:4d} campioni "
                  f"| {len(giorni)} giorni "
                  f"| {n/len(giorni):.2f} campioni/giorno")


# ==============================================================
# STEP 2 — Ordine dei campioni nel file
# ==============================================================
def step2_ordine_campioni(y, plant_ids):
    print("\n" + "="*60)
    print("STEP 2 — Come sono ordinati i campioni nel file?")
    print("="*60)

    print("\nPrimi 30 campioni (indice, pianta, classe):")
    for i in range(30):
        print(f"  [{i:4d}]  pianta={plant_ids[i]}  classe={y[i]} ({LABEL_MAP[y[i]]})")

    print("\nUltimi 20 campioni:")
    for i in range(len(y)-20, len(y)):
        print(f"  [{i:4d}]  pianta={plant_ids[i]}  classe={y[i]} ({LABEL_MAP[y[i]]})")

    # Trova i punti di transizione tra classi/piante
    print("\nTransizioni (dove cambia pianta o classe):")
    for i in range(1, len(y)):
        if y[i] != y[i-1] or plant_ids[i] != plant_ids[i-1]:
            print(f"  Indice {i:4d}: "
                  f"{plant_ids[i-1]}/classe{y[i-1]} → "
                  f"{plant_ids[i]}/classe{y[i]}")


# ==============================================================
# STEP 3 — Ricostruzione day_ids
# ==============================================================
def step3_ricostruisci_giorni(y, plant_ids):
    print("\n" + "="*60)
    print("STEP 3 — Ricostruzione day_ids")
    print("="*60)

    n = len(y)
    day_ids = np.zeros(n, dtype=int)

    for p in np.unique(plant_ids):
        for c in np.unique(y):
            # Indici globali di questo gruppo (pianta p, classe c)
            idx_gruppo = np.where((plant_ids == p) & (y == c))[0]
            n_gruppo   = len(idx_gruppo)

            giorni_classe = DAYS_PER_CLASS[c]   # es. [7,8,...,18]
            n_giorni      = len(giorni_classe)

            if n_gruppo == 0:
                continue

            # Assumiamo che i campioni siano in ordine cronologico
            # dentro ogni gruppo → dividiamo equamente per giorno
            # np.array_split divide in n_giorni parti il più uguali possibile
            parti = np.array_split(np.arange(n_gruppo), n_giorni)

            for giorno_idx, parte in enumerate(parti):
                giorno_reale = giorni_classe[giorno_idx]
                for pos_locale in parte:
                    idx_globale = idx_gruppo[pos_locale]
                    day_ids[idx_globale] = giorno_reale

            print(f"  {p} | {LABEL_MAP[c]:12s}: "
                  f"{n_gruppo} campioni → {n_giorni} giorni "
                  f"(~{n_gruppo/n_giorni:.1f} campioni/giorno)")

    return day_ids


# ==============================================================
# STEP 4 — Verifica della ricostruzione
# ==============================================================
def step4_verifica(y, plant_ids, day_ids):
    print("\n" + "="*60)
    print("STEP 4 — Verifica day_ids ricostruiti")
    print("="*60)

    print("\nDistribuzione campioni per giorno (tutti i gruppi):")
    for giorno in range(1, 31):
        mask  = day_ids == giorno
        n     = np.sum(mask)
        if n == 0:
            continue
        piante = np.unique(plant_ids[mask])
        classi = np.unique(y[mask])
        print(f"  Giorno {giorno:2d}: {n:4d} campioni | "
              f"piante={list(piante)} | "
              f"classi={[LABEL_MAP[c] for c in classi]}")

    # Controllo coerenza: ogni giorno deve appartenere a UNA sola classe
    print("\nControllo coerenza giorno→classe:")
    ok = True
    for giorno in range(1, 31):
        mask   = day_ids == giorno
        classi = np.unique(y[mask])
        if len(classi) > 1:
            print(f"  [PROBLEMA] Giorno {giorno} ha classi miste: {classi}")
            ok = False
    if ok:
        print("  OK — ogni giorno appartiene a una sola classe")


# ==============================================================
# STEP 5 — Analisi della variazione del segnale nel tempo
# ==============================================================
def step5_variazione_segnale(X, y, plant_ids, day_ids):
    print("\n" + "="*60)
    print("STEP 5 — Variazione del segnale nel tempo")
    print("="*60)

    # Riforma X in (n, 200, 2) per avere Real e Imag separati
    X3d = X.reshape(-1, 200, 2)

    # Per ogni pianta, calcola la media della parte reale
    # alla frequenza 0 (100 Hz) per ogni giorno
    # → ci aspettiamo un trend continuo se il segnale varia con lo stress
    print("\nMedia Z_reale @ 100 Hz per pianta e giorno:")
    print(f"  {'Giorno':>8} | {'Classe':>12} | {'P0':>10} | {'P1':>10} | {'P3':>10}")
    print("  " + "-"*58)

    for giorno in range(1, 31):
        mask_g = day_ids == giorno
        if not np.any(mask_g):
            continue

        # classe di questo giorno (è unica per costruzione)
        classe = y[mask_g][0]

        medie = {}
        for p in ["P0", "P1", "P3"]:
            mask_pg = mask_g & (plant_ids == p)
            if np.any(mask_pg):
                # freq 0 = 100 Hz, componente 0 = reale
                medie[p] = f"{X3d[mask_pg, 0, 0].mean():10.1f}"
            else:
                medie[p] = f"{'N/A':>10}"

        print(f"  {giorno:>8} | {LABEL_MAP[classe]:>12} | "
              f"{medie.get('P0','N/A')} | "
              f"{medie.get('P1','N/A')} | "
              f"{medie.get('P3','N/A')}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("="*60)
    print("DATASET TEMPORAL INSPECTOR")
    print("="*60)

    # ----------------------------------------------------------
    # Scelta del dataset in base a STRESS_TYPE
    # ----------------------------------------------------------
    if STRESS_TYPE == "water":
        file_path    = WATER_FILE
        day_ids_path = "day_ids_water.npy"
        label        = "Water Stress"

    elif STRESS_TYPE == "iron":
        file_path    = IRON_FILE
        day_ids_path = "day_ids_iron.npy"
        label        = "Iron Stress"

    else:
        print(f"[ERRORE] STRESS_TYPE deve essere 'water' o 'iron'.")
        print(f"         Valore trovato: '{STRESS_TYPE}'")
        sys.exit(1)

    print(f"\n[CONFIG] Dataset selezionato : {label}")
    print(f"         File               : {file_path}")
    print(f"         day_ids output     : {day_ids_path}")

    # ----------------------------------------------------------
    # Carica e analizza
    # ----------------------------------------------------------
    data      = np.load(file_path, allow_pickle=True)
    X         = data["X"]
    y         = data["y"]
    plant_ids = data["plant_ids"]

    step1_struttura_base(X, y, plant_ids)
    step2_ordine_campioni(y, plant_ids)
    day_ids = step3_ricostruisci_giorni(y, plant_ids)
    step4_verifica(y, plant_ids, day_ids)
    step5_variazione_segnale(X, y, plant_ids, day_ids)

    # ----------------------------------------------------------
    # Salva day_ids con nome coerente al dataset
    # ----------------------------------------------------------
    np.save(day_ids_path, day_ids)
    print(f"\n[SALVATO] {day_ids_path}")
    print(f"Puoi caricarlo con: day_ids = np.load('{day_ids_path}')")


if __name__ == "__main__":
    main()