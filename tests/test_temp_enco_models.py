"""
test_temp_enco_models.py
Test end-to-end per tutti i metodi di encoding temporale

Esegui con:  python -m pytest tests/test_temp_enco_models.py -v
             oppure: python tests/test_temp_enco_models.py

Cosa viene testato:
  0. Device detection (MPS, CUDA, CPU)
  1. LIFEncoder.forward() → shape e statistiche spike
  2. TemporalEncoder — i 3 encoder producono output DIVERSI
  2.5. Population size parameter — replicazione feature su tutti gli encoder
  3. Comportamento biologico LIF (più input → più spike)
  4. Gestione errori (input malformato, tipo sconosciuto)
  5. Visualizzazione raster plot singolo (LIF)
  6. Confronto visivo LIF vs Rate vs Rate_Hz
  7. Population Scaling Analysis — 3 encoder × 3 population_size con grafici realistici

Autore: Shanti Leonardo Arzu (adattamento)
Data: Marzo 2026
"""
import random
import os
import sys
from pathlib import Path

# Aggiungi src al path per le importazioni
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.temp_enco_lif_population import LIFEncoder
from src.temp_enco_dispatcher import TemporalEncoder


# ════════════════════════════════════════════════════════════════
# CONFIGURAZIONE DEL TEST
# ════════════════════════════════════════════════════════════════
GLOBAL_TEST_SEED = 42
def setup_test_seed(seed: int = GLOBAL_TEST_SEED) -> None:
    """Seed deterministico per test riproducibili su tutti i generatori."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

setup_test_seed() 



NB_STEPS = 150
DT = 3.0
N_SAMPLES = 20
N_FEATURES = 6
GAIN_LIF = 0.05
GAIN_RATE = 10.0
INPUT_SHIFT = 4.0

FEATURE_NAMES = [
    "Real @ 1kHz",
    "Imag @ 1kHz",
    "Real @ 10kHz",
    "Imag @ 10kHz",
    "Real @ 100kHz",
    "Imag @ 100kHz",
]

setup_test_seed(GLOBAL_TEST_SEED) # Per riproducibilità dei test stocastici


# ════════════════════════════════════════════════════════════════
# TEST 0 — Device Detection
# ════════════════════════════════════════════════════════════════


def test_device_detection():
    """Verifica rilevamento corretto del device."""
    print("\n" + "=" * 65)
    print("  TEST 0: Device Detection")
    print("=" * 65)

    if torch.backends.mps.is_available():
        detected = "MPS (Apple Silicon M1/M2/M3) ✓"
    elif torch.cuda.is_available():
        detected = f"CUDA ({torch.cuda.get_device_name(0)}) ✓"
    else:
        detected = "CPU (nessuna GPU disponibile)"

    print(f"  Device rilevato: {detected}")
    assert True  # Sempre passa


# ════════════════════════════════════════════════════════════════
# TEST 1 — LIFEncoder Direct
# ════════════════════════════════════════════════════════════════


def test_lif_encoder_forward():
    """Test LIFEncoder.forward() → shape e statistiche spike."""
    print("\n" + "=" * 65)
    print("  TEST 1: LIFEncoder.forward()")
    print("=" * 65)

    X_test = np.random.randn(N_SAMPLES, N_FEATURES) * 1.2
    print(f"  Input X_test:    shape={X_test.shape}, "
          f"range=[{X_test.min():.2f}, {X_test.max():.2f}]")

    encoder_lif = LIFEncoder(
        nb_steps=NB_STEPS,
        dt=DT,
        tau_syn=5.0,
        tau_mem=10.0,
        threshold=1.0,
        gain=GAIN_LIF,
        input_shift=INPUT_SHIFT,
    )

    spikes_lif = encoder_lif.forward(X_test)

    expected_shape = (N_SAMPLES, NB_STEPS, N_FEATURES)
    shape_ok = spikes_lif.shape == expected_shape
    unique_vals = spikes_lif.unique().cpu().tolist()
    binary_ok = all(v in [0.0, 1.0] for v in unique_vals)

    print(f"\n  Output shape:    {tuple(spikes_lif.shape)}")
    print(f"  Shape attesa:    {expected_shape}")
    print(f"  Shape corretta:  {'✓ OK' if shape_ok else '✗ ERRORE'}")
    print(f"  Valori unici:    {[round(v, 1) for v in unique_vals]}")
    print(f"  Binario (0/1):   {'✓ OK' if binary_ok else '✗ ERRORE'}")

    assert shape_ok, f"Shape mismatch: {spikes_lif.shape} != {expected_shape}"
    assert binary_ok, "Spike values not binary (0 or 1)"

    total_spikes = int(spikes_lif.sum().item())
    spike_density = spikes_lif.mean().item() * 100
    spikes_per_neuron = spikes_lif.sum(dim=1).mean(dim=0).cpu().numpy()

    print(f"\n  Spike totali:    {total_spikes}")
    print(f"  Densità spike:   {spike_density:.2f}%  (atteso: 10-40%)")
    print(f"  Spike medi/feature (media su {N_SAMPLES} campioni):")

    for i, (name, count) in enumerate(zip(FEATURE_NAMES, spikes_per_neuron)):
        bar = "█" * int(count / NB_STEPS * 50)
        print(f"    [{i}] {name:<18} {count:5.1f} spike  |{bar}")

    assert spike_density > 5, "Spike density too low"
    assert spike_density < 50, "Spike density too high"


# ════════════════════════════════════════════════════════════════
# TEST 2 — TemporalEncoder: 3 encoder diversi
# ════════════════════════════════════════════════════════════════


def test_temporal_encoder_distinct_outputs():
    """Test che i 3 encoder producono output DIVERSI."""
    print("\n" + "=" * 65)
    print("  TEST 2: TemporalEncoder (3 encoder distinti)")
    print("=" * 65)

    X_test = np.random.randn(N_SAMPLES, N_FEATURES) * 1.2
    results = {}

    for enc_type in ["lif", "rate_hz", "rate"]:
        enc = TemporalEncoder(
            encoding_type=enc_type,
            nb_steps=NB_STEPS,
            dt=DT,
            gain_lif=GAIN_LIF,
            gain_rate=GAIN_RATE,
        )
        np.random.seed(99)
        spikes_out = enc.encode(X_test)
        results[enc_type] = spikes_out

        n_spikes = int(spikes_out.sum().item())
        density = spikes_out.mean().item() * 100
        print(f"  '{enc_type}' → shape={tuple(spikes_out.shape)}, "
              f"spike={n_spikes} ({density:.1f}%) ✓")

    # Verifica che gli encoder producano output DIVERSI
    lif_vs_rate = not torch.equal(
        results["lif"].cpu(), results["rate"].cpu()
    )
    lif_vs_rate_hz = not torch.equal(
        results["lif"].cpu(), results["rate_hz"].cpu()
    )
    rate_vs_hz = not torch.equal(
        results["rate"].cpu(), results["rate_hz"].cpu()
    )

    print(f"\n  LIF  ≠ Rate:      {'✓ output diverso' if lif_vs_rate else '⚠️  identico'}")
    print(f"  LIF  ≠ Rate_Hz:   {'✓ output diverso' if lif_vs_rate_hz else '⚠️  identico'}")
    print(f"  Rate ≠ Rate_Hz:   {'✓ output diverso' if rate_vs_hz else '⚠️  identico'}")

    assert lif_vs_rate, "LIF and Rate should produce different outputs"
    assert lif_vs_rate_hz, "LIF and Rate_Hz should produce different outputs"


# ════════════════════════════════════════════════════════════════
# TEST 2.5 — Population Size Parameter
# ════════════════════════════════════════════════════════════════


def test_population_size_parameter():
    """Test population_size: verifica replicazione output per tutti i 3 encoder."""
    print("\n" + "=" * 65)
    print("  TEST 2.5: Population Size Parameter")
    print("=" * 65)

    X_test = np.random.randn(3, N_FEATURES)

    # Test LIFEncoder con population_size
    print("\n  LIFEncoder → population_size effects:")
    for pop_size in [1, 2, 3]:
        enc_lif = TemporalEncoder(
            encoding_type="lif",
            nb_steps=NB_STEPS,
            dt=DT,
            gain_lif=GAIN_LIF,
            input_shift=INPUT_SHIFT,
            population_size=pop_size,
        )
        spikes_lif = enc_lif.encode(X_test)
        expected_shape = (X_test.shape[0], NB_STEPS, N_FEATURES * pop_size)
        
        shape_ok = spikes_lif.shape == expected_shape
        status = "✓ OK" if shape_ok else f"✗ ERRORE (got {spikes_lif.shape})"
        print(f"    pop_size={pop_size} → shape={tuple(spikes_lif.shape)} {status}")
        assert shape_ok, f"Shape mismatch for LIF with pop_size={pop_size}"

    # Test RateEncoder Poisson con population_size
    print("\n  RateEncoder (Poisson) → population_size effects:")
    for pop_size in [1, 2]:
        enc_rate = TemporalEncoder(
            encoding_type="rate",
            nb_steps=NB_STEPS,
            dt=DT,
            gain_rate=GAIN_RATE,
            population_size=pop_size,
        )
        spikes_rate = enc_rate.encode(X_test)
        expected_shape = (X_test.shape[0], NB_STEPS, N_FEATURES * pop_size)
        
        shape_ok = spikes_rate.shape == expected_shape
        status = "✓ OK" if shape_ok else f"✗ ERRORE (got {spikes_rate.shape})"
        print(f"    pop_size={pop_size} → shape={tuple(spikes_rate.shape)} {status}")
        assert shape_ok, f"Shape mismatch for Rate with pop_size={pop_size}"

    # Test RateEncoder Hz con population_size
    print("\n  RateEncoder (Hz-based) → population_size effects:")
    for pop_size in [1, 3]:
        enc_hz = TemporalEncoder(
            encoding_type="rate_hz",
            nb_steps=NB_STEPS,
            dt=DT,
            gain_rate=GAIN_RATE,
            population_size=pop_size,
        )
        spikes_hz = enc_hz.encode(X_test)
        expected_shape = (X_test.shape[0], NB_STEPS, N_FEATURES * pop_size)
        
        shape_ok = spikes_hz.shape == expected_shape
        status = "✓ OK" if shape_ok else f"✗ ERRORE (got {spikes_hz.shape})"
        print(f"    pop_size={pop_size} → shape={tuple(spikes_hz.shape)} {status}")
        assert shape_ok, f"Shape mismatch for Hz with pop_size={pop_size}"

    print(f"\n  ✓ Population size parameter works across all 3 encoders")


# ════════════════════════════════════════════════════════════════
# TEST 3 — Comportamento biologico LIF
# ════════════════════════════════════════════════════════════════


def test_lif_biological_behavior():
    """Verifica: input alto → più spike, input basso → pochi spike."""
    print("\n" + "=" * 65)
    print("  TEST 3: Comportamento Biologico LIF")
    print("=" * 65)
    print("  (verifica: input alto → molti spike, input basso → pochi spike)")

    enc_bio = LIFEncoder(
        nb_steps=NB_STEPS, dt=DT, gain=GAIN_LIF, input_shift=INPUT_SHIFT
    )

    X_bio = np.zeros((3, N_FEATURES))
    X_bio[0, 0] = 2.0  # ALTO
    X_bio[1, 0] = 0.0  # MEDIO
    X_bio[2, 0] = -2.0  # BASSO

    spikes_bio = enc_bio.forward(X_bio)
    counts = spikes_bio[:, :, 0].sum(dim=1).cpu().numpy()

    labels = ["ALTO  ( 2.0)", "MEDIO ( 0.0)", "BASSO (-2.0)"]
    for label, count in zip(labels, counts):
        bar = "█" * int(count)
        print(f"  Input {label}: {int(count):3d} spike  |{bar}")

    bio_ok = counts[0] >= counts[1] >= counts[2]
    print(
        f"\n  Ordine corretto (alto≥medio≥basso): "
        f"{'✓ OK' if bio_ok else '✗ ERRORE BIOLOGICO'}"
    )

    assert bio_ok, (
        f"Errore biologico! Atteso alto≥medio≥basso.\n"
        f"Ottenuto: alto={counts[0]}, medio={counts[1]}, basso={counts[2]}"
    )


# ════════════════════════════════════════════════════════════════
# TEST 4 — Gestione errori
# ════════════════════════════════════════════════════════════════


def test_error_handling():
    """Test gestione errori input malformato e tipo sconosciuto."""
    print("\n" + "=" * 65)
    print("  TEST 4: Gestione Errori")
    print("=" * 65)

    enc_err = TemporalEncoder(nb_steps=NB_STEPS, dt=DT)

    # Test 1: Input 1D (dovrebbe fallire)
    try:
        enc_err.encode(np.array([0.3, -1.2, 0.8, 0.1, -0.5, 0.9]))
        print("  ✗ ERRORE: avrebbe dovuto sollevare ValueError per input 1D")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Input 1D rifiutato")

    # Test 2: Tipo sconosciuto (dovrebbe fallire)
    try:
        TemporalEncoder(encoding_type="latency")
        print("  ✗ ERRORE: avrebbe dovuto sollevare ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Tipo sconosciuto rifiutato")


# ════════════════════════════════════════════════════════════════
# TEST 5 — Raster plot
# ════════════════════════════════════════════════════════════════


def test_raster_plot():
    """Test visualizzazione raster plot."""
    print("\n" + "=" * 65)
    print("  TEST 5: Raster Plot LIF")
    print("=" * 65)

    X_plot = np.array([
        [2.1, -1.3, 0.8, -0.2, 1.5, -0.9],
        [0.3, 0.1, -1.8, 2.2, -0.5, 1.1],
        [-2.0, 1.8, 0.5, -1.5, 2.3, -0.3],
    ])

    enc_plot = TemporalEncoder(
        encoding_type="lif",
        nb_steps=NB_STEPS,
        dt=DT,
        gain_lif=GAIN_LIF,
        input_shift=INPUT_SHIFT,
    )
    spikes_plot = enc_plot.encode(X_plot)

    enc_plot.plot_spike_raster(
        spikes=spikes_plot,
        sample_idx=0,
        save_path="spike_raster_lif_test.png",
        feature_names=FEATURE_NAMES,
    )

    print(f"  ✓ Plot salvato in results/plots/")
    
    print(f"\n  Campione 0 spike per feature:")
    for i, name in enumerate(FEATURE_NAMES):
        n = int(spikes_plot[0, :, i].sum().item())
        rate = n / NB_STEPS * 100
        bar = "█" * int(rate / 2)
        print(f"    {name:<18}: {n:3d} spike ({rate:5.1f}%)  |{bar}")


# ════════════════════════════════════════════════════════════════
# TEST 6 — Confronto LIF vs Rate vs Rate_Hz
# ════════════════════════════════════════════════════════════════


def test_comparison_plot():
    """Test confronto visivo tra i 3 encoder."""
    print("\n" + "=" * 65)
    print("  TEST 6: Confronto LIF vs Rate vs Rate_Hz")
    print("=" * 65)

    X_compare = np.array([[2.1, -1.3, 0.8, -0.2, 1.5, -0.9]])

    enc_lif_cmp = TemporalEncoder(
        encoding_type="lif",
        nb_steps=NB_STEPS,
        dt=DT,
        gain_lif=GAIN_LIF,
        input_shift=INPUT_SHIFT,
    )
    enc_rate_cmp = TemporalEncoder(
        encoding_type="rate",
        nb_steps=NB_STEPS,
        dt=DT,
        gain_rate=GAIN_RATE,
    )
    enc_hz_cmp = TemporalEncoder(
        encoding_type="rate_hz",
        nb_steps=NB_STEPS,
        dt=DT,
        gain_rate=GAIN_RATE,
    )

    np.random.seed(42)
    spk_lif_c = enc_lif_cmp.encode(X_compare)
    np.random.seed(42)
    spk_rate_c = enc_rate_cmp.encode(X_compare)
    np.random.seed(42)
    spk_hz_c = enc_hz_cmp.encode(X_compare)

    # Tabella comparativa
    print(f"\n  Input: {[round(v, 1) for v in X_compare[0].tolist()]}")
    print(f"\n  {'Feature':<18} {'LIF':>8} {'Rate':>8} {'Rate_Hz':>8}")
    print(f"  {'─' * 18} {'─' * 8} {'─' * 8} {'─' * 8}")

    for i, name in enumerate(FEATURE_NAMES):
        n_lif = int(spk_lif_c[0, :, i].sum())
        n_rate = int(spk_rate_c[0, :, i].sum())
        n_hz = int(spk_hz_c[0, :, i].sum())
        print(f"  {name:<18} {n_lif:>6} sp  {n_rate:>6} sp  {n_hz:>6} sp")

    # Plot comparativo
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    cmap = plt.colormaps["tab10"]

    configs = [
        (spk_lif_c, "LIF Encoder — biologico (gain=0.05, shift=4.0, MPS)"),
        (spk_rate_c, "Rate Encoding — Poisson stocastico (gain=10.0, numpy)"),
        (spk_hz_c, "Rate Encoding Hz — deterministico + jitter ±3 (gain=10.0, numpy)"),
    ]

    for ax, (spk, title) in zip(axes, configs):
        spk_np = spk[0].cpu().numpy()

        for i in range(N_FEATURES):
            times = np.where(spk_np[:, i] > 0)[0]
            ax.scatter(
                times,
                [i] * len(times),
                marker="|",
                s=100,
                linewidths=1.6,
                color=cmap(i % 10),
                label=FEATURE_NAMES[i],
            )

        total = int(spk.sum().item())
        density = spk.mean().item() * 100
        ax.set_title(
            f"{title}   [{total} spike — {density:.1f}%]",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        ax.set_xlim(-1, NB_STEPS + 1)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_ylabel("Feature", fontsize=9)

    axes[0].legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("Timestep", fontsize=10)

    fig.suptitle(
        f"Confronto Metodi di Encoding Temporale\n"
        f"Input campione: {[round(v, 1) for v in X_compare[0].tolist()]}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    
    plots_dir = Path("results") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "spike_comparison_lif_vs_rate.png"
    
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot comparativo salvato: results/plots/spike_comparison_lif_vs_rate.png")


# ════════════════════════════════════════════════════════════════
# TEST 7 — Population Scaling Analysis
# ════════════════════════════════════════════════════════════════


def test_population_scaling_comparison():
    """Analisi realistica: 3 encoder x 3 population_size → 9 configurazioni."""
    print("\n" + "=" * 65)
    print("  TEST 7: Population Scaling Analysis")
    print("=" * 65)

    X_pop = np.array([[2.0, -1.5, 0.8, -0.3, 1.2, -0.9]])
    pop_sizes = [1, 2, 3]
    enc_types = ["lif", "rate", "rate_hz"]

    # Tabella riassuntiva
    print(f"\n  Input: {[round(v, 2) for v in X_pop[0].tolist()]}")
    print(f"\n  {'Encoder':<12} {'pop=1':>12} {'pop=2':>12} {'pop=3':>12}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    for enc_type in enc_types:
        for pop_size in pop_sizes:
            kwargs = {
                "encoding_type": enc_type,
                "nb_steps": NB_STEPS,
                "dt": DT,
                "population_size": pop_size,
            }
            if enc_type == "lif":
                kwargs["gain_lif"] = GAIN_LIF
                kwargs["input_shift"] = INPUT_SHIFT
            else:
                kwargs["gain_rate"] = GAIN_RATE
            enc_config = TemporalEncoder(**kwargs)
            np.random.seed(42)
            spikes_pop = enc_config.encode(X_pop)
            n_spikes = int(spikes_pop.sum().item())
            density = spikes_pop.mean().item() * 100
            print(f"  {enc_type:<12} {n_spikes:>6}sp({density:>5.1f}%) {n_spikes:>6}sp({density:>5.1f}%) {n_spikes:>6}sp({density:>5.1f}%)")

    # Plot 1: Heatmap di spike density con population_size
    print(f"\n  Creando visualizzazioni...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    cmap = plt.colormaps["tab10"]
    enc_titles = {
        "lif": "LIF Encoder (biologico, GPU)",
        "rate": "Rate Encoder (Poisson stocastico)",
        "rate_hz": "Rate Encoder (Hz deterministico)",
    }

    # Sottofigure: 3 encoder x 3 population sizes
    plot_idx = 0
    all_results = {}

    for enc_idx, enc_type in enumerate(enc_types):
        for pop_idx, pop_size in enumerate(pop_sizes):
            ax = axes[enc_idx, pop_idx]

            kwargs = {
                "encoding_type": enc_type,
                "nb_steps": NB_STEPS,
                "dt": DT,
                "population_size": pop_size,
            }
            if enc_type == "lif":
                kwargs["gain_lif"] = GAIN_LIF
                kwargs["input_shift"] = INPUT_SHIFT
            else:
                kwargs["gain_rate"] = GAIN_RATE
            enc_config = TemporalEncoder(**kwargs)
            np.random.seed(42)
            spikes_config = enc_config.encode(X_pop)
            all_results[f"{enc_type}_pop{pop_size}"] = spikes_config

            # Raster plot
            spk_np = spikes_config[0].cpu().numpy()
            n_features_total = spk_np.shape[1]

            for i in range(n_features_total):
                times = np.where(spk_np[:, i] > 0)[0]
                ax.scatter(
                    times,
                    [i] * len(times),
                    marker="|",
                    s=80,
                    linewidths=1.4,
                    color=cmap(i % 10),
                )

            total_spikes = int(spikes_config.sum().item())
            density = spikes_config.mean().item() * 100
            output_shape = spikes_config.shape

            ax.set_title(
                f"{enc_type.upper()} (pop_size={pop_size})\n"
                f"shape={output_shape} → {total_spikes} spike ({density:.1f}%)",
                fontsize=9,
                fontweight="bold",
            )
            ax.set_xlim(-1, NB_STEPS + 1)
            ax.set_ylim(-1, n_features_total)
            ax.set_xlabel("Timestep", fontsize=8)
            ax.set_ylabel(f"Neuron (×{pop_size if pop_size > 1 else 1})", fontsize=8)
            ax.grid(True, alpha=0.2, axis="x")

    # Riorganizza layout
    fig.suptitle(
        f"Population Scaling Effect: 3 Encoder Types × 3 Population Sizes\n"
        f"Input: {[round(v, 2) for v in X_pop[0].tolist()]} (normalized)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    plots_dir = Path("results") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "population_scaling_analysis.png"

    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Population scaling plot salvato: results/plots/population_scaling_analysis.png")

    # Plot 2: Confronto lineare popolazione vs numero spike
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, enc_type in enumerate(enc_types):
        spike_counts = []
        for pop_size in pop_sizes:
            kwargs = {
                "encoding_type": enc_type,
                "nb_steps": NB_STEPS,
                "dt": DT,
                "population_size": pop_size,
            }
            if enc_type == "lif":
                kwargs["gain_lif"] = GAIN_LIF
                kwargs["input_shift"] = INPUT_SHIFT
            else:
                kwargs["gain_rate"] = GAIN_RATE
            enc = TemporalEncoder(**kwargs)
            np.random.seed(42)
            spk = enc.encode(X_pop)
            spike_counts.append(int(spk.sum().item()))

        # Plot lineare
        ax = axes[ax_idx]
        bars = ax.bar(
            [f"pop={p}" for p in pop_sizes],
            spike_counts,
            color=[cmap(i) for i in range(len(pop_sizes))],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Aggiungi valori sopra le barre
        for bar, count in zip(bars, spike_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(f"{enc_titles[enc_type]}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Total Spike Count", fontsize=10)
        ax.set_xlabel("Population Size", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(spike_counts) * 1.15)

    fig.suptitle(
        "Spike Count vs Population Size\n(Realistic behavior with same input)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path_bars = plots_dir / "population_spike_scaling.png"
    plt.savefig(str(save_path_bars), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Spike scaling plot salvato: results/plots/population_spike_scaling.png")

    # Print dettagli numerici
    print(f"\n  Dettagli numerici per encoder:")
    for enc_type in enc_types:
        print(f"\n    {enc_titles[enc_type]}:")
        for pop_size in pop_sizes:
            kwargs = {
                "encoding_type": enc_type,
                "nb_steps": NB_STEPS,
                "dt": DT,
                "population_size": pop_size,
            }
            if enc_type == "lif":
                kwargs["gain_lif"] = GAIN_LIF
                kwargs["input_shift"] = INPUT_SHIFT
            else:
                kwargs["gain_rate"] = GAIN_RATE
            enc = TemporalEncoder(**kwargs)
            np.random.seed(42)
            spk = enc.encode(X_pop)
            total = int(spk.sum().item())
            shape = spk.shape
            print(f"      pop_size={pop_size} → shape={shape} → {total} spike")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main():
    """Esegue tutti i test."""
    print("\n" + "=" * 65)
    print("  TEST plant_temporal_encoding.py — Tutti gli Encoder")
    print("=" * 65)

    test_device_detection()
    test_lif_encoder_forward()
    test_temporal_encoder_distinct_outputs()
    test_population_size_parameter()
    test_lif_biological_behavior()
    test_error_handling()
    test_raster_plot()
    test_comparison_plot()
    test_population_scaling_comparison()

    print("\n" + "=" * 65)
    print("  TUTTI I TEST SUPERATI ✓")
    print("=" * 65)
    print(f"""
  Test Structure:
    TEST 0: Device detection           → MPS/CUDA/CPU rilevamento
    TEST 1: LIFEncoder                 → shape, binary spike, density
    TEST 2: Encoder diversity          → LIF ≠ Rate ≠ Rate_Hz
    TEST 2.5: Population size          → replicazione feature (pop_size=1,2,3)
    TEST 3: Biological behaviour       → input ↑ → spike ↑
    TEST 4: Error handling             → invalid input/type
    TEST 5: Raster plot                → visualizzazione singolo encoder
    TEST 6: Comparison plot            → 3 encoder side-by-side (pop=1)
    TEST 7: Population scaling         → 3 encoder × 3 population (9 config)

  Available Encoders:
    "lif"      → LIFEncoder      (biologico, GPU-accelerated)
    "rate"     → RateEncoder     (Poisson stocastico)
    "rate_hz"  → RateEncoder     (Hz deterministico con jitter)
  
  Tutti supportano:
    • population_size (default=1) → output shape = (*,  *, n_features * pop_size)
    • device selezionamento automatico (MPS > CUDA > CPU)
    • z-score normalized input

  Generated Visualizations:
    → spike_raster_lif_test.png              (TEST 5: LIF raster)
    → spike_comparison_lif_vs_rate.png       (TEST 6: 3 encoder comparison)
    → population_scaling_analysis.png        (TEST 7: 3×3 raster grid)
    → population_spike_scaling.png           (TEST 7: spike count bars)
  """)


if __name__ == "__main__":
    main()
