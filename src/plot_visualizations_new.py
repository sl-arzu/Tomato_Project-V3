"""
plot_visualizations_new.py
Visualizzazioni per TomatoProject v2

Genera:
- Curve di training (accuracy e loss)
- Confusion matrix normalizzata per riga
- Evoluzione media dei pesi
- Evoluzione di singoli pesi campionati
- Raster plot dell'attività neurale con struttura chiara:
  Encoding Input -> Hidden Layer -> Output Layer

Autore: Shanti Leonardo Arzu
Versione ristrutturata: Aprile 2026
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


# -----------------------------------------------------------------------------
# STILE GLOBALE
# -----------------------------------------------------------------------------

sn.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "font.size": 10
})


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def _apply_axis_style(ax, grid_axis="both"):
    ax.set_facecolor("#fafafa")

    if grid_axis == "x":
        ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.25)
        ax.grid(False, axis="y")
    elif grid_axis == "y":
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
        ax.grid(False, axis="x")
    else:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def _save_figure(fig, save_path, label):
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {label} salvato in: {save_path}")


def _prepare_spike_matrix(spike_array):
    """
    Converte un array spike in shape (nb_steps, nb_neurons).
    Accetta anche vettori 1D o matrici accidentalmente trasposte.
    """
    arr = np.asarray(spike_array)

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim != 2:
        raise ValueError(
            f"Spike array non valido: attese 1D o 2D, ricevuto ndim={arr.ndim}"
        )

    # Heuristica robusta: se sembra (neuroni, timesteps), trasponi
    if arr.shape[0] <= 32 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    return (arr > 0).astype(float)


def _compute_layer_stats(spike_matrix, total_time_ms):
    spike_counts = np.sum(spike_matrix, axis=0).astype(int)
    total_spikes = int(spike_counts.sum())
    density = 100.0 * total_spikes / max(1, spike_matrix.size)

    total_time_sec = max(total_time_ms / 1000.0, 1e-12)
    firing_rates_hz = spike_counts / total_time_sec

    return spike_counts, total_spikes, density, firing_rates_hz


def _set_time_ticks(ax, total_time_ms, nb_steps, time_step):
    if nb_steps <= 1:
        ax.set_xticks([0])
        ax.set_xticklabels(["0"])
        return

    n_ticks = min(8, nb_steps)
    tick_positions = np.linspace(0, total_time_ms, n_ticks)
    tick_labels = [f"{int(round(t))}" for t in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def _sample_weight_indices(weight_matrix, num_weights_to_plot, seed=42):
    total_weights = weight_matrix.size
    sample_size = min(num_weights_to_plot, total_weights)
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(total_weights, size=sample_size, replace=False)
    return flat_indices


def _reshape_weight_matrix(weight_matrix):
    arr = np.asarray(weight_matrix)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return arr


def _print_raster_summary(layer_stats):
    print("\n" + "=" * 88)
    print("[RASTER] Riepilogo attività neurale")
    print("-" * 88)

    for item in layer_stats:
        layer_name = item["layer_name"]
        n_neurons = item["n_neurons"]
        total_spikes = item["total_spikes"]
        density = item["density"]
        mean_rate = item["mean_rate"]

        print(
            f"  - {layer_name:<15} | neuroni: {n_neurons:>3d} | "
            f"spike totali: {total_spikes:>4d} | densità: {density:>6.2f}% | "
            f"rate medio: {mean_rate:>7.2f} Hz"
        )

    print("=" * 88)


# -----------------------------------------------------------------------------
# TRAINING CURVES
# -----------------------------------------------------------------------------

def plot_training_performance(acc_train, acc_test, loss_train, save_path):
    """
    Plotta accuracy e loss durante l'addestramento.

    Args:
        acc_train: lista accuracy training per epoch
        acc_test: lista accuracy test per epoch
        loss_train: lista loss training per epoch
        save_path: path finale della figura
    """
    if len(acc_train) == 0 or len(loss_train) == 0:
        print("[PLOT] Nessun dato disponibile per il training plot.")
        return

    acc_train = np.asarray(acc_train, dtype=float)
    acc_test = np.asarray(acc_test, dtype=float) if len(acc_test) > 0 else np.array([])
    loss_train = np.asarray(loss_train, dtype=float)

    epochs_train = np.arange(1, len(acc_train) + 1)
    epochs_test = np.arange(1, len(acc_test) + 1) if len(acc_test) > 0 else np.array([])

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.25)

    # Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        epochs_train,
        acc_train * 100.0,
        color="#1f4ed8",
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Train Accuracy"
    )

    if len(acc_test) > 0:
        ax1.plot(
            epochs_test,
            acc_test * 100.0,
            color="#e11d48",
            linewidth=2.5,
            marker="s",
            markersize=6,
            label="Test Accuracy"
        )

    ax1.set_title("Accuracy durante Training", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", framealpha=0.95)
    _apply_axis_style(ax1)

    if len(epochs_train) <= 15:
        ax1.set_xticks(epochs_train)

    ax1.annotate(
        f"{acc_train[-1] * 100:.1f}%",
        xy=(epochs_train[-1], acc_train[-1] * 100.0),
        xytext=(8, 8),
        textcoords="offset points",
        color="#1f4ed8",
        fontweight="bold"
    )

    if len(acc_test) > 0:
        ax1.annotate(
            f"{acc_test[-1] * 100:.1f}%",
            xy=(epochs_test[-1], acc_test[-1] * 100.0),
            xytext=(8, -16),
            textcoords="offset points",
            color="#e11d48",
            fontweight="bold"
        )

    # Loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        epochs_train,
        loss_train,
        color="#16a34a",
        linewidth=2.5,
        marker="D",
        markersize=5
    )

    ax2.set_title("Loss durante Training", fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Loss (MSE)", fontweight="bold")
    _apply_axis_style(ax2)

    if len(epochs_train) <= 15:
        ax2.set_xticks(epochs_train)

    ax2.annotate(
        f"{loss_train[-1]:.4f}",
        xy=(epochs_train[-1], loss_train[-1]),
        xytext=(8, 8),
        textcoords="offset points",
        color="#15803d",
        fontweight="bold"
    )

    _save_figure(fig, save_path, "Grafico training")

    print(
        f"[TRAINING] Epoche: {len(acc_train)} | "
        f"Train finale: {acc_train[-1] * 100:.2f}% | "
        f"Test finale: {(acc_test[-1] * 100):.2f}% "
        f"({'presente' if len(acc_test) > 0 else 'assente'}) | "
        f"Loss finale: {loss_train[-1]:.6f}"
    )


# -----------------------------------------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_labels, save_path):
    """
    Matrice di confusione normalizzata per riga.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("[PLOT] Nessun dato disponibile per la confusion matrix.")
        return

    label_ids = np.arange(len(class_labels))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize="true")
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    overall_acc = 100.0 * float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    fig, ax = plt.subplots(figsize=(9, 7))
    sn.heatmap(
        cm_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=1.5,
        linecolor="gray",
        cbar=True,
        cbar_kws={"label": "Normalized Accuracy"},
        annot_kws={"fontsize": 13},
        ax=ax
    )

    ax.set_title(
        f"Confusion Matrix (Test Set - Normalized by Row)\nOverall Accuracy: {overall_acc:.2f}%",
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_xticklabels(class_labels, rotation=35, ha="right")
    ax.set_yticklabels(class_labels, rotation=0)

    _save_figure(fig, save_path, "Confusion matrix")
    print(f"[CONFUSION] Accuratezza complessiva sul test set: {overall_acc:.2f}%")


# -----------------------------------------------------------------------------
# WEIGHT EVOLUTION
# -----------------------------------------------------------------------------

def plot_weights_evolution(weight_history, save_path):
    """
    Evoluzione del valore assoluto medio dei pesi.
    Nota: nel trainer attuale il weight history viene salvato a ogni batch/update,
    quindi l'asse x è uno step di training e non una vera epoch.
    """
    if "w_in" not in weight_history or len(weight_history["w_in"]) == 0:
        print("[PLOT] Nessun dato disponibile per l'evoluzione media dei pesi.")
        return

    steps = np.arange(1, len(weight_history["w_in"]) + 1)

    w_in_means = [np.mean(np.abs(w)) for w in weight_history["w_in"]]
    w_rec_means = [np.mean(np.abs(w)) for w in weight_history["w_rec"]]
    w_out_means = [np.mean(np.abs(w)) for w in weight_history["w_out"]]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        steps,
        w_in_means,
        color="#2563eb",
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="Input Weights | mean(|w|)"
    )
    ax.plot(
        steps,
        w_rec_means,
        color="#f97316",
        linewidth=2.4,
        marker="s",
        markersize=5,
        label="Recurrent Weights | mean(|w|)"
    )
    ax.plot(
        steps,
        w_out_means,
        color="#16a34a",
        linewidth=2.4,
        marker="D",
        markersize=5,
        label="Output Weights | mean(|w|)"
    )

    ax.set_title("Weight Evolution During Training", fontweight="bold")
    ax.set_xlabel("Training step", fontweight="bold")
    ax.set_ylabel("Mean Absolute Weight Value", fontweight="bold")
    ax.legend(loc="best", framealpha=0.95)
    _apply_axis_style(ax)

    if len(steps) <= 15:
        ax.set_xticks(steps)

    _save_figure(fig, save_path, "Evoluzione media pesi")

    print(
        f"[WEIGHTS] Snapshot disponibili: {len(steps)} | "
        f"ultimo mean(|w_in|)={w_in_means[-1]:.5f} | "
        f"ultimo mean(|w_rec|)={w_rec_means[-1]:.5f} | "
        f"ultimo mean(|w_out|)={w_out_means[-1]:.5f}"
    )


def plot_individual_weights_evolution(weight_history, save_path, num_weights_to_plot=10):
    """
    Visualizza la traiettoria di un sottoinsieme di pesi per ciascun layer.
    """
    if "w_in" not in weight_history or len(weight_history["w_in"]) == 0:
        print("[PLOT] Nessun dato disponibile per i pesi individuali.")
        return

    steps = np.arange(1, len(weight_history["w_in"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    layer_cfg = [
        ("w_in",  "Input Weights Evolution",     axes[0]),
        ("w_rec", "Recurrent Weights Evolution", axes[1]),
        ("w_out", "Output Weights Evolution",    axes[2]),
    ]

    for layer_key, layer_title, ax in layer_cfg:
        first_matrix = _reshape_weight_matrix(weight_history[layer_key][0])
        sampled_flat_idx = _sample_weight_indices(first_matrix, num_weights_to_plot, seed=42)
        colors = plt.cm.tab20(np.linspace(0, 1, len(sampled_flat_idx)))

        n_cols = first_matrix.shape[1]

        for color_idx, flat_idx in enumerate(sampled_flat_idx):
            row_idx = flat_idx // n_cols
            col_idx = flat_idx % n_cols

            trajectory = []
            for w in weight_history[layer_key]:
                current_w = _reshape_weight_matrix(w)
                trajectory.append(current_w[row_idx, col_idx])

            ax.plot(
                steps,
                trajectory,
                linewidth=1.7,
                alpha=0.90,
                color=colors[color_idx]
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(
            f"{layer_title} (Sample of {len(sampled_flat_idx)} weights)",
            fontweight="bold"
        )
        ax.set_ylabel("Weight Value", fontweight="bold")
        _apply_axis_style(ax)

    axes[-1].set_xlabel("Training step", fontweight="bold")

    if len(steps) <= 15:
        axes[-1].set_xticks(steps)

    _save_figure(fig, save_path, "Evoluzione pesi individuali")

    print(
        f"[WEIGHTS] Plot individuale creato con {min(num_weights_to_plot, first_matrix.size)} "
        f"pesi campionati per layer."
    )


# -----------------------------------------------------------------------------
# RASTER PLOT
# -----------------------------------------------------------------------------

def plot_network_activity(
    spr_recs,
    layer_names,
    figname,
    time_step=3.0,
    gain_factor=10.0,
    encoding_type="unknown",
    algorithm="unknown"
):
    """
    Raster plot pulito e leggibile, simile alla struttura richiesta:

    - Un subplot per layer
    - Titolo locale con: nome layer | numero neuroni | spike totali | densità
    - Statistiche per neurone sul lato destro
    - Titolo globale con encoding, algoritmo e durata totale

    Args:
        spr_recs: lista di array con shape (nb_steps, nb_neurons)
        layer_names: lista di nomi layer
        figname: path base senza estensione
        time_step: ms per timestep
        gain_factor: mantenuto per compatibilità, solo informativo
        encoding_type: "lif", "rate", ...
        algorithm: "eprop", "bptt", ...
    """
    if len(spr_recs) == 0:
        print("[RASTER] Nessun array di spike ricevuto.")
        return

    n_layers = min(len(spr_recs), len(layer_names)) if len(layer_names) > 0 else len(spr_recs)

    if n_layers == 0:
        print("[RASTER] Nessun layer valido da plottare.")
        return

    prepared_layers = []
    used_names = []

    for idx in range(n_layers):
        prepared_layers.append(_prepare_spike_matrix(spr_recs[idx]))
        used_names.append(layer_names[idx] if idx < len(layer_names) else f"Layer {idx + 1}")

    nb_steps = min(layer.shape[0] for layer in prepared_layers)
    prepared_layers = [layer[:nb_steps] for layer in prepared_layers]

    total_time_ms = float(nb_steps) * float(time_step)
    visible_time_ms = max(total_time_ms, float(time_step))
    x_max = visible_time_ms * 1.13
    time_axis_ms = np.arange(nb_steps, dtype=float) * float(time_step)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"
    ]

    fig_height = 2.0 + 1.7 * n_layers
    fig = plt.figure(figsize=(16, fig_height))
    gs = GridSpec(n_layers, 1, figure=fig, hspace=0.50)

    encoding_display = str(encoding_type).upper()
    algorithm_display = str(algorithm).upper()

    layer_stats_for_print = []

    for layer_idx, (spk_layer, layer_name) in enumerate(zip(prepared_layers, used_names)):
        ax = fig.add_subplot(gs[layer_idx, 0])

        n_neurons = spk_layer.shape[1]
        spike_counts, total_spikes, density, firing_rates_hz = _compute_layer_stats(
            spk_layer, visible_time_ms
        )

        events = []
        for neuron_idx in range(n_neurons):
            spike_idx = np.where(spk_layer[:, neuron_idx] > 0)[0]
            events.append(time_axis_ms[spike_idx] if len(spike_idx) > 0 else np.array([]))

        if total_spikes > 0:
            ax.eventplot(
                events,
                lineoffsets=np.arange(n_neurons),
                linelengths=0.82,
                linewidths=1.4 if n_neurons <= 12 else 1.1,
                colors=[palette[i % len(palette)] for i in range(n_neurons)],
                orientation="horizontal"
            )
        else:
            ax.text(
                0.5,
                0.50,
                "Nessuno spike registrato",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
                fontweight="bold"
            )

        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.6, max(0.6, n_neurons - 0.4))

        if n_neurons <= 12:
            y_ticks = np.arange(n_neurons)
        else:
            tick_step = max(1, n_neurons // 10)
            y_ticks = np.arange(0, n_neurons, tick_step)

        ax.set_yticks(y_ticks)
        ax.set_ylabel(f"{layer_name}\nNeuron id", fontweight="bold")
        _set_time_ticks(ax, visible_time_ms, nb_steps, time_step)
        _apply_axis_style(ax, grid_axis="x")

        header = (
            f"{layer_name:<18} | {n_neurons} neurons | "
            f"{total_spikes} spikes ({density:.1f}%)"
        )
        ax.set_title(header, loc="left", fontweight="bold", fontsize=10, pad=6)

        if n_neurons <= 60:
            stats_x = visible_time_ms * 1.01
            for neuron_idx in range(n_neurons):
                stat_text = f"{spike_counts[neuron_idx]:>3d} sp | {firing_rates_hz[neuron_idx]:>6.1f} Hz"
                ax.text(
                    stats_x,
                    neuron_idx,
                    stat_text,
                    fontsize=7,
                    color="#1e3a8a",
                    va="center",
                    family="monospace",
                    fontweight="bold"
                )

        if layer_idx == n_layers - 1:
            ax.set_xlabel("Time (ms)", fontweight="bold")

        layer_stats_for_print.append({
            "layer_name": layer_name,
            "n_neurons": int(n_neurons),
            "total_spikes": int(total_spikes),
            "density": float(density),
            "mean_rate": float(np.mean(firing_rates_hz)) if len(firing_rates_hz) > 0 else 0.0
        })

    fig.suptitle(
        "Neural Activity Analysis - Raster Plot Visualization\n"
        f"Encoding: {encoding_display}   |   Algorithm: {algorithm_display}   |   "
        f"Total Time: {int(round(visible_time_ms))} ms ({nb_steps} timesteps)",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    pdf_path = figname + ".pdf"

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    _print_raster_summary(layer_stats_for_print)
    print(f"[RASTER] Gain factor ricevuto: {gain_factor}")
    print(f"[PLOT] Raster plot salvato in: {pdf_path}")

