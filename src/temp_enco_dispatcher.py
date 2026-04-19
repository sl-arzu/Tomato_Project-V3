"""
Temporal Encoder Dispatcher

Dispatcher centrale che seleziona automaticamente il tipo di encoder:
  • "lif"      → LIFEncoder (biologico, GPU/MPS support)
  • "rate"     → RateEncoder Poisson (stocastico)
  • "rate_hz"  → RateEncoder Hz-based (deterministico)

Tutti gli encoder supportano population_size per duplicare gli output.

Autore: Shanti Leonardo Arzu
"""

from typing import Optional
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.temp_enco_lif_population import LIFEncoder
from src.temp_enco_rate import RateEncoder


class TemporalEncoder:
    """
    Central dispatcher for temporal encoding methods.
    
    Seleziona automaticamente l'encoder corretto in base a encoding_type.
    Interfaccia unica per tutti i tipi di encoding.
    
    Output shape: (n_samples, nb_steps, n_features * population_size)
    """

    VALID_ENCODING_TYPES = {"lif", "rate", "rate_hz"}

    def __init__(
        self,
        encoding_type: str = "lif",
        nb_steps: int = 150,
        dt: float = 3.0,
        gain_lif: float = 0.05,
        gain_rate: float = 10.0,
        tau_syn: float = 5.0,
        tau_mem: float = 10.0,
        threshold: float = 1.0,
        input_shift: float = 4.0,
        tau_ref: float = 5.0,
        noise_std: float = 0.2,
        population_size: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            encoding_type: "lif" | "rate" | "rate_hz"
            nb_steps: numero timestep della sequenza
            dt: durata timestep (ms)
            gain_lif: amplificazione LIF (default 0.05)
            gain_rate: amplificazione rate (default 10.0)
            tau_syn: sinaptica time costante (ms)
            tau_mem: membrana time costante (ms)
            threshold: soglia spike LIF
            input_shift: traslazione input LIF (default 4.0)
            tau_ref: periodo refrattario LIF (ms)
            noise_std: rumore std (0=deterministico)
            population_size: neuroni per feature output (default 1)
                Output totale: n_features * population_size
        """
        # Validazione
        if encoding_type not in self.VALID_ENCODING_TYPES:
            raise ValueError(
                f"encoding_type '{encoding_type}' non valido. "
                f"Scegli tra: {sorted(self.VALID_ENCODING_TYPES)}"
            )

        self.encoding_type = encoding_type
        self.nb_steps = nb_steps
        self.dt = dt
        self.population_size = population_size

        # Istanzia l'encoder corretto
        if encoding_type == "lif":
            self._encoder = LIFEncoder(
                nb_steps=nb_steps,
                dt=dt,
                tau_syn=tau_syn,
                tau_mem=tau_mem,
                threshold=threshold,
                gain=gain_lif,
                input_shift=input_shift,
                tau_ref=tau_ref,
                noise_std=noise_std,
                population_size=population_size,
                seed=seed,
            )
            self._mode = "lif"
        else:
            self._encoder = RateEncoder(
                nb_steps=nb_steps,
                dt=dt,
                gain=gain_rate,
                population_size=population_size,
            )
            self._mode = encoding_type

        # Log configurazione
        self._log_init(encoding_type, gain_lif, gain_rate)

    def _log_init(self, encoding_type, gain_lif, gain_rate):
        """Log configurazione encoder."""
        print(
            f"[TemporalEncoder] type={encoding_type} "
            f"→ {type(self._encoder).__name__}"
        )
        gain_str = f"lif={gain_lif}" if encoding_type == "lif" else f"rate={gain_rate}"
        print(
            f"[TemporalEncoder] nb_steps={self.nb_steps}, dt={self.dt}ms, "
            f"gain={gain_str}, pop_size={self.population_size}"
        )

    def encode(self, X: np.ndarray) -> torch.Tensor:
        """
        Codifica input continui in spike temporali.
        
        Args:
            X: (n_samples, n_features) z-score normalized
            
        Returns:
            Spike tensor (n_samples, nb_steps, n_features * population_size)
        """
        if X.ndim != 2:
            raise ValueError(
                f"X deve avere shape (n_samples, n_features), ricevuto: {X.shape}"
            )

        if isinstance(self._encoder, LIFEncoder):
            with torch.no_grad():
                return self._encoder(X)
        else:
            # RateEncoder
            if self._mode == "rate":
                return self._encoder.encode_poisson(X)
            else:  # rate_hz
                return self._encoder.encode_hz(X)

    def plot_spike_raster(
        self,
        spikes: torch.Tensor,
        sample_idx: int = 0,
        save_path: str = "spike_raster.png",
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Visualizza raster plot degli spike per un campione.
        
        Args:
            spikes: (n_samples, nb_steps, n_features) spike tensor
            sample_idx: campione da visualizzare
            save_path: percorso salvataggio PNG
            feature_names: nomi feature (opzionale)
        """
        # Crea cartella se manca
        plots_dir = Path("results") / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if "/" not in save_path and "\\" not in save_path:
            full_path = plots_dir / save_path
        else:
            full_path = Path(save_path)

        spikes_cpu = spikes[sample_idx].cpu().numpy()
        n_features = spikes_cpu.shape[1]

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

        # Pannello 1: Raster
        ax_raster = fig.add_subplot(gs[0])
        cmap = plt.colormaps["tab10"]

        for i in range(n_features):
            spike_times = np.where(spikes_cpu[:, i] > 0)[0]
            ax_raster.scatter(
                spike_times,
                [i] * len(spike_times),
                marker="|",
                s=120,
                linewidths=1.8,
                color=cmap(i % 10),
                label=feature_names[i],
            )

        ax_raster.set_xlabel("Timestep", fontsize=11)
        ax_raster.set_ylabel("Feature (Neurone)", fontsize=11)
        ax_raster.set_title(
            f"Spike Raster [{self.encoding_type.upper()}] "
            f"Sample #{sample_idx} ({self.nb_steps}×{n_features})",
            fontsize=12,
        )
        ax_raster.set_yticks(range(n_features))
        ax_raster.set_yticklabels(feature_names, fontsize=9)
        ax_raster.set_xlim(-1, self.nb_steps + 1)
        ax_raster.grid(True, alpha=0.25, axis="x")
        ax_raster.legend(loc="upper right", fontsize=8, ncol=2)

        # Pannello 2: Conteggio spike
        ax_bar = fig.add_subplot(gs[1])
        spike_counts = spikes_cpu.sum(axis=0)
        spike_rates = spike_counts / self.nb_steps * 100

        bars = ax_bar.bar(
            range(n_features),
            spike_counts,
            color=[cmap(i % 10) for i in range(n_features)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.7,
        )

        for bar, count, rate in zip(bars, spike_counts, spike_rates):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{int(count)}\n({rate:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax_bar.set_xlabel("Feature (Neurone)", fontsize=10)
        ax_bar.set_ylabel("Spike Count", fontsize=10)
        ax_bar.set_xticks(range(n_features))
        ax_bar.set_xticklabels(feature_names, fontsize=8, rotation=15)
        ax_bar.set_title("Spike Count per Feature", fontsize=10)
        ax_bar.grid(True, alpha=0.25, axis="y")

        plt.savefig(str(full_path), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[TemporalEncoder] Raster plot salvato: {full_path}")

