"""
LIF Encoder (Leaky Integrate-and-Fire)

Encoder biologicamente ispirato che converte segnali continui in sequenze di spike
temporali simulando neuroni LIF reali.

Supporta: MPS (Apple M1/M2/M3), CUDA, CPU
Autore: Shanti Leonardo Arzu
"""

import numpy as np
import torch
import torch.nn as nn
from src.snn_gradient_surrogate import spike_function
from typing import Optional


class LIFEncoder(nn.Module):
    """
    Leaky Integrate-and-Fire neuron encoder con supporto popolazione.
    
    Ogni feature diventa un neurone LIF indipendente.
    Con population_size > 1, ogni feature è replicata in neuroni indipendenti.
    
    Output shape: (n_samples, nb_steps, n_features * population_size)
    """

    def __init__(
        self,
        nb_steps: int = 100,
        dt: float = 1.0,
        tau_syn: float = 5.0,
        tau_mem: float = 10.0,
        tau_ref: float = 5.0,
        threshold: float = 1.0,
        gain: float = 0.05,
        input_shift: float = 4.0,
        noise_std: float = 0.3,
        seed: Optional[int] = 42,
        population_size: int = 1,
    ):
        """
        Args:
            nb_steps: numero di timestep della simulazione
            dt: durata timestep (ms)
            tau_syn: costante sinaptica (ms)
            tau_mem: costante membrana (ms)
            tau_ref: periodo refrattario (ms)
            threshold: soglia di spike
            gain: amplificazione input (default 0.05 per LIF)
            input_shift: traslazione input per zona eccitativa (default 4.0)
            noise_std: deviazione rumore (0=deterministico, range 0.05-0.3)
            seed: seed per riproducibilità (None=stocastico globale)
            population_size: neuroni per feature (default 1, nessuna duplicazione)
        """
        super().__init__()

        self.nb_steps = nb_steps
        self.dt = dt
        self.threshold = threshold
        self.gain = gain
        self.input_shift = input_shift
        self.tau_ref = tau_ref
        self.ref_steps = max(1, int(round(tau_ref / dt)))
        self.noise_std = noise_std
        self.seed = seed
        self.population_size = population_size

        # Selezione device automatica (PRIMA della RNG)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Configurazione RNG (sul device corretto per compatibilità)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            self._rng = torch.Generator(device=self.device)
            self._rng.manual_seed(self.seed)
        else:
            # Nessun seed: usa il RNG globale di torch (NON reimposta il seed)
            self._rng = None

        # Parametri decadimento biologico
        self.alpha = float(np.exp(-dt / tau_syn))
        self.beta = float(np.exp(-dt / tau_mem))

        # Soglia minima di firing (debug)
        i_min_fire = (1 - self.alpha) * (1 - self.beta)
        x_min_fire = i_min_fire / gain - input_shift

        # Log configurazione
        self._log_config(tau_syn, tau_mem, i_min_fire, x_min_fire)

    def _log_config(self, tau_syn, tau_mem, i_min_fire, x_min_fire):
        """Log parametri durante init."""
        print(f"[LIFEncoder] Device: {self.device}")
        print(f"[LIFEncoder] α={self.alpha:.4f}  β={self.beta:.4f}")
        print(f"[LIFEncoder] I_min={i_min_fire:.4f}  X_min={x_min_fire:.2f}")
        max_hz = 1000 / (self.dt * (1 + self.ref_steps))
        print(
            f"[LIFEncoder] τ_ref={self.tau_ref}ms  ref_steps={self.ref_steps}  "
            f"max_freq≈{max_hz:.0f}Hz  noise={self.noise_std}"
        )
        print(f"[LIFEncoder] dt={self.dt}ms  steps={self.nb_steps}  pop_size={self.population_size}")

    def forward(self, X: np.ndarray) -> torch.Tensor:
        """
        Simula neuroni LIF per T timestep.
        
        Dinamica per ogni timestep:
        1. Aggiorna sinapsi: syn = α·syn + I_input + noise
        2. Aggiorna membrana: mem = β·mem + syn
        3. Applica maschera refrattaria
        4. Genera spike se mem > soglia
        5. Soft reset: mem -= spike * soglia
        6. Gestisci periodo refrattario
        
        Args:
            X: (n_samples, n_features) - z-score normalized
            
        Returns:
            (n_samples, nb_steps, n_features*population_size) spike tensor
        """
        n_samples, n_features = X.shape

        # Tensore input su device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Replica per popolazione
        if self.population_size > 1:
            X_tensor = torch.repeat_interleave(X_tensor, self.population_size, dim=1)

        # Corrente input: I = (X + shift) × gain
        I_input = (X_tensor + self.input_shift) * self.gain
        n_features_expanded = I_input.shape[1]

        # Inizializza stato neuronale (sul device corretto)
        if self.noise_std > 0.0:
            # Genera rumore direttamente sul device
            if self._rng is not None:
                mem = torch.rand(I_input.shape, dtype=torch.float32, generator=self._rng, device=self.device)
            else:
                mem = torch.rand(I_input.shape, dtype=torch.float32, device=self.device)
            mem = mem * self.threshold * 0.5
        else:
            mem = torch.zeros_like(I_input)

        syn = torch.zeros_like(I_input)  # Corrente sinaptica
        refr = torch.zeros_like(I_input)  # Timer refrattario

        # Registra spike per T timestep
        spike_record = torch.zeros(
            n_samples, self.nb_steps, n_features_expanded,
            dtype=torch.float32, device=self.device
        )

        for t in range(self.nb_steps):
            # Rumore gaussiano (opzionale, generato sul device corretto)
            if self.noise_std > 0.0:
                if self._rng is not None:
                    noise = torch.randn(I_input.shape, dtype=torch.float32, generator=self._rng, device=self.device)
                else:
                    noise = torch.randn(I_input.shape, dtype=torch.float32, device=self.device)
                noise = noise * self.noise_std
            else:
                noise = torch.zeros_like(I_input)

            # Sinapsi
            syn = self.alpha * syn + I_input + noise

            # Membrana
            mem = self.beta * mem + syn

            # Refrattario: 0=attivo, >0=bloccato
            active_mask = (refr <= 0).float()
            mem_effective = mem * active_mask

            # Spike (surrogate gradient)
            spike = spike_function(mem_effective, self.threshold)

            # Reset soft (mem -= soglia per firing rate più biologico)
            mem = mem_effective - spike * self.threshold

            # Aggiorna timer refrattario
            refr = torch.clamp(refr - 1.0, min=0.0) + spike * float(self.ref_steps)

            # Registra
            spike_record[:, t, :] = spike

        return spike_record

    def get_params_summary(self) -> dict:
        """Dizionario parametri per logging/riproducibilità."""
        return {
            "nb_steps": self.nb_steps,
            "dt_ms": self.dt,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "threshold": self.threshold,
            "gain": self.gain,
            "input_shift": self.input_shift,
            "tau_ref_ms": self.tau_ref,
            "ref_steps": self.ref_steps,
            "max_hz": round(1000 / (self.dt * (1 + self.ref_steps)), 1),
            "noise_std": self.noise_std,
            "population_size": self.population_size,
            "device": str(self.device),
        }

    
    