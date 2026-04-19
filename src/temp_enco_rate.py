"""
Rate Encoder - Encoding basato su frequenza

Supporta due metodi di rate coding:
1. Poisson: stocastico, probabilità proporzionale all'ampiezza
2. Hz: deterministico, spike a frequenza fissa + jitter

Autore: Shanti Leonardo Arzu
"""

import numpy as np
import torch


class RateEncoder:
    """
    Rate coding encoder (Poisson + Hz deterministic).
    
    Converte input continui in spike basati su rate coding.
    Se population_size > 1, ogni feature è replicata in neuroni indipendenti.
    
    Output shape: (n_samples, nb_steps, n_features * population_size)
    """

    def __init__(
        self,
        nb_steps: int = 150,
        dt: float = 3.0,
        gain: float = 10.0,
        population_size: int = 1,
    ):
        """
        Args:
            nb_steps: numero di timestep
            dt: durata timestep (ms)
            gain: amplificazione input (calibrato per z-score + shift +3.0)
            population_size: neuroni per feature (default 1)
        """
        self.nb_steps = nb_steps
        self.dt = dt
        self.gain = gain
        self.population_size = population_size

    def encode_poisson(self, X: np.ndarray) -> torch.Tensor:
        """
        Poisson rate encoding (stocastico).
        
        Algoritmo:
        1. X_pos = X + 3.0  (porta z-score [-3,+3] in [0,6])
        2. prob = clip(X_pos × gain / 100, 0, 1)
        3. Spike se random() < prob
        4. Replica per population_size
        
        Args:
            X: (n_samples, n_features)
            
        Returns:
            Spike tensor (n_samples, nb_steps, n_features * population_size)
        """
        n_samples, n_features = X.shape

        # Shift per ampiezza positiva
        X_positive = X + 3.0
        spike_prob = np.clip((X_positive * self.gain) / 100.0, 0.0, 1.0)

        # Lancia dado per ogni timestep
        spikes = np.zeros((n_samples, self.nb_steps, n_features), dtype=np.float32)
        for t in range(self.nb_steps):
            spikes[:, t, :] = (np.random.rand(n_samples, n_features) < spike_prob).astype(np.float32)

        # Replica per popolazione
        if self.population_size > 1:
            spikes = np.repeat(spikes, self.population_size, axis=2)

        return torch.tensor(spikes, dtype=torch.float32)

    def encode_hz(self, X: np.ndarray) -> torch.Tensor:
        """
        Deterministic Hz-based rate encoding (con jitter).
        
        Algoritmo:
        1. X_pos = X + 3.0
        2. hz = clip(X_pos × gain, 1, 100)
        3. interval = periodo in timestep
        4. Spike ogni interval timestep + jitter ±3
        5. Replica per population_size
        
        Args:
            X: (n_samples, n_features)
            
        Returns:
            Spike tensor (n_samples, nb_steps, n_features * population_size)
        """
        n_samples, n_features = X.shape

        # Shift → Hz
        X_positive = X + 3.0
        spike_hz = np.clip(X_positive * self.gain, 1.0, 100.0)

        # Hz → intervallo timestep
        spike_intervals = np.round((1000.0 / spike_hz) / self.dt).astype(int)

        spikes = np.zeros((n_samples, self.nb_steps, n_features), dtype=np.float32)

        # Posiziona spike agli intervalli
        for i in range(n_samples):
            for j in range(n_features):
                interval = spike_intervals[i, j] + np.random.randint(-3, 4)
                interval = max(interval, 1)
                spikes[i, interval::interval, j] = 1.0

        # Replica per popolazione
        if self.population_size > 1:
            spikes = np.repeat(spikes, self.population_size, axis=2)

        return torch.tensor(spikes, dtype=torch.float32)
