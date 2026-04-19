"""
plant_feature_selector.py
Feature selection and normalization utilities for tomato stress datasets.
"""

from __future__ import annotations
import numpy as np

class PlantFeatureSelector:
    """Seleziona le frequenze specifiche di bioimpedenza e applica la normalizzazione."""

    # Mappatura delle frequenze ottimali in base allo stress (Elastic Net)
    _FREQ_BY_STRESS = {
        "water": [0, 1, 2],         # 100-120 Hz
        "iron": [197, 198, 199]     # 4.7-10 MHz
    }

    def __init__(self, stress_type="water", custom_freq_indices=None):
        stress_key = str(stress_type).lower()
        self.stress_type = stress_key
        
        # Inizializza gli indici delle frequenze (custom o da default)
        if custom_freq_indices is not None:
            self.freq_indices = [int(i) for i in custom_freq_indices]
        elif stress_key in self._FREQ_BY_STRESS:
            self.freq_indices = list(self._FREQ_BY_STRESS[stress_key])
        else:
            raise ValueError(f"stress_type '{stress_type}' non supportato. Usa: {list(self._FREQ_BY_STRESS.keys())}")

        # x2 perché ogni frequenza ha due componenti: Parte Reale e Parte Immaginaria
        self.nb_features = len(self.freq_indices) * 2

    def _build_feature_indices(self, n_raw_features):
        """
        Mappa le frequenze scelte sulle colonne del dataset flat (es. 400 feature).
        Layout atteso del dataset flat: [Reale(0..N), Immaginaria(0..N)]
        """
        if n_raw_features % 2 != 0:
            raise ValueError(f"Il numero di feature grezze deve essere pari (Reale+Immaginaria), ricevuto: {n_raw_features}.")

        n_freqs = n_raw_features // 2
        
        # Controllo validità degli indici scelti
        invalid = [f for f in self.freq_indices if f < 0 or f >= n_freqs]
        if invalid:
            raise ValueError(f"Indici frequenza fuori range (max {n_freqs-1}): {invalid}")

        # Concatena gli indici della parte reale e quelli della parte immaginaria (shiftati di +n_freqs)
        real_cols = self.freq_indices
        imag_cols = [f + n_freqs for f in self.freq_indices]
        return real_cols + imag_cols

    def select_features(self, X):
        """Estrae le feature specifiche dallo spazio vettoriale della bioimpedenza."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"L'input deve essere 2D (n_samples, n_features), ricevuto: {X.shape}")

        # Se il dataset è già ridotto, ignora l'estrazione
        if X.shape[1] == self.nb_features:
            return X

        # Calcola le colonne da estrarre ed esegue lo slicing
        selected_cols = self._build_feature_indices(X.shape[1])
        return X[:, selected_cols]

    def normalize_features(self, X_train, X_test, X_val=None):
        """
        Normalizzazione Z-Score. 
        Calcola media e varianza SOLO sul Training set per prevenire il Data Leakage.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        # Calcolo statistiche sul Train
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        
        # Sostituisce eventuali std=0 con 1.0 per evitare warning/errori di divisione per zero
        std_safe = np.where(std == 0.0, 1.0, std)

        # Normalizzazione Train e Test
        X_train_n = (X_train - mean) / std_safe
        X_test_n = (X_test - mean) / std_safe

        norm_params = {
            "method": "zscore",
            "mean": mean,
            "std": std_safe,
            "features_selected": self.freq_indices,
            "selected_columns": self._build_feature_indices(X_train.shape[1]),
        }

        # Normalizza e restituisce X_val se presente (70/15/15 split)
        if X_val is not None:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            X_val_n = (X_val_arr - mean) / std_safe
            return X_train_n, X_val_n, X_test_n, norm_params
            
        # Altrimenti restituisce solo Train/Test (70/30 split o LOPO)
        return X_train_n, X_test_n, norm_params
    