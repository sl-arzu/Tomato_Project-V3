"""
plant_data_management.py
Gestione del caricamento, preprocessing e splitting dei dataset per lo stress delle piante.
"""
"""
plant_data_management.py
Data loading, preprocessing, and splitting for plant stress datasets
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from src.data_processing_plant_feature_selector import PlantFeatureSelector
from src.temp_enco_dispatcher import TemporalEncoder

class PlantDataManager:
    """Gestisce caricamento, preprocessing, normalizzazione e encoding dei dataset."""

    def __init__(self, stress_type="water", encoding_params=None):
        self.stress_type = stress_type
        self.feature_selector = PlantFeatureSelector(stress_type)

        # Default SNN encoding parameters
        if encoding_params is None:
            encoding_params = {"encoding_type": "rate", "nb_steps": 100, "dt": 1.0, "gain": 10.0}

        self.temporal_encoder = TemporalEncoder(**encoding_params)

    def load_npz_data(self, file_path):
        """Carica Features (X), Labels (y) e Plant IDs da un file .npz."""
        data = np.load(file_path, allow_pickle=True)
        return data["X"], data["y"], data["plant_ids"]

    def _apply_encoding(self, *datasets):
        """
        Appiattisce e applica il temporal encoding ai dataset passati.
        Args: *datasets: tuple di array numpy (es: X_train, X_val, X_test)
        """
        encoded_datasets = []
        for X in datasets:
            X_flat = X.reshape(X.shape[0], -1)
            X_encoded = self.temporal_encoder.encode(X_flat)
            encoded_datasets.append(X_encoded)
        return tuple(encoded_datasets)

    def prepare_dataset_standard_split(self, file_path, test_size=0.15, val_size=0.15):
        """
        Prepara dataset con split standard randomico. 
        Se val_size=0.0 esegue solo Train/Test (es. 70/30). Altrimenti Train/Val/Test (es. 70/15/15).
        """
        X, y, _ = self.load_npz_data(file_path)
        X_selected = self.feature_selector.select_features(X)

        # Split: isola subito il Test set
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X_selected, y, test_size=test_size, stratify=y, random_state=42
        )

        if val_size > 0.0:
            # Calcola la proporzione reale per il validation dal set rimanente
            val_fraction = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_temp, y_train_temp, test_size=val_fraction, stratify=y_train_temp, random_state=42
            )
            # Usa il nuovo normalize_features (gestisce 3 input)
            X_train_n, X_val_n, X_test_n, norm_params = self.feature_selector.normalize_features(X_train, X_test, X_val) # type: ignore
            
            # Temporal Encoding
            X_train_enc, X_val_enc, X_test_enc = self._apply_encoding(X_train_n, X_val_n, X_test_n)

            # PyTorch Datasets
            ds_train = TensorDataset(X_train_enc, torch.tensor(y_train, dtype=torch.long))
            ds_val = TensorDataset(X_val_enc, torch.tensor(y_val, dtype=torch.long))
            ds_test = TensorDataset(X_test_enc, torch.tensor(y_test, dtype=torch.long))
            
            split_sizes = {"train": 1.0 - test_size - val_size, "val": val_size, "test": test_size}
            return ds_train, ds_val, ds_test, self._build_metadata(norm_params, split_sizes, "standard_with_val")
        
        else:
            # Solo Train e Test (es. 70/30)
            X_train_n, X_test_n, norm_params = self.feature_selector.normalize_features(X_train_temp, X_test) # type: ignore
            X_train_enc, X_test_enc = self._apply_encoding(X_train_n, X_test_n)

            ds_train = TensorDataset(X_train_enc, torch.tensor(y_train_temp, dtype=torch.long))
            ds_test = TensorDataset(X_test_enc, torch.tensor(y_test, dtype=torch.long))
            
            split_sizes = {"train": 1.0 - test_size, "test": test_size}
            return ds_train, ds_test, self._build_metadata(norm_params, split_sizes, "standard_train_test")

    def prepare_dataset_leave_one_plant_split(self, file_path, leave_plant, val_size=0.5):
        """
        Split LOPO: train su tutte le piante, escluso 'leave_plant'.
        Se val_size > 0.0 la pianta esclusa è divisa in Val/Test. Se val_size=0.0 è tutta Test.
        """
        X, y, plant_ids = self.load_npz_data(file_path)
        X_selected = self.feature_selector.select_features(X)

        # Isola train (tutte le piante tranne una) e test (solo la pianta target)
        train_mask = plant_ids != leave_plant
        test_mask = plant_ids == leave_plant

        X_train, y_train = X_selected[train_mask], y[train_mask]
        X_test_full, y_test_full = X_selected[test_mask], y[test_mask]

        if val_size > 0.0:
            X_val, X_test, y_val, y_test = train_test_split(
                X_test_full, y_test_full, test_size=(1.0 - val_size), stratify=y_test_full, random_state=42
            )
            X_train_n, X_val_n, X_test_n, norm_params = self.feature_selector.normalize_features(X_train, X_test, X_val) # type: ignore
            X_train_enc, X_val_enc, X_test_enc = self._apply_encoding(X_train_n, X_val_n, X_test_n)
            
            ds_train = TensorDataset(X_train_enc, torch.tensor(y_train, dtype=torch.long))
            ds_val = TensorDataset(X_val_enc, torch.tensor(y_val, dtype=torch.long))
            ds_test = TensorDataset(X_test_enc, torch.tensor(y_test, dtype=torch.long))
            
            return ds_train, ds_val, ds_test, self._build_metadata(norm_params, "lopo_val_test", leave_plant)
        
        else:
            X_train_n, X_test_n, norm_params = self.feature_selector.normalize_features(X_train, X_test_full) # type: ignore
            X_train_enc, X_test_enc = self._apply_encoding(X_train_n, X_test_n)

            ds_train = TensorDataset(X_train_enc, torch.tensor(y_train, dtype=torch.long))
            ds_test = TensorDataset(X_test_enc, torch.tensor(y_test_full, dtype=torch.long))
            
            return ds_train, ds_test, self._build_metadata(norm_params, "lopo_train_test", leave_plant)

    def _build_metadata(self, norm_params, split_strategy, extra_info=None):
        """Helper interno per generare i metadati uniformati senza duplicare codice."""
        return {
            "nb_inputs": self.feature_selector.nb_features,
            "nb_outputs": 3,
            "nb_steps": self.temporal_encoder.nb_steps,
            "dt": self.temporal_encoder.dt,
            "stress_type": self.stress_type,
            "normalization_params": norm_params,
            "split_strategy": split_strategy,
            "extra": extra_info
        }