"""
learning_evaluator.py - Valutazione del Modello

Inferenza finale sul test set: raccoglie predizioni and attività neurale.
Output: metriche di classificazione + spike rasters per visualizzazione.

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List

from src.learning_eprop import run_snn


class ModelEvaluator:
    """Valuta il modello SNN sul test set e estrae attività neurale."""
    
    def __init__(
        self, 
        layers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        nb_outputs: int
    ):
        self.layers = layers
        self.device = device
        self.nb_outputs = nb_outputs

    def collect_predictions_and_activity(
        self, 
        test_loader: DataLoader, 
        run_snn_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Passa sull'intero test set e raccoglie:
        1. Etichette vere (y_true)
        2. Etichette predette (y_pred) → per confusion matrix
        3. Spike input layer
        4. Spike hidden layer
        5. Spike output layer
        
        Ritorna arrays NumPy per visualizzazione e metriche.
        """
        y_true_list = []
        y_pred_list = []
        
        spk_input_list = []
        spk_hidden_list = []
        spk_readout_list = []
        
        for x_local, y_local in test_loader:
            x_local = x_local.to(self.device)
            bs = x_local.shape[0]
            
            # Copia kwargs e inizializza contatori refrattari se necessario
            kwargs = run_snn_kwargs.copy() # kwargs è un dizionario con parametri per run_snn
            if kwargs.get("ref_per_timesteps", 0) > 0:
                kwargs["ref_counter_hidden"] = torch.zeros((bs, kwargs["nb_hidden"]), device=self.device)
                kwargs["ref_counter_readout"] = torch.zeros((bs, self.nb_outputs), device=self.device)
                
            # Inferenza pura (nessun calcolo di gradienti)
            with torch.no_grad():
                spk_rec_readout, other_recs, _ = run_snn(
                    inputs=x_local, layers=self.layers, trainable=False, yt=None, **kwargs
                )
                
                # other_recs = [mem_hidden, spk_hidden, mem_output]
                spk_rec_hidden = other_recs[1]
                
                # Predizione: neurone di output con più spike
                n_spikes = torch.sum(spk_rec_readout, dim=1)
                _, am = torch.max(n_spikes, 1)
                
                # Accumula predizioni per metriche
                y_true_list.extend(y_local.cpu().numpy())
                y_pred_list.extend(am.cpu().numpy())
                
                # Accumula attività neurale per raster plots
                spk_input_list.append(x_local.cpu().numpy())
                spk_hidden_list.append(spk_rec_hidden.cpu().numpy())
                spk_readout_list.append(spk_rec_readout.cpu().numpy())
                
        # Converte liste in arrays
        y_true_arr = np.array(y_true_list)
        y_pred_arr = np.array(y_pred_list)
        
        return y_true_arr, y_pred_arr, spk_input_list, spk_hidden_list, spk_readout_list
