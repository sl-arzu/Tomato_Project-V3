"""
learning_trainer.py - SNN Training Orchestrator

Training loop con support per algoritmi intercambiabili (E-prop, BPTT).
Gestisce optimizer, loss, accuracy, weight history.

Autore: Shanti Leonardo Arzu | Data: Aprile 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Any, List, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.learning_eprop import run_snn as run_snn_eprop
from src.learning_bptt import run_snn as run_snn_bptt


class SNNTrainer:
    """Orchestrates training with selectable algorithm (E-prop or BPTT)."""
    
    def __init__(
        self, 
        layers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        nb_outputs: int,
        tau_mem_ms: float,
        max_time_ms: float,
        lr: float = 0.0015,
        algorithm: str = "eprop",
        scheduler_patience: int = 5,      # Epoche prima di ridurre LR
        scheduler_factor: float = 0.5,    # Fattore di riduzione
        scheduler_min_lr: float = 1e-6   # LR minimo
    ):
        self.layers = layers
        self.device = device
        self.nb_outputs = nb_outputs
        self.tau_mem_ms = tau_mem_ms
        self.max_time_ms = max_time_ms
        self.algorithm = algorithm.lower()
        
        # ── ALGORITHM DISPATCHER ──
        self.run_snn = (
            run_snn_eprop if self.algorithm == "eprop" 
            else run_snn_bptt if self.algorithm == "bptt" 
            else run_snn_eprop
        )
        
        self.optimizer = torch.optim.Adamax(self.layers, lr=lr, betas=(0.9, 0.995))
        # Scheduler per riduzione adattativa del learning rate
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',              # Monitora loss (minimo è meglio)
            factor=scheduler_factor, # Es: 0.5 = riduce LR del 50%
            patience=scheduler_patience,  # Epoche senza miglioramento
            threshold=1e-4,          # Cambiamento minimo considerato miglioramento
            cooldown=1,              # Epoche da aspettare dopo una riduzione
            min_lr=scheduler_min_lr,  # LR non scende sotto questo valore
        )
    
        self.history = {
            "loss": [],
            "train_acc": [],
            "test_acc": []
        }
        self.weight_history = {
            "w_in": [],
            "w_out": [],
            "w_rec": []
        }

    def evaluate(self, test_loader: DataLoader, run_snn_kwargs: Dict[str, Any]) -> float:
        """Compute accuracy on test/validation set (optional, can be None)."""
        accs = []
        
        for x_local, y_local in test_loader:
            x_local = x_local.to(self.device)
            y_local = y_local.to(self.device)
            bs = x_local.shape[0]
            
            kwargs = run_snn_kwargs.copy()
            if kwargs.get("ref_per_timesteps", 0) > 0:
                kwargs["ref_counter_hidden"] = torch.zeros((bs, kwargs["nb_hidden"]), device=self.device)
                kwargs["ref_counter_readout"] = torch.zeros((bs, self.nb_outputs), device=self.device)
            
            with torch.no_grad():
                spk_rec_readout, _, _ = self.run_snn(
                    inputs=x_local, layers=self.layers, trainable=False, **kwargs
                )
                n_spikes = torch.sum(spk_rec_readout, dim=1)
                _, am = torch.max(n_spikes, 1)
                acc = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(acc)
        
        return float(np.mean(accs)) if accs else 0.0


    def fit(self, train_loader: DataLoader, epochs: int, run_snn_kwargs: Dict[str, Any], test_loader: Optional[DataLoader] = None) -> Tuple[Dict[str, List[float]], Dict[str, List[np.ndarray]]]:
        """
        Training loop. test_loader is optional:
        - Pass test_loader for train/val/test evaluation
        - Leave None for train/test 70/30 split only
        """
        pbar = tqdm(range(epochs), desc="Training")
        
        for epoch in pbar:
            local_loss = []
            accs = []
            
            for x_local, y_local in train_loader:
                x_local = x_local.to(self.device)
                y_local = y_local.to(self.device)
                bs = x_local.shape[0]
                
                self.optimizer.zero_grad()
                
                kwargs = run_snn_kwargs.copy()
                if kwargs.get("ref_per_timesteps", 0) > 0:
                    kwargs["ref_counter_hidden"] = torch.zeros((bs, kwargs["nb_hidden"]), device=self.device)
                    kwargs["ref_counter_readout"] = torch.zeros((bs, self.nb_outputs), device=self.device)
                
                spk_rec_readout, _, self.layers = self.run_snn(
                    inputs=x_local, layers=self.layers, trainable=True, yt=y_local, **kwargs
                )
                
                out_rate = torch.sum(spk_rec_readout, dim=1) * self.tau_mem_ms / self.max_time_ms
                target_one_hot = F.one_hot(y_local, num_classes=self.nb_outputs).float()
                loss_val = torch.mean((out_rate - target_one_hot) ** 2)
                local_loss.append(loss_val.item())
                
                _, am = torch.max(out_rate, 1)
                acc_batch = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(acc_batch)
                
                self.optimizer.step()
            
            with torch.no_grad():
                self.weight_history["w_in"].append(self.layers[0].detach().cpu().numpy().copy())
                self.weight_history["w_out"].append(self.layers[1].detach().cpu().numpy().copy())
                self.weight_history["w_rec"].append(self.layers[2].detach().cpu().numpy().copy())
            
            mean_loss = np.mean(local_loss)
            train_acc = np.mean(accs)
            self.history["loss"].append(mean_loss)
            self.history["train_acc"].append(train_acc)
            
            test_acc = 0.0
            if test_loader is not None:
                test_acc = self.evaluate(test_loader, run_snn_kwargs)
                self.history["test_acc"].append(test_acc)
                
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} | Loss: {mean_loss:.4f} | Train: {train_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%"
            )
            # ─── AGGIORNA SCHEDULER ───
            # Monitora il loss per decidere se ridurre il LR
            self.scheduler.step(mean_loss)
            
            # Opzionale: mostra il LR corrente
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} | Loss: {mean_loss:.4f} | LR: {current_lr:.2e} | Train: {train_acc * 100:.2f}% | Test: {test_acc * 100:.2f}%"
            )

            
        return self.history, self.weight_history