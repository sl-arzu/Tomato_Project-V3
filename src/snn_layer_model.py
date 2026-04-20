"""
snn_model.py
Definizione dei layer per la Spiking Neural Network (SNN).

Questo modulo contiene l'implementazione matematica dei neuroni LIF 
(Leaky Integrate-and-Fire) utilizzati per costruire la rete neurale.
Include sia layer feedforward (per l'output) sia layer ricorrenti (per l'hidden).

Autore: Shanti Leonardo Arzu (Adattamento modulare)
Data: Aprile 2026
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from src.snn_gradient_surrogate import spike_function


def compute_decay_factors(dt: float, tau_mem: float, tau_mem_rec: float, 
                          tau_syn: float, tau_trace: float, tau_trace_out: float, 
                          no_synapse: bool = False) -> Dict[str, float]:
    """
    Calcola i fattori di decadimento esponenziale (alpha, beta) a partire dalle 
    costanti di tempo (tau) in millisecondi.
    
    Formula base discretizzata: V[t] = V[t-1] * exp(-dt/tau)
    """
    decay_factors = {
        "alpha": float(np.exp(-dt / tau_syn)) if not no_synapse else 0.0,
        "beta": float(np.exp(-dt / tau_mem)),
        "beta_rec": float(np.exp(-dt / tau_mem_rec)),
        "beta_trace": float(np.exp(-dt / tau_trace)),
        "beta_trace_out": float(np.exp(-dt / tau_trace_out)),
    }
    return decay_factors


def update_refractory_period_counter(spk: torch.Tensor, counter: torch.Tensor, ref_per_timesteps: int) -> torch.Tensor:
    """
    Aggiorna il contatore del periodo refrattario per i neuroni.
    - Se il contatore è > 0, diminuisce di 1.
    - Se c'è stato uno spike (spk > 0), il contatore viene impostato a ref_per_timesteps.
    """
    bs_current = spk.shape[0]
    counter_view = counter[:bs_current, :]
    counter_view[counter_view > 0.0] -= 1
    counter_view[spk > 0.0] = ref_per_timesteps
    return counter_view


class FeedforwardLayer:
    """
    Layer Feedforward (Spiking).
    Riceve input (o hidden spikes) e produce spike in output senza ricorrenza.
    """
    
    @staticmethod
    def create_layer(nb_inputs: int, nb_outputs: int, scale: float, device: torch.device) -> torch.Tensor:
        """Inizializza i pesi del layer (distribuzione normale scalata)."""
        ff_layer = torch.empty(
            (nb_outputs, nb_inputs), device=device, dtype=torch.float32, requires_grad=True
        )
        torch.nn.init.normal_(ff_layer, mean=0.0, std=scale / np.sqrt(nb_inputs))
        return ff_layer

    @staticmethod
    def compute_activity(
        batch_size: int,
        nb_neurons: int,
        input_activity: torch.Tensor,
        nb_steps: int,
        alpha: float,
        beta: float,
        device: torch.device,
        lower_bound: Optional[float] = None,
        ref_per_counter: Optional[torch.Tensor] = None,
        ref_per_timesteps: int = 1,
        use_linear_decay: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola l'attività temporale (forward pass) del layer feedforward.
        
        Ritorna:
            spk_rec: (batch_size, nb_steps, nb_neurons) - Registro degli spike (0/1)
            mem_rec: (batch_size, nb_steps, nb_neurons) - Registro dei potenziali di membrana
            n_spike: (batch_size, nb_neurons) - Conteggio totale spike per ogni neurone
        """
        syn = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)
        mem = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)
        n_spike = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)

        mem_rec = []
        spk_rec = []

        for t in range(nb_steps):
            # Controllo soglia con spike function (differenziabile per BPTT)
            out = spike_function(mem, threshold=1.0)
            n_spike[out == 1] += 1
            
            # rst è il segnale di reset (sganciato dal gradiente)
            rst = out.detach()

            if ref_per_counter is not None:
                update_refractory_period_counter(rst, ref_per_counter, ref_per_timesteps)
                # mask: True se il neurone NON è in refrattario
                mask = ref_per_counter[:batch_size, :] == 0.0
                new_syn = alpha * syn
                new_syn[mask] = alpha * syn[mask] + input_activity[:, t][mask]
            else:
                new_syn = alpha * syn + input_activity[:, t]

            if use_linear_decay:
                new_mem = ((mem - torch.sign(mem) * beta) + syn) * (1.0 - rst)
            else:
                new_mem = (beta * mem + syn) * (1.0 - rst)

            if lower_bound is not None:
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)
            
            # Aggiorna stato per il prossimo timestep
            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return spk_rec, mem_rec, n_spike


class RecurrentLayer:
    """
    Layer Ricorrente (Spiking / RSNN).
    Combina segnale in ingresso (feedforward) con segnale ricorrente (dal layer stesso al t-1).
    """

    @staticmethod
    def create_layer(nb_inputs: int, nb_outputs: int, fwd_scale: float, rec_scale: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inizializza i pesi feedforward e i pesi ricorrenti."""
        ff_layer = torch.empty(
            (nb_outputs, nb_inputs), device=device, dtype=torch.float32, requires_grad=True
        )
        torch.nn.init.normal_(ff_layer, mean=0.0, std=fwd_scale / np.sqrt(nb_inputs))
        
        rec_layer = torch.empty(
            (nb_outputs, nb_outputs), device=device, dtype=torch.float32, requires_grad=True
        )
        torch.nn.init.normal_(rec_layer, mean=0.0, std=rec_scale / np.sqrt(nb_inputs))
        
        return ff_layer, rec_layer

    @staticmethod
    def compute_activity(
        batch_size: int,
        nb_neurons: int,
        input_activity: torch.Tensor,
        rec_weights: torch.Tensor,
        nb_steps: int,
        alpha: float,
        beta_rec: float,
        device: torch.device,
        lower_bound: Optional[float] = None,
        ref_per_counter: Optional[torch.Tensor] = None,
        ref_per_timesteps: int = 1,
        use_linear_decay: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcola l'attività temporale (forward pass) del layer ricorrente.
        
        Ritorna:
            spk_rec: (batch_size, nb_steps, nb_neurons)
            mem_rec: (batch_size, nb_steps, nb_neurons)
        """
        syn = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)
        mem = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)
        out = torch.zeros((batch_size, nb_neurons), device=device, dtype=torch.float32)

        mem_rec = []
        spk_rec = []

        for t in range(nb_steps):
            # L'input totale include la componente feedforward + la ricorrente (moltiplicazione matriciale out @ W_rec.T)
            h1 = input_activity[:, t] + torch.einsum("ab,bc->ac", (out, rec_weights.t()))
            
            # Controllo soglia con spike function (differenziabile per BPTT)
            out = spike_function(mem, threshold=1.0)
            rst = out.detach()

            if ref_per_counter is not None:
                update_refractory_period_counter(rst, ref_per_counter, ref_per_timesteps)
                mask = ref_per_counter[:batch_size, :] == 0.0
                new_syn = alpha * syn
                new_syn[mask] = alpha * syn[mask] + h1[mask]
            else:
                new_syn = alpha * syn + h1

            if use_linear_decay:
                new_mem = ((mem - torch.sign(mem) * beta_rec) + syn) * (1.0 - rst)
            else:
                new_mem = (beta_rec * mem + syn) * (1.0 - rst)

            if lower_bound is not None:
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)
            
            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        return spk_rec, mem_rec