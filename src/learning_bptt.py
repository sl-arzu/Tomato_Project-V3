"""
learning_bptt.py - BPTT Algorithm (Backpropagation Through Time)

Forward pass con RecurrentLayer + FeedforwardLayer.
Backward pass con PyTorch autograd (BPTT standard).

"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from src.snn_layer_model import FeedforwardLayer, RecurrentLayer


def run_snn(
    inputs: torch.Tensor,
    layers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    decay: Dict[str, float],
    nb_steps: int,
    nb_hidden: int,
    nb_outputs: int,
    device: torch.device,
    lower_bound: Optional[float] = None,
    ref_per_timesteps: int = 1,
    tau_mem_ms: float = 60.0,
    max_time_ms: float = 3000.0,
    gamma: float = 0.3,
    threshold: float = 1.0,
    trainable: bool = False,
    yt: Optional[torch.Tensor] = None,
    ref_counter_hidden: Optional[torch.Tensor] = None,
    ref_counter_readout: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Forward + backward (BPTT) pass.
    
    Returns: (spk_output, [mem_hidden, spk_hidden, mem_output], layers)
    """
    w_in, w_out, w_rec = layers
    bs = inputs.shape[0]

    # ── LAYER 1: INPUT → HIDDEN (Recurrent) ──
    h1 = torch.einsum("bti,hi->bth", inputs, w_in)
    
    if ref_per_timesteps > 0 and ref_counter_hidden is not None:
        spk_rec_hidden, mem_rec_hidden = RecurrentLayer.compute_activity(
            batch_size=bs, nb_neurons=nb_hidden, input_activity=h1, rec_weights=w_rec,
            nb_steps=nb_steps, alpha=decay["alpha"], beta_rec=decay["beta_rec"],
            device=device, lower_bound=lower_bound,
            ref_per_counter=ref_counter_hidden, ref_per_timesteps=ref_per_timesteps
        )
    else:
        spk_rec_hidden, mem_rec_hidden = RecurrentLayer.compute_activity(
            batch_size=bs, nb_neurons=nb_hidden, input_activity=h1, rec_weights=w_rec,
            nb_steps=nb_steps, alpha=decay["alpha"], beta_rec=decay["beta_rec"],
            device=device, lower_bound=lower_bound
        )

    # ── LAYER 2: HIDDEN → OUTPUT (Feedforward) ──
    h2 = torch.einsum("bth,oh->bto", spk_rec_hidden, w_out)
    
    if ref_per_timesteps > 0 and ref_counter_readout is not None:
        spk_rec_readout, mem_rec_readout, n_spike = FeedforwardLayer.compute_activity(
            batch_size=bs, nb_neurons=nb_outputs, input_activity=h2,
            nb_steps=nb_steps, alpha=decay["alpha"], beta=decay["beta"],
            device=device, lower_bound=lower_bound,
            ref_per_counter=ref_counter_readout, ref_per_timesteps=ref_per_timesteps
        )
    else:
        spk_rec_readout, mem_rec_readout, n_spike = FeedforwardLayer.compute_activity(
            batch_size=bs, nb_neurons=nb_outputs, input_activity=h2,
            nb_steps=nb_steps, alpha=decay["alpha"], beta=decay["beta"],
            device=device, lower_bound=lower_bound
        )

    other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout]

    # ── BPTT BACKWARD (PyTorch Autograd) ──
    if trainable and yt is not None:
        y_out_rate = n_spike * tau_mem_ms / max_time_ms

        # Binary cross-entropy loss (spike_rate vs target)
        y_out_rate_clamped = torch.clamp(y_out_rate, min=1e-7, max=1.0)
        target_one_hot = F.one_hot(yt, num_classes=nb_outputs).float().to(device)
        loss = F.binary_cross_entropy(y_out_rate_clamped, target_one_hot, reduction='mean')

        # Zero gradients before backward
        if w_in.grad is not None:
            w_in.grad.zero_()
        if w_rec.grad is not None:
            w_rec.grad.zero_()
        if w_out.grad is not None:
            w_out.grad.zero_()

        # Autograd computes gradients through entire computation graph
        loss.backward()

    return spk_rec_readout, other_recs, layers
