"""
learning_eprop.py - E-prop Algorithm (Eligibility Propagation)

Forward pass con RecurrentLayer + FeedforwardLayer.
Backward pass con tracce di eleggibilità locali (no BPTT).

"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from src.snn_layer_model import FeedforwardLayer, RecurrentLayer


def grads_batch(
    x_seq: torch.Tensor,
    y_out: torch.Tensor,
    y_true: torch.Tensor,
    v_seq: torch.Tensor,
    z_seq: torch.Tensor,
    w_in: torch.Tensor,
    w_rec: torch.Tensor,
    w_out: torch.Tensor,
    gamma: float,
    threshold: float,
    beta_trace: float,
    beta_trace_out: float,
    device: torch.device
) -> None:
    """Accumula gradienti E-prop sui pesi (w_in, w_rec, w_out) modificandoli in-place."""
    
    # Inizializza gradienti se assenti
    if w_in.grad is None:
        w_in.grad = torch.zeros_like(w_in)
    if w_rec.grad is None:
        w_rec.grad = torch.zeros_like(w_rec)
    if w_out.grad is None:
        w_out.grad = torch.zeros_like(w_out)

    nb_steps, _, nb_inputs = x_seq.shape
    nb_hidden = v_seq.shape[2]
    nb_outputs = w_out.shape[0]

    # ── SURROGATE GRADIENT ──
    h = gamma * torch.max(torch.zeros_like(v_seq), 1 - torch.abs((v_seq - threshold) / threshold))

    # ── ERROR & LEARNING SIGNAL ──
    target_one_hot = F.one_hot(y_true, num_classes=nb_outputs).float().to(device)
    err = y_out - target_one_hot
    err_time = err.unsqueeze(0).expand(nb_steps, -1, -1)
    L = torch.einsum("tbo,or->brt", err_time, w_out)

    # ── EXPONENTIAL DECAY FILTERS ──
    beta_conv = torch.tensor(
        [beta_trace_out ** (nb_steps - i - 1) for i in range(nb_steps)]
    ).float().view(1, 1, -1).to(device)
    
    beta_rec_conv = torch.tensor(
        [beta_trace ** (nb_steps - i - 1) for i in range(nb_steps)]
    ).float().view(1, 1, -1).to(device)

    # ── INPUT TRACES: x_t * h_t * exponential_decay ──
    input_perm = x_seq.permute(1, 2, 0)
    trace_in = F.conv1d(
        input_perm,
        beta_rec_conv.expand(nb_inputs, -1, -1),
        padding=nb_steps,
        groups=nb_inputs,
    )[:, :, 1 : nb_steps + 1]
    trace_in = trace_in.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    h_perm = h.permute(1, 2, 0)
    trace_in = torch.einsum("brt,brit->brit", h_perm, trace_in)

    # ── RECURRENT TRACES: z_t * h_t * exponential_decay ──
    z_perm = z_seq.permute(1, 2, 0)
    trace_rec = F.conv1d(
        z_perm,
        beta_rec_conv.expand(nb_hidden, -1, -1),
        padding=nb_steps,
        groups=nb_hidden,
    )[:, :, :nb_steps]
    trace_rec = trace_rec.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    trace_rec = torch.einsum("brt,brit->brit", h_perm, trace_rec)

    # ── OUTPUT TRACES: z_t * exponential_decay ──
    trace_out = F.conv1d(
        z_perm, 
        beta_conv.expand(nb_hidden, -1, -1), 
        padding=nb_steps, 
        groups=nb_hidden
    )[:, :, 1 : nb_steps + 1]

    # ── ACCUMULATE GRADIENTS ──
    w_in.grad += torch.sum(L.unsqueeze(2) * trace_in, dim=(0, 3))
    w_rec.grad += torch.sum(L.unsqueeze(2) * trace_rec, dim=(0, 3))
    w_out.grad += torch.einsum("tbo,brt->or", err_time, trace_out)


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
    Forward + backward (E-prop) pass.
    
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

    # ── E-PROP BACKWARD (Training) ──
    if trainable and yt is not None:
        y_out_rate = n_spike * tau_mem_ms / max_time_ms
        
        grads_batch(
            x_seq=inputs.permute(1, 0, 2),
            y_out=y_out_rate,
            y_true=yt,
            v_seq=mem_rec_hidden.permute(1, 0, 2),
            z_seq=spk_rec_hidden.permute(1, 0, 2),
            w_in=w_in,
            w_rec=w_rec,
            w_out=w_out,
            gamma=gamma,
            threshold=threshold,
            beta_trace=decay["beta_trace"],
            beta_trace_out=decay["beta_trace_out"],
            device=device
        )

    return spk_rec_readout, other_recs, layers
