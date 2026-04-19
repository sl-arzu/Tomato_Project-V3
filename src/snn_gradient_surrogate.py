"""
surrogate_gradient.py
Implementazione del gradiente surrogato per neuroni spiking

Problema: la funzione di spike è un step (0 → 1 alla soglia).
          La sua derivata è ZERO ovunque tranne alla soglia dove è INFINITA.

Soluzione: definiamo un'operazione custom con:
  FORWARD  → comportamento reale (step function, output 0 o 1)
  BACKWARD → derivata approssimata (triangolare, valori finiti)

Questo è lo standard in tutte le SNN moderne (Intel Loihi, UZH, IBM TrueNorth).

Riferimento:
  - Bellec et al. 2020: "A solution to the learning dilemma for
    recurrent networks of spiking neurons"
"""

from typing import cast
import torch
import torch.nn as nn


class SurrogateSpikeFunction(torch.autograd.Function):
    """
    Funzione di spike con gradiente surrogato triangolare.

    FORWARD:  spike = 1 se mem >= threshold, altrimenti 0
              (funzione step_function, output binario)

    BACKWARD: usa derivata triangolare invece di quella vera
              d_surrogate = max(0, 1 - |mem - threshold|)
              (piecewise linear, non differenziabile ma finita)

    Perché triangolare?
      - Veloce: solo operazioni elementari (no exp, no division)
      - Locale: gradiente ≠ 0 solo nell'intorno della soglia
      - Stabile: non esplode mai (output tra 0 e 1)
      - Equivalente al surrogate usato nel paper E-prop
        (Bellec et al. 2020, Eq. S1)
    """

    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        """
        Calcola lo spike durante il forward pass.

        Args:
            ctx:        contesto PyTorch per salvare dati riusabili nel backward
            mem:        potenziale di membrana (qualsiasi shape)
            threshold:  soglia di spike (default: 1.0)

        Returns:
            spike: tensore binario (0.0 o 1.0), stessa shape di mem
        """
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Calcola il gradiente surrogato durante il backward pass.

        Args:
            ctx:         contesto con i dati salvati nel forward
            grad_output: gradiente dai layer successivi

        Returns:
            grad_mem:       gradiente per mem
            grad_threshold: None (non è learnable)
        """
        mem, = ctx.saved_tensors
        threshold = ctx.threshold

        # Derivata triangolare (surrogate)
        # Visualizzazione:
        #   1.0 ─────────────/\──────────────
        #                   /  \
        #   0.0 ────────────    ────────────
        #           threshold-1  threshold+1
        surrogate_derivative = torch.clamp(
            1.0 - torch.abs(mem - threshold),
            min=0.0
        )

        grad_mem = grad_output * surrogate_derivative
        return grad_mem, None


def spike_function(mem: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """
    Interfaccia semplificata per SurrogateSpikeFunction.

    Args:
        mem:       potenziale di membrana corrente
        threshold: soglia di firing (default 1.0)

    Returns:
        spike: tensore float32 con valori 0.0 o 1.0
    """
    return cast(torch.Tensor, SurrogateSpikeFunction.apply(mem, threshold))
