"""Memory-lean multi-step LIF with the exact SpikingJelly LIFNode semantics.

Supported configuration (the one used by every LIF in the Spike-DINO ViT and
projection head): decay_input=True, hard reset to v_reset=0, detach_reset=True,
ATan surrogate, multi-step input [T, ...], membrane state reset to zero before
every independent forward. The forward is bit-identical to
``neuron.LIFNode(..., backend='torch', step_mode='m')``; the backward is the
same BPTT expressed analytically, so autograd only stores the T charged
membrane tensors instead of every intermediate of the Python loop.
"""

import math
from typing import List, Tuple

import torch
from torch import nn
from spikingjelly.activation_based import neuron, surrogate

# TorchScript versions fuse each time step's elementwise chain into fewer
# kernels; the eager versions are the readable reference. Equality between
# them (and with SpikingJelly) is checked in tests and on the server GPU.
USE_SCRIPT = True


def _forward_eager(x_seq: torch.Tensor, tau: float, v_threshold: float
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.zeros_like(x_seq[0])
    spikes: List[torch.Tensor] = []
    charged: List[torch.Tensor] = []
    for step in range(x_seq.shape[0]):
        h = v + (x_seq[step] - v) / tau
        spike = (h - v_threshold >= 0).to(x_seq.dtype)
        v = (1. - spike) * h
        spikes.append(spike)
        charged.append(h)
    return torch.stack(spikes), torch.stack(charged)


def _backward_eager(grad_spikes: torch.Tensor, charged: torch.Tensor, tau: float,
                    v_threshold: float, alpha: float) -> torch.Tensor:
    grad_x = torch.empty_like(charged)
    grad_v = torch.zeros_like(charged[0])
    scale = math.pi / 2 * alpha
    for step in range(charged.shape[0] - 1, -1, -1):
        h = charged[step]
        shifted = h - v_threshold
        spike = (shifted >= 0).to(h.dtype)
        surrogate_grad = alpha / 2 / (1 + (scale * shifted).pow(2))
        grad_h = grad_spikes[step] * surrogate_grad + grad_v * (1. - spike)
        grad_x[step] = grad_h / tau
        grad_v = grad_h * (1. - 1. / tau)
    return grad_x


_forward_script = torch.jit.script(_forward_eager)
_backward_script = torch.jit.script(_backward_eager)


class _FusedLIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, tau, v_threshold, alpha):
        spikes, charged = (_forward_script if USE_SCRIPT else _forward_eager)(
            x_seq, tau, v_threshold)
        ctx.save_for_backward(charged)
        ctx.tau, ctx.v_threshold, ctx.alpha = tau, v_threshold, alpha
        return spikes

    @staticmethod
    def backward(ctx, grad_spikes):
        (charged,) = ctx.saved_tensors
        grad_x = (_backward_script if USE_SCRIPT else _backward_eager)(
            grad_spikes.contiguous(), charged, ctx.tau, ctx.v_threshold, ctx.alpha)
        return grad_x, None, None, None


class FusedLIF(nn.Module):
    """Drop-in replacement for the supported SpikingJelly LIF configuration."""

    def __init__(self, tau=2., v_threshold=1., alpha=2.):
        super().__init__()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.alpha = float(alpha)

    def forward(self, x_seq):
        return _FusedLIFFunction.apply(x_seq, self.tau, self.v_threshold,
                                       self.alpha)

    # Deliberately no reset(): the neuron is stateless across calls (every
    # forward starts from v=0), so functional.reset_net has nothing to clear.

    def extra_repr(self):
        return (f'tau={self.tau}, v_threshold={self.v_threshold}, '
                f'alpha={self.alpha}, reset=hard0-detached')


def _supported(module):
    return (isinstance(module, neuron.LIFNode) and module.step_mode == 'm'
            and module.decay_input and module.v_reset == 0.
            and module.detach_reset
            and isinstance(module.surrogate_function, surrogate.ATan)
            and module.surrogate_function.spiking)


def replace_lif(model):
    """Swap every supported LIFNode for FusedLIF; refuse anything else."""
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, neuron.BaseNode):
                if not _supported(child):
                    raise ValueError(f'Unsupported spiking neuron {name}: {child}')
                setattr(parent, name, FusedLIF(
                    child.tau, child.v_threshold,
                    child.surrogate_function.alpha))
                replaced += 1
    return replaced
