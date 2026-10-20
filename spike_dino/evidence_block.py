"""Experimental observation-normalized spike block for event self-distillation.

This is a research prototype, not a validated DINOv3-equivalent architecture.
Learned synapses receive binary spikes. Coincidence accumulators, normalized
evidence, normalization, and residual membrane carriers remain real currents.
"""

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .spiking_vit import _lif


class EvidenceCurrent(NamedTuple):
    attended: torch.Tensor
    contrast: torch.Tensor
    coincidence_mass: torch.Tensor
    visible_key_count: torch.Tensor


def normalized_spike_evidence(q, k, v, observed):
    """Query-normalized positive-kernel attention across visible T×N keys.

    Q/K/V: [T,B,H,N,D], binary in the intended forward circuit.
    observed: boolean [T,B,N], built from the STUDENT'S OBSERVABLE input;
    hidden bins and non-event special tokens must be false. No batch mixing.
    KᵀV is an integer current accumulator, without intermediate thresholding.
    """
    if q.ndim != 5 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('Expected aligned [T,B,H,N,D] Q/K/V')
    t, b, _h, n, _d = q.shape
    if observed.dtype != torch.bool or observed.shape != (t, b, n):
        raise ValueError('observed must be boolean [T,B,N]')
    support = observed[:, :, None, :, None]
    # masked_fill, rather than multiplication, prevents even non-finite
    # unobserved values from leaking into the visible evidence accumulator.
    key = k.float().masked_fill(~support, 0)
    value = v.float().masked_fill(~support, 0)
    query = q.float()
    context = torch.einsum('tbhnc,tbhnd->bhcd', key, value)
    key_sum = key.sum((0, 3))
    mass = (query * key_sum[None, :, :, None]).sum(-1, keepdim=True)
    numerator = torch.einsum('tbhnc,bhcd->tbhnd', query, context)
    attended = numerator / mass.clamp_min(1)
    count = observed.sum((0, 2)).to(value.dtype)
    reference = value.sum((0, 3)) / count[:, None, None].clamp_min(1)
    contrast = (attended - reference[None, :, :, None]) * (mass > 0)
    return EvidenceCurrent(attended, contrast, mass, count)


def rms_current(current, eps=1e-6):
    """Within-token current scaling, never a cross-record normalization."""
    return current * torch.rsqrt(current.square().mean(-1, keepdim=True) + eps)


def energy_matched_residual(carrier, branch, depth):
    """Bound each residual update RMS by carrier RMS / sqrt(2*depth).

    Scales are detached for an explicit identity carrier derivative. This
    bounds forward update energy; it does not prove a stable deep Jacobian.
    """
    if depth < 1 or carrier.shape != branch.shape:
        raise ValueError('Positive depth and aligned carrier/branch required')
    source_rms = carrier.detach().square().mean(-1, keepdim=True).sqrt()
    branch_rms = branch.detach().square().mean(-1, keepdim=True).sqrt()
    gain = (source_rms / branch_rms.clamp_min(1e-6)).clamp_max(1)
    return carrier + gain * branch / math.sqrt(2 * depth)


class SpikeTokenProjection(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.linear = nn.Linear(source, target, bias=False)
        self.norm = nn.LayerNorm(target)
        self.spike = _lif()

    def forward(self, spikes):
        return self.spike(self.norm(self.linear(spikes)))


class EvidenceSpikeBlock(nn.Module):
    """Visible-evidence attention + bipolar spikes + bounded membrane writes.

    Selected by block_variant='evidence'; the standard backbone is unchanged.
    Its components require separate controls before claiming training benefits.
    """

    def __init__(self, dim=384, heads=6, depth=12, mlp_ratio=4,
                 encoding='channel_centered'):
        super().__init__()
        if dim % heads or depth < 1:
            raise ValueError('Width must divide heads; depth must be positive')
        if encoding not in ('channel_centered', 'reference_contrast'):
            raise ValueError('Unknown evidence encoding')
        self.dim, self.heads, self.depth = dim, heads, depth
        self.encoding = encoding
        self.input_norm = nn.LayerNorm(dim)
        self.input_spike = _lif()
        self.q = SpikeTokenProjection(dim, dim)
        self.k = SpikeTokenProjection(dim, dim)
        self.v = SpikeTokenProjection(dim, dim)
        self.evidence_on = _lif(threshold=.5)
        self.evidence_off = _lif(threshold=.5)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ffn_input_norm = nn.LayerNorm(dim)
        self.ffn_input_spike = _lif()
        self.fc1 = SpikeTokenProjection(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim, bias=False)

    def attention_current(self, carrier, observed):
        if carrier.ndim != 4 or carrier.shape[-1] != self.dim:
            raise ValueError('Expected carrier [T,B,N,C]')
        t, b, n, _c = carrier.shape
        spikes = self.input_spike(self.input_norm(carrier))
        split = lambda x: x.reshape(t, b, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        evidence = normalized_spike_evidence(
            split(self.q(spikes)), split(self.k(spikes)),
            split(self.v(spikes)), observed)
        source = (evidence.contrast if self.encoding == 'reference_contrast'
                  else evidence.attended)
        current = source.transpose(2, 3).reshape(t, b, n, self.dim)
        if self.encoding == 'channel_centered':
            # Keep the whole query-selected evidence pattern. Subtracting
            # the entire visible-V reference harmed view correspondence in
            # the first real-event audit; it remains a separate ablation.
            current = current - current.mean(-1, keepdim=True)
        normalized = rms_current(current)
        on = self.evidence_on(F.relu(normalized))
        off = self.evidence_off(F.relu(-normalized))
        # Two binary accumulations share the same matrix. No analog
        # attention activation is fed to a learned synapse.
        current = rms_current(self.proj(on) - self.proj(off))
        has_evidence = (evidence.coincidence_mass > 0).any(2)
        return current * has_evidence, evidence

    def forward(self, carrier, observed):
        current, _evidence = self.attention_current(carrier, observed)
        carrier = energy_matched_residual(carrier, current, self.depth)
        spikes = self.ffn_input_spike(self.ffn_input_norm(carrier))
        current = rms_current(self.fc2(self.fc1(spikes)))
        return energy_matched_residual(carrier, current, self.depth)
