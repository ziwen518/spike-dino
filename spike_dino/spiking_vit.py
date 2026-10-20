"""Directly trainable spike-driven ViT prototype for event streams.

This module supplies a ViT-S/16-sized SNN *backbone*, not a finished DINOv3
implementation. The linear QK/V rule is deliberately distinct from softmax
attention and positional currents are not DINOv3 RoPE. Both require controlled
equivalence experiments before making a DINOv3 claim.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import neuron, surrogate


def _lif(threshold=1.0):
    return neuron.LIFNode(tau=2.0, v_threshold=threshold, v_reset=0.0,
                          detach_reset=True, surrogate_function=surrogate.ATan(),
                          backend='torch', step_mode='m')


class SpikeOrBridge(torch.autograd.Function):
    """Binary OR; iBOT bridge gradients train new context spikes only."""

    @staticmethod
    def forward(ctx, old_spikes, context_spikes):
        ctx.save_for_backward(old_spikes, context_spikes)
        return torch.maximum(old_spikes, context_spikes)

    @staticmethod
    def backward(ctx, gradient):
        old_spikes, context_spikes = ctx.saved_tensors
        del context_spikes
        # Do not repeatedly backpropagate through the positional-mask state
        # at every block. The bridge exists to learn event-bearing attention.
        return None, gradient * (1 - old_spikes) * .5


class UnitBoundedBackward(torch.autograd.Function):
    """Identity forward; bound each token's branch-input gradient to spike units."""

    @staticmethod
    def forward(ctx, membrane):
        return membrane

    @staticmethod
    def backward(ctx, gradient):
        maximum = gradient.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
        return gradient / maximum


class SpikeExciteInhibit(torch.autograd.Function):
    """Binary residual update with identity-path surrogate gradient."""

    @staticmethod
    def forward(ctx, old_spikes, excitation, inhibition):
        excited = torch.maximum(old_spikes, excitation)
        ctx.save_for_backward(old_spikes, excitation, inhibition, excited)
        return excited * (1 - inhibition)

    @staticmethod
    def backward(ctx, gradient):
        old_spikes, excitation, inhibition, excited = ctx.saved_tensors
        del excitation
        # Straight-through identity gives a deep binary skip path. Branches
        # use the local logical derivatives and remain trainable.
        return (gradient,
                gradient * (1 - old_spikes) * (1 - inhibition),
                -gradient * excited)


class TokenBatchNorm(nn.BatchNorm1d):
    """Normalize channels across time, batch and tokens, then reshape."""

    def forward(self, x):
        t, b, n, c = x.shape
        return super().forward(x.reshape(-1, c)).reshape(t, b, n, c)


class SpikeLinear(nn.Module):
    """The output crossing into the next synaptic layer is a binary spike."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = TokenBatchNorm(out_features)
        self.spike = _lif()

    def forward(self, x):
        return self.spike(self.norm(self.linear(x)))


class SpikeDrivenAttention(nn.Module):
    """Binary Q/K/V with explicit projection-interface choices."""

    def __init__(self, dim, heads, variant='matrix_recode'):
        super().__init__()
        if dim % heads:
            raise ValueError('Embedding width must divide attention heads')
        if variant not in ('matrix_recode', 'matrix_legacy', 'channel_gate',
                           'token_pair', 'token_pair_centered',
                           'matrix_value_skip', 'matrix_visible_keys',
                           'matrix_cls_wta', 'matrix_cls_wta_temporal',
                           'matrix_count', 'spikformer_count'):
            raise ValueError(f'Unknown spike attention variant: {variant}')
        self.variant = variant
        self.heads = heads
        self.head_dim = dim // heads
        self.q = SpikeLinear(dim, dim)
        self.k = SpikeLinear(dim, dim)
        self.v = SpikeLinear(dim, dim)
        self.context_spike = _lif(threshold=.005)
        self.attention_output_spike = _lif(threshold=.49)
        self.count_output_spike = _lif(threshold=.5)
        self.pair_spike = _lif(threshold=.49)
        self.pair_value_spike = _lif(threshold=.005)
        self.proj = SpikeLinear(dim, dim)

    def forward(self, x, output_current=False, masked_tokens=None):
        t, b, n, c = x.shape
        shape = (t, b, n, self.heads, self.head_dim)
        q = self.q(x).reshape(shape).permute(0, 1, 3, 2, 4)
        k = self.k(x).reshape(shape).permute(0, 1, 3, 2, 4)
        v = self.v(x).reshape(shape).permute(0, 1, 3, 2, 4)
        if self.variant == 'matrix_visible_keys' and masked_tokens is not None:
            if masked_tokens.shape != (b, n) or masked_tokens.dtype != torch.bool:
                raise ValueError('masked_tokens must be boolean [B,N]')
            # A masked query remains active; masked locations cannot supply
            # K/V content to any query. This is an event-specific candidate,
            # not a claim of equivalence to DINOv3's dense self-attention.
            key_mask = masked_tokens[None, :, None, :, None]
            k = k.masked_fill(key_mask, 0)
            v = v.masked_fill(key_mask, 0)
        if self.variant in ('token_pair', 'token_pair_centered'):
            # Encode a token-to-token QK coincidence matrix, then collect
            # visible V spikes separately for every query position. The
            # normalized count is an internal current; pair_value_spike
            # re-encodes it before the learned output projection.
            pair = self.pair_spike(q @ k.transpose(-1, -2))
            value_current = (pair @ v) / pair.sum(-1, keepdim=True).clamp_min(1)
            if self.variant == 'token_pair_centered':
                # An OR-like readout activates on ubiquitous values. Compare
                # each query's attended rate against its own recording's
                # population rate before encoding the next binary spike.
                value_current = value_current - v.mean(-2, keepdim=True)
            out = self.pair_value_spike(value_current)
        elif self.variant == 'channel_gate':
            # Spike-driven Transformer-style channel gate. K/V coincidence
            # is integrated, re-encoded, then multiplied by binary Q.
            gate = self.context_spike((k * v).sum(dim=-2, keepdim=True))
            out = q * gate
        elif self.variant in ('matrix_count', 'spikformer_count'):
            # Operator controls: retain integer coincidence counts until
            # after Q(K^T V). The official Spikformer SSA uses scale .125
            # and threshold .5 here, without an intermediate context LIF.
            # This matches its aggregation operator, not its full backbone
            # (our projection norms and stem still differ).
            out = q @ (k.transpose(-1, -2) @ v)
            out = (self.count_output_spike(out * .125)
                   if self.variant == 'spikformer_count' else
                   self.attention_output_spike(out / max(n, 1)))
        else:
            # K^T V is a different attention function from ViT softmax.
            context = self.context_spike((k.transpose(-1, -2) @ v) / max(n, 1))
            out = q @ context
            if self.variant in ('matrix_cls_wta', 'matrix_cls_wta_temporal'):
                # Experimental CLS-only winner-take-all spatial selection.
                # Forward communication selects one binary V patch per
                # head/time; soft competition supplies a surrogate gradient
                # to binary Q/K. Patch queries keep the original matrix path.
                patch_k = k[:, :, :, 1:]
                patch_v = v[:, :, :, 1:]
                overlap = (q[:, :, :, :1] @
                           patch_k.transpose(-1, -2)).squeeze(-2)
                value_count = patch_v.sum(-1)
                valid = (patch_k.sum(-1) + value_count) > 0
                if masked_tokens is not None:
                    if (masked_tokens.shape != (b, n) or
                            masked_tokens.dtype != torch.bool):
                        raise ValueError('masked_tokens must be boolean [B,N]')
                    valid = valid & ~masked_tokens[None, :, None, 1:]
                if self.variant == 'matrix_cls_wta_temporal':
                    # One winner is shared across T bins. Accumulating
                    # coincidence evidence reduces framewise winner jitter.
                    overlap = overlap.sum(0, keepdim=True)
                    value_count = value_count.sum(0, keepdim=True)
                    valid = valid.any(0, keepdim=True)
                has_key = valid.any(-1, keepdim=True)
                hard_score = overlap * (self.head_dim + 1) + value_count
                winner = hard_score.masked_fill(~valid, -1e6).argmax(-1)
                hard = torch.nn.functional.one_hot(
                    winner, patch_k.shape[-2]).to(patch_v.dtype) * has_key
                soft_score = (overlap / math.sqrt(self.head_dim) +
                              value_count / self.head_dim)
                soft = torch.softmax(
                    soft_score.masked_fill(~valid, -1e4), dim=-1) * has_key
                selector = hard.detach() - soft.detach() + soft
                selected = selector.unsqueeze(-2) @ patch_v
                out = torch.cat((selected, out[:, :, :, 1:]), dim=-2)
            if self.variant in ('matrix_recode', 'matrix_value_skip',
                                'matrix_visible_keys', 'matrix_cls_wta',
                                'matrix_cls_wta_temporal'):
                # Q @ context can count several coincident spikes. Re-encode
                # before its learned projection to keep that synapse binary.
                out = self.attention_output_spike(out)
        out = out.transpose(2, 3).reshape(t, b, n, c)
        projected = (self.proj.norm(self.proj.linear(out))
                     if output_current else self.proj(out))
        if self.variant == 'matrix_value_skip':
            # Both operands are binary. Preserve the same-position value
            # spike when matrix aggregation/projection erases it; masked
            # tokens can still receive observed context through projected.
            projected = torch.maximum(projected,
                                      v.transpose(2, 3).reshape(t, b, n, c))
        return projected


class SpikeTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, attention_variant='matrix_recode',
                 residual_policy='accumulate', masked_context_bridge='none',
                 residual_gradient_policy='full'):
        super().__init__()
        if residual_policy not in ('accumulate', 'recode_norm', 'layer_scale',
                                   'centered_layer_scale',
                                   'moment_preserving', 'signed_current',
                                   'signed_layer_scale',
                                   'binary_ei'):
            raise ValueError('Unknown residual policy')
        if residual_policy == 'binary_ei' and masked_context_bridge != 'none':
            raise ValueError('Binary EI blocks do not use a masked-only bridge')
        if residual_policy in ('signed_current', 'signed_layer_scale') and masked_context_bridge != 'none':
            raise ValueError('Signed-current blocks do not use a binary mask bridge')
        if residual_policy in ('signed_current', 'signed_layer_scale') and attention_variant == 'matrix_value_skip':
            raise ValueError('Signed-current blocks require attention without binary V OR')
        self.residual_policy = residual_policy
        if residual_gradient_policy not in ('full', 'branch_detach',
                                            'masked_branch_detach',
                                            'bounded_branch'):
            raise ValueError('Unknown residual gradient policy')
        self.residual_gradient_policy = residual_gradient_policy
        if masked_context_bridge not in ('none', 'spike_or'):
            raise ValueError('Unknown masked context bridge')
        self.masked_context_bridge = masked_context_bridge
        if masked_context_bridge == 'spike_or':
            self.mask_bridge_spike = _lif(threshold=.25)
        self.attn_input = _lif(threshold=.25)
        self.attn = SpikeDrivenAttention(dim, heads, attention_variant)
        self.mlp_input = _lif(threshold=.25)
        self.fc1 = SpikeLinear(dim, dim * mlp_ratio)
        self.fc2 = SpikeLinear(dim * mlp_ratio, dim)
        if residual_policy == 'recode_norm':
            self.attn_residual_norm = nn.LayerNorm(dim)
            self.attn_residual_spike = _lif(threshold=.5)
            self.mlp_residual_norm = nn.LayerNorm(dim)
            self.mlp_residual_spike = _lif(threshold=.5)
        elif residual_policy in ('layer_scale', 'centered_layer_scale',
                                 'moment_preserving', 'signed_layer_scale'):
            # Match the small residual-branch initialization used by DINOv3.
            self.attn_scale = nn.Parameter(torch.full((dim,), 1e-5))
            self.mlp_scale = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, membrane, masked_tokens=None):
        if self.residual_policy == 'binary_ei':
            # The first block receives tokenizer spikes; every later block
            # receives this binary output. Learned synapses see spikes only.
            attention = self.attn(self._branch_input(membrane,
                                                     masked_tokens))
            excited = torch.maximum(membrane, attention)
            inhibition = self.fc2(self.fc1(self._branch_input(
                excited, masked_tokens)))
            return SpikeExciteInhibit.apply(membrane, attention, inhibition)
        initial_membrane = membrane
        # Residual addition stores a real membrane current. Every branch
        # entering a learned synaptic transform is encoded by an LIF node.
        branch_membrane = self._branch_input(membrane, masked_tokens)
        attention_input = self.attn_input(branch_membrane)
        if self.masked_context_bridge == 'spike_or' and masked_tokens is not None:
            # Keep masked position queries trainable through Q weights while
            # stopping their state gradient from multiplying across blocks.
            attention_input = torch.where(
                masked_tokens[None, :, :, None],
                attention_input.detach(), attention_input)
        attention = self.attn(
            attention_input,
            output_current=self.residual_policy in ('signed_current',
                                                    'signed_layer_scale'),
            masked_tokens=masked_tokens)
        if self.masked_context_bridge == 'spike_or' and masked_tokens is not None:
            if masked_tokens.shape != membrane.shape[1:3]:
                raise ValueError('Masked token positions must match the token grid')
            # Attention is already binary. Encode the existing residual
            # state into spikes and OR in attention only at masked patches.
            # This gives observed-event context a binary route around the
            # tiny LayerScale current used to stabilize the deep backbone.
            old_spikes = self.mask_bridge_spike(membrane)
            bridged = SpikeOrBridge.apply(old_spikes, attention)
        else:
            bridged = None
        if self.residual_policy == 'centered_layer_scale':
            attention = attention - attention.mean(dim=-1, keepdim=True)
        membrane = membrane + (attention * self.attn_scale
                               if self.residual_policy in ('layer_scale',
                                                           'centered_layer_scale',
                                                           'moment_preserving',
                                                           'signed_layer_scale')
                               else attention)
        if bridged is not None:
            membrane = torch.where(masked_tokens[None, :, :, None],
                                   bridged, membrane)
        if self.residual_policy == 'recode_norm':
            membrane = self.attn_residual_spike(self.attn_residual_norm(membrane))
        branch_membrane = self._branch_input(membrane, masked_tokens)
        hidden_spikes = self.fc1(self.mlp_input(branch_membrane))
        feedforward = (self.fc2.norm(self.fc2.linear(hidden_spikes))
                       if self.residual_policy in ('signed_current',
                                                   'signed_layer_scale') else
                       self.fc2(hidden_spikes))
        if self.residual_policy == 'centered_layer_scale':
            feedforward = feedforward - feedforward.mean(dim=-1, keepdim=True)
        membrane = membrane + (feedforward * self.mlp_scale
                               if self.residual_policy in ('layer_scale',
                                                           'centered_layer_scale',
                                                           'moment_preserving',
                                                           'signed_layer_scale')
                               else feedforward)
        if self.residual_policy == 'recode_norm':
            membrane = self.mlp_residual_spike(self.mlp_residual_norm(membrane))
        if self.residual_policy == 'moment_preserving':
            original_mean = initial_membrane.mean(dim=-1, keepdim=True)
            original_std = (initial_membrane.var(dim=-1, keepdim=True,
                                                 unbiased=False) + 1e-4).sqrt()
            updated_mean = membrane.mean(dim=-1, keepdim=True)
            updated_std = (membrane.var(dim=-1, keepdim=True,
                                        unbiased=False) + 1e-4).sqrt()
            membrane = ((membrane - updated_mean) *
                        (original_std / updated_std) + original_mean)
        return membrane

    def _branch_input(self, membrane, masked_tokens):
        if self.residual_gradient_policy == 'branch_detach':
            return membrane.detach()
        if self.residual_gradient_policy == 'bounded_branch':
            return UnitBoundedBackward.apply(membrane)
        if (self.residual_gradient_policy == 'masked_branch_detach' and
                masked_tokens is not None):
            return torch.where(masked_tokens[None, :, :, None],
                               membrane.detach(), membrane)
        return membrane


class SpikingVisionTransformer(nn.Module):
    """Event [T,B,2,H,W] -> binary CLS and spatial patch token streams.

    Default depth/width/head count matches the ViT-S family. The exact
    parameter count includes SNN norms and token currents, so it is reported
    rather than assumed equal to the ANN checkpoint.
    """

    def __init__(self, image_size=128, patch_size=16, in_channels=2,
                 dim=384, depth=12, heads=6, mlp_ratio=4,
                 attention_variant='matrix_recode',
                 cls_aggregation='attention_only',
                 residual_policy='accumulate',
                 masked_context_bridge='none',
                 residual_gradient_policy='full',
                 block_variant='standard'):
        super().__init__()
        if image_size % patch_size or patch_size < 1 or depth < 1:
            raise ValueError('Image size must divide patch size; depth must be positive')
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        if block_variant not in ('standard', 'evidence'):
            raise ValueError('Unknown transformer block variant')
        if block_variant == 'evidence' and (
                attention_variant != 'matrix_recode' or
                residual_policy != 'accumulate' or
                masked_context_bridge != 'none' or
                residual_gradient_policy != 'full'):
            raise ValueError('Evidence blocks use their own attention/residual; '
                             'leave standard block options at their defaults')
        self.block_variant = block_variant
        if cls_aggregation not in ('attention_only', 'spike_pool',
                                   'event_weighted_pool',
                                   'spike_quadrant_pool'):
            raise ValueError('Unknown CLS aggregation policy')
        if cls_aggregation == 'spike_quadrant_pool' and dim % 4:
            raise ValueError('Quadrant CLS pool requires width divisible by four')
        self.cls_aggregation = cls_aggregation
        self.residual_policy = residual_policy
        self.masked_context_bridge = masked_context_bridge
        self.residual_gradient_policy = residual_gradient_policy
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, patch_size, bias=False)
        self.patch_norm = nn.BatchNorm2d(dim)
        self.cls_current = nn.Parameter(torch.zeros(1, 1, dim))
        grid = image_size // patch_size
        self.position_current = nn.Parameter(torch.zeros(1, 1 + grid * grid, dim))
        self.mask_current = nn.Parameter(torch.zeros(1, 1, dim))
        self.token_spike = _lif()
        if cls_aggregation in ('spike_pool', 'event_weighted_pool',
                               'spike_quadrant_pool'):
            self.cls_pool_norm = nn.LayerNorm(dim)
            self.cls_pool_spike = _lif(threshold=.5)
        if block_variant == 'evidence':
            # Local import avoids a module cycle: the isolated prototype
            # shares only the LIF factory with this file.
            from .evidence_block import EvidenceSpikeBlock
            self.blocks = nn.ModuleList(
                EvidenceSpikeBlock(dim, heads, depth, mlp_ratio)
                for _ in range(depth))
        else:
            self.blocks = nn.ModuleList(SpikeTransformerBlock(dim, heads, mlp_ratio,
                                                           attention_variant,
                                                           residual_policy,
                                                           masked_context_bridge,
                                                           residual_gradient_policy)
                                    for _ in range(depth))
        self.cls_output_norm = nn.BatchNorm1d(dim)
        self.cls_output_spike = _lif(threshold=.25)
        self.patch_output_norm = TokenBatchNorm(dim)
        self.patch_output_spike = _lif(threshold=.25)
        # CLS must emit a query spike before the attention path can import
        # any event information. A near-zero current silences that path.
        nn.init.constant_(self.cls_current, 1.5)
        # Positional currents must survive spike thresholding. A near-zero
        # positional code makes every masked patch query identical, so iBOT
        # cannot learn different spatial targets.
        nn.init.trunc_normal_(self.position_current, std=.3)
        # A masked token must emit query spikes to import visible context.
        nn.init.constant_(self.mask_current, 1.3)

    def forward(self, frames, patch_mask=None):
        if frames.ndim != 5 or frames.shape[2] != self.patch_embed.in_channels:
            raise ValueError('Expected event frames [T,B,C,H,W]')
        t, b, _, h, w = frames.shape
        if h % self.patch_size or w % self.patch_size or h > self.image_size or w > self.image_size:
            raise ValueError('Input resolution must fit the configured patch grid')
        n = (h // self.patch_size) * (w // self.patch_size)
        if patch_mask is not None:
            if patch_mask.dtype != torch.bool or patch_mask.shape not in (
                    (b, n), (t, b, n)):
                raise ValueError('patch_mask must be [B,P] or [T,B,P] boolean')
            time_mask = (patch_mask[None].expand(t, -1, -1)
                         if patch_mask.ndim == 2 else patch_mask)
            spatial_mask = time_mask.any(0)
            if (self.block_variant == 'standard' and patch_mask.ndim == 3 and
                    any(block.attn.variant == 'matrix_visible_keys'
                        for block in self.blocks) and
                    not torch.equal(time_mask,
                                    spatial_mask[None].expand_as(time_mask))):
                raise ValueError('Visible-key attention requires full-time spatial masks')
        else:
            time_mask = spatial_mask = None
        # Input event histograms can be real-valued currents. The first
        # synaptic stage encodes them into spikes before token processing.
        x = self.patch_embed(frames.flatten(0, 1))
        if patch_mask is not None:
            # Mask before batch normalization: otherwise masked event content
            # would leak to visible tokens through the batch statistics.
            gh, gw = h // self.patch_size, w // self.patch_size
            x = x.reshape(t, b, self.dim, gh, gw)
            x = x.masked_fill(time_mask.reshape(t, b, 1, gh, gw), 0)
            x = x.flatten(0, 1)
        x = self.patch_norm(x).flatten(2).transpose(1, 2)
        x = x.reshape(t, b, n, self.dim)
        if patch_mask is not None:
            x = torch.where(time_mask[:, :, :, None], self.mask_current, x)
        cls = self.cls_current.expand(t, b, -1, -1)
        grid = self.image_size // self.patch_size
        patch_positions = self.position_current[:, 1:].reshape(1, grid, grid, self.dim)
        patch_positions = F.interpolate(patch_positions.permute(0, 3, 1, 2),
                                        size=(h // self.patch_size, w // self.patch_size),
                                        mode='bicubic', align_corners=False)
        patch_positions = patch_positions.flatten(2).transpose(1, 2)
        positions = torch.cat((self.position_current[:, :1], patch_positions), dim=1)
        tokens = torch.cat((cls, x), dim=2) + positions
        tokens = self.token_spike(tokens)
        if self.cls_aggregation in ('spike_pool', 'event_weighted_pool',
                                    'spike_quadrant_pool'):
            # The standard CLS query receives almost no identifiable event
            # signal at initialization with sparse linear spike attention.
            # Both policies pool already-binary patch spikes and re-encode
            # before the first learned block. Event weighting is an
            # experimental input-supported current, not a binary interlayer
            # operation or an equivalence claim about DINOv3 CLS attention.
            if self.cls_aggregation == 'spike_quadrant_pool':
                # Preserve a coarse 2×2 spatial layout without adding ANN
                # weights. Each channel quarter receives binary spikes from
                # one quadrant, then the current is re-encoded by CLS LIF.
                grid_spikes = tokens[:, :, 1:].float().reshape(
                    t * b, h // self.patch_size, w // self.patch_size,
                    self.dim).permute(0, 3, 1, 2)
                quadrants = F.adaptive_avg_pool2d(
                    grid_spikes, (2, 2)).reshape(t, b, self.dim, 4)
                quarter = self.dim // 4
                pooled_current = torch.cat([
                    quadrants[:, :, index * quarter:(index + 1) * quarter,
                              index] for index in range(4)], dim=-1)
            elif self.cls_aggregation == 'event_weighted_pool':
                event_mass = frames.float().abs().sum(2)
                event_mass = F.avg_pool2d(
                    event_mass.flatten(0, 1).unsqueeze(1),
                    self.patch_size, self.patch_size).reshape(t, b, n).sum(0)
                total_mass = event_mass.sum(-1, keepdim=True)
                weights = torch.where(
                    total_mass > 0,
                    event_mass / total_mass.clamp_min(1e-12),
                    torch.full_like(event_mass, 1. / n))
                pooled_current = (tokens[:, :, 1:].float() *
                                  weights[None, :, :, None]).sum(2)
            else:
                pooled_current = tokens[:, :, 1:].float().mean(dim=2)
            pooled = self.cls_pool_spike(self.cls_pool_norm(pooled_current))
            tokens = torch.cat((pooled.unsqueeze(2), tokens[:, :, 1:]), dim=2)
        masked_tokens = (torch.cat((torch.zeros(b, 1, device=frames.device,
                                                  dtype=torch.bool), spatial_mask),
                                   dim=1) if patch_mask is not None else None)
        if self.block_variant == 'evidence':
            # Only actually visible input events can supply K/V content.
            # AND with ~time_mask prevents even hidden event existence from
            # affecting the support. CLS is a query, never an event source.
            support_mass = F.avg_pool2d(
                frames.float().abs().sum(2).flatten(0, 1).unsqueeze(1),
                self.patch_size, self.patch_size).reshape(t, b, n)
            observed = support_mass > 0
            if time_mask is not None:
                observed = observed & ~time_mask
            observed = torch.cat((torch.zeros(t, b, 1, dtype=torch.bool,
                                              device=frames.device), observed), 2)
            for block in self.blocks:
                tokens = block(tokens, observed)
        else:
            for block in self.blocks:
                tokens = block(tokens, masked_tokens)
        cls = self.cls_output_norm(tokens[:, :, 0].reshape(t * b, self.dim))
        cls = self.cls_output_spike(cls.reshape(t, b, self.dim))
        patch = self.patch_output_spike(self.patch_output_norm(tokens[:, :, 1:]))
        return cls, patch

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
