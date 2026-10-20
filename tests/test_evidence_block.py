import math

import torch
from torch import nn
from spikingjelly.activation_based import functional

from spike_dino.evidence_block import (
    EvidenceSpikeBlock, energy_matched_residual, normalized_spike_evidence)
from spike_dino.dinov3 import SpikingDinoV3Prototype


def _inputs():
    torch.manual_seed(173)
    q, k, v = [(torch.rand(4, 2, 3, 5, 8) > .6).float()
               for _ in range(3)]
    observed = torch.rand(4, 2, 5) > .3
    return q, k, v, observed


def test_evidence_matches_explicit_spatiotemporal_query_key_attention():
    q, k, v, observed = _inputs()
    actual = normalized_spike_evidence(q, k, v, observed)
    flatten = lambda x: x.permute(1, 2, 0, 3, 4).reshape(2, 3, 20, 8)
    queries, keys, values = map(flatten, (q, k, v))
    support = observed.permute(1, 0, 2).reshape(2, 20)
    weights = (queries @ keys.transpose(-1, -2)) * support[:, None, None]
    dense = (weights / weights.sum(-1, keepdim=True).clamp_min(1)) @ values
    expected = dense.reshape(2, 3, 4, 5, 8).permute(2, 0, 1, 3, 4)
    torch.testing.assert_close(actual.attended, expected)
    assert actual.attended.min() >= 0 and actual.attended.max() <= 1


def test_duplicating_observed_key_value_evidence_does_not_change_attention():
    q, k, v, observed = _inputs()
    original = normalized_spike_evidence(q, k, v, observed)
    duplicate = normalized_spike_evidence(
        torch.cat((q, q), 3), torch.cat((k, k), 3),
        torch.cat((v, v), 3), torch.cat((observed, observed), 2))
    torch.testing.assert_close(original.attended, duplicate.attended[:, :, :, :5])
    torch.testing.assert_close(original.contrast, duplicate.contrast[:, :, :, :5])
    torch.testing.assert_close(original.coincidence_mass * 2,
                               duplicate.coincidence_mass[:, :, :, :5])


def test_hidden_time_bins_cannot_supply_content_or_mix_recordings():
    q, k, v, observed = _inputs()
    observed[0, 0, 2] = False
    original = normalized_spike_evidence(q, k, v, observed)
    corrupted_k, corrupted_v = k.clone(), v.clone()
    corrupted_k[0, 0, :, 2] = float('nan')
    corrupted_v[0, 0, :, 2] = float('inf')
    hidden = normalized_spike_evidence(q, corrupted_k, corrupted_v, observed)
    torch.testing.assert_close(original.contrast, hidden.contrast)
    changed = v.clone()
    changed[:, 1] = 1 - changed[:, 1]
    other_recording = normalized_spike_evidence(q, k, changed, observed)
    torch.testing.assert_close(original.contrast[:, 0],
                               other_recording.contrast[:, 0])
    no_support = normalized_spike_evidence(q, k, v, torch.zeros_like(observed))
    assert not no_support.attended.any()
    assert not no_support.contrast.any()


def test_residual_bounds_update_energy_and_keeps_identity_derivative():
    carrier = torch.randn(4, 2, 5, 24, requires_grad=True)
    branch = (torch.randn_like(carrier) * 100).requires_grad_()
    output = energy_matched_residual(carrier, branch, depth=12)
    update_rms = (output - carrier).square().mean(-1).sqrt()
    carrier_rms = carrier.square().mean(-1).sqrt()
    assert torch.all(update_rms <= carrier_rms / math.sqrt(24) + 1e-6)
    output.sum().backward()
    torch.testing.assert_close(carrier.grad, torch.ones_like(carrier))


def test_block_synapses_receive_binary_streams_and_backward_is_finite():
    torch.manual_seed(179)
    block = EvidenceSpikeBlock(dim=24, heads=3, depth=2)
    seen = []
    handles = [module.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach()))
        for module in block.modules() if isinstance(module, nn.Linear)]
    carrier = torch.randn(4, 2, 5, 24, requires_grad=True)
    observed = torch.ones(4, 2, 5, dtype=torch.bool)
    observed[:, :, 0] = False  # CLS remains a query, never an event source.
    observed[:2, :, 2] = False  # Visible bins of patch 2 remain usable.
    output = block(carrier, observed)
    for handle in handles:
        handle.remove()
    assert output.shape == carrier.shape
    assert seen and all(torch.all((x == 0) | (x == 1)) for x in seen)
    assert not torch.equal(output, carrier)
    output.square().mean().backward()
    assert torch.isfinite(carrier.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in block.parameters())
    functional.reset_net(block)
    current, _ = block.attention_current(carrier.detach(),
                                        torch.zeros_like(observed))
    assert not current.any()


def test_event_backbone_hides_event_existence_in_masked_time_bins():
    torch.manual_seed(191)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=24, depth=2, heads=3,
        block_variant='evidence', norm_policy='layer_norm',
        cls_aggregation='spike_pool', head_hidden=48, bottleneck=12,
        prototypes=32).train()
    original = torch.rand(4, 2, 2, 32, 32)
    changed = original.clone()
    changed[:2, :, :, :8, :8] = 1000
    time_mask = torch.zeros(4, 2, 16, dtype=torch.bool)
    time_mask[:2, :, 0] = True
    functional.reset_net(model.student)
    cls_a, patch_a = model.student.backbone(original, time_mask)
    functional.reset_net(model.student)
    cls_b, patch_b = model.student.backbone(changed, time_mask)
    functional.reset_net(model.student)
    torch.testing.assert_close(cls_a, cls_b, rtol=0, atol=0)
    torch.testing.assert_close(patch_a, patch_b, rtol=0, atol=0)
    views = [original, torch.rand_like(original)]
    masks = [time_mask.any(0), time_mask.any(0)]
    metrics = model(views, masks, temporal_masks=[time_mask, time_mask])
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics['loss'].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.student.parameters())
