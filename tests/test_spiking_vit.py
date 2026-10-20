import pytest
import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional

from spike_dino.spiking_vit import (SpikeExciteInhibit, SpikeOrBridge,
                                   SpikeDrivenAttention, SpikeTransformerBlock,
                                   SpikingVisionTransformer,
                                   UnitBoundedBackward)


torch.set_num_threads(2)


def _small():
    return SpikingVisionTransformer(image_size=32, patch_size=8,
                                     dim=48, depth=2, heads=4)


def test_token_shape_spikes_and_gradients():
    torch.manual_seed(12)
    model = _small()
    frames = torch.rand(4, 2, 2, 32, 32)
    cls, patches = model(frames)
    assert cls.shape == (4, 2, 48)
    assert patches.shape == (4, 2, 16, 48)
    assert all(torch.all((v == 0) | (v == 1)) for v in (cls, patches))
    assert patches.float().mean() > 0  # reject a silent backbone
    (cls.float().mean() + patches.float().mean()).backward()
    assert model.patch_embed.weight.grad is not None
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert model.patch_embed.weight.grad.abs().sum() > 0


def test_mask_hides_all_timesteps_without_batchnorm_leakage():
    torch.manual_seed(19)
    model = _small()
    a = torch.rand(4, 2, 2, 32, 32)
    b = a.clone()
    b[:, :, :, :8, :8] = 100.0
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 0] = True
    functional.reset_net(model)
    cls_a, patch_a = model(a, mask)
    functional.reset_net(model)
    cls_b, patch_b = model(b, mask)
    torch.testing.assert_close(cls_a, cls_b, rtol=0, atol=0)
    torch.testing.assert_close(patch_a, patch_b, rtol=0, atol=0)


def test_temporal_patch_mask_hides_only_selected_event_bins():
    torch.manual_seed(23)
    model = _small()
    a = torch.rand(4, 2, 2, 32, 32)
    b = a.clone()
    b[0, :, :, :8, :8] = 100.
    mask = torch.zeros(4, 2, 16, dtype=torch.bool)
    mask[0, :, 0] = True
    functional.reset_net(model)
    cls_a, patch_a = model(a, mask)
    functional.reset_net(model)
    cls_b, patch_b = model(b, mask)
    torch.testing.assert_close(cls_a, cls_b, rtol=0, atol=0)
    torch.testing.assert_close(patch_a, patch_b, rtol=0, atol=0)


def test_vit_small_parameter_scale():
    model = SpikingVisionTransformer()
    assert 21_000_000 <= model.parameter_count() < 22_000_000


def test_attention_projection_synapse_is_binary_except_legacy_control():
    torch.manual_seed(43)
    frames = torch.rand(4, 2, 2, 32, 32)
    for variant in ('matrix_recode', 'channel_gate', 'matrix_legacy'):
        model = SpikingVisionTransformer(image_size=32, patch_size=8, dim=64,
                                         depth=2, heads=4,
                                         attention_variant=variant).train()
        seen = []
        model.blocks[0].attn.proj.linear.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach()))
        model(frames)
        input_values = seen[0]
        assert input_values.any()
        is_binary = bool(torch.all((input_values == 0) | (input_values == 1)))
        assert is_binary == (variant != 'matrix_legacy')


def test_event_resolution_keeps_dinov3_global_and_local_token_counts():
    model = SpikingVisionTransformer(image_size=128, patch_size=8,
                                     dim=48, depth=1, heads=4)
    global_cls, global_patch = model(torch.rand(2, 2, 2, 128, 128))
    assert global_cls.shape == (2, 2, 48)
    assert global_patch.shape == (2, 2, 256, 48)
    functional.reset_net(model)
    local_cls, local_patch = model(torch.rand(2, 2, 2, 56, 56))
    assert local_cls.shape == (2, 2, 48)
    assert local_patch.shape == (2, 2, 49, 48)


def test_spike_pooled_cls_reads_patch_events_as_binary_spikes():
    torch.manual_seed(77)
    model = SpikingVisionTransformer(image_size=32, patch_size=8,
                                     dim=48, depth=2, heads=4,
                                     cls_aggregation='spike_pool').train()
    seen = []
    model.cls_pool_spike.register_forward_hook(
        lambda _module, _inputs, output: seen.append(output.detach()))
    cls, _patch = model(torch.rand(4, 3, 2, 32, 32))
    pooled = seen[0]
    assert torch.all((pooled == 0) | (pooled == 1))
    assert pooled.any()
    assert pooled.float().mean(0).std(0, unbiased=False).mean() > 0
    cls.float().mean().backward()
    assert model.patch_embed.weight.grad.abs().sum() > 0


def test_normalized_residual_is_reencoded_before_next_block():
    torch.manual_seed(79)
    model = SpikingVisionTransformer(image_size=32, patch_size=8,
                                     dim=48, depth=3, heads=4,
                                     residual_policy='recode_norm').train()
    seen = []
    for block in model.blocks:
        block.register_forward_hook(
            lambda _module, _inputs, output: seen.append(output.detach()))
    cls, patch = model(torch.rand(4, 3, 2, 32, 32))
    assert len(seen) == 3
    assert all(torch.all((tensor == 0) | (tensor == 1)) for tensor in seen)
    assert cls.any() and patch.any()
    cls.float().mean().backward()
    assert model.patch_embed.weight.grad.abs().sum() > 0


def test_layer_scale_initializes_small_trainable_residual_branches():
    torch.manual_seed(83)
    model = SpikingVisionTransformer(image_size=32, patch_size=8,
                                     dim=48, depth=3, heads=4,
                                     residual_policy='layer_scale').train()
    assert all(torch.allclose(block.attn_scale,
                              torch.full_like(block.attn_scale, 1e-5))
               for block in model.blocks)
    cls, _patch = model(torch.rand(4, 3, 2, 32, 32))
    cls.float().mean().backward()
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert torch.isfinite(model.blocks[0].attn_scale.grad).all()


def test_centered_layer_scale_keeps_finite_spiking_interfaces():
    torch.manual_seed(85)
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=3, heads=4,
        residual_policy='centered_layer_scale').train()
    cls, patch = model(torch.rand(4, 3, 2, 32, 32))
    assert torch.all((cls == 0) | (cls == 1))
    assert torch.all((patch == 0) | (patch == 1))
    (cls.mean() + patch.mean()).backward()
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert torch.isfinite(model.blocks[0].attn_scale.grad).all()
    assert torch.isfinite(model.blocks[0].mlp_scale.grad).all()


def test_moment_preserving_residual_controls_per_token_mean_and_variance():
    torch.manual_seed(87)
    block = SpikeTransformerBlock(
        dim=48, heads=4, residual_policy='moment_preserving').train()
    old = torch.randint(0, 2, (4, 3, 17, 48)).float().requires_grad_()
    with torch.no_grad():
        block.attn_scale.fill_(.3)
        block.mlp_scale.fill_(.3)
    updated = block(old)
    torch.testing.assert_close(updated.mean(-1), old.mean(-1), atol=1e-5, rtol=0)
    torch.testing.assert_close(updated.var(-1, unbiased=False),
                               old.var(-1, unbiased=False), atol=1e-3, rtol=0)
    updated.square().mean().backward()
    assert torch.isfinite(old.grad).all()
    assert torch.isfinite(block.attn_scale.grad).all()


def test_signed_current_residual_receives_spikes_and_keeps_signed_updates():
    for residual_policy in ('signed_current', 'signed_layer_scale'):
        torch.manual_seed(88)
        model = SpikingVisionTransformer(
            image_size=32, patch_size=8, dim=48, depth=2, heads=4,
            residual_policy=residual_policy).train()
        synapse_inputs = []
        attention_currents = []
        handles = [
            model.blocks[0].attn.proj.linear.register_forward_pre_hook(
                lambda _module, inputs: synapse_inputs.append(inputs[0].detach())),
            model.blocks[0].fc2.linear.register_forward_pre_hook(
                lambda _module, inputs: synapse_inputs.append(inputs[0].detach())),
            model.blocks[0].attn.register_forward_hook(
                lambda _module, _inputs, output: attention_currents.append(
                    output.detach()))]
        cls, patch = model(torch.rand(4, 3, 2, 32, 32))
        for handle in handles:
            handle.remove()
        assert len(synapse_inputs) == 2
        assert all(torch.all((x == 0) | (x == 1)) for x in synapse_inputs)
        assert attention_currents[0].min() < 0 < attention_currents[0].max()
        assert torch.all((cls == 0) | (cls == 1))
        assert torch.all((patch == 0) | (patch == 1))
        (cls.mean() + patch.mean()).backward()
        assert torch.isfinite(model.patch_embed.weight.grad).all()
        assert torch.isfinite(model.blocks[0].attn.proj.linear.weight.grad).all()
        assert torch.isfinite(model.blocks[0].fc2.linear.weight.grad).all()


def test_branch_detach_preserves_forward_and_trains_each_residual_branch():
    torch.manual_seed(97)
    frames = torch.rand(4, 3, 2, 32, 32)
    full = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=3, heads=4,
        residual_policy='layer_scale', residual_gradient_policy='full').train()
    detached = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=3, heads=4,
        residual_policy='layer_scale',
        residual_gradient_policy='branch_detach').train()
    detached.load_state_dict(full.state_dict())
    full_cls, full_patch = full(frames)
    detached_cls, detached_patch = detached(frames)
    torch.testing.assert_close(full_cls, detached_cls)
    torch.testing.assert_close(full_patch, detached_patch)
    (detached_cls.mean() + detached_patch.mean()).backward()
    assert torch.isfinite(detached.patch_embed.weight.grad).all()
    assert all(block.attn_scale.grad is not None and
               torch.isfinite(block.attn_scale.grad).all()
               for block in detached.blocks)


def test_bounded_branch_backward_preserves_small_gradients_and_spike_forward():
    x = torch.tensor([[[.1, .9, .2], [.5, .0, .7]]],
                     requires_grad=True)
    output = UnitBoundedBackward.apply(x)
    torch.testing.assert_close(output, x)
    output.backward(torch.tensor([[[.2, -.5, .1], [200., -50., 0.]]]))
    torch.testing.assert_close(x.grad[0, 0],
                               torch.tensor([.2, -.5, .1]))
    torch.testing.assert_close(x.grad[0, 1],
                               torch.tensor([1., -.25, 0.]))


def test_binary_excitation_inhibition_truth_table_and_identity_gradient():
    old = torch.tensor([0., 0., 1., 1.], requires_grad=True)
    excite = torch.tensor([0., 1., 0., 1.], requires_grad=True)
    inhibit = torch.tensor([0., 0., 1., 1.], requires_grad=True)
    output = SpikeExciteInhibit.apply(old, excite, inhibit)
    torch.testing.assert_close(output, torch.tensor([0., 1., 0., 0.]))
    output.sum().backward()
    torch.testing.assert_close(old.grad, torch.ones_like(old))
    torch.testing.assert_close(excite.grad, torch.tensor([1., 1., 0., 0.]))
    torch.testing.assert_close(inhibit.grad, torch.tensor([0., -1., -1., -1.]))


def test_binary_ei_blocks_change_spikes_and_keep_finite_gradients():
    torch.manual_seed(107)
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=3, heads=4,
        residual_policy='binary_ei',
        residual_gradient_policy='bounded_branch').train()
    outputs = []
    handles = [block.register_forward_hook(
        lambda _module, _inputs, output: outputs.append(output.detach()))
        for block in model.blocks]
    cls, patch = model(torch.rand(4, 3, 2, 32, 32))
    for handle in handles:
        handle.remove()
    assert all(torch.all((value == 0) | (value == 1)) for value in outputs)
    assert torch.all((cls == 0) | (cls == 1))
    assert torch.all((patch == 0) | (patch == 1))
    assert (outputs[0] - outputs[-1]).abs().mean() > 0
    (cls.mean() + patch.mean()).backward()
    assert torch.isfinite(model.patch_embed.weight.grad).all()
    assert torch.isfinite(model.blocks[0].attn.q.linear.weight.grad).all()


def test_token_pair_attention_keeps_binary_synapse_interface_and_gradients():
    torch.set_num_threads(2)
    for variant in ('token_pair', 'token_pair_centered'):
        torch.manual_seed(109)
        model = SpikingVisionTransformer(
            image_size=32, patch_size=8, dim=48, depth=2, heads=4,
            attention_variant=variant, residual_policy='layer_scale').train()
        attended = []
        handle = model.blocks[0].attn.pair_value_spike.register_forward_hook(
            lambda _module, _inputs, output: attended.append(output.detach()))
        cls, patch = model(torch.rand(4, 3, 2, 32, 32))
        handle.remove()
        assert attended and attended[0].shape == (4, 3, 4, 17, 12)
        assert torch.all((attended[0] == 0) | (attended[0] == 1))
        (cls.mean() + patch.mean()).backward()
        for weight in (model.patch_embed.weight,
                       model.blocks[0].attn.q.linear.weight,
                       model.blocks[0].attn.k.linear.weight,
                       model.blocks[0].attn.v.linear.weight):
            assert weight.grad is not None
            assert torch.isfinite(weight.grad).all()


def test_visible_key_attention_excludes_masked_kv_but_keeps_masked_queries():
    torch.manual_seed(111)
    attention = SpikeDrivenAttention(48, 4, 'matrix_visible_keys').train()
    values = {}
    handles = []
    for name in ('q', 'k', 'v'):
        handles.append(getattr(attention, name).register_forward_hook(
            lambda _module, _inputs, output, name=name:
            values.__setitem__(name, output)))
    handles.append(attention.context_spike.register_forward_pre_hook(
        lambda _module, inputs: values.__setitem__('context_current', inputs[0])))
    handles.append(attention.proj.linear.register_forward_pre_hook(
        lambda _module, inputs: values.__setitem__('proj_input', inputs[0])))
    x = torch.rand(4, 2, 6, 48, requires_grad=True)
    mask = torch.tensor([[False, True, False, False, True, False],
                         [False, False, True, False, False, True]])
    output = attention(x, masked_tokens=mask)
    for handle in handles:
        handle.remove()
    k = values['k'].reshape(4, 2, 6, 4, 12).permute(0, 1, 3, 2, 4)
    v = values['v'].reshape(4, 2, 6, 4, 12).permute(0, 1, 3, 2, 4)
    selector = (~mask)[None, :, None, :, None]
    expected = ((k * selector).transpose(-1, -2) @ (v * selector)) / 6
    torch.testing.assert_close(values['context_current'], expected)
    assert torch.all((values['proj_input'] == 0) |
                     (values['proj_input'] == 1))
    assert output.shape == x.shape
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_visible_key_attention_rejects_partially_observed_time_bins():
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        attention_variant='matrix_visible_keys')
    mask = torch.zeros(4, 2, 16, dtype=torch.bool)
    mask[:2, :, 0] = True
    with pytest.raises(ValueError, match='full-time spatial masks'):
        model(torch.rand(4, 2, 2, 32, 32), mask)


def test_event_weighted_cls_pool_uses_event_support_and_binary_readout():
    torch.manual_seed(119)
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        cls_aggregation='event_weighted_pool',
        residual_policy='layer_scale').train()
    token_spikes = []
    pool_currents = []
    handles = [
        model.token_spike.register_forward_hook(
            lambda _module, _inputs, output: token_spikes.append(output.detach())),
        model.cls_pool_norm.register_forward_pre_hook(
            lambda _module, inputs: pool_currents.append(inputs[0].detach()))]
    frames = torch.rand(4, 2, 2, 32, 32)
    frames[:, :, :, :8, :8] *= 10
    cls, patch = model(frames)
    for handle in handles:
        handle.remove()
    support = F.avg_pool2d(frames.abs().sum(2).flatten(0, 1).unsqueeze(1),
                           8, 8).reshape(4, 2, 16).sum(0)
    weights = support / support.sum(-1, keepdim=True)
    expected = (token_spikes[0][:, :, 1:] * weights[None, :, :, None]).sum(2)
    torch.testing.assert_close(pool_currents[0], expected)
    assert (pool_currents[0] - token_spikes[0][:, :, 1:].mean(2)).abs().sum() > 0
    assert torch.all((cls == 0) | (cls == 1))
    assert torch.all((patch == 0) | (patch == 1))


@pytest.mark.parametrize('variant', ('matrix_cls_wta',
                                     'matrix_cls_wta_temporal'))
def test_cls_wta_keeps_patch_attention_and_binary_projection_interface(variant):
    torch.manual_seed(127)
    original = SpikeDrivenAttention(48, 4, 'matrix_recode').train()
    competitive = SpikeDrivenAttention(48, 4, variant).train()
    competitive.load_state_dict(original.state_dict())
    inputs = torch.rand(4, 2, 6, 48, requires_grad=True)
    captured = {}
    handles = [
        original.proj.linear.register_forward_pre_hook(
            lambda _module, values: captured.__setitem__('base', values[0].detach())),
        competitive.proj.linear.register_forward_pre_hook(
            lambda _module, values: captured.__setitem__('wta', values[0].detach()))]
    original(inputs)
    result = competitive(inputs)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(captured['base'][:, :, 1:],
                               captured['wta'][:, :, 1:])
    assert torch.all((captured['wta'] == 0) | (captured['wta'] == 1))
    assert result.shape == inputs.shape
    result[:, :, 0].mean().backward()
    for weight in (competitive.q.linear.weight,
                   competitive.k.linear.weight,
                   competitive.v.linear.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()


def test_quadrant_cls_pool_retains_spatial_spike_layout():
    torch.manual_seed(131)
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        cls_aggregation='spike_quadrant_pool').train()
    seen = {}
    handles = [
        model.token_spike.register_forward_hook(
            lambda _module, _inputs, output: seen.__setitem__(
                'tokens', output.detach())),
        model.cls_pool_norm.register_forward_pre_hook(
            lambda _module, inputs: seen.__setitem__(
                'pooled', inputs[0].detach()))]
    cls, patches = model(torch.rand(4, 2, 2, 32, 32))
    for handle in handles:
        handle.remove()
    token_grid = seen['tokens'][:, :, 1:].reshape(8, 4, 4, 48)
    quadrants = F.adaptive_avg_pool2d(
        token_grid.permute(0, 3, 1, 2), (2, 2)).reshape(4, 2, 48, 4)
    expected = torch.cat([
        quadrants[:, :, i * 12:(i + 1) * 12, i] for i in range(4)], -1)
    torch.testing.assert_close(seen['pooled'], expected)
    assert torch.all((cls == 0) | (cls == 1))
    assert torch.all((patches == 0) | (patches == 1))


@pytest.mark.parametrize('variant,scale', (('matrix_count', 1 / 6),
                                           ('spikformer_count', .125)))
def test_count_attention_keeps_kv_counts_until_query_aggregation(variant, scale):
    torch.manual_seed(137)
    attention = SpikeDrivenAttention(48, 4, variant).train()
    values, context_calls = {}, []
    handles = []
    for name in ('q', 'k', 'v'):
        handles.append(getattr(attention, name).register_forward_hook(
            lambda _module, _inputs, output, name=name:
            values.__setitem__(name, output)))
    neuron = (attention.count_output_spike if variant == 'spikformer_count'
              else attention.attention_output_spike)
    handles.append(neuron.register_forward_pre_hook(
        lambda _module, inputs: values.__setitem__('current', inputs[0])))
    handles.append(attention.context_spike.register_forward_hook(
        lambda *_args: context_calls.append(True)))
    result = attention(torch.rand(4, 2, 6, 48))
    for handle in handles:
        handle.remove()
    split = lambda value: value.reshape(4, 2, 6, 4, 12).permute(0, 1, 3, 2, 4)
    q, k, v = [split(values[name]) for name in ('q', 'k', 'v')]
    torch.testing.assert_close(values['current'],
                               (q @ (k.transpose(-1, -2) @ v)) * scale)
    assert not context_calls
    assert torch.all((result == 0) | (result == 1))


def test_matrix_value_skip_preserves_binary_value_spikes():
    torch.set_num_threads(2)
    torch.manual_seed(113)
    model = SpikingVisionTransformer(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        attention_variant='matrix_value_skip',
        residual_policy='layer_scale').train()
    values = []
    outputs = []
    handles = [
        model.blocks[0].attn.v.spike.register_forward_hook(
            lambda _module, _inputs, output: values.append(output.detach())),
        model.blocks[0].attn.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(output.detach()))]
    cls, patch = model(torch.rand(4, 3, 2, 32, 32))
    for handle in handles:
        handle.remove()
    assert len(values) == len(outputs) == 1
    assert torch.all((outputs[0] == 0) | (outputs[0] == 1))
    assert torch.all(outputs[0] >= values[0])
    (cls.mean() + patch.mean()).backward()
    assert torch.isfinite(model.blocks[0].attn.v.linear.weight.grad).all()


def test_masked_spike_bridge_imports_visible_event_changes():
    torch.set_num_threads(2)
    torch.manual_seed(89)
    frames = torch.rand(4, 3, 2, 32, 32)
    mask = torch.zeros(3, 16, dtype=torch.bool)
    mask[:, :4] = True
    scores = {}
    for variant in ('none', 'spike_or'):
        torch.manual_seed(91)
        model = SpikingVisionTransformer(
            image_size=32, patch_size=8, dim=48, depth=2, heads=4,
            residual_policy='layer_scale',
            masked_context_bridge=variant).train()
        full = model(frames, mask)[1][:, mask]
        functional.reset_net(model)
        empty = model(torch.zeros_like(frames), mask)[1][:, mask]
        scores[variant] = float((full - empty).abs().mean().detach())
        if variant == 'spike_or':
            assert torch.all((full == 0) | (full == 1))
    assert scores['spike_or'] > scores['none']


def test_binary_or_bridge_has_bounded_split_surrogate_gradient():
    old = torch.tensor([0., 0., 1., 1.], requires_grad=True)
    context = torch.tensor([0., 1., 0., 1.], requires_grad=True)
    output = SpikeOrBridge.apply(old, context)
    torch.testing.assert_close(output, torch.tensor([0., 1., 1., 1.]))
    output.sum().backward()
    assert old.grad is None
    torch.testing.assert_close(context.grad, torch.tensor([.5, .5, 0., 0.]))
