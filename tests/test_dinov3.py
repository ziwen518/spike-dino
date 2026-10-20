import torch
from unittest.mock import patch
from spikingjelly.activation_based import functional

from spike_dino.dinov3 import (SpikingDinoV3Prototype, SpikingProjectionHead,
                               gram_loss, koleo_loss, sinkhorn_knopp,
                               spike_diversity_retention)
from spike_dino.pretrain_v3 import patch_masks, partial_temporal_masks


torch.set_num_threads(2)


def test_sinkhorn_balances_batch_and_prototypes():
    torch.manual_seed(1)
    result = sinkhorn_knopp(torch.randn(12, 8), temperature=.2, iterations=50)
    torch.testing.assert_close(result.sum(-1), torch.ones(12), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(result.sum(0), torch.full((8,), 1.5),
                               atol=1e-3, rtol=1e-3)


def test_spiking_dino_global_local_backward_and_ema():
    torch.manual_seed(3)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=2, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=32, teacher_momentum=.5)
    views = [torch.rand(4, 3, 2, 32, 32), torch.rand(4, 3, 2, 32, 32),
             torch.rand(4, 3, 2, 16, 16)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    losses = model(views, masks)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['loss'].backward()
    assert model.student.backbone.patch_embed.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    student = model.student.backbone.patch_embed.weight
    teacher = model.teacher.backbone.patch_embed.weight
    old_teacher = teacher.detach().clone()
    with torch.no_grad():
        student.add_(.1)
    model.after_optimizer_step()
    torch.testing.assert_close(teacher, (old_teacher + student) / 2)


def test_dino_head_binary_input_shortcut_trains_both_student_and_teacher():
    torch.manual_seed(4)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=2, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        dino_shortcut='input_or').train()
    assert model.student.dino_head.shortcut == 'input_or'
    assert model.teacher.dino_head.shortcut == 'input_or'
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    metrics = model(views, masks)
    assert torch.isfinite(metrics['loss'])
    metrics['loss'].backward()
    assert torch.isfinite(model.student.dino_head.fc1.weight.grad).all()


def test_gram_loss_matches_geometry_and_stops_teacher_gradient():
    torch.manual_seed(5)
    student = torch.randn(2, 4, 7, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    loss = gram_loss(student, teacher)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss = gram_loss(student, teacher * 1.1 + torch.randn_like(teacher) * .01)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_optional_gram_refinement_uses_frozen_snn_snapshot():
    torch.manual_seed(8)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=1, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=16)
    model.enable_gram_refinement()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    metrics = model(views, masks)
    assert torch.isfinite(metrics['gram'])
    metrics['loss'].backward()
    assert all(p.grad is None for p in model.gram_teacher.parameters())


def test_joint_balance_pooling_matches_official_crop_scope():
    torch.manual_seed(10)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=1, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=16, balance_mode='joint')
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.ones(3, 16, dtype=torch.bool) for _ in range(2)]
    with patch('spike_dino.dinov3.sinkhorn_knopp', wraps=sinkhorn_knopp) as balance:
        model(views, masks)
    assert balance.call_count == 2
    assert balance.call_args_list[0].args[0].shape[0] == 6  # both global crops
    assert balance.call_args_list[1].args[0].shape[0] == 96  # both patch sets


def test_mask_sample_probability_and_selected_patch_head():
    torch.manual_seed(11)
    assert not patch_masks(4, 16, .1, .5, 'cpu', probability=0).any()
    full = patch_masks(4, 16, .1, .5, 'cpu', probability=1)
    assert full.any(dim=1).all()
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=1, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=16)
    frames = torch.rand(4, 4, 2, 32, 32)
    out = model.student(frames, mask=full, patch_head_mask=full)
    assert out['patch_scores'].shape == (int(full.sum()), 16)


def test_koleo_must_not_mix_two_views_of_same_recording():
    rates = torch.eye(4)
    per_crop = koleo_loss(rates) + koleo_loss(rates)
    mixed = koleo_loss(torch.cat([rates, rates]))
    assert per_crop < 0
    assert mixed > 10  # duplicate same-recording views have distance zero


def test_masked_patch_predictions_are_live_and_context_dependent():
    torch.manual_seed(5)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=2, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=32)
    a = torch.rand(4, 3, 2, 32, 32)
    b = torch.rand_like(a)
    mask = torch.rand(3, 16) < .4
    first = model.student(a, mask, mask)['patch_scores'].detach()
    functional.reset_net(model.student)
    second = model.student(b, mask, mask)['patch_scores'].detach()
    assert first.std(0).mean() > .01
    assert (first - second).abs().mean() > .01


def test_batch_stat_teacher_can_eval_without_silent_targets():
    torch.manual_seed(2)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=2, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=32,
                                  norm_policy='batch_stat').train()
    assert not model.teacher.training
    assert model.teacher.backbone.patch_norm.running_mean is None
    views = [torch.rand(4, 4, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(4, 16) < .4 for _ in range(2)]
    metrics = model(views, masks)
    assert metrics['teacher_target_information'] > .01
    metrics['loss'].backward()
    assert model.student.backbone.patch_embed.weight.grad.abs().sum() > 0


def test_layer_norm_teacher_is_eval_batch_independent():
    torch.manual_seed(7)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=2, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=32,
                                  norm_policy='layer_norm').train()
    assert not model.teacher.training
    fixed = torch.rand(4, 1, 2, 32, 32)
    a = torch.cat((fixed, torch.rand_like(fixed)), dim=1)
    b = torch.cat((fixed, torch.zeros_like(fixed)), dim=1)
    with torch.no_grad():
        first = model.teacher(a)['cls_scores'][0]
        functional.reset_net(model.teacher)
        second = model.teacher(b)['cls_scores'][0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_distinct_dino_and_ibot_heads_train_with_different_prototype_sizes():
    torch.manual_seed(29)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        dino_hidden=64, dino_bottleneck=12, dino_prototypes=17,
        ibot_hidden=80, ibot_bottleneck=20, ibot_prototypes=29,
        norm_policy='layer_norm').train()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    out = model.student(views[0], masks[0], masks[0])
    assert out['cls_scores'].shape == (3, 17)
    assert out['patch_scores'].shape == (int(masks[0].sum()), 29)
    functional.reset_net(model.student)
    losses = model(views, masks)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert model.student.dino_head.prototypes.grad is not None
    assert model.student.ibot_head.prototypes.grad is not None


def test_ema_center_targets_update_from_teacher_logits_and_remain_finite():
    torch.manual_seed(37)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=2, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        target_policy='ema_center', norm_policy='layer_norm').train()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    metrics = model(views, masks)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert model.dino_center.abs().sum() > 0
    assert model.ibot_center.abs().sum() > 0
    metrics['loss'].backward()
    assert model.student.backbone.patch_embed.weight.grad.abs().sum() > 0
    assert 'dino_center' in model.state_dict()


def test_ibot_binary_input_shortcut_preserves_patch_difference_when_mlp_silences():
    torch.manual_seed(41)
    head = SpikingProjectionHead(16, hidden=32, bottleneck=8,
                                 prototypes=12, shortcut='input_or').eval()
    with torch.no_grad():
        head.fc2.weight.zero_()
        spikes = torch.zeros(4, 2, 16)
        spikes[:, 0, :4] = 1
        spikes[:, 1, 4:8] = 1
        out = head(spikes)
    assert out.shape == (2, 12)
    assert not torch.equal(out[0], out[1])
    functional.reset_net(head)
    head.shortcut = 'none'
    without = head(spikes)
    torch.testing.assert_close(without[0], without[1])


def test_spiking_head_diversity_retention_penalizes_lost_token_variation():
    torch.manual_seed(47)
    input_spikes = (torch.rand(4, 16, 12) < .4).float()
    weak_output = (input_spikes[..., :8] * .1).requires_grad_()
    strong_output = input_spikes[..., :8]
    weak_gap = spike_diversity_retention(input_spikes, weak_output)
    strong_gap = spike_diversity_retention(input_spikes, strong_output)
    assert weak_gap > strong_gap >= 0
    weak_gap.backward()
    assert weak_output.grad.abs().sum() > 0
    spatial_input = torch.tensor([[[1., 0.], [0., 1.],
                                   [1., 0.], [0., 1.]]])
    position_only_output = torch.tensor([[[1., 0.], [1., 0.],
                                          [0., 1.], [0., 1.]]])
    positions = torch.tensor([0, 0, 1, 1])
    assert spike_diversity_retention(spatial_input,
                                     position_only_output) == 0
    assert spike_diversity_retention(
        spatial_input, position_only_output, positions, 2) > .4


def test_grouped_spike_diversity_zero_variance_has_finite_gradient():
    input_spikes = torch.tensor([[[0., 1.], [1., 0.],
                                  [0., 1.], [1., 0.]]])
    output_spikes = torch.zeros_like(input_spikes).requires_grad_()
    gap = spike_diversity_retention(input_spikes, output_spikes,
                                    torch.tensor([0, 0, 1, 1]), 2)
    assert gap > 0
    gap.backward()
    assert torch.isfinite(output_spikes.grad).all()


def test_ibot_diversity_term_is_finite_and_trainable():
    torch.manual_seed(49)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=2, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        norm_policy='layer_norm', ibot_diversity_weight=1.,
        ibot_diversity_source='visible').train()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    metrics = model(views, masks)
    assert 'ibot_diversity_gap' in metrics
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics['loss'].backward()
    assert model.student.ibot_head.fc2.weight.grad.abs().sum() > 0


def test_ibot_alignment_diagnostic_compares_matched_and_shuffled_targets():
    torch.manual_seed(51)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        norm_policy='layer_norm').train()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.ones(3, 16, dtype=torch.bool) for _ in range(2)]
    result = model(views, masks, diagnostics=True, context_ablation=True)
    torch.testing.assert_close(result['ibot_context_advantage'],
                               result['ibot_shuffled_patch_ce'] -
                               result['ibot_matched_patch_ce'])
    assert torch.isfinite(result['ibot_context_advantage'])
    assert result['ibot_same_position_pairs'] == 96
    assert torch.isfinite(result['ibot_same_position_advantage'])
    assert torch.isfinite(result['ibot_zero_event_same_position_advantage'])
    torch.testing.assert_close(
        result['teacher_patch_within_position_information'] +
        result['teacher_patch_position_information'],
        result['teacher_patch_target_information'])


def test_aligned_teacher_patch_target_averages_two_event_views():
    torch.manual_seed(53)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        norm_policy='layer_norm',
        patch_teacher_aggregation='matched_cross_view_mean').train()
    views = [torch.rand(4, 2, 2, 32, 32) for _ in range(2)]
    masks = [torch.zeros(2, 16, dtype=torch.bool) for _ in range(2)]
    masks[0][:, [0, 1, 4]] = True
    masks[1][:, [1, 4, 5]] = True
    teacher_logits = []
    handle = model.teacher.ibot_head.register_forward_hook(
        lambda _module, _inputs, output: teacher_logits.append(output.detach()))
    with patch('spike_dino.dinov3.sinkhorn_knopp', wraps=sinkhorn_knopp) as spy:
        metrics = model(views, masks)
    handle.remove()
    union = masks[0] | masks[1]
    expected = torch.cat([
        (teacher_logits[0][mask[union]] +
         teacher_logits[1][mask[union]]) * .5 for mask in masks])
    torch.testing.assert_close(spy.call_args_list[-1].args[0], expected)
    assert all(torch.isfinite(value) for value in metrics.values())
    metrics['loss'].backward()
    assert torch.isfinite(model.student.backbone.patch_embed.weight.grad).all()


def test_partial_time_iBOT_path_backpropagates_with_spatial_patch_targets():
    torch.manual_seed(57)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        norm_policy='layer_norm', residual_policy='layer_scale').train()
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .4 for _ in range(2)]
    temporal = partial_temporal_masks(masks, 4)
    result = model(views, masks, temporal_masks=temporal)
    assert all(torch.isfinite(value) for value in result.values())
    result['loss'].backward()
    assert model.student.backbone.patch_embed.weight.grad.abs().sum() > 0


def test_visible_patch_diversity_does_not_change_masked_logits_with_layer_norm():
    torch.manual_seed(53)
    model = SpikingDinoV3Prototype(
        image_size=32, patch_size=8, dim=48, depth=1, heads=4,
        head_hidden=96, bottleneck=24, prototypes=32,
        norm_policy='layer_norm').train()
    view = torch.rand(4, 3, 2, 32, 32)
    mask = torch.rand(3, 16) < .4
    masked_only = model.student(view, mask, mask)['patch_scores']
    functional.reset_net(model.student)
    for source in ('visible', 'visible_position'):
        functional.reset_net(model.student)
        with_visible = model.student(view, mask, mask,
                                     patch_diversity=True,
                                     patch_diversity_source=source)
        torch.testing.assert_close(masked_only, with_visible['patch_scores'],
                                   rtol=0, atol=0)
        assert torch.isfinite(with_visible['patch_diversity_gap'])


def test_separate_unmasked_cls_keeps_ibot_targets_and_trains_both_heads():
    torch.manual_seed(103)
    kwargs = dict(image_size=32, patch_size=8, dim=48, depth=2, heads=4,
                  head_hidden=96, bottleneck=24, prototypes=32,
                  norm_policy='layer_norm', residual_policy='layer_scale',
                  residual_gradient_policy='full')
    shared = SpikingDinoV3Prototype(**kwargs).train()
    separate = SpikingDinoV3Prototype(
        **kwargs, global_cls_policy='unmasked_separate',
        ibot_residual_gradient_policy='bounded_branch').train()
    separate.load_state_dict(shared.state_dict())
    views = [torch.rand(4, 3, 2, 32, 32) for _ in range(2)]
    masks = [torch.rand(3, 16) < .35 for _ in range(2)]
    temporal = partial_temporal_masks(masks, 4)
    shared_result = shared(views, masks, temporal_masks=temporal)
    separate_result = separate(views, masks, temporal_masks=temporal)
    torch.testing.assert_close(shared_result['ibot'], separate_result['ibot'])
    assert torch.isfinite(separate_result['dino'])
    separate_result['loss'].backward()
    assert separate.student.dino_head.fc1.weight.grad is not None
    assert separate.student.ibot_head.fc1.weight.grad is not None
    assert torch.isfinite(separate.student.backbone.patch_embed.weight.grad).all()
