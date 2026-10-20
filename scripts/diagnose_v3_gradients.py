"""Identify non-finite gradients before any optimizer update on real events."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spike_dino.data import ManifestEventDataset
from spike_dino.dinov3 import (SpikingDinoV3Prototype, _independent,
                               sinkhorn_knopp)
from spike_dino.pretrain_v3 import (FixedEpochBatchSampler, fixed_count_masks,
                                    partial_temporal_masks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--anomaly', action='store_true')
    p.add_argument('--residual-policy', choices=('accumulate', 'recode_norm',
                                                 'layer_scale'))
    p.add_argument('--cls-aggregation', choices=('attention_only', 'spike_pool'))
    p.add_argument('--target-policy', choices=('sinkhorn', 'ema_center'))
    p.add_argument('--patch-teacher-aggregation',
                   choices=('single_view', 'matched_cross_view_mean'))
    p.add_argument('--masked-context-bridge', choices=('none', 'spike_or'))
    p.add_argument('--residual-gradient-policy',
                   choices=('full', 'branch_detach', 'masked_branch_detach',
                            'bounded_branch'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--ibot-shortcut', choices=('none', 'input_or'))
    p.add_argument('--ibot-diversity-weight', type=float)
    p.add_argument('--ibot-diversity-source',
                   choices=('masked', 'visible', 'visible_position'))
    p.add_argument('--metrics-only', action='store_true')
    p.add_argument('--gradient-conflict', action='store_true',
                   help='Compare global DINO/KoLeo and local iBOT gradients by backbone block')
    p.add_argument('--teacher-observation-audit', action='store_true',
                   help='Compare full teacher patch targets with observed-only and hidden-only events')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8'))
    if args.residual_policy is not None:
        c['residual_policy'] = args.residual_policy
    if args.cls_aggregation is not None:
        c['cls_aggregation'] = args.cls_aggregation
    if args.target_policy is not None:
        c['target_policy'] = args.target_policy
    if args.patch_teacher_aggregation is not None:
        c['patch_teacher_aggregation'] = args.patch_teacher_aggregation
    if args.masked_context_bridge is not None:
        c['masked_context_bridge'] = args.masked_context_bridge
    if args.residual_gradient_policy is not None:
        c['residual_gradient_policy'] = args.residual_gradient_policy
    if args.ibot_shortcut is not None:
        c['ibot_shortcut'] = args.ibot_shortcut
    if args.ibot_diversity_weight is not None:
        c['ibot_diversity_weight'] = args.ibot_diversity_weight
    if args.ibot_diversity_source is not None:
        c['ibot_diversity_source'] = args.ibot_diversity_source
    torch.manual_seed(c['seed'])
    torch.set_num_threads(2)
    model = SpikingDinoV3Prototype(
        image_size=c['size'], patch_size=c['patch_size'],
        dim=c['dim'], depth=c['depth'], heads=c['heads'],
        block_variant=c.get('block_variant', 'standard'),
        attention_variant=c['attention_variant'],
        cls_aggregation=c['cls_aggregation'],
        residual_policy=c['residual_policy'],
        masked_context_bridge=c.get('masked_context_bridge', 'none'),
        residual_gradient_policy=c.get('residual_gradient_policy', 'full'),
        global_cls_policy=c.get('global_cls_policy', 'masked'),
        ibot_residual_gradient_policy=c.get(
            'ibot_residual_gradient_policy', 'inherit'),
        head_hidden=c['head_hidden'], head_layers=c['head_layers'],
        prototype_readout=c['prototype_readout'],
        bottleneck=c['bottleneck'], prototypes=c['prototypes'],
        norm_policy=c['norm_policy'],
        target_policy=c.get('target_policy', 'sinkhorn'),
        patch_teacher_aggregation=c.get('patch_teacher_aggregation',
                                        'single_view'),
        ibot_shortcut=c.get('ibot_shortcut', 'none'),
        dino_shortcut=c.get('dino_shortcut', 'none'),
        ibot_diversity_weight=c.get('ibot_diversity_weight', 0.),
        ibot_diversity_source=c.get('ibot_diversity_source', 'masked'),
        gram_weight=c.get('gram_weight', 1.)).to(
            args.device).train()
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=True)['model']
        if any(key.startswith('gram_teacher.') for key in state):
            model.enable_gram_refinement()
        model.load_state_dict(state)
    dataset = ManifestEventDataset(
        c['dataset_root'], c['manifest_path'], seed=c['seed'],
        time_steps=c['time_steps'], size=c['size'],
        local_crops=c['local_crops'], local_size=c['local_size'],
        global_crop_scale=c['global_crop_scale'],
        local_crop_scale=c['local_crop_scale'],
        global_temporal_crop_range=c['global_time_range'],
        local_temporal_crop_range=c['local_time_range'],
        global_view_policy=c['global_view_policy'])
    sampler = FixedEpochBatchSampler(len(dataset), c['batch_size'],
                                     c['seed'], 0, 1)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(c['seed'] + 1))
    batch = next(iter(loader))
    views = [v.permute(1, 0, 2, 3, 4).to(args.device) for v in batch]
    masks = fixed_count_masks(c['batch_size'],
                              (c['size'] // c['patch_size']) ** 2,
                              c['mask_min'], c['mask_max'], args.device,
                              c['mask_probability'])
    temporal_masks = (partial_temporal_masks(masks, c['time_steps'])
                      if c.get('mask_observation_policy', 'full_spatial') ==
                      'partial_time' else None)
    if args.teacher_observation_audit:
        if temporal_masks is None:
            raise ValueError('Teacher observation audit needs partial-time masks')
        grid = c['size'] // c['patch_size']
        patch = c['patch_size']
        full_logits = []
        observed_logits = []
        hidden_logits = []
        with torch.no_grad():
            for index, view in enumerate(views[:2]):
                hidden_pixels = temporal_masks[index].reshape(
                    c['time_steps'], c['batch_size'], grid, grid)
                hidden_pixels = hidden_pixels.repeat_interleave(
                    patch, 2).repeat_interleave(patch, 3)
                selected_pixels = masks[index].reshape(
                    c['batch_size'], grid, grid)
                selected_pixels = selected_pixels.repeat_interleave(
                    patch, 1).repeat_interleave(patch, 2)
                observed = view.masked_fill(hidden_pixels[:, :, None], 0)
                hidden_only = view.masked_fill(
                    (selected_pixels[None] & ~hidden_pixels)[:, :, None], 0)
                for target_view, bucket in (
                        (view, full_logits), (observed, observed_logits),
                        (hidden_only, hidden_logits)):
                    bucket.append(_independent(
                        model.teacher, target_view,
                        patch_head_mask=masks[index])['patch_scores'])
            full = sinkhorn_knopp(torch.cat(full_logits), model.teacher_temp)
            observed = sinkhorn_knopp(
                torch.cat(observed_logits), model.teacher_temp)
            hidden = sinkhorn_knopp(
                torch.cat(hidden_logits), model.teacher_temp)
            def divergence(alternative):
                return float((full * (full.clamp_min(1e-12).log() -
                                      alternative.clamp_min(1e-12).log()))
                             .sum(-1).mean())
            result = {
                'full_to_observed_kl': divergence(observed),
                'full_to_hidden_kl': divergence(hidden),
                'full_observed_top1_agreement': float(
                    (full.argmax(-1) == observed.argmax(-1)).float().mean()),
                'full_hidden_top1_agreement': float(
                    (full.argmax(-1) == hidden.argmax(-1)).float().mean()),
                'selected_tokens': int(full.shape[0]),
            }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
        return
    if args.metrics_only:
        activations = {}
        handles = []
        for side, encoder in (('teacher', model.teacher),
                              ('student', model.student)):
            head = encoder.ibot_head
            for name in ('lif1', 'lif_mid', 'lif2'):
                module = getattr(head, name)
                if module is None:
                    continue
                def record(_module, _inputs, output, key=f'{side}_{name}'):
                    activations.setdefault(key, []).append(
                        output.detach().float().mean(0).cpu())
                handles.append(module.register_forward_hook(record))
        with torch.no_grad():
            metrics = model(views, masks, diagnostics=True,
                            context_ablation=True,
                            temporal_masks=temporal_masks)
        for handle in handles:
            handle.remove()
        result = {key: float(value.detach())
                  for key, value in metrics.items()}
        for name, parts in activations.items():
            joined = torch.cat(parts)
            result[name + '_feature_std'] = float(joined.std(0, unbiased=False).mean())
            result[name + '_zero_vector_fraction'] = float((joined == 0).all(-1).float().mean())
        result['teacher_ibot_prototype_std'] = float(
            model.teacher.ibot_head.prototypes.std())
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
        return
    if args.anomaly:
        with torch.autograd.detect_anomaly():
            metrics = model(views, masks, return_components=True,
                            temporal_masks=temporal_masks)
            metrics['_dino_raw'].backward()
        print(json.dumps({'anomaly': 'none'}))
        return
    if args.gradient_conflict:
        metrics = model(views, masks, return_components=True,
                        temporal_masks=temporal_masks)
        selected = [(name, value) for name, value in
                    model.student.named_parameters()
                    if name.startswith('backbone.')]
        values = [value for _, value in selected]
        global_grads = torch.autograd.grad(
            metrics['_dino_raw'] + model.koleo_weight *
            metrics['_koleo_raw'], values, retain_graph=True,
            allow_unused=True)
        local_loss = metrics['_ibot_raw']
        if '_diversity_raw' in metrics:
            local_loss = local_loss + model.ibot_diversity_weight * \
                metrics['_diversity_raw']
        local_grads = torch.autograd.grad(
            local_loss, values, retain_graph=False, allow_unused=True)
        groups = {}
        for (name, _), global_grad, local_grad in zip(
                selected, global_grads, local_grads):
            if global_grad is None or local_grad is None:
                continue
            parts = name.split('.')
            group = ('block_' + parts[2] if len(parts) > 2 and
                     parts[1] == 'blocks' else parts[1])
            accumulator = groups.setdefault(group, [0., 0., 0.])
            a = global_grad.double().flatten()
            b = local_grad.double().flatten()
            accumulator[0] += float(a @ b)
            accumulator[1] += float(a @ a)
            accumulator[2] += float(b @ b)
        result = {group: {'cosine': dot / max((a2 * b2) ** .5, 1e-30),
                          'global_norm': a2 ** .5,
                          'local_norm': b2 ** .5,
                          'local_to_global_norm': (b2 / max(a2, 1e-30)) ** .5}
                  for group, (dot, a2, b2) in groups.items()}
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
        return
    metrics = model(views, masks, return_components=True,
                    temporal_masks=temporal_masks)
    named = list(model.student.named_parameters())
    result = {}
    if 'ibot_diversity_gap' in metrics:
        result['ibot_diversity_gap'] = float(metrics['ibot_diversity_gap'])
    for name, scalar in (('dino', metrics['_dino_raw']),
                         ('ibot', metrics['_ibot_raw']),
                         ('koleo', metrics['_koleo_raw']),
                         ('total', metrics['loss'])):
        grads = torch.autograd.grad(scalar, [value for _, value in named],
                                    retain_graph=True, allow_unused=True)
        bad = []
        max_finite = 0.
        top = []
        for (parameter_name, _value), grad in zip(named, grads):
            if grad is None:
                continue
            finite = torch.isfinite(grad)
            if not finite.all():
                bad.append({'name': parameter_name,
                            'nonfinite': int((~finite).sum()),
                            'size': grad.numel()})
            if finite.any():
                peak = float(grad[finite].abs().max())
                max_finite = max(max_finite, peak)
                top.append((parameter_name, peak))
        result[name] = {'value': float(scalar.detach()),
                        'nonfinite_tensors': len(bad),
                        'first_bad': bad[:3],
                        'max_finite_abs_grad': max_finite,
                        'top_finite': sorted(top, key=lambda row: row[1],
                                             reverse=True)[:8]}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
