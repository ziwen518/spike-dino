"""Read-only representation-path audit on training recordings only.

Compare matched augmented views, shuffled views, spiking layer activity and
teacher assignment information. This does not train or inspect the test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional

from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import (EventCLRViewTransform, ManifestEventDataset,
                             SENSOR_SIZES, read_event_record)
from spike_dino.dinov3 import SpikingDinoV3Prototype, sinkhorn_knopp
from spike_dino.pretrain_v3 import linear_warmup_value


def model_from_checkpoint(payload):
    c = payload['config']
    teacher_temp = linear_warmup_value(
        max(payload['steps'] - 1, 0), c.get('teacher_temp_start', .04),
        c.get('teacher_temp_final', .04),
        c.get('teacher_temp_warmup_steps', 0))
    model = SpikingDinoV3Prototype(
        image_size=c['size'], patch_size=c.get('patch_size', 16),
        dim=c['dim'], depth=c['depth'], heads=c['heads'],
        block_variant=c.get('block_variant', 'standard'),
        attention_variant=c.get('attention_variant', 'matrix_legacy'),
        cls_aggregation=c.get('cls_aggregation', 'attention_only'),
        residual_policy=c.get('residual_policy', 'accumulate'),
        masked_context_bridge=c.get('masked_context_bridge', 'none'),
        residual_gradient_policy=c.get('residual_gradient_policy', 'full'),
        global_cls_policy=c.get('global_cls_policy', 'masked'),
        ibot_residual_gradient_policy=c.get(
            'ibot_residual_gradient_policy', 'inherit'),
        head_hidden=c['head_hidden'], head_layers=c.get('head_layers', 2),
        dino_shortcut=c.get('dino_shortcut', 'none'),
        prototype_readout=c.get('prototype_readout', 'cosine'),
        bottleneck=c['bottleneck'], prototypes=c['prototypes'],
        dino_prototypes=c.get('dino_prototypes'),
        ibot_prototypes=c.get('ibot_prototypes'),
        dino_hidden=c.get('dino_hidden'), ibot_hidden=c.get('ibot_hidden'),
        dino_bottleneck=c.get('dino_bottleneck'),
        ibot_bottleneck=c.get('ibot_bottleneck'),
        norm_policy=c.get('norm_policy', 'ema_bn'),
        target_policy=c.get('target_policy', 'sinkhorn'),
        patch_teacher_aggregation=c.get('patch_teacher_aggregation',
                                        'single_view'),
        gram_weight=c.get('gram_weight', 1.))
    model.teacher_temp = teacher_temp
    if any(key.startswith('gram_teacher.') for key in payload['model']):
        model.enable_gram_refinement()
    model.load_state_dict(payload['model'])
    return model


def activity(tensor):
    value = tensor.detach().float()
    return {'mean': float(value.mean()),
            'sample_std': float(value.mean(0).std(0, unbiased=False).mean()),
            'zero_fraction': float((value == 0).float().mean()),
            'one_fraction': float((value == 1).float().mean())}


def matched_cosine(a, b):
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    matrix = a @ b.T
    return {'matched': float(matrix.diag().mean()),
            'mismatched': float((matrix.sum() - matrix.diag().sum()) /
                                (matrix.numel() - len(matrix))),
            'retrieval_top1': float((matrix.argmax(-1) ==
                                    torch.arange(len(matrix), device=matrix.device)).float().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--manifest-path', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--view-policy', choices=('strong', 'temporal', 'spatial',
                                             'roll', 'crop', 'flip', 'polarity',
                                             'none', 'shared_core',
                                             'shared_temporal_windows',
                                             'shared_time_independent_space',
                                             'shared_time_shifted_space',
                                             'shared_core_nested_local'),
                   default='strong')
    p.add_argument('--zero-input-audit', action='store_true')
    p.add_argument('--temporal-order-audit', action='store_true')
    p.add_argument('--attention-path-audit', action='store_true')
    p.add_argument('--attention-variant-override',
                   choices=('matrix_recode', 'channel_gate', 'token_pair',
                            'token_pair_centered', 'matrix_value_skip',
                            'matrix_visible_keys', 'matrix_cls_wta',
                            'matrix_cls_wta_temporal', 'matrix_count',
                            'spikformer_count'))
    p.add_argument('--cls-aggregation-override',
                   choices=('event_weighted_pool', 'spike_quadrant_pool'))
    p.add_argument('--residual-scale-multiplier', type=float, default=1.)
    p.add_argument('--residual-scale-block-start', type=int, default=0)
    p.add_argument('--residual-scale-block-stop', type=int)
    p.add_argument('--residual-policy-override',
                   choices=('layer_scale', 'centered_layer_scale',
                            'moment_preserving'))
    p.add_argument('--readout-stage-audit', action='store_true')
    p.add_argument('--dino-head-shortcut', choices=('none', 'input_or'),
                   default=None)
    p.add_argument('--teacher-temp-override', type=float)
    p.add_argument('--cls-aggregation', choices=('attention_only', 'spike_pool',
                                                'event_weighted_pool',
                                                'spike_quadrant_pool'),
                   default='attention_only')
    p.add_argument('--residual-policy', choices=('accumulate', 'recode_norm',
                                                 'layer_scale'),
                   default='accumulate')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    torch.set_num_threads(2)
    manifest = load_manifest(args.manifest_path)
    payload = (torch.load(args.checkpoint, map_location='cpu', weights_only=True)
               if args.checkpoint else None)
    if payload:
        c = payload['config']
        if c.get('manifest_sha256') is not None and c['manifest_sha256'] != manifest['sha256']:
            p.error('Checkpoint manifest does not match supplied manifest')
        torch.manual_seed(c['seed'])
        model = model_from_checkpoint(payload)
    else:
        torch.manual_seed(42)
        model = SpikingDinoV3Prototype(
            image_size=128, patch_size=8, dim=384, depth=12, heads=6,
            head_hidden=2048, bottleneck=256, prototypes=65536,
            norm_policy='layer_norm', cls_aggregation=args.cls_aggregation,
            residual_policy=args.residual_policy)
        c = {'size': 128, 'time_steps': 4, 'seed': 42}
    is_evidence = getattr(model.teacher.backbone, 'block_variant', 'standard') == 'evidence'
    if is_evidence and (args.attention_path_audit or args.attention_variant_override or
                        args.residual_policy_override or
                        args.residual_scale_multiplier != 1):
        p.error('Evidence blocks support generic CLS/patch path audits; '
                'use their dedicated current audit for internal operator controls')
    if args.attention_variant_override:
        for block in model.teacher.backbone.blocks:
            block.attn.variant = args.attention_variant_override
    if args.cls_aggregation_override:
        if not hasattr(model.teacher.backbone, 'cls_pool_spike'):
            p.error('CLS aggregation override requires existing pool weights')
        model.teacher.backbone.cls_aggregation = args.cls_aggregation_override
    if args.dino_head_shortcut is not None:
        model.teacher.dino_head.shortcut = args.dino_head_shortcut
    if args.teacher_temp_override is not None:
        if args.teacher_temp_override <= 0:
            p.error('Teacher temperature must be positive')
        model.teacher_temp = args.teacher_temp_override
    if args.residual_policy_override:
        model.teacher.backbone.residual_policy = args.residual_policy_override
        for block in model.teacher.backbone.blocks:
            if not hasattr(block, 'attn_scale'):
                p.error('Residual override requires LayerScale weights')
            block.residual_policy = args.residual_policy_override
    if args.residual_scale_multiplier <= 0:
        p.error('Residual scale multiplier must be positive')
    block_stop = (len(model.teacher.backbone.blocks)
                  if args.residual_scale_block_stop is None else
                  args.residual_scale_block_stop)
    if not 0 <= args.residual_scale_block_start < block_stop <= len(
            model.teacher.backbone.blocks):
        p.error('Residual scale block range must fit the backbone')
    if args.residual_scale_multiplier != 1:
        with torch.no_grad():
            for block in model.teacher.backbone.blocks[
                    args.residual_scale_block_start:block_stop]:
                if not hasattr(block, 'attn_scale'):
                    p.error('Residual scale multiplier requires LayerScale blocks')
                block.attn_scale.mul_(args.residual_scale_multiplier)
                block.mlp_scale.mul_(args.residual_scale_multiplier)
    model = model.to(args.device).eval()
    records = manifest['train']
    indices = np.linspace(0, len(records) - 1, args.batch_size, dtype=int)
    families = {'strong': ('temporal', 'spatial', 'polarity'),
                'temporal': ('temporal',), 'spatial': ('spatial',),
                'roll': ('roll',), 'crop': ('crop',),
                'flip': ('flip',), 'polarity': ('polarity',),
                'none': (), 'shared_core': (),
                'shared_temporal_windows': (),
                'shared_time_independent_space': (),
                'shared_time_shifted_space': (),
                'shared_core_nested_local': ()}[args.view_policy]
    transform = EventCLRViewTransform(
        time_steps=c['time_steps'], size=c['size'],
        crop_scale=(.32, 1.), temporal_crop_range=(.5, 1.),
        families=families,
        sensor_size=SENSOR_SIZES[manifest['dataset']])
    views = [[], []]
    shared_dataset = (ManifestEventDataset(
        args.dataset_root, args.manifest_path, time_steps=c['time_steps'],
        size=c['size'], local_crops=0, global_crop_scale=(.32, 1.),
        global_temporal_crop_range=(.5, 1.),
        global_view_policy=args.view_policy)
        if args.view_policy in ('shared_core', 'shared_temporal_windows',
                                'shared_time_independent_space',
                                'shared_time_shifted_space',
                                'shared_core_nested_local')
        else None)
    for index in indices:
        if shared_dataset is not None:
            for j, view in enumerate(shared_dataset[(int(index), 0)]):
                views[j].append(view)
            continue
        record = records[int(index)]
        events = read_event_record(args.dataset_root / 'raw' / record['path'],
                                   manifest['dataset'])
        for j in range(2):
            seed = 1_000_003 * c['seed'] + int(index) * 1000 + j
            views[j].append(transform(events, np.random.default_rng(seed),
                                      torch.Generator().manual_seed(seed)))
    views = [torch.stack(v, dim=1).to(args.device) for v in views]
    observations = {}
    cls_path = {}
    patch_path = {}
    token_patch_path = {}
    attention_path = {}
    attention_input_path = {}
    attention_preproj_path = {}
    attention_projection_path = {}
    readout_stage_path = {}
    handles = []
    backbone = model.teacher.backbone
    head = model.teacher.dino_head
    modules = {'token': backbone.token_spike,
               'cls_output': backbone.cls_output_spike,
               'head_hidden_1': head.lif1,
               'head_bottleneck': head.lif2}
    if head.lif_mid is not None:
        modules['head_hidden_2'] = head.lif_mid
    for i, block in enumerate(backbone.blocks):
        if is_evidence:
            modules[f'block_{i:02d}_q'] = block.q.spike
            modules[f'block_{i:02d}_attention_on'] = block.evidence_on
            modules[f'block_{i:02d}_attention_off'] = block.evidence_off
            continue
        modules[f'block_{i:02d}_q'] = block.attn.q.spike
        modules[f'block_{i:02d}_context'] = block.attn.context_spike
        modules[f'block_{i:02d}_attention_output'] = (
            block.attn.count_output_spike
            if block.attn.variant == 'spikformer_count' else
            block.attn.attention_output_spike)
        if block.attn.variant in ('token_pair', 'token_pair_centered'):
            modules[f'block_{i:02d}_pair'] = block.attn.pair_spike
            modules[f'block_{i:02d}_pair_value'] = block.attn.pair_value_spike
    current = [0]
    for name, module in modules.items():
        def record(_module, _input, output, name=name):
            observations.setdefault(str(current[0]), {})[name] = activity(output)
        handles.append(module.register_forward_hook(record))
    def record_token(_module, _input, output):
        token_patch_path[str(current[0])] = (
            output[:, :, 1:].detach().float().mean((0, 2)).cpu())
    handles.append(backbone.token_spike.register_forward_hook(record_token))
    for i, block in enumerate(backbone.blocks):
        def record_cls(_module, _input, output, i=i):
            cls_path.setdefault(str(current[0]), {})[str(i)] = (
                output[:, :, 0].detach().float().mean(0).cpu())
            patch_path.setdefault(str(current[0]), {})[str(i)] = (
                output[:, :, 1:].detach().float().mean((0, 2)).cpu())
        handles.append(block.register_forward_hook(record_cls))
        if args.attention_path_audit:
            def record_attention(_module, _input, output, i=i):
                attention_path.setdefault(str(current[0]), {})[str(i)] = {
                    'cls': output[:, :, 0].detach().float().mean(0).cpu(),
                    'patch': output[:, :, 1:].detach().float().mean((0, 2)).cpu()}
            def record_attention_input(_module, _input, output, i=i):
                attention_input_path.setdefault(str(current[0]), {})[str(i)] = {
                    'cls': output[:, :, 0].detach().float().mean(0).cpu(),
                    'patch': output[:, :, 1:].detach().float().mean((0, 2)).cpu()}
            handles.append(block.attn.register_forward_hook(record_attention))
            handles.append(block.attn_input.register_forward_hook(
                record_attention_input))
            preproj = (block.attn.count_output_spike
                       if block.attn.variant == 'spikformer_count' else
                       block.attn.pair_value_spike
                       if block.attn.variant in ('token_pair',
                                                 'token_pair_centered') else
                       block.attn.attention_output_spike
                       if block.attn.variant in ('matrix_recode',
                                                 'matrix_value_skip',
                                                 'matrix_cls_wta',
                                                 'matrix_cls_wta_temporal',
                                                 'matrix_visible_keys',
                                                 'matrix_count') else None)
            if preproj is not None:
                def record_preproj(_module, _input, output, i=i):
                    # [T,B,H,N,D] -> [B,H*D], retaining individual channels.
                    attention_preproj_path.setdefault(str(current[0]), {})[str(i)] = {
                        'cls': output[:, :, :, 0].detach().float().mean(0).flatten(1).cpu(),
                        'patch': output[:, :, :, 1:].detach().float().mean((0, 3)).flatten(1).cpu()}
                handles.append(preproj.register_forward_hook(record_preproj))
            for stage, projection_module in (
                    ('q', block.attn.q.spike),
                    ('k', block.attn.k.spike),
                    ('v', block.attn.v.spike),
                    ('linear', block.attn.proj.linear),
                    ('norm', block.attn.proj.norm)):
                def record_projection(_module, _input, output, i=i,
                                      stage=stage):
                    attention_projection_path.setdefault(
                        str(current[0]), {}).setdefault(str(i), {})[stage] = {
                        'cls': output[:, :, 0].detach().float().mean(0).cpu(),
                        'patch': output[:, :, 1:].detach().float().mean((0, 2)).cpu()}
                handles.append(projection_module.register_forward_hook(
                    record_projection))
    if args.readout_stage_audit:
        def record_cls_norm(_module, _input, output):
            readout_stage_path.setdefault(str(current[0]), {})['cls_norm'] = (
                output.reshape(c['time_steps'], len(indices), -1)
                .detach().float().mean(0).cpu())
        def record_patch_norm(_module, _input, output):
            readout_stage_path.setdefault(str(current[0]), {})['patch_norm'] = (
                output.detach().float().mean((0, 2)).cpu())
        handles.append(backbone.cls_output_norm.register_forward_hook(
            record_cls_norm))
        handles.append(backbone.patch_output_norm.register_forward_hook(
            record_patch_norm))
    outputs = []
    with torch.no_grad():
        for j, view in enumerate(views):
            current[0] = j
            functional.reset_net(model.teacher)
            outputs.append(model.teacher(view, compute_patch_head=False))
            functional.reset_net(model.teacher)
        if args.zero_input_audit:
            current[0] = 2
            zero_output = model.teacher(torch.zeros_like(views[0]),
                                        compute_patch_head=False)
            functional.reset_net(model.teacher)
        if args.temporal_order_audit:
            current[0] = 3
            reverse_output = model.teacher(views[0].flip(0),
                                           compute_patch_head=False)
            functional.reset_net(model.teacher)
    for handle in handles:
        handle.remove()
    cls = [o['cls_spikes'].float().mean(0) for o in outputs]
    logits = [o['cls_scores'] for o in outputs]
    targets = sinkhorn_knopp(torch.cat(logits), model.teacher_temp)
    marginal = targets.mean(0)
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(-1).mean()
    information = -(marginal * marginal.clamp_min(1e-12).log()).sum() - entropy
    input_flat = [v.permute(1, 0, 2, 3, 4).reshape(len(indices), -1)
                  for v in views]
    result = {'dataset': manifest['dataset'], 'records': len(indices),
              'block_variant': getattr(backbone, 'block_variant', 'standard'),
              'view_policy': args.view_policy,
              'checkpoint_steps': payload['steps'] if payload else 0,
              'checkpoint': str(args.checkpoint) if args.checkpoint else None,
              'attention_variant_override': args.attention_variant_override,
              'cls_aggregation_override': args.cls_aggregation_override,
              'residual_scale_multiplier': args.residual_scale_multiplier,
              'residual_scale_block_range': [args.residual_scale_block_start,
                                             block_stop],
              'residual_policy_override': args.residual_policy_override,
              'dino_head_shortcut': model.teacher.dino_head.shortcut,
              'teacher_temperature': model.teacher_temp,
              'training_split_only': True,
              'input_view_cosine': matched_cosine(*input_flat),
              'cls_view_cosine': matched_cosine(*cls),
              'logit_view_cosine': matched_cosine(*logits),
              'logit_sample_std': float(torch.cat(logits).std(0, unbiased=False).mean()),
              'target_information_nat': float(information),
              'target_entropy_nat': float(entropy),
              'activity': observations}
    result['token_patch_view_cosine'] = matched_cosine(
        token_patch_path['0'], token_patch_path['1'])
    result['block_patch_view_cosine'] = [matched_cosine(
        patch_path['0'][str(i)], patch_path['1'][str(i)])
        for i in range(len(backbone.blocks))]
    result['block_cls_view_cosine'] = [matched_cosine(
        cls_path['0'][str(i)], cls_path['1'][str(i)])
        for i in range(len(backbone.blocks))]
    if args.zero_input_audit:
        zero_cls = zero_output['cls_spikes'].float().mean(0)
        real_cls = cls[0]
        result['input_ablation'] = {
            'final_cls_l1': float((real_cls - zero_cls).abs().mean()),
            'final_cls_exact_equal_fraction': float((real_cls == zero_cls).float().mean()),
            'per_block_cls_l1': [float((cls_path['0'][str(i)] -
                                        cls_path['2'][str(i)]).abs().mean())
                                 for i in range(len(backbone.blocks))]}
    if args.temporal_order_audit:
        reverse_cls = reverse_output['cls_spikes'].float().mean(0)
        reverse_logits = reverse_output['cls_scores']
        result['temporal_order_audit'] = {
            'operation': 'reverse the four time bins of the same view',
            'input_l1': float((views[0] - views[0].flip(0)).abs().mean()),
            'final_cls_l1': float((cls[0] - reverse_cls).abs().mean()),
            'final_cls_exact_equal_fraction': float((cls[0] == reverse_cls).float().mean()),
            'final_cls_cosine': matched_cosine(cls[0], reverse_cls),
            'prototype_logits_l1': float((logits[0] - reverse_logits).abs().mean()),
            'per_block_cls_l1': [float((cls_path['0'][str(i)] -
                                        cls_path['3'][str(i)]).abs().mean())
                                 for i in range(len(backbone.blocks))]}
    if args.attention_path_audit:
        result['attention_path_audit'] = []
        for i in range(len(backbone.blocks)):
            key = str(i)
            entry = {'block': i}
            paths = [('input', attention_input_path),
                     ('output', attention_path)]
            if attention_preproj_path:
                paths.append(('preproj', attention_preproj_path))
            for source, path in paths:
                for token in ('cls', 'patch'):
                    first = path['0'][key][token]
                    second = path['1'][key][token]
                    entry[f'{source}_{token}_view_cosine'] = matched_cosine(
                        first, second)
                    entry[f'{source}_{token}_sample_std'] = float(
                        first.std(0, unbiased=False).mean())
                    if args.zero_input_audit:
                        entry[f'{source}_{token}_zero_input_l1'] = float(
                            (first - path['2'][key][token]).abs().mean())
            for stage in ('q', 'k', 'v', 'linear', 'norm'):
                for token in ('cls', 'patch'):
                    first = attention_projection_path['0'][key][stage][token]
                    second = attention_projection_path['1'][key][stage][token]
                    entry[f'projection_{stage}_{token}_view_cosine'] = (
                        matched_cosine(first, second))
                    entry[f'projection_{stage}_{token}_sample_std'] = float(
                        first.std(0, unbiased=False).mean())
            result['attention_path_audit'].append(entry)
    if args.readout_stage_audit:
        result['readout_stage_audit'] = {}
        for name, post in (('cls', cls),
                           ('patch', [o['patch_spikes'].float().mean((0, 2))
                                      for o in outputs])):
            pre = [readout_stage_path[str(j)][name + '_norm']
                   for j in (0, 1)]
            result['readout_stage_audit'][name] = {
                'norm_view_cosine': matched_cosine(*pre),
                'norm_sample_std': float(pre[0].std(0, unbiased=False).mean()),
                'spike_view_cosine': matched_cosine(*post),
                'spike_sample_std': float(post[0].std(0, unbiased=False).mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
