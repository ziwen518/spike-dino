"""Locate where visible-event information reaches or misses masked patches."""
import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from spike_dino.data import ManifestEventDataset
from spike_dino.pretrain_v3 import FixedEpochBatchSampler, fixed_count_masks
from scripts.diagnose_v3_representation import model_from_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--masked-context-bridge', choices=('none', 'spike_or'),
                   default='none')
    p.add_argument('--attention-variant-override',
                   choices=('matrix_visible_keys',))
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output file')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    c = payload['config']
    c['masked_context_bridge'] = args.masked_context_bridge
    torch.manual_seed(c['seed'])
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.student.backbone
    if args.attention_variant_override is not None:
        for block in backbone.blocks:
            block.attn.variant = args.attention_variant_override
    dataset = ManifestEventDataset(
        c['dataset_root'], c['manifest_path'], seed=c['seed'],
        time_steps=c['time_steps'], size=c['size'], local_crops=0,
        global_crop_scale=c['global_crop_scale'],
        local_crop_scale=c['local_crop_scale'],
        global_temporal_crop_range=c['global_time_range'],
        local_temporal_crop_range=c['local_time_range'],
        global_view_policy=c['global_view_policy'])
    sampler = FixedEpochBatchSampler(len(dataset), c['batch_size'],
                                     c['seed'], 0, 1)
    batch = next(iter(DataLoader(dataset, batch_sampler=sampler,
                                 num_workers=0,
                                 generator=torch.Generator().manual_seed(
                                     c['seed'] + 1))))
    frames = batch[0].permute(1, 0, 2, 3, 4).to(args.device)
    mask = fixed_count_masks(
        c['batch_size'], (c['size'] // c['patch_size']) ** 2,
        c['mask_min'], c['mask_max'], args.device,
        c['mask_probability'])[0]
    captures = {'real': {}, 'zero': {}}
    current = ['real']
    handles = []
    def hook(name, selector):
        def save(_module, _input, output):
            captures[current[0]][name] = selector(output).detach().float().cpu()
        return save
    handles.append(backbone.token_spike.register_forward_hook(
        hook('token_masked', lambda out: out[:, :, 1:][:, mask])))
    handles.append(backbone.token_spike.register_forward_hook(
        hook('token_visible', lambda out: out[:, :, 1:][:, ~mask])))
    for index, block in enumerate(backbone.blocks):
        handles.append(block.attn.register_forward_hook(
            hook(f'attn_{index}', lambda out: out[:, :, 1:][:, mask])))
        handles.append(block.register_forward_hook(
            hook(f'block_{index}', lambda out: out[:, :, 1:][:, mask])))
    handles.append(backbone.patch_output_spike.register_forward_hook(
        hook('final_masked', lambda out: out[:, mask])))
    with torch.no_grad():
        for label, events in (('real', frames), ('zero', torch.zeros_like(frames))):
            current[0] = label
            functional.reset_net(backbone)
            backbone(events, mask)
            functional.reset_net(backbone)
    for handle in handles:
        handle.remove()
    names = list(captures['real'])
    results = {}
    for name in names:
        real = captures['real'][name]
        zero = captures['zero'][name]
        results[name] = {'real_mean': float(real.mean()),
                         'zero_mean': float(zero.mean()),
                         'event_removal_l1': float((real - zero).abs().mean()),
                         'changed_fraction': float((real != zero).float().mean())}
    result = {'checkpoint': str(args.checkpoint),
              'masked_context_bridge': args.masked_context_bridge,
              'attention_variant_override': args.attention_variant_override,
              'mask_tokens': int(mask.sum()),
              'training_split_only': True, 'layers': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
