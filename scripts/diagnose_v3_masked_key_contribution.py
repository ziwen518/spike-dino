"""Measure how masked keys contribute to binary KᵀV attention currents.

Read-only, one deterministic training batch. No label or official test access.
The calculation precedes the attention context LIF and learned projection.
"""

import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from scripts.diagnose_v3_representation import model_from_checkpoint
from spike_dino.data import ManifestEventDataset
from spike_dino.pretrain_v3 import FixedEpochBatchSampler, fixed_count_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.student.backbone
    if any(block.attn.variant not in ('matrix_recode', 'matrix_legacy',
                                      'matrix_value_skip')
           for block in backbone.blocks):
        parser.error('Audit currently supports KᵀV matrix attention only')
    dataset = ManifestEventDataset(
        config['dataset_root'], config['manifest_path'],
        seed=config['seed'], time_steps=config['time_steps'],
        size=config['size'], local_crops=0,
        global_crop_scale=config['global_crop_scale'],
        local_crop_scale=config['local_crop_scale'],
        global_temporal_crop_range=config['global_time_range'],
        local_temporal_crop_range=config['local_time_range'],
        global_view_policy=config['global_view_policy'])
    sampler = FixedEpochBatchSampler(len(dataset), config['batch_size'],
                                     config['seed'], 0, 1)
    batch = next(iter(DataLoader(
        dataset, batch_sampler=sampler, num_workers=0,
        generator=torch.Generator().manual_seed(config['seed'] + 1))))
    frames = batch[0].permute(1, 0, 2, 3, 4).to(args.device)
    masks = fixed_count_masks(
        config['batch_size'], (config['size'] // config['patch_size']) ** 2,
        config['mask_min'], config['mask_max'], args.device,
        config['mask_probability'])
    patch_mask = masks[0]
    token_mask = torch.cat((torch.zeros(config['batch_size'], 1,
                                        dtype=torch.bool, device=args.device),
                            patch_mask), dim=1)
    captured = [{} for _ in backbone.blocks]
    handles = []
    for index, block in enumerate(backbone.blocks):
        for name in ('q', 'k', 'v'):
            def hook(_module, _inputs, output, index=index, name=name):
                captured[index][name] = output.detach()
            handles.append(getattr(block.attn, name).register_forward_hook(hook))
    with torch.no_grad():
        functional.reset_net(backbone)
        backbone(frames, patch_mask)
        functional.reset_net(backbone)
    for handle in handles:
        handle.remove()
    rows = []
    with torch.no_grad():
        for index, (block, item) in enumerate(zip(backbone.blocks, captured)):
            t, b, n, c = item['q'].shape
            h, d = block.attn.heads, block.attn.head_dim
            reshape = lambda value: value.reshape(t, b, n, h, d).permute(0, 1, 3, 2, 4)
            q, k, v = (reshape(item[name]).float() for name in ('q', 'k', 'v'))
            selector = token_mask[None, :, None, :, None]
            km, vm = k * selector, v * selector
            context_all = (k.transpose(-1, -2) @ v) / n
            context_masked = (km.transpose(-1, -2) @ vm) / n
            queries_all = q @ context_all
            queries_masked = q @ context_masked
            # q_i @ k_i times v_i is the exact diagonal/self contribution
            # to query i before the context LIF; no N×N tensor is needed.
            queries_self = (q * k).sum(-1, keepdim=True) * v / n
            masked_queries = selector.expand_as(queries_all)
            all_sum = queries_all[masked_queries].sum()
            masked_sum = queries_masked[masked_queries].sum()
            self_sum = queries_self[masked_queries].sum()
            visible_selector = (~token_mask)[None, :, None, :, None]
            rows.append({
                'block': index,
                'masked_token_fraction': float(token_mask.float().mean()),
                'k_rate_masked': float(k[selector.expand_as(k)].mean()),
                'k_rate_visible': float(k[visible_selector.expand_as(k)].mean()),
                'v_rate_masked': float(v[selector.expand_as(v)].mean()),
                'v_rate_visible': float(v[visible_selector.expand_as(v)].mean()),
                'masked_key_fraction_of_masked_query_current':
                    float(masked_sum / all_sum.clamp_min(1e-9)),
                'self_key_fraction_of_masked_query_current':
                    float(self_sum / all_sum.clamp_min(1e-9)),
                'other_masked_key_fraction_of_masked_query_current':
                    float((masked_sum - self_sum) /
                          all_sum.clamp_min(1e-9)),
                'all_masked_query_current_mean':
                    float(queries_all[masked_queries].mean()),
                'masked_key_current_mean':
                    float(queries_masked[masked_queries].mean()),
            })
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_steps': payload['steps'],
        'dataset': config['dataset'],
        'training_split_only': True,
        'measurement_stage': 'linear Q@(K^T V)/N before context LIF; fractions do not decompose post-LIF attention',
        'global_view_index': 0,
        'masked_patch_tokens': int(patch_mask.sum()),
        'total_patch_tokens': int(patch_mask.numel()),
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
