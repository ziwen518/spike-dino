"""Same-checkpoint iBOT objective audit with and without masked K/V.

Uses one deterministic, unlabeled training batch. The attention override is
read-only and untrained; a positive event response does not prove accuracy.
"""

import argparse
import json
from pathlib import Path

import torch
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
    model.ibot_diversity_weight = config.get('ibot_diversity_weight', 0.)
    model.ibot_diversity_source = config.get('ibot_diversity_source', 'masked')
    if config.get('mask_observation_policy', 'full_spatial') != 'full_spatial':
        parser.error('This audit requires full-spatial masks')
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
    views = [value.permute(1, 0, 2, 3, 4).to(args.device) for value in batch[:2]]
    masks = fixed_count_masks(
        config['batch_size'], (config['size'] // config['patch_size']) ** 2,
        config['mask_min'], config['mask_max'], args.device,
        config['mask_probability'])
    modes = {}
    with torch.no_grad():
        for name in ('original', 'visible_keys'):
            variant = ('matrix_visible_keys' if name == 'visible_keys'
                       else config['attention_variant'])
            for block in model.student.backbone.blocks:
                block.attn.variant = variant
            metrics = model(views, masks, diagnostics=True,
                            context_ablation=True)
            modes[name] = {key: float(value) for key, value in metrics.items()}
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_steps': payload['steps'],
        'dataset': config['dataset'],
        'training_split_only': True,
        'masked_tokens': int(sum(mask.sum() for mask in masks)),
        'modes': modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
