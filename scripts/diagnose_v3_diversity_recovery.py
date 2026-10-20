"""Controlled short forks from one collapsed iBOT checkpoint on training data."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spike_dino.data import ManifestEventDataset
from spike_dino.dinov3 import SpikingDinoV3Prototype
from spike_dino.pretrain_v3 import FixedEpochBatchSampler, fixed_count_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weight', type=float, choices=(0., 1.), required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists() or args.steps < 1:
        p.error('Choose a new output file and positive step count')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    c = payload['config']
    torch.manual_seed(c['seed'])
    torch.set_num_threads(2)
    model = SpikingDinoV3Prototype(
        image_size=c['size'], patch_size=c['patch_size'],
        dim=c['dim'], depth=c['depth'], heads=c['heads'],
        block_variant=c.get('block_variant', 'standard'),
        attention_variant=c['attention_variant'],
        cls_aggregation=c['cls_aggregation'], residual_policy=c['residual_policy'],
        head_hidden=c['head_hidden'], head_layers=c['head_layers'],
        prototype_readout=c['prototype_readout'], bottleneck=c['bottleneck'],
        prototypes=c['prototypes'], norm_policy=c['norm_policy'],
        ibot_reduction=c['ibot_reduction'], balance_mode=c['balance_mode'],
        target_policy=c.get('target_policy', 'sinkhorn'),
        patch_teacher_aggregation=c.get('patch_teacher_aggregation', 'single_view'),
        ibot_diversity_weight=args.weight,
        ibot_diversity_source='visible').to(args.device).train()
    model.load_state_dict(payload['model'])
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=c['lr'])
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['cpu_rng'])
    if str(args.device).startswith('cuda'):
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    dataset = ManifestEventDataset(
        c['dataset_root'], c['manifest_path'], seed=c['seed'],
        time_steps=c['time_steps'], size=c['size'],
        local_crops=c['local_crops'], local_size=c['local_size'],
        global_crop_scale=c['global_crop_scale'],
        local_crop_scale=c['local_crop_scale'],
        global_temporal_crop_range=c['global_time_range'],
        local_temporal_crop_range=c['local_time_range'],
        global_view_policy=c['global_view_policy'])
    start = payload['steps']
    sampler = FixedEpochBatchSampler(len(dataset), c['batch_size'],
                                     c['seed'], start, start + args.steps)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(c['seed'] + 1),
                        pin_memory=str(args.device).startswith('cuda'))
    rows = []
    for step, batch in enumerate(loader, start=start + 1):
        views = [v.permute(1, 0, 2, 3, 4).to(args.device) for v in batch]
        masks = fixed_count_masks(
            c['batch_size'], (c['size'] // c['patch_size']) ** 2,
            c['mask_min'], c['mask_max'], args.device, c['mask_probability'])
        optimizer.zero_grad(set_to_none=True)
        metrics = model(views, masks)
        metrics['loss'].backward()
        if not bool(torch.stack([
                torch.isfinite(parameter.grad).all()
                for parameter in model.student.parameters()
                if parameter.grad is not None]).all()):
            raise FloatingPointError(f'Non-finite gradient at step {step}')
        optimizer.step()
        model.after_optimizer_step()
        row = {'step': step, **{key: float(value.detach())
                               for key, value in metrics.items()}}
        rows.append(row)
        if step == start + 1 or step % 10 == 0 or step == start + args.steps:
            print(json.dumps({key: row[key] for key in (
                'step', 'ibot', 'teacher_patch_target_information',
                'masked_student_logit_std', 'teacher_target_information')}
                | {'ibot_diversity_gap': row.get('ibot_diversity_gap', 0.)}),
                flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'parent_checkpoint': str(args.checkpoint),
        'parent_steps': start, 'new_steps': args.steps,
        'auxiliary_weight': args.weight,
        'auxiliary_source': 'visible', 'training_split_only': True,
        'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
