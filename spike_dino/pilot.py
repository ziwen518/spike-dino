"""A bounded real-data CIFAR10-DVS training pilot, not a benchmark runner."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import Experiment, diagnostics
from .data import Cifar10DVSPilotDataset


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--method', choices=['dino', 'spike_dino', 'spikeclr', 'temporal_ntxent'], default='dino')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--time-steps', type=int, default=4)
    parser.add_argument('--size', type=int, default=48)
    parser.add_argument('--width', type=float, default=.125)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--augmentation', choices=['pilot', 'eventclr'], default='pilot')
    parser.add_argument('--output-target-rate', type=float, default=None,
                        help='Optional prototype firing-rate target for strict Spike-DINO')
    parser.add_argument('--input-coding', choices=['occupancy', 'bernoulli'], default='occupancy')
    parser.add_argument('--teacher-input-coding', choices=['occupancy','bernoulli'])
    parser.add_argument('--population', type=int, default=1)
    parser.add_argument('--local-crops', type=int, default=0,
                        help='Number of smaller student-only views for DINO methods')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.limit < args.batch_size or args.workers < 0:
        parser.error('Invalid steps, batch size, sample limit, or worker count')
    if not 0 <= args.local_crops <= 6 or (args.local_crops and
        (args.method not in ('dino', 'spike_dino') or args.augmentation != 'eventclr')):
        parser.error('Local crops require DINO and the EventCLR view policy')
    if args.population < 1 or (args.population != 1 and args.method != 'spike_dino'):
        parser.error('Population coding requires strict Spike-DINO')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = args.dataset_root / f'split_seed{args.seed}.json'
    if not split_path.is_file():
        parser.error(f'Missing fixed split manifest: {split_path}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Cifar10DVSPilotDataset(args.dataset_root, seed=args.seed,
                                     time_steps=args.time_steps, size=args.size, limit=args.limit,
                                     augmentation=args.augmentation, local_crops=args.local_crops)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        generator=torch.Generator().manual_seed(args.seed), drop_last=True,
                        pin_memory=args.device.startswith('cuda'))
    model = Experiment(args.method, width=args.width, hidden=256, bottleneck=64, prototypes=128,
                       output_target_rate=args.output_target_rate,
                       input_coding=args.input_coding,
                       population=args.population,
                       teacher_input_coding=args.teacher_input_coding).to(args.device).train()
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-3, weight_decay=1e-4)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(purpose='real_event_pipeline_pilot_not_benchmark', split_sha256=sha256(split_path),
                  sampled_manifest_sha256=hashlib.sha256(json.dumps(dataset.records, sort_keys=True).encode()).hexdigest(),
                  sampled_class_counts=dict(sorted(Counter(record['label'] for record in dataset.records).items())),
                  split='train', augmentation=('EventCLR policy parameters with independent event copies'
                                               if args.augmentation == 'eventclr'
                                               else 'weak frame-space pilot transform; not exact SpikeCLR policy'),
                  hidden=256, bottleneck=64, prototypes=128, lr=1e-3, weight_decay=1e-4,
                  torch=str(torch.__version__))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch_views = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_views = next(iterator)
        views = [view.transpose(0, 1).contiguous().to(args.device, non_blocking=True)
                 for view in batch_views]
        optimizer.zero_grad(set_to_none=True)
        loss, features, teacher_logits = model(views)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3., error_if_nonfinite=True)
        metrics = {'step': step, 'loss': loss.item(), 'gradient_norm': float(grad_norm),
                   **diagnostics(features, teacher_logits)}
        if args.method == 'spike_dino':
            metrics.update(model.student.last_rates)
            metrics['prototype_threshold'] = model.student.head.prototype_threshold.item()
        if teacher_logits:
            distribution = torch.cat([model.criterion.teacher_distribution(z)
                                      for z in teacher_logits])
            entropy = lambda probability: -(probability * probability.clamp_min(1e-12).log()).sum(-1).mean().item()
            metrics.update(teacher_sample_entropy=entropy(distribution),
                           teacher_batch_entropy=entropy(distribution.mean(0)),
                           prototype_usage=distribution.argmax(-1).unique().numel() / distribution.shape[-1])
        optimizer.step()
        model.after_optimizer_step(teacher_logits)
        with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'step': args.steps, 'torch_rng': torch.get_rng_state()},
               args.output / 'checkpoint.pt')


if __name__ == '__main__':
    main()
