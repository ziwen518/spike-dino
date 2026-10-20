"""Resumable event-only CIFAR10-DVS pretraining on one device.

This is a research training runner, not an exact reproduction of DINO or SpikeCLR.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import Experiment, diagnostics
from .data import Cifar10DVSPilotDataset, ManifestEventDataset
from .benchmark_manifest import load_manifest


def cosine(start, end, fraction):
    fraction = min(max(fraction, 0.), 1.)
    return end + .5 * (start - end) * (1 + math.cos(math.pi * fraction))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('cifar10dvs', 'ncaltech101',
                                              'nmnist', 'dvsgesture'),
                        default='cifar10dvs')
    parser.add_argument('--manifest-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['spike_dino', 'dino', 'spikeclr', 'temporal_ntxent'],
                        default='spike_dino')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--width', type=float, default=.125)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--prototypes', type=int, default=1024)
    parser.add_argument('--population', type=int, default=1)
    parser.add_argument('--time-steps', type=int, default=4)
    parser.add_argument('--size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--local-crops', type=int, default=2)
    parser.add_argument('--input-coding', choices=['occupancy', 'bernoulli'], default='occupancy')
    parser.add_argument('--teacher-input-coding', choices=['occupancy','bernoulli'])
    parser.add_argument('--output-target-rate', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--teacher-temp-start', type=float, default=.04)
    parser.add_argument('--teacher-temp-end', type=float, default=.07)
    parser.add_argument('--teacher-temp-warmup-epochs', type=int, default=30)
    parser.add_argument('--ema-start', type=float, default=.996)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int,
                        help='Stop after this many epochs while keeping the planned schedule')
    return parser


def validate(args, parser):
    if (args.epochs < 1 or args.batch_size < 2 or args.width <= 0 or args.workers < 0 or
        args.time_steps < 1 or args.size < 16 or args.local_crops < 0 or args.local_crops > 6 or
        args.hidden < 1 or args.bottleneck < 1 or args.prototypes < 2 or args.population < 1 or
        args.torch_threads < 1 or
        args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr or
        not 0 <= args.ema_start <= 1):
        parser.error('Invalid training configuration')
    if args.method not in ('spike_dino', 'dino') and args.local_crops:
        parser.error('Only DINO methods support local crops')
    if args.dataset != 'cifar10dvs' and args.manifest_path is None:
        parser.error('Other datasets require --manifest-path')
    if args.method != 'spike_dino' and args.output_target_rate is not None:
        parser.error('Output firing-rate control requires spike_dino')
    if args.method != 'spike_dino' and args.input_coding != 'occupancy':
        parser.error('Input coding is only configurable for spike_dino')
    if args.method != 'spike_dino' and args.teacher_input_coding is not None:
        parser.error('Teacher input coding is only configurable for spike_dino')
    if args.method != 'spike_dino' and args.population != 1:
        parser.error('Population coding is only configurable for spike_dino')
    if args.resume and not args.resume.is_file():
        parser.error('Resume checkpoint does not exist')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error('stop-after-epoch must be within 1..epochs')


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.manifest_sha256 = (load_manifest(args.manifest_path, args.dataset)['sha256']
                            if args.manifest_path is not None else None)
    validate(args, parser)
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error('Output directory exists and is not empty; pass --resume to continue')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    dataset = (ManifestEventDataset(
        args.dataset_root, args.manifest_path, seed=args.seed,
        time_steps=args.time_steps, size=args.size,
        local_crops=args.local_crops, local_size=args.size // 2,
        global_crop_scale=(.1, 1.0), local_crop_scale=(.05, .4),
        global_temporal_crop_range=(.1, 1.0),
        local_temporal_crop_range=(.1, 1.0))
        if args.manifest_path is not None else
        Cifar10DVSPilotDataset(args.dataset_root, seed=args.seed,
                               time_steps=args.time_steps, size=args.size,
                               augmentation='eventclr', local_crops=args.local_crops))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers, generator=generator,
                        pin_memory=args.device.startswith('cuda'))
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    model = Experiment(args.method, width=args.width, hidden=args.hidden,
                       bottleneck=args.bottleneck, prototypes=args.prototypes,
                       projection_dim=128, teacher_momentum=args.ema_start,
                       output_target_rate=args.output_target_rate,
                       input_coding=args.input_coding,
                       population=args.population,
                       teacher_input_coding=args.teacher_input_coding).to(args.device).train()
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(train_records=len(dataset), steps_per_epoch=steps_per_epoch,
                  train_manifest_sha256=hashlib.sha256(json.dumps(dataset.records, sort_keys=True).encode()).hexdigest(),
                  projection_dim=128,
                  data_protocol='fixed event manifest; EventCLR-policy views',
                  status='research schedule; not an exact DINO or SpikeCLR reproduction')
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        previous = checkpoint['config']
        previous.setdefault('dataset', 'cifar10dvs')
        previous.setdefault('manifest_path', None)
        previous.setdefault('manifest_sha256', None)
        compatible = ('method','width','hidden','bottleneck','prototypes','population',
                      'dataset','manifest_path','manifest_sha256',
                      'time_steps','size','batch_size','local_crops',
                      'input_coding','teacher_input_coding','output_target_rate','seed','epochs','lr','min_lr',
                      'weight_decay','warmup_epochs','teacher_temp_start','teacher_temp_end',
                      'teacher_temp_warmup_epochs','ema_start','steps_per_epoch',
                      'train_manifest_sha256','torch_threads')
        if any(previous[key] != config[key] for key in compatible):
            raise ValueError('Resume checkpoint was created with a different training configuration')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        generator.set_state(checkpoint['loader_rng'])
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for epoch in range(start_epoch, args.epochs):
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
        started = time.monotonic()
        aggregate = {}
        counts = {}
        model.train()
        dataset.set_epoch(epoch)
        for batch_index, batch_views in enumerate(loader):
            step = epoch * steps_per_epoch + batch_index
            if args.warmup_epochs and step < args.warmup_epochs * steps_per_epoch:
                lr = args.lr * (step + 1) / (args.warmup_epochs * steps_per_epoch)
            else:
                post = step - args.warmup_epochs * steps_per_epoch
                duration = max(total_steps - args.warmup_epochs * steps_per_epoch, 1)
                lr = cosine(args.lr, args.min_lr, post / duration)
            for group in optimizer.param_groups:
                group['lr'] = lr
            if model.teacher is not None:
                model.teacher_momentum = cosine(args.ema_start, 1., step / max(total_steps - 1, 1))
                warmup = max(args.teacher_temp_warmup_epochs * steps_per_epoch, 1)
                model.criterion.teacher_temp = (args.teacher_temp_start +
                    (args.teacher_temp_end - args.teacher_temp_start) * min(step / warmup, 1.))
            views = [view.transpose(0,1).contiguous().to(args.device, non_blocking=True)
                     for view in batch_views]
            optimizer.zero_grad(set_to_none=True)
            loss, features, teacher_logits = model(views)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {batch_index}')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.,
                                                       error_if_nonfinite=True)
            metrics = {'loss': loss.item(), 'gradient_norm': float(gradient),
                       **diagnostics(features, teacher_logits)}
            if teacher_logits and batch_index % 25 == 0:
                distribution = torch.cat([model.criterion.teacher_distribution(z)
                                          for z in teacher_logits])
                sample_entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(-1).mean()
                batch_distribution = distribution.mean(0)
                batch_entropy = -(batch_distribution * batch_distribution.clamp_min(1e-12).log()).sum()
                metrics.update(teacher_sample_entropy=sample_entropy.item(),
                               teacher_batch_entropy=batch_entropy.item(),
                               prototype_usage=distribution.argmax(-1).unique().numel() / distribution.shape[-1])
            if args.method == 'spike_dino':
                metrics.update(model.student.last_rates)
                metrics['teacher_input_spike_rate'] = model.teacher.last_rates['input_spike_rate']
                metrics['prototype_threshold'] = model.student.head.prototype_threshold.item()
                if batch_index % 25 == 0:
                    backbone_grads = [parameter.grad.detach().float().square().sum()
                                      for parameter in model.student.backbone.parameters()
                                      if parameter.grad is not None]
                    metrics['backbone_gradient_norm'] = torch.stack(backbone_grads).sum().sqrt().item()
            optimizer.step()
            model.after_optimizer_step(teacher_logits)
            for key, value in metrics.items():
                aggregate[key] = aggregate.get(key, 0.) + value
                counts[key] = counts.get(key, 0) + 1
            if batch_index % 25 == 0:
                print(json.dumps({'epoch':epoch, 'batch':batch_index,
                                  'batches':steps_per_epoch, 'loss':metrics['loss'],
                                  'elapsed_seconds':time.monotonic()-started}), flush=True)
        summary = {key: value / counts[key] for key, value in aggregate.items()}
        summary.update(epoch=epoch, lr=lr, ema=model.teacher_momentum if model.teacher else None,
                       teacher_temp=model.criterion.teacher_temp if model.teacher else None,
                       elapsed_seconds=time.monotonic()-started)
        with (args.output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(summary) + '\n')
        checkpoint = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                      'config':config, 'epoch':epoch, 'torch_rng':torch.get_rng_state(),
                      'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                      'loader_rng':generator.get_state()}
        temp = args.output / 'last.tmp'
        torch.save(checkpoint,temp)
        temp.replace(args.output / 'last.pt')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
