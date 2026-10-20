"""Frozen CLS-feature diagnostic for spiking DINOv3 CIFAR10-DVS checkpoints.

Both labeled cohorts come from the original training partition. The official
test partition is not opened. This is a fixed diagnostic, not a paper score.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from .data import Cifar10DVSPilotDataset
from .benchmark_manifest import load_manifest
from .dinov3 import SpikingDinoV3Prototype
from .evaluate import FixedEventDataset, linear_probe, weighted_knn


def cohorts(root, seed=42, train_limit=512, validation_per_class=100,
            manifest_path=None):
    split = (load_manifest(manifest_path, 'cifar10dvs') if manifest_path is not None
             else json.loads((Path(root) / f'split_seed{seed}.json').read_text(
                 encoding='utf-8')))
    selected = Cifar10DVSPilotDataset(root, seed=seed, limit=train_limit,
                                      manifest_path=manifest_path).records
    selected_paths = {record['path'] for record in selected}
    test_paths = {record['path'] for record in split['test']}
    if selected_paths & test_paths:
        raise RuntimeError('Labeled probe cohort overlaps official test data')
    remaining = [record for record in split['train']
                 if record['path'] not in selected_paths]
    rng = np.random.default_rng(seed + 17)
    validation = []
    for label in range(10):
        candidates = [record for record in remaining if record['label'] == label]
        if len(candidates) < validation_per_class:
            raise ValueError(f'Not enough validation records in class {label}')
        indices = rng.choice(len(candidates), size=validation_per_class, replace=False)
        validation.extend(candidates[int(index)] for index in indices)
    validation_paths = {record['path'] for record in validation}
    if selected_paths & validation_paths or validation_paths & test_paths:
        raise RuntimeError('Evaluation cohorts overlap')
    return selected, validation


def select_feature(cls, patch, source='cls'):
    if source not in ('cls', 'patch_mean', 'concat'):
        raise ValueError(source)
    cls_rate = cls.float().mean(0)
    patch_rate = patch.float().mean((0, 2))
    if source == 'cls':
        return cls_rate
    if source == 'patch_mean':
        return patch_rate
    return torch.cat((cls_rate, patch_rate), dim=-1)


@torch.no_grad()
def extract(backbone, loader, device, mode='eval', feature_source='cls'):
    if mode not in ('eval', 'train_batch'):
        raise ValueError(mode)
    backbone.train(mode == 'train_batch')
    features, labels = [], []
    for frames, targets in loader:
        events = frames.transpose(0, 1).contiguous().to(device)
        functional.reset_net(backbone)
        try:
            cls, patch = backbone(events)
            features.append(select_feature(cls, patch, feature_source).cpu())
            labels.append(targets.clone())
        finally:
            functional.reset_net(backbone)
    return torch.cat(features), torch.cat(labels)


def feature_stats(features):
    x = features.float()
    centered = x - x.mean(0)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if total > 0:
        p = singular / total
        rank = float((-(p * p.clamp_min(1e-12).log()).sum()).exp())
    else:
        rank = 0.0
    return {'mean_feature_std': float(x.std(0, unbiased=False).mean()),
            'mean_feature_norm': float(x.norm(dim=1).mean()),
            'zero_fraction': float((x == 0).float().mean()),
            'unique_feature_vectors': int(torch.unique(x, dim=0).shape[0]),
            'effective_rank': rank}


@torch.no_grad()
def batch_context_difference(backbone, frames, device, mode='eval',
                             feature_source='cls'):
    """Change companion events while keeping the first recording fixed."""

    if frames.ndim != 5 or frames.shape[1] < 2:
        raise ValueError('Need at least two records')
    backbone.train(mode == 'train_batch')
    first = frames.to(device)
    second = first.clone()
    second[:, 1] = 0
    functional.reset_net(backbone)
    a = select_feature(*backbone(first), feature_source)[0]
    functional.reset_net(backbone)
    b = select_feature(*backbone(second), feature_source)[0]
    functional.reset_net(backbone)
    return float((a - b).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--train-limit', type=int, default=512)
    p.add_argument('--validation-per-class', type=int, default=100)
    p.add_argument('--modes', nargs='+', choices=('eval', 'train_batch'),
                   default=['eval', 'train_batch'])
    p.add_argument('--random-init', action='store_true')
    p.add_argument('--cohort-seed', type=int,
                   help='Fix labeled probe/validation cohort across pretraining seeds')
    p.add_argument('--probe-seed', type=int,
                   help='Fix linear-probe initialization across pretraining seeds')
    p.add_argument('--feature-source', choices=('cls', 'patch_mean', 'concat'),
                   default='cls')
    p.add_argument('--cls-aggregation-override',
                   choices=('event_weighted_pool', 'spike_quadrant_pool'))
    args = p.parse_args()
    if args.output.exists():
        p.error('Output file already exists')
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    if config.get('synthetic'):
        p.error('Synthetic checkpoints cannot be evaluated on event recordings')
    if config.get('dataset', 'cifar10dvs') != 'cifar10dvs':
        p.error('This frozen-feature diagnostic currently supports CIFAR10-DVS only')
    if config.get('manifest_path') is not None:
        manifest = load_manifest(config['manifest_path'], 'cifar10dvs')
        if manifest['sha256'] != config['manifest_sha256']:
            p.error('Checkpoint and current manifest hashes differ')
    torch.manual_seed(config['seed'])
    model = SpikingDinoV3Prototype(image_size=config['size'],
                                   patch_size=config.get('patch_size', 16),
                                   dim=config['dim'], depth=config['depth'],
                                   heads=config['heads'],
                                   block_variant=config.get('block_variant', 'standard'),
                                   attention_variant=config.get('attention_variant',
                                                                'matrix_legacy'),
                                   cls_aggregation=config.get('cls_aggregation',
                                                              'attention_only'),
                                   residual_policy=config.get('residual_policy',
                                                              'accumulate'),
                                   masked_context_bridge=config.get('masked_context_bridge',
                                                                    'none'),
                                   residual_gradient_policy=config.get(
                                       'residual_gradient_policy', 'full'),
                                   head_hidden=config['head_hidden'],
                                   head_layers=config.get('head_layers', 2),
                                   prototype_readout=config.get('prototype_readout', 'cosine'),
                                   ibot_reduction=config.get('ibot_reduction', 'token'),
                                   ibot_shortcut=config.get('ibot_shortcut', 'none'),
                                   dino_shortcut=config.get('dino_shortcut', 'none'),
                                   ibot_diversity_weight=config.get('ibot_diversity_weight', 0.),
                                   ibot_diversity_source=config.get('ibot_diversity_source', 'masked'),
                                   bottleneck=config['bottleneck'],
                                   prototypes=config['prototypes'],
                                   dino_prototypes=config.get('dino_prototypes'),
                                   ibot_prototypes=config.get('ibot_prototypes'),
                                   dino_hidden=config.get('dino_hidden'),
                                   ibot_hidden=config.get('ibot_hidden'),
                                   dino_bottleneck=config.get('dino_bottleneck'),
                                   ibot_bottleneck=config.get('ibot_bottleneck'),
                                   balance_mode=config.get('balance_mode', 'joint'),
                                   target_policy=config.get('target_policy', 'sinkhorn'),
                                   norm_policy=config.get('norm_policy', 'ema_bn'),
                                   gram_weight=config.get('gram_weight', 1.))
    if not args.random_init:
        if any(key.startswith('gram_teacher.') for key in payload['model']):
            model.enable_gram_refinement()
        model.load_state_dict(payload['model'])
    if args.cls_aggregation_override is not None:
        if not hasattr(model.teacher.backbone, 'cls_pool_spike'):
            p.error('CLS aggregation override requires existing pool weights')
        model.teacher.backbone.cls_aggregation = args.cls_aggregation_override
    cohort_seed = (config['seed'] if args.cohort_seed is None
                   else args.cohort_seed)
    probe_seed = (config['seed'] if args.probe_seed is None
                  else args.probe_seed)
    selected, validation = cohorts(args.dataset_root, cohort_seed,
                                   args.train_limit, args.validation_per_class,
                                   config.get('manifest_path'))
    train_loader = DataLoader(FixedEventDataset(args.dataset_root, selected,
                                               config['time_steps'], config['size']),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)
    validation_loader = DataLoader(FixedEventDataset(args.dataset_root, validation,
                                                    config['time_steps'], config['size']),
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers)
    report = {'checkpoint': str(args.checkpoint),
              'random_init': args.random_init,
              'cohort_seed': cohort_seed, 'probe_seed': probe_seed,
              'feature_source': args.feature_source,
              'cls_aggregation_override': args.cls_aggregation_override,
              'pretrain_steps': 0 if args.random_init else payload['steps'],
              'norm_policy': config.get('norm_policy', 'ema_bn'),
              'train_records': len(selected),
              'validation_records': len(validation),
              'validation_source': 'unused labeled subset of pretraining train partition',
              'official_test_accessed': False, 'results': {}}
    first_frames = next(iter(train_loader))[0][:2].transpose(0, 1).contiguous()
    for mode in args.modes:
        backbone = deepcopy(model.teacher.backbone).to(args.device)
        train_features, train_labels = extract(backbone, train_loader, args.device,
                                               mode, args.feature_source)
        valid_features, valid_labels = extract(backbone, validation_loader,
                                              args.device, mode,
                                              args.feature_source)
        knn = weighted_knn(train_features, train_labels, valid_features)
        linear = linear_probe(train_features, train_labels, valid_features,
                              seed=probe_seed)
        report['results'][mode] = {
            **feature_stats(valid_features),
            'knn_top1': float((knn == valid_labels).float().mean()),
            'linear_top1': float((linear == valid_labels).float().mean()),
            'batch_context_mean_absolute_difference': batch_context_difference(
                backbone, first_frames, args.device, mode,
                args.feature_source),
        }
        del backbone
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
