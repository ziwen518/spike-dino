"""Short-run trainer for the experimental spiking DINOv3 objective.

This is a research runner, not the official DINOv3 7B/large-batch recipe.
Use --synthetic to validate compute without reading event data.
"""

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler

from .data import Cifar10DVSPilotDataset, ManifestEventDataset
from .benchmark_manifest import load_manifest
from .dinov3 import SpikingDinoV3Prototype
from .gradient_fusion import (balanced_backbone_gradients,
                              global_preserving_backbone_gradients)


def patch_masks(batch, patches, low, high, device, probability=.5):
    if not 0 < low <= high <= 1:
        raise ValueError('Mask ratios must lie in (0, 1]')
    if not 0 <= probability <= 1:
        raise ValueError('Mask sample probability must lie in [0, 1]')
    masks = torch.zeros(batch, patches, dtype=torch.bool, device=device)
    for row in masks:
        if torch.rand((), device=device) >= probability:
            continue
        count = max(1, min(patches, round(patches *
                                       float(torch.empty(()).uniform_(low, high)))))
        row[torch.randperm(patches, device=device)[:count]] = True
    return masks


def _block_mask(grid, count, rng):
    """Event-grid adaptation of DINOv3 rectangular patch masking."""

    mask = np.zeros((grid, grid), dtype=bool)
    remaining = count
    attempts_without_progress = 0
    while remaining and attempts_without_progress < 20:
        min_area = min(4, remaining)
        max_area = max(min_area, min(remaining, grid * grid // 2))
        area = rng.uniform(min_area, max_area)
        aspect = math.exp(rng.uniform(math.log(.3), math.log(1 / .3)))
        height = max(1, min(grid - 1, round(math.sqrt(area * aspect))))
        width = max(1, min(grid - 1, round(math.sqrt(area / aspect))))
        top = int(rng.integers(0, grid - height + 1))
        left = int(rng.integers(0, grid - width + 1))
        region = mask[top:top + height, left:left + width]
        added = int((~region).sum())
        if 0 < added <= remaining:
            region[:] = True
            remaining -= added
            attempts_without_progress = 0
        else:
            attempts_without_progress += 1
    if remaining:
        free = np.flatnonzero(~mask.reshape(-1))
        mask.reshape(-1)[rng.choice(free, size=remaining, replace=False)] = True
    return torch.from_numpy(mask.reshape(-1))


def fixed_count_masks(batch, patches, low, high, device, probability=.5):
    """Mask exactly floor(2B*p) global views with stratified mask ratios."""

    if not 0 < low <= high <= 1 or not 0 < probability <= 1:
        raise ValueError('Invalid mask ratio or probability')
    grid = math.isqrt(patches)
    if grid * grid != patches:
        raise ValueError('Patch grid must be square')
    total = 2 * batch
    selected = int(total * probability)
    if selected < 1:
        raise ValueError('At least one global sample must be masked')
    masks = torch.zeros(total, patches, dtype=torch.bool)
    positions = torch.randperm(total)[:selected]
    ratios = torch.linspace(low, high, selected + 1)[1:]
    for position, ratio in zip(positions.tolist(), ratios.tolist()):
        count = max(1, min(patches, int(patches * ratio)))
        seed = int(torch.randint(0, 2**31 - 1, ()).item())
        masks[position] = _block_mask(grid, count, np.random.default_rng(seed))
    return list(masks.reshape(2, batch, patches).to(device).unbind(0))


def partial_temporal_masks(spatial_masks, time_steps):
    """Hide half the time bins of each selected patch, leaving time evidence."""
    if time_steps < 2:
        raise ValueError('Partial temporal masking needs at least two time bins')
    masks = []
    for spatial in spatial_masks:
        if spatial.dtype != torch.bool or spatial.ndim != 2:
            raise ValueError('Expected spatial masks [batch, patches]')
        order = torch.rand((time_steps, *spatial.shape),
                           device=spatial.device).argsort(dim=0).argsort(dim=0)
        masks.append(spatial[None] & (order < time_steps // 2))
    return masks


class FixedEpochBatchSampler(Sampler):
    """Reproducible per-epoch orders, resumable at any completed batch."""

    def __init__(self, records, batch_size, seed, start_step, stop_step):
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step
        if records // batch_size < 1 or start_step < 0 or stop_step < start_step:
            raise ValueError('Invalid dataset size or step interval')

    def __len__(self):
        return self.stop_step - self.start_step

    def __iter__(self):
        steps_per_epoch = self.records // self.batch_size
        current_epoch = -1
        order = None
        for step in range(self.start_step, self.stop_step):
            epoch, offset = divmod(step, steps_per_epoch)
            if epoch != current_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.records, generator=generator).tolist()
                current_epoch = epoch
            start = offset * self.batch_size
            yield [(index, epoch) for index in order[start:start + self.batch_size]]


def linear_warmup_value(step, start, final, warmup_steps):
    """Step-indexed linear warmup followed by a flat value."""
    if warmup_steps < 0 or step < 0:
        raise ValueError('Step and warmup duration must be nonnegative')
    if warmup_steps == 0:
        return final
    return start + (final - start) * min(step / warmup_steps, 1.)


def save_checkpoint(model, optimizer, args, completed, output):
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'steps': completed, 'config': config,
               'cpu_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all()
               if str(args.device).startswith('cuda') else None}
    temporary = output / 'last.pt.tmp'
    torch.save(payload, temporary)
    temporary.replace(output / 'last.pt')
    # Evaluation milestones omit optimizer state and RNG to save disk space.
    torch.save({'model': model.state_dict(), 'steps': completed,
                'config': config}, output / f'eval_step_{completed:06d}.pt')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('cifar10dvs', 'ncaltech101',
                                         'nmnist', 'dvsgesture'),
                   default='cifar10dvs')
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--manifest-path', type=Path)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--local-size', type=int)
    p.add_argument('--global-crop-scale', type=float, nargs=2, default=(.1, 1.0))
    p.add_argument('--local-crop-scale', type=float, nargs=2, default=(.05, .4))
    p.add_argument('--global-time-range', type=float, nargs=2, default=(.1, 1.0))
    p.add_argument('--local-time-range', type=float, nargs=2, default=(.1, 1.0))
    p.add_argument('--view-validity', choices=('none', 'resample_empty'),
                   default='none')
    p.add_argument('--global-view-policy',
                   choices=('independent', 'shared_core',
                            'shared_temporal_windows',
                            'shared_time_independent_space',
                            'shared_time_shifted_space',
                            'shared_core_nested_local'),
                   default='independent')
    p.add_argument('--max-view-attempts', type=int, default=4)
    p.add_argument('--time-steps', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--local-crops', type=int, default=8)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--dim', type=int, default=384)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--block-variant', choices=('standard', 'evidence'),
                   default='standard',
                   help='Experimental full block replacement; evidence uses its own attention and residual')
    p.add_argument('--attention-variant',
                   choices=('matrix_recode', 'matrix_legacy', 'channel_gate',
                            'token_pair', 'token_pair_centered',
                            'matrix_value_skip', 'matrix_count',
                            'spikformer_count'),
                   default='matrix_recode')
    p.add_argument('--cls-aggregation',
                   choices=('attention_only', 'spike_pool',
                            'spike_quadrant_pool'),
                   default='attention_only')
    p.add_argument('--residual-policy', choices=('accumulate', 'recode_norm',
                                                 'layer_scale',
                                                 'centered_layer_scale',
                                                 'moment_preserving',
                                                 'signed_current',
                                                 'signed_layer_scale',
                                                 'binary_ei'),
                   default='accumulate')
    p.add_argument('--masked-context-bridge', choices=('none', 'spike_or'),
                   default='none')
    p.add_argument('--residual-gradient-policy',
                   choices=('full', 'branch_detach', 'masked_branch_detach',
                            'bounded_branch'),
                   default='full',
                   help='Optional stop-gradient at each spiking residual branch input')
    p.add_argument('--head-hidden', type=int, default=2048)
    p.add_argument('--head-layers', type=int, choices=(2, 3), default=3)
    p.add_argument('--prototype-readout', choices=('linear', 'cosine'), default='linear')
    p.add_argument('--ibot-reduction', choices=('sample', 'token'), default='sample')
    p.add_argument('--ibot-shortcut', choices=('none', 'input_or'),
                   default='none')
    p.add_argument('--dino-shortcut', choices=('none', 'input_or'),
                   default='none')
    p.add_argument('--ibot-diversity-weight', type=float, default=0.)
    p.add_argument('--ibot-diversity-source',
                   choices=('masked', 'visible', 'visible_position'),
                   default='masked')
    p.add_argument('--bottleneck', type=int, default=256)
    p.add_argument('--prototypes', type=int, default=65536)
    p.add_argument('--dino-prototypes', type=int)
    p.add_argument('--ibot-prototypes', type=int)
    p.add_argument('--dino-hidden', type=int)
    p.add_argument('--ibot-hidden', type=int)
    p.add_argument('--dino-bottleneck', type=int)
    p.add_argument('--ibot-bottleneck', type=int)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--checkpoint-every', type=int, default=50)
    p.add_argument('--freeze-last-layer-epochs', type=int, default=0,
                   help='Keep student DINO and iBOT prototype weights fixed for early full epochs')
    p.add_argument('--no-checkpoint', action='store_true',
                   help='Only for short engineering checks; retain metrics but no model weights')
    p.add_argument('--resume', type=Path)
    p.add_argument('--anomaly-step', type=int,
                   help='Diagnostic-only autograd trace for one optimizer step')
    p.add_argument('--diagnose-detach-branches', action='store_true',
                   help='At the anomaly step, test backbone residual-branch gradient isolation')
    p.add_argument('--diagnose-mask-detach-branches', action='store_true',
                   help='At the anomaly step, isolate only masked token branch gradients')
    p.add_argument('--diagnose-bounded-branches', action='store_true',
                   help='At the anomaly step, bound branch-input gradients')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-warmup-steps', type=int, default=0,
                   help='Linear warmup from zero, then keep --lr flat')
    p.add_argument('--teacher-temp-start', type=float, default=.04)
    p.add_argument('--teacher-temp-final', type=float, default=.04)
    p.add_argument('--teacher-temp-warmup-steps', type=int, default=0)
    p.add_argument('--teacher-momentum', type=float, default=.996)
    p.add_argument('--adam-beta2', type=float, default=.999)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mask-min', type=float, default=.1)
    p.add_argument('--mask-max', type=float, default=.5)
    p.add_argument('--mask-probability', type=float, default=.5)
    p.add_argument('--mask-policy', choices=('fixed_block', 'independent_random'),
                   default='fixed_block')
    p.add_argument('--mask-observation-policy',
                   choices=('full_spatial', 'partial_time'),
                   default='full_spatial')
    p.add_argument('--global-cls-policy',
                   choices=('masked', 'unmasked_separate'), default='masked',
                   help='Use a separate complete global view for DINO CLS')
    p.add_argument('--ibot-residual-gradient-policy',
                   choices=('inherit', 'full', 'bounded_branch'),
                   default='inherit',
                   help='Backward route for the separate masked iBOT pass')
    p.add_argument('--gradient-fusion',
                   choices=('sum', 'balanced_blocks',
                            'balanced_blocks_stem_global',
                            'global_preserving'),
                   default='sum',
                   help='Optional per-block task gradient aggregation')
    p.add_argument('--gram-anchor-phase', action='store_true',
                   help='Snapshot the EMA SNN teacher when resuming for Gram refinement')
    p.add_argument('--gram-weight', type=float, default=1.)
    p.add_argument('--balance-mode', choices=('joint', 'per_view'), default='joint')
    p.add_argument('--target-policy', choices=('sinkhorn', 'ema_center'),
                   default='sinkhorn')
    p.add_argument('--patch-teacher-aggregation',
                   choices=('single_view', 'matched_cross_view_mean'),
                   default='single_view',
                   help='Fuse aligned teacher patch logits across two global event views')
    p.add_argument('--norm-policy', choices=('ema_bn', 'batch_stat', 'layer_norm'),
                   default='ema_bn')
    return p


def main():
    args = parser().parse_args()
    args.manifest_sha256 = (load_manifest(args.manifest_path, args.dataset)['sha256']
                            if args.manifest_path is not None else None)
    if (args.steps < 1 or args.batch_size < 2 or args.local_crops < 0 or
            args.local_crops > 8 or args.checkpoint_every < 1 or
            args.max_view_attempts < 1 or args.freeze_last_layer_epochs < 0):
        raise ValueError('Invalid steps, batch size or local crop count')
    if args.ibot_diversity_weight < 0:
        raise ValueError('iBOT diversity weight must be nonnegative')
    if args.mask_observation_policy == 'partial_time' and args.time_steps < 2:
        raise ValueError('Partial temporal masking needs at least two time bins')
    if args.no_checkpoint and (args.resume is not None or args.steps > 10):
        raise ValueError('--no-checkpoint is only for at most 10 fresh smoke steps')
    if args.gradient_fusion != 'sum' and args.anomaly_step is not None:
        raise ValueError('Gradient fusion and anomaly-step diagnostics cannot be combined')
    if args.gram_anchor_phase and args.resume is None:
        raise ValueError('Gram anchor phase requires a trained resume checkpoint')
    if args.gram_weight < 0:
        raise ValueError('Gram weight must be nonnegative')
    if (args.lr <= 0 or args.lr_warmup_steps < 0 or
            args.teacher_temp_start <= 0 or args.teacher_temp_final <= 0 or
            args.teacher_temp_warmup_steps < 0 or
            not 0 <= args.teacher_momentum < 1 or
            not 0 < args.adam_beta2 < 1 or args.weight_decay < 0):
        raise ValueError('Invalid optimizer or teacher schedule settings')
    optional_widths = (args.dino_prototypes, args.ibot_prototypes,
                       args.dino_hidden, args.ibot_hidden,
                       args.dino_bottleneck, args.ibot_bottleneck)
    if any(value is not None and value < 1 for value in optional_widths):
        raise ValueError('Head dimensions must be positive')
    local_size = args.local_size if args.local_size is not None else args.size // 2
    if (args.patch_size < 1 or args.size % args.patch_size or
            local_size % args.patch_size or args.size < 32 or local_size < args.patch_size):
        raise ValueError('Global and local sizes must be divisible by patch size')
    for pair in (args.global_crop_scale, args.local_crop_scale,
                 args.global_time_range, args.local_time_range):
        if not 0 < pair[0] <= pair[1] <= 1:
            raise ValueError('Crop ranges must lie in (0, 1]')
    if not args.synthetic and args.dataset_root is None:
        raise ValueError('Provide --dataset-root for real event data')
    if not args.synthetic and args.dataset != 'cifar10dvs' and args.manifest_path is None:
        raise ValueError('Other event datasets require --manifest-path')
    if args.view_validity != 'none' and args.manifest_path is None and not args.synthetic:
        raise ValueError('Event-view resampling requires a fixed manifest')
    if args.global_view_policy != 'independent' and args.manifest_path is None and not args.synthetic:
        raise ValueError('Shared-core global views require a fixed manifest')
    if (args.patch_teacher_aggregation == 'matched_cross_view_mean' and
            args.global_view_policy not in
            ('shared_core', 'shared_temporal_windows',
             'shared_core_nested_local')):
        raise ValueError('Cross-view patch targets require aligned global crop grids')
    if args.freeze_last_layer_epochs and args.synthetic:
        raise ValueError('Prototype-freeze epochs require a real fixed training set')
    if args.block_variant == 'evidence' and args.gradient_fusion != 'sum':
        raise ValueError('Evaluate evidence blocks with ordinary loss gradients before adding fusion rules')
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise FileExistsError('Choose a new, empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    model = SpikingDinoV3Prototype(image_size=args.size, patch_size=args.patch_size,
                                   dim=args.dim, depth=args.depth, heads=args.heads,
                                   block_variant=args.block_variant,
                                   attention_variant=args.attention_variant,
                                   cls_aggregation=args.cls_aggregation,
                                   residual_policy=args.residual_policy,
                                   masked_context_bridge=args.masked_context_bridge,
                                   residual_gradient_policy=args.residual_gradient_policy,
                                   head_hidden=args.head_hidden,
                                   head_layers=args.head_layers,
                                   prototype_readout=args.prototype_readout,
                                   ibot_reduction=args.ibot_reduction,
                                   ibot_shortcut=args.ibot_shortcut,
                                   dino_shortcut=args.dino_shortcut,
                                   ibot_diversity_weight=args.ibot_diversity_weight,
                                   ibot_diversity_source=args.ibot_diversity_source,
                                   global_cls_policy=args.global_cls_policy,
                                   ibot_residual_gradient_policy=args.ibot_residual_gradient_policy,
                                   bottleneck=args.bottleneck,
                                   prototypes=args.prototypes,
                                   dino_prototypes=args.dino_prototypes,
                                   ibot_prototypes=args.ibot_prototypes,
                                   dino_hidden=args.dino_hidden,
                                   ibot_hidden=args.ibot_hidden,
                                   dino_bottleneck=args.dino_bottleneck,
                                   ibot_bottleneck=args.ibot_bottleneck,
                                   balance_mode=args.balance_mode,
                                   target_policy=args.target_policy,
                                   patch_teacher_aggregation=args.patch_teacher_aggregation,
                                   norm_policy=args.norm_policy,
                                   teacher_temp=args.teacher_temp_start,
                                   teacher_momentum=args.teacher_momentum,
                                   gram_weight=args.gram_weight).to(args.device).train()
    optimizer = torch.optim.AdamW(
        model.student.parameters(), lr=args.lr,
        betas=(.9, args.adam_beta2), weight_decay=args.weight_decay)
    start_step = 0
    if args.resume is not None:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        original = previous['config']
        stable = ('dataset', 'dataset_root', 'manifest_path', 'manifest_sha256',
                  'synthetic', 'device', 'size', 'patch_size',
                  'local_size', 'global_crop_scale', 'local_crop_scale',
                  'global_time_range', 'local_time_range',
                  'view_validity', 'max_view_attempts', 'global_view_policy',
                  'time_steps',
                  'batch_size', 'local_crops', 'depth', 'dim', 'heads', 'block_variant',
                  'cls_aggregation', 'residual_policy', 'ibot_shortcut',
                  'dino_shortcut',
                  'ibot_diversity_weight',
                  'ibot_diversity_source',
                  'masked_context_bridge', 'residual_gradient_policy',
                  'head_hidden', 'head_layers', 'bottleneck', 'prototypes',
                  'dino_prototypes', 'ibot_prototypes', 'dino_hidden',
                  'ibot_hidden', 'dino_bottleneck', 'ibot_bottleneck',
                  'lr', 'lr_warmup_steps', 'teacher_temp_start',
                  'teacher_temp_final', 'teacher_temp_warmup_steps',
                  'teacher_momentum', 'adam_beta2', 'weight_decay',
                  'seed', 'mask_min', 'mask_max', 'mask_probability',
                  'freeze_last_layer_epochs',
                  'balance_mode', 'target_policy',
                  'patch_teacher_aggregation', 'norm_policy', 'mask_policy',
                  'mask_observation_policy', 'global_cls_policy',
                  'ibot_residual_gradient_policy', 'gradient_fusion',
                  'gram_weight')
        original.setdefault('prototype_readout', 'cosine')
        original.setdefault('ibot_reduction', 'token')
        original.setdefault('attention_variant', 'matrix_legacy')
        original.setdefault('block_variant', 'standard')
        original.setdefault('cls_aggregation', 'attention_only')
        original.setdefault('residual_policy', 'accumulate')
        original.setdefault('ibot_shortcut', 'none')
        original.setdefault('dino_shortcut', 'none')
        original.setdefault('ibot_diversity_weight', 0.)
        original.setdefault('ibot_diversity_source', 'masked')
        original.setdefault('masked_context_bridge', 'none')
        original.setdefault('residual_gradient_policy', 'full')
        original.setdefault('global_cls_policy', 'masked')
        original.setdefault('ibot_residual_gradient_policy', 'inherit')
        original.setdefault('gradient_fusion', 'sum')
        original.setdefault('gram_weight', 1.)
        original.setdefault('lr_warmup_steps', 0)
        original.setdefault('teacher_temp_start', .04)
        original.setdefault('teacher_temp_final', .04)
        original.setdefault('teacher_temp_warmup_steps', 0)
        original.setdefault('teacher_momentum', .996)
        original.setdefault('adam_beta2', .999)
        original.setdefault('weight_decay', .01)
        original.setdefault('mask_policy', 'independent_random')
        original.setdefault('mask_observation_policy', 'full_spatial')
        original.setdefault('manifest_path', None)
        original.setdefault('manifest_sha256', None)
        original.setdefault('dataset', 'cifar10dvs')
        original.setdefault('view_validity', 'none')
        original.setdefault('global_view_policy', 'independent')
        original.setdefault('target_policy', 'sinkhorn')
        original.setdefault('patch_teacher_aggregation', 'single_view')
        original.setdefault('freeze_last_layer_epochs', 0)
        original.setdefault('max_view_attempts', 4)
        for key in ('dino_prototypes', 'ibot_prototypes', 'dino_hidden',
                    'ibot_hidden', 'dino_bottleneck', 'ibot_bottleneck'):
            original.setdefault(key, None)
        original.setdefault('patch_size', 16)
        original.setdefault('local_size', None)
        original.setdefault('global_crop_scale', (.1, 1.0))
        original.setdefault('local_crop_scale', (.05, .4))
        original.setdefault('global_time_range', (.1, 1.0))
        original.setdefault('local_time_range', (.1, 1.0))
        current = {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}
        for key in ('global_crop_scale', 'local_crop_scale',
                    'global_time_range', 'local_time_range'):
            original[key] = list(original[key])
            current[key] = list(current[key])
        stable += ('prototype_readout', 'ibot_reduction', 'attention_variant')
        if any(original[key] != current[key] for key in stable):
            raise ValueError('Resume configuration differs from checkpoint')
        checkpoint_has_gram = any(
            key.startswith('gram_teacher.') for key in previous['model'])
        if checkpoint_has_gram and not args.gram_anchor_phase:
            raise ValueError('Resume checkpoint contains a Gram teacher')
        if checkpoint_has_gram:
            model.enable_gram_refinement()
        model.load_state_dict(previous['model'])
        if args.gram_anchor_phase and not checkpoint_has_gram:
            model.enable_gram_refinement()
        optimizer.load_state_dict(previous['optimizer'])
        start_step = previous['steps']
        if start_step >= args.steps:
            raise ValueError('Requested step target must exceed completed steps')
        existing_log = args.output / 'metrics.jsonl'
        if existing_log.exists():
            rows = [json.loads(line) for line in existing_log.read_text(
                encoding='utf-8').splitlines() if line.strip()]
            if rows and rows[-1]['step'] != start_step:
                raise ValueError('Output log and checkpoint steps differ; use a new output directory')
        torch.set_rng_state(previous['cpu_rng'])
        if str(args.device).startswith('cuda'):
            torch.cuda.set_rng_state_all(previous['cuda_rng'])
    dataset = None
    if not args.synthetic:
        dataset_kwargs = dict(seed=args.seed, time_steps=args.time_steps,
                              size=args.size, local_crops=args.local_crops,
                              local_size=local_size,
                              global_crop_scale=args.global_crop_scale,
                              local_crop_scale=args.local_crop_scale,
                              global_temporal_crop_range=args.global_time_range,
                              local_temporal_crop_range=args.local_time_range)
        dataset = (ManifestEventDataset(args.dataset_root, args.manifest_path,
                                        view_validity=args.view_validity,
                                        max_view_attempts=args.max_view_attempts,
                                        global_view_policy=args.global_view_policy,
                                        **dataset_kwargs)
                   if args.manifest_path is not None else
                   Cifar10DVSPilotDataset(args.dataset_root,
                                          augmentation='eventclr', **dataset_kwargs))
        sampler = FixedEpochBatchSampler(len(dataset), args.batch_size,
                                         args.seed, start_step, args.steps)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed + 1),
                            pin_memory=str(args.device).startswith('cuda'))
        iterator = iter(loader)
    args.train_records = len(dataset) if dataset is not None else None
    args.steps_per_epoch = (len(dataset) // args.batch_size
                            if dataset is not None else None)
    (args.output / 'config.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value
         for key, value in vars(args).items()}, indent=2), encoding='utf-8')
    history = []
    log_path = args.output / 'metrics.jsonl'
    for step in range(start_step, args.steps):
        started = time.monotonic()
        effective_lr = linear_warmup_value(
            step, 0., args.lr, args.lr_warmup_steps)
        model.teacher_temp = linear_warmup_value(
            step, args.teacher_temp_start, args.teacher_temp_final,
            args.teacher_temp_warmup_steps)
        for group in optimizer.param_groups:
            group['lr'] = effective_lr
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        if args.synthetic:
            sizes = [args.size, args.size] + [local_size] * args.local_crops
            views = [torch.rand(args.time_steps, args.batch_size, 2, size, size,
                                device=args.device) for size in sizes]
        else:
            batch = next(iterator)
            views = [v.permute(1, 0, 2, 3, 4).to(args.device,
                                                 non_blocking=True) for v in batch]
        patch_count = (args.size // args.patch_size) ** 2
        if args.mask_policy == 'fixed_block':
            masks = fixed_count_masks(args.batch_size, patch_count, args.mask_min,
                                      args.mask_max, args.device,
                                      args.mask_probability)
        else:
            masks = [patch_masks(args.batch_size, patch_count, args.mask_min,
                                 args.mask_max, args.device, args.mask_probability)
                     for _ in range(2)]
            if not any(mask.any() for mask in masks):
                masks[0][0, torch.randint(patch_count, ())] = True
        temporal_masks = (partial_temporal_masks(masks, args.time_steps)
                          if args.mask_observation_policy == 'partial_time'
                          else None)
        optimizer.zero_grad(set_to_none=True)
        empty_global = torch.stack([
            (view == 0).all(dim=(0, 2, 3, 4)).float().mean()
            for view in views[:2]]).mean()
        empty_local = (torch.stack([
            (view == 0).all(dim=(0, 2, 3, 4)).float().mean()
            for view in views[2:]]).mean() if len(views) > 2 else
            empty_global.new_tensor(0.))
        context = (torch.autograd.detect_anomaly()
                   if args.anomaly_step == step + 1 else nullcontext())
        branch_hooks = []
        changed_policies = []
        if args.diagnose_detach_branches and args.anomaly_step == step + 1:
            for block in model.student.backbone.blocks:
                for branch_input in (block.attn_input, block.mlp_input):
                    branch_hooks.append(branch_input.register_forward_pre_hook(
                        lambda _module, inputs: (inputs[0].detach(),)))
        if args.diagnose_mask_detach_branches and args.anomaly_step == step + 1:
            for block in model.student.backbone.blocks:
                changed_policies.append((block, block.residual_gradient_policy))
                block.residual_gradient_policy = 'masked_branch_detach'
        if args.diagnose_bounded_branches and args.anomaly_step == step + 1:
            for block in model.student.backbone.blocks:
                changed_policies.append((block, block.residual_gradient_policy))
                block.residual_gradient_policy = 'bounded_branch'
        with context:
            metrics = model(views, masks, temporal_masks=temporal_masks,
                            return_components=(args.anomaly_step == step + 1 or
                                               args.gradient_fusion != 'sum'))
            for hook in branch_hooks:
                hook.remove()
            for block, policy in changed_policies:
                block.residual_gradient_policy = policy
            if args.anomaly_step == step + 1:
                patch_weight = model.student.backbone.patch_embed.weight
                for name in ('_dino_raw', '_ibot_raw', '_koleo_raw',
                             '_diversity_raw', 'loss'):
                    if name not in metrics:
                        continue
                    with torch.autograd.set_detect_anomaly(False):
                        grad = torch.autograd.grad(
                            metrics[name], patch_weight, retain_graph=True,
                            allow_unused=True)[0]
                    print('GRAD_DIAG', name,
                          'finite=', bool(torch.isfinite(grad).all()),
                          'max=', float(grad.nan_to_num().abs().max()),
                          flush=True)
                selected = {
                    name: parameter for name, parameter in
                    model.student.named_parameters()
                    if name in ('backbone.patch_embed.weight',
                                'backbone.mask_current',
                                'backbone.position_current',
                                'ibot_head.fc1.weight',
                                'ibot_head.fc2.weight') or
                    (name.startswith('backbone.blocks.') and
                     name.endswith('attn_scale'))
                }
                with torch.autograd.set_detect_anomaly(False):
                    grad_list = torch.autograd.grad(
                        metrics['_ibot_raw'], tuple(selected.values()),
                        retain_graph=True, allow_unused=True)
                for name, grad in zip(selected, grad_list):
                    if grad is not None:
                        print('IBOT_LAYER_GRAD', name,
                              'finite=', bool(torch.isfinite(grad).all()),
                              'max=', float(grad.nan_to_num().abs().max()),
                              flush=True)
                for name in ('_dino_raw', '_ibot_raw', '_koleo_raw',
                             '_diversity_raw'):
                    metrics.pop(name, None)
            fusion_diagnostics = None
            if args.gradient_fusion != 'sum':
                global_loss = (metrics['_dino_raw'] + model.koleo_weight *
                               metrics['_koleo_raw'])
                if '_gram_raw' in metrics:
                    global_loss = global_loss + model.gram_weight * metrics['_gram_raw']
                local_loss = metrics['_ibot_raw']
                if '_diversity_raw' in metrics:
                    local_loss = (local_loss + model.ibot_diversity_weight *
                                  metrics['_diversity_raw'])
                if args.gradient_fusion == 'global_preserving':
                    fusion_diagnostics = global_preserving_backbone_gradients(
                        model.student, global_loss, local_loss)
                else:
                    fusion_diagnostics = balanced_backbone_gradients(
                        model.student, global_loss, local_loss,
                        protect_stem=(args.gradient_fusion ==
                                      'balanced_blocks_stem_global'))
                for name in ('_dino_raw', '_ibot_raw', '_koleo_raw',
                             '_diversity_raw', '_gram_raw'):
                    metrics.pop(name, None)
            else:
                metrics['loss'].backward()
        gradients_finite = torch.stack([
            torch.isfinite(parameter.grad).all()
            for parameter in model.student.parameters()
            if parameter.grad is not None]).all()
        if not bool(gradients_finite):
            raise FloatingPointError(
                f'Non-finite student gradient at step {step + 1}; '
                'optimizer and EMA teacher were not updated')
        freeze_prototypes = (args.steps_per_epoch is not None and
                             step // args.steps_per_epoch <
                             args.freeze_last_layer_epochs)
        if freeze_prototypes:
            model.student.dino_head.prototypes.grad = None
            model.student.ibot_head.prototypes.grad = None
        optimizer.step()
        model.after_optimizer_step()
        row = {'step': step + 1, **{name: float(value.detach())
                                   for name, value in metrics.items()}}
        row['lr_effective'] = effective_lr
        row['teacher_temp_effective'] = model.teacher_temp
        row['teacher_momentum_effective'] = model.teacher_momentum
        if fusion_diagnostics is not None:
            for group in ('backbone.patch_embed', 'backbone.blocks.0',
                          'backbone.blocks.11'):
                if group in fusion_diagnostics:
                    norms = fusion_diagnostics[group]
                    row['task_grad_ratio_' + group.replace('.', '_')] = (
                        norms['local_norm'] /
                        max(norms['global_norm'], 1e-30))
                    if 'task_cosine' in norms:
                        row['task_grad_cosine_' + group.replace('.', '_')] = (
                            norms['task_cosine'])
                        row['task_local_scale_' + group.replace('.', '_')] = (
                            norms['local_scale'])
        if args.steps_per_epoch is not None:
            row['epoch'] = step // args.steps_per_epoch + 1
            row['step_in_epoch'] = step % args.steps_per_epoch + 1
        row['empty_global_fraction'] = float(empty_global)
        row['empty_local_fraction'] = float(empty_local)
        active_patch_mass = []
        removed_patch_mass = []
        for index, view in enumerate(views[:2]):
            grid_activity = torch.nn.functional.avg_pool2d(
                view.flatten(0, 1), args.patch_size,
                stride=args.patch_size).sum(1).reshape(
                    args.time_steps, args.batch_size, patch_count)
            observation_mask = (temporal_masks[index] if temporal_masks is not None
                                else masks[index][None].expand(args.time_steps,
                                                               -1, -1))
            active_patch_mass.append(grid_activity.sum())
            removed_patch_mass.append(grid_activity[observation_mask].sum())
        row['masked_event_fraction'] = float(
            torch.stack(removed_patch_mass).sum() /
            torch.stack(active_patch_mass).sum().clamp_min(1e-6))
        row['masked_time_fraction'] = float(torch.stack([
            (temporal_masks[i] if temporal_masks is not None else
             masks[i][None].expand(args.time_steps, -1, -1)).float().mean()
            for i in range(2)]).mean())
        row['prototypes_frozen'] = bool(freeze_prototypes)
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
            row['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        row['elapsed_seconds'] = round(time.monotonic() - started, 2)
        history.append(row)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        completed = step + 1
        if not args.no_checkpoint and (completed % args.checkpoint_every == 0 or
                                       completed == args.steps):
            save_checkpoint(model, optimizer, args, completed, args.output)
    all_rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
    (args.output / 'metrics.json').write_text(json.dumps(all_rows, indent=2),
                                              encoding='utf-8')


if __name__ == '__main__':
    main()
