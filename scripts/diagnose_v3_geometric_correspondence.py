"""Measure patch-feature equivariance under known DVS global-crop shifts.

The ground truth is shared raw sensor coordinates, not semantic part labels.
Uses only the fixed training partition and keeps teacher states independent.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional
from torch.nn import functional as F
from torchvision.transforms import RandomResizedCrop

from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import ManifestEventDataset
from scripts.diagnose_v3_representation import model_from_checkpoint


def crop_pair(index, seed, size, crop_scale):
    base = seed * 1_000_003 + index * 1000
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base + 73)
        first = RandomResizedCrop.get_params(
            torch.empty(2, size, size), scale=crop_scale,
            ratio=(3 / 4, 4 / 3))
    second = ManifestEventDataset._shift_global_crop(
        first, (size, size), base + 74)
    return first, second


def patch_mass(view, patch_size):
    total = view.float().sum((0, 1))[None, None]
    return F.avg_pool2d(total, patch_size, patch_size).flatten()


def patch_activity(view, patch_size):
    return patch_mass(view, patch_size) > 0


def geometric_targets(first, second, grid, device):
    top0, left0, height0, width0 = first
    top1, left1, height1, width1 = second
    index = torch.arange(grid * grid, device=device)
    row, col = torch.div(index, grid, rounding_mode='floor'), index % grid
    raw_y = top0 + (row.float() + .5) * height0 / grid
    raw_x = left0 + (col.float() + .5) * width0 / grid
    mapped_y = (raw_y - top1) * grid / height1 - .5
    mapped_x = (raw_x - left1) * grid / width1 - .5
    inside = ((mapped_y >= 0) & (mapped_y < grid) &
              (mapped_x >= 0) & (mapped_x < grid))
    nearest_y = mapped_y.round().long().clamp(0, grid - 1)
    nearest_x = mapped_x.round().long().clamp(0, grid - 1)
    return mapped_y, mapped_x, nearest_y * grid + nearest_x, inside


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--manifest-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--records', type=int, default=32)
    p.add_argument('--view-seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    if config.get('dataset', 'cifar10dvs') != 'cifar10dvs':
        p.error('Current geometric audit requires CIFAR10-DVS')
    manifest = load_manifest(args.manifest_path)
    if manifest['dataset'] != 'cifar10dvs':
        p.error('Manifest must be CIFAR10-DVS')
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).teacher.to(args.device).eval()
    grid = config['size'] // config['patch_size']
    dataset = ManifestEventDataset(
        args.dataset_root, args.manifest_path, seed=args.view_seed,
        time_steps=config['time_steps'], size=config['size'], local_crops=0,
        global_crop_scale=tuple(config['global_crop_scale']),
        global_temporal_crop_range=tuple(config['global_time_range']),
        global_view_policy='shared_time_shifted_space')
    indices = np.linspace(0, len(dataset) - 1, args.records, dtype=int)
    rows = []
    all_distances = []
    all_same_index_distances = []
    all_matched_cosine = []
    all_other_cosine = []
    high_activity_distances = []
    with torch.no_grad():
        for index in indices:
            first_view, second_view = dataset[(int(index), 0)]
            features = []
            for view in (first_view, second_view):
                functional.reset_net(model)
                output = model(view[:, None].to(args.device),
                               compute_patch_head=False)
                features.append(F.normalize(
                    output['patch_spikes'].float().mean(0)[0],
                    dim=-1, eps=1e-6))
                functional.reset_net(model)
            first_crop, second_crop = crop_pair(
                int(index), args.view_seed, config['size'],
                tuple(config['global_crop_scale']))
            mapped_y, mapped_x, target, inside = geometric_targets(
                first_crop, second_crop, grid, args.device)
            mass0 = patch_mass(first_view,
                               config['patch_size']).to(args.device)
            mass1 = patch_mass(second_view,
                               config['patch_size']).to(args.device)
            activity0, activity1 = mass0 > 0, mass1 > 0
            valid = inside & activity0 & activity1[target]
            high0 = mass0 >= torch.quantile(mass0, .75)
            high1 = mass1 >= torch.quantile(mass1, .75)
            valid_high = valid & high0 & high1[target]
            similarity = features[0] @ features[1].T
            predicted = similarity.argmax(-1)
            predicted_y = torch.div(predicted, grid, rounding_mode='floor')
            predicted_x = predicted % grid
            source = torch.arange(grid * grid, device=args.device)
            source_y = torch.div(source, grid, rounding_mode='floor')
            source_x = source % grid
            distance = ((predicted_y.float() - mapped_y).square() +
                        (predicted_x.float() - mapped_x).square()).sqrt()
            same_index_distance = ((source_y.float() - mapped_y).square() +
                                   (source_x.float() - mapped_x).square()).sqrt()
            selected_distance = distance[valid].cpu().tolist()
            all_distances.extend(selected_distance)
            high_activity_distances.extend(distance[valid_high].cpu().tolist())
            all_same_index_distances.extend(
                same_index_distance[valid].cpu().tolist())
            matched = similarity[source, target]
            all_matched_cosine.extend(matched[valid].cpu().tolist())
            other = similarity.mean(-1)
            all_other_cosine.extend(other[valid].cpu().tolist())
            rows.append({'index': int(index),
                         'valid_active_patch_count': int(valid.sum()),
                         'valid_high_activity_patch_count': int(valid_high.sum()),
                         'recall_at_1_patch': float((distance[valid] <= 1).float().mean())
                         if valid.any() else None})
    if not all_distances:
        raise RuntimeError('No active patches with a valid geometric match')
    distances = np.asarray(all_distances)
    high_distances = np.asarray(high_activity_distances)
    same = np.asarray(all_same_index_distances)
    report = {
        'checkpoint': str(args.checkpoint), 'steps': payload['steps'],
        'records': len(rows), 'valid_active_patches': len(distances),
        'view_seed': args.view_seed,
        'training_split_only': True, 'official_test_accessed': False,
        'ground_truth': 'raw sensor-coordinate mapping of shifted crops',
        'not_semantic_correspondence': True,
        'patch_grid': [grid, grid],
        'recall_at_1_patch': float(np.mean(distances <= 1)),
        'recall_at_2_patches': float(np.mean(distances <= 2)),
        'high_activity_definition': 'top quartile event mass within each view',
        'high_activity_valid_patches': len(high_distances),
        'high_activity_recall_at_1_patch': float(np.mean(high_distances <= 1))
        if len(high_distances) else None,
        'high_activity_recall_at_2_patches': float(np.mean(high_distances <= 2))
        if len(high_distances) else None,
        'median_patch_error': float(np.median(distances)),
        'same_output_index_recall_at_1': float(np.mean(same <= 1)),
        'matched_patch_cosine': float(np.mean(all_matched_cosine)),
        'mean_other_patch_cosine': float(np.mean(all_other_cosine)),
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
