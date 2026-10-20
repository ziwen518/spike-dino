"""Audit event-space overlap of paired DVS global crops without labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import RandomResizedCrop

from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import (ManifestEventDataset, SENSOR_SIZES,
                             random_crop_time, read_event_record)


def sample_crop(frame, scale, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RandomResizedCrop.get_params(
            frame, scale=scale, ratio=(3 / 4, 4 / 3))


def paired_content(events, first, second):
    x, y = events['x'], events['y']
    def inside(crop):
        top, left, height, width = crop
        return ((x >= left) & (x < left + width) &
                (y >= top) & (y < top + height))
    a, b = inside(first), inside(second)
    shared = int((a & b).sum())
    counts = [int(a.sum()), int(b.sum())]
    fractions = [shared / count if count else 0. for count in counts]
    top_a, left_a, height_a, width_a = first
    top_b, left_b, height_b, width_b = second
    overlap_height = max(0, min(top_a + height_a, top_b + height_b) -
                         max(top_a, top_b))
    overlap_width = max(0, min(left_a + width_a, left_b + width_b) -
                        max(left_a, left_b))
    area = overlap_height * overlap_width
    return {'common_event_count': shared,
            'event_counts': counts,
            'minimum_shared_event_fraction': min(fractions),
            'area_overlap_over_smaller': area / min(height_a * width_a,
                                                    height_b * width_b),
            'center_displacement_pixels': float(((top_a + height_a / 2 -
                                                  top_b - height_b / 2) ** 2 +
                                                 (left_a + width_a / 2 -
                                                  left_b - width_b / 2) ** 2) ** .5)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--manifest-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--records', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--global-crop-scale', type=float, nargs=2,
                   default=(.32, 1.))
    p.add_argument('--global-time-range', type=float, nargs=2,
                   default=(.5, 1.))
    args = p.parse_args()
    manifest = load_manifest(args.manifest_path)
    width, height = SENSOR_SIZES[manifest['dataset']]
    frame = torch.empty(2, height, width)
    indices = np.linspace(0, len(manifest['train']) - 1,
                          args.records, dtype=int)
    rows = []
    for index in indices:
        record = manifest['train'][int(index)]
        events = read_event_record(
            args.dataset_root / 'raw' / record['path'],
            manifest['dataset'])
        base = args.seed * 1_000_003 + int(index) * 1000
        core = random_crop_time(events, np.random.default_rng(base + 71),
                                *args.global_time_range)
        first = sample_crop(frame, args.global_crop_scale, base + 73)
        independent = sample_crop(frame, args.global_crop_scale, base + 74)
        shifted = ManifestEventDataset._shift_global_crop(
            first, (height, width), base + 74)
        rows.append({'index': int(index),
                     'core_events': int(len(core['x'])),
                     'independent': paired_content(core, first, independent),
                     'shifted': paired_content(core, first, shifted)})
    summary = {}
    for policy in ('independent', 'shifted'):
        summary[policy] = {}
        for key in ('minimum_shared_event_fraction',
                    'area_overlap_over_smaller',
                    'center_displacement_pixels'):
            values = np.array([row[policy][key] for row in rows])
            summary[policy][key] = {
                'mean': float(values.mean()),
                'median': float(np.median(values)),
                'minimum': float(values.min()),
                'p10': float(np.quantile(values, .1)),
            }
    report = {'dataset': manifest['dataset'], 'train_split_only': True,
              'records': len(rows), 'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
