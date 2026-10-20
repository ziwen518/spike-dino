"""Read-only DVS global/local view-support audit on training recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional

from scripts.diagnose_v3_representation import (matched_cosine,
                                                 model_from_checkpoint)
from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import ManifestEventDataset


@torch.no_grad()
def cls_features(backbone, views, device):
    features = []
    for view in views:
        functional.reset_net(backbone)
        cls, _patch = backbone(view.to(device))
        functional.reset_net(backbone)
        features.append(cls.float().mean(0).cpu())
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--records', type=int, default=32)
    parser.add_argument('--local-crops', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    manifest = load_manifest(args.manifest_path)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.teacher.backbone
    indices = np.linspace(0, len(manifest['train']) - 1,
                          args.records, dtype=int)
    common = dict(
        seed=config['seed'], time_steps=config['time_steps'],
        size=config['size'], local_size=config['local_size'],
        local_crops=args.local_crops,
        global_crop_scale=config['global_crop_scale'],
        local_crop_scale=config['local_crop_scale'],
        global_temporal_crop_range=config['global_time_range'],
        local_temporal_crop_range=config['local_time_range'],
        view_validity='resample_empty', max_view_attempts=4)
    outputs = {}
    global_reference = None
    for policy in ('shared_core', 'shared_core_nested_local'):
        dataset = ManifestEventDataset(
            args.dataset_root, args.manifest_path,
            global_view_policy=policy, **common)
        records = [dataset[(int(index), 0)] for index in indices]
        views = [torch.stack([record[i] for record in records], dim=1)
                 for i in range(2 + args.local_crops)]
        if global_reference is None:
            global_reference = views[:2]
        elif not all(torch.equal(a, b) for a, b in
                     zip(global_reference, views[:2])):
            raise RuntimeError('Global views changed between local policies')
        cls = cls_features(backbone, views, args.device)
        locals_report = []
        for index in range(args.local_crops):
            local = views[index + 2]
            locals_report.append({
                'index': index,
                'empty_recording_fraction': float(
                    (local == 0).all((0, 2, 3, 4)).float().mean()),
                'nonzero_cell_fraction': float((local != 0).float().mean()),
                'mean_event_mass': float(local.float().sum((0, 2, 3, 4)).mean()),
                'global0_local_cls_cosine': matched_cosine(
                    cls[0], cls[index + 2]),
                'global1_local_cls_cosine': matched_cosine(
                    cls[1], cls[index + 2]),
            })
        outputs[policy] = {
            'global0_global1_cls_cosine': matched_cosine(cls[0], cls[1]),
            'local_views': locals_report,
        }
    result = {
        'checkpoint': str(args.checkpoint),
        'records': args.records,
        'local_crops': args.local_crops,
        'training_split_only': True,
        'official_test_accessed': False,
        'same_global_views': True,
        'policies': outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
