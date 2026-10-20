"""Export fixed train-internal spiking-layer features for comparable PCA plots."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from spike_dino.evaluate import FixedEventDataset
from spike_dino.evaluate_v3 import cohorts
from scripts.diagnose_v3_representation import model_from_checkpoint


LAYERS = (0, 3, 6, 9, 11)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cohort-seed', type=int, default=42)
    p.add_argument('--validation-per-class', type=int, default=100)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    if config.get('dataset', 'cifar10dvs') != 'cifar10dvs':
        p.error('This plot export currently supports CIFAR10-DVS')
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.student.backbone
    _, records = cohorts(args.dataset_root, args.cohort_seed, 512,
                         args.validation_per_class,
                         config.get('manifest_path'))
    loader = DataLoader(FixedEventDataset(args.dataset_root, records,
                                          config['time_steps'], config['size']),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    captured = {}
    features = {}

    def record_token(_module, _inputs, output):
        captured['token_patch_mean'] = output[:, :, 1:].float().mean((0, 2)).cpu()

    def record_block(index):
        def hook(_module, _inputs, output):
            captured[f'block_{index}_cls'] = output[:, :, 0].float().mean(0).cpu()
            captured[f'block_{index}_patch_mean'] = (
                output[:, :, 1:].float().mean((0, 2)).cpu())
        return hook

    handles = [backbone.token_spike.register_forward_hook(record_token)]
    handles.extend(backbone.blocks[index].attn_input.register_forward_hook(
        record_block(index)) for index in LAYERS)
    labels = []
    try:
        with torch.no_grad():
            for frames, targets in loader:
                captured.clear()
                events = frames.transpose(0, 1).contiguous().to(args.device)
                functional.reset_net(backbone)
                cls, patch = backbone(events)
                captured['final_cls'] = cls.float().mean(0).cpu()
                captured['final_patch_mean'] = patch.float().mean((0, 2)).cpu()
                for name, value in captured.items():
                    features.setdefault(name, []).append(value.numpy())
                labels.append(targets.numpy())
                functional.reset_net(backbone)
    finally:
        for handle in handles:
            handle.remove()
        functional.reset_net(backbone)
    output = {name: np.concatenate(parts).astype(np.float32)
              for name, parts in features.items()}
    output['labels'] = np.concatenate(labels).astype(np.int64)
    output['record_paths'] = np.asarray([record['path'] for record in records])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        'checkpoint': str(args.checkpoint), 'steps': payload['steps'],
        'cohort_seed': args.cohort_seed,
        'validation_per_class': args.validation_per_class,
        'records': len(records),
        'record_source': 'unused labeled subset of pretraining train partition',
        'official_test_accessed': False,
        'features': {name: list(value.shape) for name, value in output.items()
                     if name not in ('labels', 'record_paths')},
        'interfaces': 'token and block features are binary spikes; final readouts are time-averaged binary spikes',
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
