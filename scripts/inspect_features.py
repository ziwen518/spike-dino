"""Inspect frozen SNN features with batch statistics versus running statistics."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from spike_dino.core import Experiment
from spike_dino.data import Cifar10DVSPilotDataset
from spike_dino.evaluate import FixedEventDataset


def summarize(features):
    x = features.float()
    singular = torch.linalg.svdvals(x - x.mean(0))
    probabilities = singular / singular.sum().clamp_min(1e-12)
    return {'feature_std': x.std(0, unbiased=False).mean().item(),
            'zero_fraction': (x == 0).float().mean().item(),
            'feature_norm': x.norm(dim=1).mean().item(),
            'unique_features': len(torch.unique(x, dim=0)),
            'effective_rank': (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    model = Experiment(config['method'], width=config['width'], hidden=config['hidden'],
                       bottleneck=config['bottleneck'], prototypes=config['prototypes'],
                       output_target_rate=config.get('output_target_rate'),
                       input_coding=config.get('input_coding','occupancy'),
                       population=config.get('population',1),
                       teacher_input_coding=config.get('teacher_input_coding'))
    model.load_state_dict(payload['model'])
    encoder = model.student.to(args.device)
    selected = Cifar10DVSPilotDataset(args.dataset_root, seed=config['seed'],
                                      limit=config['limit']).records
    by_class = {label: [record for record in selected if record['label']==label][:4]
                for label in range(10)}
    records = [record for group in by_class.values() for record in group]
    loader = DataLoader(FixedEventDataset(args.dataset_root, records,
                                         config['time_steps'], config['size']),
                        batch_size=len(records), shuffle=False)
    frames, _ = next(iter(loader))
    x = frames.transpose(0,1).contiguous().to(args.device)
    report = {}
    for mode in ('train','eval'):
        encoder.train(mode == 'train')
        torch.manual_seed(123)
        functional.reset_net(encoder)
        with torch.no_grad():
            features = encoder.encode(x).mean(0)
        report[mode] = summarize(features)
        report[mode]['input_spike_rate'] = encoder.last_rates.get('input_spike_rate')
        report[mode]['backbone_output_rate'] = encoder.last_rates.get('backbone_output_rate')
        functional.reset_net(encoder)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
