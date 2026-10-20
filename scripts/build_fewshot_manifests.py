"""Freeze EventCLR-style per-class label selections for paired methods."""

import argparse
import hashlib
import json
from pathlib import Path

from spike_dino.benchmark_manifest import load_manifest, select_labeled, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shots', nargs='+', type=float, required=True)
    parser.add_argument('--mode', choices=('count', 'percent'), default='count')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    source = load_manifest(args.manifest)
    experiments = []
    for shot in args.shots:
        amount = int(shot) if args.mode == 'count' else shot
        for index in range(args.runs):
            subset_seed = args.seed + index * 1000
            experiments.append(select_labeled(source, amount, subset_seed, args.mode))
    payload = {'schema': 1, 'dataset': source['dataset'],
               'source_manifest_sha256': source['sha256'],
               'protocol': 'EventCLR-style RandomState per-class selection; '
                           'same records for SpikeCLR and Spike-DINO',
               'experiments': experiments}
    payload['sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write_manifest(payload, args.output)
    print(json.dumps({'dataset': source['dataset'], 'experiments': len(experiments),
                      'sha256': payload['sha256'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
