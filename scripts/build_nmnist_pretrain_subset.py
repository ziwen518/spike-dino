"""Freeze SpikeCLR-style stratified 10K N-MNIST SSL subset."""

import argparse
import hashlib
import json
from pathlib import Path

from spike_dino.benchmark_manifest import load_manifest, select_labeled, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = load_manifest(args.manifest, 'nmnist')
    subset = select_labeled(source, 1000, args.seed, 'count')
    if len(subset['train']) != 10000:
        raise ValueError('N-MNIST 10K subset must have 1000 recordings per class')
    payload = {'schema': 1, 'dataset': 'nmnist',
               'source_manifest_sha256': source['sha256'],
               'protocol': 'SpikeCLR-style 1000 recordings/class; labels used '
                           'only for stratified subset construction, not SSL loss',
               'seed': args.seed, 'train': subset['train']}
    payload['sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write_manifest(payload, args.output)
    print(json.dumps({'train': len(payload['train']),
                      'sha256': payload['sha256'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
