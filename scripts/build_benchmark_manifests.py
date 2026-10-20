"""Build fixed four-dataset manifests without altering historical splits."""

import argparse
import json
from pathlib import Path

from spike_dino.benchmark_manifest import build_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        choices=('cifar10dvs', 'ncaltech101', 'nmnist', 'dvsgesture'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-small-fixture', action='store_true')
    args = parser.parse_args()
    result = build_manifest(args.root, args.dataset, args.seed,
                            strict=not args.allow_small_fixture)
    write_manifest(result, args.output)
    print(json.dumps({'dataset': result['dataset'],
                      'train': len(result['train']), 'test': len(result['test']),
                      'sha256': result['sha256'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
