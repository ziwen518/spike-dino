"""Read one real training recording from each fixed event benchmark manifest."""

import argparse
import json
from pathlib import Path
import numpy as np

from spike_dino.data import ManifestEventDataset


FOLDERS = {'cifar10dvs': 'CIFAR10DVS', 'ncaltech101': 'NCALTECH101',
           'nmnist': 'NMNIST', 'dvsgesture': 'DVSGesture'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'))
    parser.add_argument('--manifest-root', type=Path,
                        default=Path('configs/splits'))
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    results = []
    for dataset, folder in FOLDERS.items():
        source = ManifestEventDataset(
            args.dataset_root / folder,
            args.manifest_root / f'{dataset}_seed42.json',
            size=128, local_size=56, time_steps=4, local_crops=1)
        indices = np.linspace(0, len(source) - 1,
                              num=min(args.samples, len(source)), dtype=int)
        activity = []
        shapes = None
        for index in indices:
            views = source[int(index)]
            shapes = [list(view.shape) for view in views]
            activity.append([float((view != 0).float().mean()) for view in views])
        activity = np.asarray(activity)
        results.append({'dataset': dataset, 'records': len(source),
                        'sample_count': len(indices),
                        'selection': 'evenly spaced indices of sorted train manifest',
                        'first_record': source.records[int(indices[0])]['path'],
                        'shapes': shapes,
                        'view_nonzero_median': np.median(activity, axis=0).tolist(),
                        'view_nonzero_min': activity.min(axis=0).tolist(),
                        'view_nonzero_max': activity.max(axis=0).tolist(),
                        'empty_view_count': int((activity == 0).sum())})
    serialized = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
