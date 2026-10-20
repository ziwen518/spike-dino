"""Portable, immutable event benchmark and few-shot record manifests.

The sorted-file split is reproducible across hosts. It is deliberately named
as such: the public EventCLR loader does not sort files inside a class, so an
identical RNG seed alone cannot recover its original file-level split.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


CIFAR_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')


def _record(raw, path, label):
    return {'path': path.relative_to(raw).as_posix(), 'label': int(label)}


def _stratified_split(groups, ratio, seed):
    rng = np.random.RandomState(seed)
    train, test = [], []
    for label in sorted(groups):
        records = sorted(groups[label], key=lambda row: row['path'])
        indices = np.arange(len(records))
        rng.shuffle(indices)
        boundary = int(len(records) * ratio)
        train.extend(records[int(index)] for index in indices[:boundary])
        test.extend(records[int(index)] for index in indices[boundary:])
    return train, test


def build_manifest(root, dataset, seed=42, strict=True):
    root = Path(root)
    raw = root / 'raw'
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    dataset = dataset.lower()
    groups = {}
    if dataset == 'cifar10dvs':
        for label, name in enumerate(CIFAR_CLASSES):
            groups[label] = [_record(raw, p, label)
                             for p in sorted((raw / name).rglob('*.aedat'))]
        train, test = _stratified_split(groups, .9, seed)
        source_format = 'original AEDAT2'
        split_policy = 'sorted files; per-class RandomState(seed).shuffle; 90/10'
        if strict and any(len(records) != 1000 for records in groups.values()):
            raise ValueError('CIFAR10-DVS requires 1000 files per class')
    elif dataset == 'ncaltech101':
        files = sorted(raw.rglob('*.bin'))
        names = sorted({path.parent.name for path in files})
        labels = {name: index for index, name in enumerate(names)}
        for path in files:
            label = labels[path.parent.name]
            groups.setdefault(label, []).append(_record(raw, path, label))
        train, test = _stratified_split(groups, .9, seed)
        source_format = 'original 5-byte bin'
        split_policy = 'sorted files; per-class RandomState(seed).shuffle; 90/10'
        if strict and (len(files) != 8709 or len(names) != 101):
            raise ValueError('N-Caltech101 requires 8709 files in 101 classes')
    elif dataset == 'nmnist':
        train = [_record(raw, path, int(path.parent.name))
                 for path in sorted((raw / 'Train').rglob('*.bin'))]
        test = [_record(raw, path, int(path.parent.name))
                for path in sorted((raw / 'Test').rglob('*.bin'))]
        source_format = 'original 5-byte bin'
        split_policy = 'official Train/Test directories'
        if strict and (len(train), len(test)) != (60000, 10000):
            raise ValueError('N-MNIST requires 60000 train and 10000 test files')
    elif dataset == 'dvsgesture':
        train = [_record(raw, path, int(path.stem)) for path in
                 sorted((raw / 'ibmGestureTrain').rglob('*.npy'))]
        test = [_record(raw, path, int(path.stem)) for path in
                sorted((raw / 'ibmGestureTest').rglob('*.npy'))]
        source_format = 'Tonic pre-cut NPY; original timestamps in milliseconds'
        split_policy = 'Tonic train/test archives; subjects 01-23 / 24-29'
        train_subjects = {Path(row['path']).parent.name.split('_')[0] for row in train}
        test_subjects = {Path(row['path']).parent.name.split('_')[0] for row in test}
        if train_subjects & test_subjects:
            raise ValueError('DVS-Gesture train/test subjects overlap')
        if strict and (len(train), len(test)) != (1077, 264):
            raise ValueError('Tonic DVS-Gesture requires 1077 train and 264 test files')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not train or not test:
        raise ValueError(f'{dataset} has an empty split')
    paths = [row['path'] for row in train + test]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate or overlapping event records')
    train = sorted(train, key=lambda row: row['path'])
    test = sorted(test, key=lambda row: row['path'])
    payload = {'schema': 1, 'dataset': dataset, 'seed': seed,
               'source_format': source_format, 'split_policy': split_policy,
               'ordering': 'lexicographic relative POSIX path',
               'train': train, 'test': test}
    payload['sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def select_labeled(manifest, samples_per_class, seed, mode='count'):
    """Match EventCLR's RandomState per-class selection on a fixed manifest."""

    if mode not in ('count', 'percent'):
        raise ValueError(mode)
    rng = np.random.RandomState(seed)
    grouped = {}
    for row in manifest['train']:
        grouped.setdefault(row['label'], []).append(row)
    selected = []
    for label in sorted(grouped):
        records = grouped[label]
        if mode == 'percent':
            if not 0 < samples_per_class <= 1:
                raise ValueError('Percent must be inside (0, 1]')
            amount = (len(records) if samples_per_class == 1 else
                      max(1, int(len(records) * samples_per_class)))
        else:
            if samples_per_class < 1:
                raise ValueError('Count must be positive')
            amount = min(int(samples_per_class), len(records))
        indices = rng.choice(len(records), size=amount, replace=False)
        selected.extend(records[int(index)] for index in indices)
    return {'dataset': manifest['dataset'], 'source_manifest_sha256': manifest['sha256'],
            'seed': seed, 'samples_per_class': samples_per_class, 'mode': mode,
            'actual_per_class': {str(label): count for label, count in
                                 Counter(row['label'] for row in selected).items()},
            'train': selected}


def write_manifest(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != serialized:
            raise FileExistsError(f'Refusing to overwrite different manifest: {path}')
        return
    path.write_text(serialized, encoding='utf-8')


def load_manifest(path, dataset=None):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = payload.get('sha256')
    content = {key: value for key, value in payload.items() if key != 'sha256'}
    actual = hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if expected != actual:
        raise ValueError(f'Manifest SHA-256 mismatch: {path}')
    if dataset is not None and payload.get('dataset') != dataset:
        raise ValueError(f'Manifest dataset mismatch: {path}')
    seen = set()
    for row in payload['train'] + payload['test']:
        relative = Path(row['path'])
        if (relative.is_absolute() or '..' in relative.parts or
                '\\' in row['path'] or ':' in row['path'] or row['path'] in seen):
            raise ValueError(f'Unsafe or duplicate manifest path: {row["path"]}')
        seen.add(row['path'])
    return payload
