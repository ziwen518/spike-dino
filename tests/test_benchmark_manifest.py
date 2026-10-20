import json
from pathlib import Path

import pytest

from spike_dino.benchmark_manifest import (CIFAR_CLASSES, build_manifest,
                                           load_manifest, select_labeled,
                                           write_manifest)
from spike_dino.data import Cifar10DVSPilotDataset


def _touch(root, relative):
    path = root / 'raw' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_cifar_split_and_shots_are_fixed_without_test_leakage(tmp_path):
    for name in CIFAR_CLASSES:
        for index in range(10):
            _touch(tmp_path, f'{name}/{index:04d}.aedat')
    first = build_manifest(tmp_path, 'cifar10dvs', strict=False)
    second = build_manifest(tmp_path, 'cifar10dvs', strict=False)
    assert first == second
    assert len(first['train']) == 90 and len(first['test']) == 10
    assert {row['path'] for row in first['train']}.isdisjoint(
        row['path'] for row in first['test'])
    shots = select_labeled(first, 3, seed=42)
    assert len(shots['train']) == 30
    assert set(shots['actual_per_class'].values()) == {3}
    assert {row['path'] for row in shots['train']}.isdisjoint(
        row['path'] for row in first['test'])
    target = tmp_path / 'manifests' / 'cifar.json'
    write_manifest(first, target)
    write_manifest(first, target)
    assert load_manifest(target, 'cifar10dvs') == first
    assert len(Cifar10DVSPilotDataset(tmp_path, manifest_path=target)) == 90
    with pytest.raises(FileExistsError):
        write_manifest({**first, 'seed': 99}, target)
    target.write_text(target.read_text(encoding='utf-8').replace(
        '"label": 0', '"label": 1', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_manifest(target, 'cifar10dvs')


def test_ncaltech_split_clamps_fewshot_to_class_size(tmp_path):
    for name, count in (('a', 5), ('b', 3), ('c', 2)):
        for index in range(count):
            _touch(tmp_path, f'Caltech101/{name}/{index}.bin')
    manifest = build_manifest(tmp_path, 'ncaltech101', strict=False)
    assert len(manifest['train']) == 7 and len(manifest['test']) == 3
    shots = select_labeled(manifest, 3, seed=7)
    assert shots['actual_per_class'] == {'0': 3, '1': 2, '2': 1}
    assert shots == json.loads(json.dumps(shots))


def test_official_train_test_boundaries_for_nmnist_and_gesture(tmp_path):
    for relative in ('Train/0/a.bin', 'Train/1/b.bin',
                     'Test/0/c.bin', 'Test/1/d.bin'):
        _touch(tmp_path, relative)
    nmnist = build_manifest(tmp_path, 'nmnist', strict=False)
    assert len(nmnist['train']) == len(nmnist['test']) == 2
    for relative in ('ibmGestureTrain/user01_lab/0.npy',
                     'ibmGestureTrain/user02_lab/1.npy',
                     'ibmGestureTest/user24_lab/0.npy'):
        _touch(tmp_path, relative)
    gesture = build_manifest(tmp_path, 'dvsgesture', strict=False)
    assert (len(gesture['train']), len(gesture['test'])) == (2, 1)
    _touch(tmp_path, 'ibmGestureTest/user01_lab/1.npy')
    with pytest.raises(ValueError, match='subjects overlap'):
        build_manifest(tmp_path, 'dvsgesture', strict=False)
