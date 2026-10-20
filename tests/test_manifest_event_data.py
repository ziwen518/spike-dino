import numpy as np
import torch

from spike_dino.benchmark_manifest import build_manifest, write_manifest
from spike_dino.data import (ManifestEventDataset, eventclr_representation,
                             read_5byte_events, read_gesture_events)


def _five_byte(path, count=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.zeros((count, 5), dtype=np.uint8)
    events[:, 0] = np.arange(count) % 30
    events[:, 1] = np.arange(count) % 30
    events[:, 2] = (np.arange(count) % 2) * 128
    events[:, 3] = np.arange(count) // 256
    events[:, 4] = np.arange(count) % 256
    events.tofile(path)


def _gesture(path, count=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((count, 4), dtype=np.int64)
    values[:, 0] = np.arange(count) % 128
    values[:, 1] = np.arange(count) % 128
    values[:, 2] = np.arange(count) % 2
    values[:, 3] = np.arange(count)
    np.save(path, values)


def test_nmnist_manifest_reads_original_events_and_multicrop(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin')
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    dataset = ManifestEventDataset(root, path, size=32, local_size=16,
                                   time_steps=4, local_crops=1)
    assert len(read_5byte_events(root / 'raw/Train/0/a.bin')['t']) == 200
    views = dataset[0]
    assert [tuple(view.shape) for view in views] == [
        (4, 2, 32, 32), (4, 2, 32, 32), (4, 2, 16, 16)]
    assert all(torch.isfinite(view).all() for view in views)


def test_gesture_millisecond_timestamps_and_subject_split(tmp_path):
    root = tmp_path / 'gesture'
    _gesture(root / 'raw/ibmGestureTrain/user01_lab/0.npy')
    _gesture(root / 'raw/ibmGestureTest/user24_lab/0.npy')
    path = tmp_path / 'gesture.json'
    write_manifest(build_manifest(root, 'dvsgesture', strict=False), path)
    events = read_gesture_events(root / 'raw/ibmGestureTrain/user01_lab/0.npy')
    assert events['t'][1] - events['t'][0] == 1000
    dataset = ManifestEventDataset(root, path, size=32, local_size=16,
                                   time_steps=4, local_crops=1)
    assert len(dataset[0]) == 3


def test_ncaltech_sensor_shape_uses_240_by_180(tmp_path):
    root = tmp_path / 'ncaltech'
    for index in range(10):
        _five_byte(root / f'raw/Caltech101/classA/{index}.bin')
    path = tmp_path / 'ncaltech.json'
    write_manifest(build_manifest(root, 'ncaltech101', strict=False), path)
    dataset = ManifestEventDataset(root, path, size=32, local_size=16,
                                   time_steps=4, local_crops=0)
    assert len(dataset) == 9
    assert dataset.global_transform.sensor_size == (240, 180)
    assert dataset[0][0].shape == (4, 2, 32, 32)
    events = {'t': np.array([0, 1]), 'x': np.array([239, 0]),
              'y': np.array([179, 0]), 'p': np.array([1, 0])}
    assert eventclr_representation(events, 4, 32, (240, 180)).sum() > 0


def test_event_view_resampling_is_deterministic_and_bounded(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/empty.bin', count=0)
    _five_byte(root / 'raw/Test/0/test.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    dataset = ManifestEventDataset(root, path, size=32, local_size=16,
                                   local_crops=1, view_validity='resample_empty',
                                   max_view_attempts=3)
    first, second = dataset[0], dataset[0]
    assert len(first) == 3
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
        assert not a.any()  # bounded retries cannot invent events


def test_shared_core_global_views_preserve_content_and_vary_by_epoch(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin', count=2000)
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    dataset = ManifestEventDataset(root, path, size=32, local_crops=0,
                                   global_view_policy='shared_core')
    first = dataset[(0, 0)]
    repeat = dataset[(0, 0)]
    next_epoch = dataset[(0, 1)]
    for a, b in zip(first, repeat):
        torch.testing.assert_close(a, b)
        assert a.any()
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first[0], next_epoch[0])
    overlap = ((first[0] > 0) & (first[1] > 0)).sum()
    union = ((first[0] > 0) | (first[1] > 0)).sum()
    assert overlap / union > .5


def test_shared_temporal_windows_are_distinct_reproducible_global_views(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin', count=2000)
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    dataset = ManifestEventDataset(
        root, path, size=32, local_crops=0,
        global_view_policy='shared_temporal_windows')
    first = dataset[(0, 0)]
    repeat = dataset[(0, 0)]
    assert len(first) == 2
    for a, b in zip(first, repeat):
        torch.testing.assert_close(a, b)
        assert a.any()
    assert not torch.equal(first[0], first[1])


def test_shared_time_independent_space_changes_only_spatial_pairing(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin', count=2000)
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    common = dict(size=32, local_crops=0, global_crop_scale=(.32, 1.))
    shared = ManifestEventDataset(root, path,
                                  global_view_policy='shared_core', **common)
    displaced = ManifestEventDataset(
        root, path, global_view_policy='shared_time_independent_space',
        **common)
    first = displaced[(0, 0)]
    repeat = displaced[(0, 0)]
    for a, b in zip(first, repeat):
        torch.testing.assert_close(a, b)
        assert a.any()
    torch.testing.assert_close(first[0], shared[(0, 0)][0])
    assert not torch.equal(first[1], shared[(0, 0)][1])


def test_shared_time_shifted_space_preserves_crop_content(tmp_path):
    original = (10, 10, 20, 20)
    shifted = ManifestEventDataset._shift_global_crop(
        original, (40, 40), seed=42)
    assert shifted == ManifestEventDataset._shift_global_crop(
        original, (40, 40), seed=42)
    assert shifted != original
    assert shifted[2:] == original[2:]
    assert abs(shifted[0] - original[0]) + abs(shifted[1] - original[1]) == 3

    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin', count=2000)
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    common = dict(size=32, local_crops=0, global_crop_scale=(.32, 1.))
    shared = ManifestEventDataset(root, path,
                                  global_view_policy='shared_core', **common)
    displaced = ManifestEventDataset(
        root, path, global_view_policy='shared_time_shifted_space',
        **common)
    first = displaced[(0, 0)]
    repeat = displaced[(0, 0)]
    for a, b in zip(first, repeat):
        torch.testing.assert_close(a, b)
        assert a.any()
    torch.testing.assert_close(first[0], shared[(0, 0)][0])
    assert not torch.equal(first[1], shared[(0, 0)][1])


def test_nested_local_views_share_global_core_and_are_deterministic(tmp_path):
    root = tmp_path / 'nmnist'
    _five_byte(root / 'raw/Train/0/a.bin', count=2000)
    _five_byte(root / 'raw/Test/0/b.bin')
    path = tmp_path / 'nmnist.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), path)
    common = dict(size=32, local_size=16, local_crops=2,
                  global_crop_scale=(.5, 1.), local_crop_scale=(.4, .8),
                  view_validity='resample_empty', max_view_attempts=4)
    nested = ManifestEventDataset(
        root, path, global_view_policy='shared_core_nested_local', **common)
    shared = ManifestEventDataset(
        root, path, global_view_policy='shared_core', **common)
    first = nested[(0, 0)]
    repeat = nested[(0, 0)]
    later = nested[(0, 1)]
    assert len(first) == 4
    assert [tuple(view.shape) for view in first] == [
        (4, 2, 32, 32), (4, 2, 32, 32),
        (4, 2, 16, 16), (4, 2, 16, 16)]
    for a, b in zip(first, repeat):
        torch.testing.assert_close(a, b)
    for a, b in zip(first[:2], shared[(0, 0)][:2]):
        torch.testing.assert_close(a, b)
    assert all(view.any() for view in first)
    assert any(not torch.equal(a, b) for a, b in zip(first, later))
