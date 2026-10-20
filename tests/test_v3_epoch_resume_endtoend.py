import json
import sys

import numpy as np
import pytest
import torch

from spike_dino.benchmark_manifest import build_manifest, write_manifest
from spike_dino.pretrain_v3 import main


def _events(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((200, 5), dtype=np.uint8)
    values[:, 0] = np.arange(200) % 30
    values[:, 1] = np.arange(200) % 30
    values[:, 2] = (np.arange(200) % 2) * 128
    values[:, 4] = np.arange(200)
    values.tofile(path)


@pytest.mark.parametrize('with_schedule', (False, True))
@pytest.mark.parametrize('block_variant', ('standard', 'evidence'))
def test_real_event_resume_crosses_epoch_without_changing_weights(
        tmp_path, monkeypatch, with_schedule, block_variant):
    root = tmp_path / 'nmnist'
    for label in range(2):
        for index in range(2):
            _events(root / f'raw/Train/{label}/{index}.bin')
        _events(root / f'raw/Test/{label}/0.bin')
    manifest = tmp_path / 'manifest.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), manifest)
    common = ['--dataset', 'nmnist', '--dataset-root', str(root),
              '--manifest-path', str(manifest), '--device', 'cpu',
              '--size', '32', '--patch-size', '16', '--local-size', '16',
              '--time-steps', '4', '--batch-size', '2', '--local-crops', '1',
              '--depth', '1', '--dim', '48', '--heads', '4',
              '--head-hidden', '96', '--bottleneck', '24', '--prototypes', '32',
              '--workers', '0', '--checkpoint-every', '2',
              '--freeze-last-layer-epochs', '1',
              '--norm-policy', 'layer_norm']
    common += ['--block-variant', block_variant]
    if with_schedule:
        common += ['--lr-warmup-steps', '2',
                   '--teacher-temp-start', '.04',
                   '--teacher-temp-final', '.07',
                   '--teacher-temp-warmup-steps', '2',
                   '--teacher-momentum', '.994',
                   '--adam-beta2', '.99',
                   '--weight-decay', '.04']
    full = tmp_path / 'full'
    resumed = tmp_path / 'resumed'
    for output, steps, resume in ((full, 4, None), (resumed, 2, None),
                                  (resumed, 4, resumed / 'last.pt')):
        argv = ['pretrain_v3', *common, '--output', str(output),
                '--steps', str(steps)]
        if resume is not None:
            argv += ['--resume', str(resume)]
        monkeypatch.setattr(sys, 'argv', argv)
        main()
    uninterrupted = torch.load(full / 'last.pt', map_location='cpu', weights_only=True)
    interrupted = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=True)
    assert uninterrupted['steps'] == interrupted['steps'] == 4
    assert all(torch.equal(value, interrupted['model'][name])
               for name, value in uninterrupted['model'].items())
    rows = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3, 4]
    assert [row['epoch'] for row in rows] == [1, 1, 2, 2]
    assert [row['step_in_epoch'] for row in rows] == [1, 2, 1, 2]
    assert [row['prototypes_frozen'] for row in rows] == [True, True, False, False]
    if with_schedule:
        assert [row['lr_effective'] for row in rows] == [0., .00005, .0001, .0001]
        assert [round(row['teacher_temp_effective'], 4)
                for row in rows] == [.04, .055, .07, .07]
        assert all(row['teacher_momentum_effective'] == .994 for row in rows)
    frozen = torch.load(full / 'eval_step_000002.pt', map_location='cpu',
                        weights_only=True)['model']
    for head in ('dino_head', 'ibot_head'):
        suffix = f'{head}.prototypes'
        torch.testing.assert_close(frozen[f'student.{suffix}'],
                                   frozen[f'teacher.{suffix}'], rtol=0, atol=0)
        assert not torch.equal(uninterrupted['model'][f'student.{suffix}'],
                               uninterrupted['model'][f'teacher.{suffix}'])
