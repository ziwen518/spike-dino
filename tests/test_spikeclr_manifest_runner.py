import json
import sys

import numpy as np

from spike_dino.benchmark_manifest import build_manifest, write_manifest
from spike_dino.pretrain import main


def _events(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((200, 5), dtype=np.uint8)
    data[:, 0] = np.arange(200) % 30
    data[:, 1] = np.arange(200) % 30
    data[:, 2] = (np.arange(200) % 2) * 128
    data[:, 4] = np.arange(200)
    data.tofile(path)


def test_spikeclr_uses_fixed_manifest_on_non_cifar_dataset(tmp_path, monkeypatch):
    root = tmp_path / 'nmnist'
    for label in range(2):
        for index in range(2):
            _events(root / f'raw/Train/{label}/{index}.bin')
        _events(root / f'raw/Test/{label}/0.bin')
    manifest = tmp_path / 'nmnist_manifest.json'
    write_manifest(build_manifest(root, 'nmnist', strict=False), manifest)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', [
        'pretrain', '--dataset', 'nmnist', '--dataset-root', str(root),
        '--manifest-path', str(manifest), '--output', str(output),
        '--method', 'spikeclr', '--local-crops', '0', '--device', 'cpu',
        '--epochs', '1', '--batch-size', '2', '--width', '.125',
        '--size', '32', '--time-steps', '4', '--workers', '0',
        '--torch-threads', '2'])
    main()
    config = json.loads((output / 'config.json').read_text())
    row = json.loads((output / 'epochs.jsonl').read_text().splitlines()[0])
    assert config['dataset'] == 'nmnist'
    assert config['train_records'] == 4
    assert config['manifest_sha256'] == json.loads(manifest.read_text())['sha256']
    assert np.isfinite(row['loss'])
    assert (output / 'last.pt').exists()
