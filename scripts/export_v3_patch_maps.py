"""Export DVS input and teacher patch-spike grids for DINO-style RGB PCA."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from spike_dino.evaluate import FixedEventDataset
from spike_dino.evaluate_v3 import cohorts
from scripts.diagnose_v3_representation import model_from_checkpoint


LAYERS = (0, 3, 6, 9, 11)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--selection-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    if config.get('dataset', 'cifar10dvs') != 'cifar10dvs':
        p.error('This export currently supports CIFAR10-DVS')
    torch.set_num_threads(2)
    _, records = cohorts(args.dataset_root, 42, 512, 100,
                         config.get('manifest_path'))
    dataset = FixedEventDataset(args.dataset_root, records,
                                config['time_steps'], config['size'])
    if args.selection_file.exists():
        selected = json.loads(args.selection_file.read_text(encoding='utf-8'))
        if selected['cohort_seed'] != 42 or selected['selection_rule'] != 'median_event_mass_per_class':
            p.error('Selection file does not match the fixed PCA cohort')
        lookup = {record['path']: index for index, record in enumerate(records)}
        indices = [lookup[item['path']] for item in selected['records']]
        if [records[index]['label'] for index in indices] != [
                item['label'] for item in selected['records']]:
            p.error('Selected labels changed')
    else:
        loader = DataLoader(dataset, batch_size=16, shuffle=False,
                            num_workers=args.workers)
        masses = []
        for frames, _targets in loader:
            masses.extend(frames.abs().sum((1, 2, 3, 4)).tolist())
        indices = []
        for label in range(10):
            candidates = [index for index, record in enumerate(records)
                          if record['label'] == label]
            midpoint = float(np.median([masses[index] for index in candidates]))
            indices.append(min(candidates,
                               key=lambda index: (abs(masses[index] - midpoint),
                                                  records[index]['path'])))
        selected = {
            'cohort_seed': 42, 'validation_per_class': 100,
            'selection_rule': 'median_event_mass_per_class',
            'record_source': 'unused labeled subset of pretraining train partition',
            'official_test_accessed': False,
            'records': [{'path': records[index]['path'],
                         'label': records[index]['label'],
                         'event_mass': masses[index]} for index in indices],
        }
        args.selection_file.parent.mkdir(parents=True, exist_ok=True)
        args.selection_file.write_text(json.dumps(selected, indent=2),
                                       encoding='utf-8')
    frames = torch.stack([dataset[index][0] for index in indices], dim=1)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.teacher.backbone
    captured = {}

    def record_token(_module, _inputs, output):
        captured['token_patch'] = output[:, :, 1:].float().mean(0).cpu()

    def record_block(index):
        def hook(_module, _inputs, output):
            captured[f'block_{index}_patch'] = (
                output[:, :, 1:].float().mean(0).cpu())
        return hook

    handles = [backbone.token_spike.register_forward_hook(record_token)]
    handles.extend(backbone.blocks[index].attn_input.register_forward_hook(
        record_block(index)) for index in LAYERS)
    try:
        with torch.no_grad():
            functional.reset_net(backbone)
            _cls, patch = backbone(frames.to(args.device))
            captured['final_patch'] = patch.float().mean(0).cpu()
            functional.reset_net(backbone)
    finally:
        for handle in handles:
            handle.remove()
        functional.reset_net(backbone)
    grid = config['size'] // config['patch_size']
    if captured['final_patch'].shape[1] != grid * grid:
        raise ValueError('Patch grid does not match checkpoint configuration')
    event_mass = frames.abs().sum((0, 2))
    activity = torch.nn.functional.avg_pool2d(
        event_mass[:, None], config['patch_size'],
        stride=config['patch_size'])[:, 0]
    output = {name: value.numpy().astype(np.float32)
              for name, value in captured.items()}
    output.update({
        'input_frames': frames.permute(1, 0, 2, 3, 4).numpy().astype(np.float32),
        'patch_activity': activity.numpy().astype(np.float32),
        'labels': np.asarray([records[index]['label'] for index in indices],
                             dtype=np.int64),
        'record_paths': np.asarray([records[index]['path'] for index in indices]),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        'checkpoint': str(args.checkpoint), 'steps': payload['steps'],
        'records': len(indices), 'size': config['size'],
        'patch_size': config['patch_size'], 'grid': grid,
        'features': {name: list(value.shape) for name, value in output.items()
                     if name.endswith('_patch')},
        'teacher_features': True, 'feature_interface': 'time-averaged binary spikes',
        'selection_file': str(args.selection_file),
        'official_test_accessed': False,
    }
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
