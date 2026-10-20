"""Audit whether masked patch positions survive the first spike encoder.

This is a read-only parameter/encoder diagnostic, without event data or labels.
It does not measure masked patch prediction quality.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional

from scripts.diagnose_v3_representation import model_from_checkpoint


@torch.no_grad()
def mask_code_metrics(backbone, time_steps):
    grid = backbone.image_size // backbone.patch_size
    positions = backbone.position_current[:, 1:]
    current = backbone.mask_current + positions
    current = current.expand(time_steps, -1, -1)
    functional.reset_net(backbone.token_spike)
    spikes = backbone.token_spike(current)
    functional.reset_net(backbone.token_spike)
    flat = spikes.permute(1, 0, 2).reshape(grid * grid, -1).float()
    normalized = F.normalize(flat, dim=-1)
    similarity = normalized @ normalized.T
    count = flat.shape[0]
    other = (similarity.sum() - similarity.diag().sum()) / (count * (count - 1))
    return {
        'patch_grid': [grid, grid],
        'time_steps': time_steps,
        'unique_binary_position_codes': int(torch.unique(flat, dim=0).shape[0]),
        'mean_position_code_cosine': float(other),
        'mean_position_bit_variance': float(flat.var(0, unbiased=False).mean()),
        'spike_rate_by_time': [float(v) for v in spikes.float().mean((1, 2))],
        'mask_current_mean': float(backbone.mask_current.mean()),
        'mask_current_std': float(backbone.mask_current.std(unbiased=False)),
        'position_current_std': float(positions.std(unbiased=False)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-steps', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = model_from_checkpoint(payload).eval()
    result = {'checkpoint': str(args.checkpoint),
              'checkpoint_steps': payload['steps'],
              'teacher': mask_code_metrics(model.teacher.backbone,
                                           args.time_steps),
              'student': mask_code_metrics(model.student.backbone,
                                           args.time_steps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
