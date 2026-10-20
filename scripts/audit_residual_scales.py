"""Read-only learned residual-current scale audit for SNN checkpoints."""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    rows = []
    for prefix in ('student', 'teacher'):
        for block in range(checkpoint['config']['depth']):
            row = {'model': prefix, 'block': block}
            for branch in ('attn', 'mlp'):
                key = f'{prefix}.backbone.blocks.{block}.{branch}_scale'
                if key not in state:
                    continue
                values = state[key].float()
                row[branch] = {
                    'mean': float(values.mean()),
                    'mean_abs': float(values.abs().mean()),
                    'minimum': float(values.min()),
                    'maximum': float(values.max()),
                    'negative_fraction': float((values < 0).float().mean()),
                    'above_quarter_fraction': float(
                        (values.abs() >= .25).float().mean()),
                }
            rows.append(row)
    report = {'checkpoint': str(args.checkpoint),
              'steps': checkpoint['steps'], 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
