"""Read-only checkpoint finite-value and parameter-change diagnostic."""
import argparse
import json

import torch

from spike_dino.dinov3 import SpikingDinoV3Prototype


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    c = checkpoint['config']
    torch.manual_seed(c['seed'])
    model = SpikingDinoV3Prototype(
        image_size=c['size'], patch_size=c.get('patch_size', 16),
        dim=c['dim'], depth=c['depth'], heads=c['heads'],
        block_variant=c.get('block_variant', 'standard'),
        attention_variant=c.get('attention_variant', 'matrix_legacy'),
        cls_aggregation=c.get('cls_aggregation', 'attention_only'),
        residual_policy=c.get('residual_policy', 'accumulate'),
        masked_context_bridge=c.get('masked_context_bridge', 'none'),
        head_hidden=c['head_hidden'], head_layers=c.get('head_layers', 2),
        prototype_readout=c.get('prototype_readout', 'cosine'),
        bottleneck=c['bottleneck'], prototypes=c['prototypes'],
        dino_prototypes=c.get('dino_prototypes'),
        ibot_prototypes=c.get('ibot_prototypes'),
        dino_hidden=c.get('dino_hidden'), ibot_hidden=c.get('ibot_hidden'),
        dino_bottleneck=c.get('dino_bottleneck'),
        ibot_bottleneck=c.get('ibot_bottleneck'),
        norm_policy=c.get('norm_policy', 'ema_bn'),
        target_policy=c.get('target_policy', 'sinkhorn'))
    initial = model.state_dict()
    rows = []
    for name, value in checkpoint['model'].items():
        if not value.is_floating_point():
            continue
        finite = torch.isfinite(value)
        old = initial[name]
        rows.append({'name': name, 'nonfinite': int((~finite).sum()),
                     'size': value.numel(),
                     'max_abs_change': float((value[finite] - old[finite]).abs().max())
                     if finite.any() else None})
    result = {'steps': checkpoint['steps'],
              'nonfinite_total': sum(row['nonfinite'] for row in rows),
              'student_nonfinite': sum(row['nonfinite'] for row in rows
                                       if row['name'].startswith('student.')),
              'teacher_nonfinite': sum(row['nonfinite'] for row in rows
                                       if row['name'].startswith('teacher.')),
              'nonfinite_tensor_count': sum(bool(row['nonfinite']) for row in rows),
              'first_nonfinite_tensors': [row for row in rows if row['nonfinite']][:20],
              'largest_changes': sorted(rows,
                                        key=lambda row: row['max_abs_change'] or 0,
                                        reverse=True)[:20]}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
