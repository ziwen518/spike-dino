"""Plot controlled CIFAR10-DVS Spike-DINO patch8 learning evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (
        args.reports_dir / 'v3_patch8_base_6ep_metrics.jsonl').read_text(
            encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 3372 or [row['step'] for row in rows] != list(
            range(1, 3373)):
        raise ValueError('Expected contiguous six-epoch patch8 metrics')
    epoch_steps = 562
    epochs = list(range(1, 7))
    target = {
        name: [float(np.mean([row[key] for row in rows[
            index * epoch_steps:(index + 1) * epoch_steps]]))
               for index in range(6)]
        for name, key in (
            ('cls', 'teacher_target_information'),
            ('patch', 'teacher_patch_target_information'))}
    probe_files = {
        1: {'cls': 'v3_grid_patch8_1ep_cls_frozen_20261017.json',
            'patch': 'v3_grid_patch8_1ep_patch_mean_frozen_20261017.json'},
        3: {'cls': 'v3_patch8_base_3ep_cls_frozen_20261018.json',
            'patch': 'v3_patch8_base_3ep_patch_mean_frozen_20261018.json'},
        6: {'cls': 'v3_patch8_base_6ep_cls_frozen_20261018.json',
            'patch': 'v3_patch8_base_6ep_patch_mean_frozen_20261018.json'},
    }
    probes = {}
    for epoch, sources in probe_files.items():
        probes[epoch] = {}
        for source, filename in sources.items():
            report = json.loads((args.reports_dir / filename).read_text(
                encoding='utf-8'))
            if report['pretrain_steps'] != epoch * epoch_steps or report[
                    'official_test_accessed']:
                raise ValueError(f'Probe protocol mismatch: {filename}')
            values = report['results']['eval']
            probes[epoch][source] = {
                'knn_top1': float(values['knn_top1']),
                'linear_top1': float(values['linear_top1']),
                'effective_rank': float(values['effective_rank']),
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    axes[0].plot(epochs, target['cls'], 'o-', color='#2848b8',
                 label='CLS teacher target')
    axes[0].plot(epochs, target['patch'], 's-', color='#ba5b26',
                 label='Patch teacher target')
    axes[0].set_ylabel('Information (nat)')
    axes[0].set_title('Teacher target information')
    axes[0].legend(fontsize=8)
    observed_epochs = sorted(probes)
    styles = (
        ('cls', 'knn_top1', 'CLS kNN', '#2848b8', 'o-'),
        ('cls', 'linear_top1', 'CLS linear', '#6385db', 'o--'),
        ('patch', 'knn_top1', 'Patch kNN', '#ba5b26', 's-'),
        ('patch', 'linear_top1', 'Patch linear', '#db9866', 's--'),
    )
    for source, key, label, color, style in styles:
        axes[1].plot(observed_epochs,
                     [100 * probes[e][source][key] for e in observed_epochs],
                     style, color=color, label=label, linewidth=1.7)
    axes[1].set_ylabel('Frozen top-1 (%)')
    axes[1].set_title('Internal frozen probes (epochs 1, 3, 6)')
    axes[1].legend(fontsize=8, loc='best')
    for source, label, color, marker in (
            ('cls', 'CLS', '#2848b8', 'o'),
            ('patch', 'Patch', '#ba5b26', 's')):
        axes[2].plot(observed_epochs,
                     [probes[e][source]['effective_rank']
                      for e in observed_epochs], marker + '-',
                     color=color, label=label)
    axes[2].set_ylabel('Effective rank')
    axes[2].set_title('Feature diversity')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Pretraining epoch')
        axis.set_xticks(epochs)
        axis.grid(alpha=.25)
    fig.suptitle('Spike-DINO patch8 | CIFAR10-DVS | seed42 | official test unused',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(args.output_dir / 'patch8_learning_curve_1_3_6.png', dpi=240,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    (args.output_dir / 'patch8_learning_curve_1_3_6.json').write_text(
        json.dumps({'epochs': epochs, 'target_information_nat': target,
                    'frozen_probes': probes,
                    'validation_source': 'unused labeled training partition',
                    'official_test_accessed': False}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
