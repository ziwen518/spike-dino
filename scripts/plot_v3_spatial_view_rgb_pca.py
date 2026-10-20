"""Compare DINO-style RGB PCA patch maps under paired DVS view policies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spike_dino.benchmark_manifest import CIFAR_CLASSES
from scripts.plot_v3_patch_rgb_pca import (event_input_rgb, fit_rgb_pca,
                                           foreground_mask, project_rgb)


NAMES = (
    ('patch8_1ep', 'Same spatial crop'),
    ('patch8_shiftedviews_1ep', 'Shifted spatial crop'),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    loaded = {}
    for name, _title in NAMES:
        source = args.input_dir / f'{name}_cifar10dvs_seed42_patchmaps.npz'
        with np.load(source, allow_pickle=False) as handle:
            loaded[name] = {key: handle[key].copy() for key in handle.files}
    base, shifted = (loaded[name] for name, _ in NAMES)
    for key in ('labels', 'record_paths', 'input_frames', 'patch_activity'):
        if not np.array_equal(base[key], shifted[key]):
            raise ValueError(f'Models do not share {key}')
    activity = base['patch_activity']
    if activity.shape != (10, 16, 16):
        raise ValueError('Expected ten records on the 16x16 patch grid')
    masks = np.stack([foreground_mask(row) for row in activity])
    inputs = [event_input_rgb(row) for row in base['input_frames']]
    maps = {}
    metadata = {'records': 10, 'patch_grid': [16, 16],
                'pca_basis': 'fitted separately for each model; colors not aligned',
                'foreground': 'Otsu threshold on log event activity, not classifier'}
    for name, title in NAMES:
        features = loaded[name]['final_patch']
        if features.shape != (10, 256, 384):
            raise ValueError(f'Unexpected patch features: {name}')
        pca = fit_rgb_pca(features, masks)
        maps[name] = [project_rgb(pca, row, 16) for row in features]
        metadata[name] = {
            'title': title,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'foreground_patch_count': masks.sum((1, 2)).tolist(),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for masked in (False, True):
        for page in range(2):
            fig, axes = plt.subplots(5, 4, figsize=(13, 15))
            for row in range(5):
                index = page * 5 + row
                panels = [inputs[index], masks[index]]
                for name, _title in NAMES:
                    image = maps[name][index].copy()
                    if masked:
                        image[~masks[index]] = 0
                    panels.append(image)
                for col, (axis, panel) in enumerate(zip(axes[row], panels)):
                    if col == 1:
                        axis.imshow(panel, cmap='gray', vmin=0, vmax=1,
                                    interpolation='nearest')
                    else:
                        axis.imshow(panel, interpolation='nearest',
                                    vmin=0, vmax=1)
                    axis.axis('off')
                    if row == 0:
                        axis.set_title(('DVS input', 'Event activity',
                                        NAMES[0][1], NAMES[1][1])[col],
                                       fontsize=10)
                    if col == 0:
                        axis.text(-.08, .5, CIFAR_CLASSES[base['labels'][index]],
                                  transform=axis.transAxes, va='center',
                                  ha='right', fontsize=10)
            fig.suptitle('Teacher patch RGB-PCA | CIFAR10-DVS | 128px / patch8 '
                         '| equal 562 training steps\n'
                         'PCA fitted separately per model; compare spatial '
                         'structure, not exact colors', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .94))
            stem = ('shifted_view_rgb_pca_event_mask' if masked
                    else 'shifted_view_rgb_pca_all')
            fig.savefig(args.output_dir / f'{stem}_{page+1:02d}.png',
                        dpi=220, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    (args.output_dir / 'shifted_view_rgb_pca_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
