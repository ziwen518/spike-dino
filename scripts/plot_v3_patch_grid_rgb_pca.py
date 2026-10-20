"""Compare DINO-style RGB patch maps at native 8x8 and 16x16 DVS grids."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    loaded = {}
    for label in ('patch16_1ep', 'patch8_1ep'):
        path = args.input_dir / f'{label}_cifar10dvs_seed42_patchmaps.npz'
        with np.load(path, allow_pickle=False) as contents:
            loaded[label] = {key: contents[key].copy()
                             for key in contents.files}
    small, large = loaded.values()
    for key in ('labels', 'record_paths', 'input_frames'):
        if not np.array_equal(small[key], large[key]):
            raise ValueError(f'Patch-grid models do not share {key}')
    labels = small['labels']
    inputs = [event_input_rgb(frame) for frame in small['input_frames']]
    masks = {}
    maps = {}
    metadata = {}
    for label, contents in loaded.items():
        activity = contents['patch_activity']
        grid = activity.shape[-1]
        masks[label] = np.stack([foreground_mask(row) for row in activity])
        pca = fit_rgb_pca(contents['final_patch'], masks[label])
        maps[label] = [project_rgb(pca, features, grid)
                       for features in contents['final_patch']]
        metadata[label] = {
            'grid': [grid, grid], 'patch_size': 128 // grid,
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'foreground_patches_per_record': masks[label].sum((1, 2)).tolist(),
        }
    if metadata['patch16_1ep']['grid'] != [8, 8] or metadata['patch8_1ep']['grid'] != [16, 16]:
        raise ValueError('Expected 8x8 and 16x16 grids')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    titles = ('DVS input', '8x8 activity', 'patch16 RGB-PCA',
              '16x16 activity', 'patch8 RGB-PCA')
    for masked in (False, True):
        for page in range(2):
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            for row in range(5):
                index = page * 5 + row
                panels = [inputs[index], masks['patch16_1ep'][index],
                          maps['patch16_1ep'][index].copy(),
                          masks['patch8_1ep'][index],
                          maps['patch8_1ep'][index].copy()]
                if masked:
                    panels[2][~masks['patch16_1ep'][index]] = 0
                    panels[4][~masks['patch8_1ep'][index]] = 0
                for col, (ax, panel) in enumerate(zip(axes[row], panels)):
                    if col in (1, 3):
                        ax.imshow(panel, cmap='gray', vmin=0, vmax=1,
                                  interpolation='nearest')
                    else:
                        ax.imshow(panel, interpolation='nearest',
                                  vmin=0, vmax=1)
                    ax.axis('off')
                    if row == 0:
                        ax.set_title(titles[col], fontsize=11)
                    if col == 0:
                        ax.text(-.08, .5, CIFAR_CLASSES[labels[index]],
                                transform=ax.transAxes, va='center',
                                ha='right', fontsize=10)
            fig.suptitle('DVS teacher patch RGB-PCA at equal 562 training steps | '
                         + ('Otsu activity mask' if masked else 'all patches') +
                         '\nPCA basis fit separately per grid; colors are not comparable'
                         '\n128/16 gives 8x8 patches; 128/8 gives 16x16 patches',
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .92))
            stem = ('patch_grid_rgb_pca_event_mask' if masked
                    else 'patch_grid_rgb_pca_all')
            fig.savefig(args.output_dir / f'{stem}_{page+1:02d}.png',
                        dpi=220, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    report = {
        'dataset': 'CIFAR10-DVS', 'records': 10,
        'selection': 'same median event-mass recording per class',
        'training_steps': 562, 'models': metadata,
        'official_test_accessed': False,
        'pca': 'per-model 3 PCs, whiten, sigmoid(2*x) on event-activity-selected patches',
        'foreground': 'Otsu on log1p event patch activity; not a trained classifier',
    }
    (args.output_dir / 'patch_grid_rgb_pca_metadata.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'files': [
        'patch_grid_rgb_pca_all_01.png', 'patch_grid_rgb_pca_all_02.png',
        'patch_grid_rgb_pca_event_mask_01.png',
        'patch_grid_rgb_pca_event_mask_02.png',
        'patch_grid_rgb_pca_metadata.json']}))


if __name__ == '__main__':
    main()
