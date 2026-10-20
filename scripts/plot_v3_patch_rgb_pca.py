"""Render DINO-style RGB maps from spatial SNN patch features on DVS data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA

from spike_dino.benchmark_manifest import CIFAR_CLASSES


MODELS = (
    ('base3ep', 'Full-spatial baseline'),
    ('balanced3ep', 'Temporal + balanced'),
    ('stemglobal3ep', 'Temporal + stem guard'),
    ('gram3ep', 'Temporal + SNN Gram'),
)
LAYERS = ('block_0_patch', 'block_6_patch', 'block_11_patch',
          'final_patch')


def foreground_mask(activity):
    log_activity = np.log1p(activity.astype(np.float64))
    if np.ptp(log_activity) == 0:
        return np.ones_like(activity, dtype=bool)
    return log_activity > threshold_otsu(log_activity)


def event_input_rgb(frames):
    polarity = np.log1p(np.clip(frames.astype(np.float64).sum(0), 0, None))
    scaled = []
    for channel in polarity:
        positive = channel[channel > 0]
        scale = np.percentile(positive, 99) if len(positive) else 1.
        scaled.append(np.clip(channel / max(scale, 1e-9), 0, 1))
    p0, p1 = scaled
    return np.clip(np.stack((p0 + .1 * p1,
                             .45 * p0 + .85 * p1,
                             .12 * p0 + p1), -1), 0, 1)


def fit_rgb_pca(features, masks):
    selected = np.concatenate([row[mask.reshape(-1)]
                               for row, mask in zip(features, masks)])
    if len(selected) < 4 or float(selected.std(0).mean()) < 1e-8:
        raise ValueError('Patch features do not have enough foreground variance')
    pca = PCA(n_components=3, whiten=True, random_state=42)
    pca.fit(selected)
    return pca


def project_rgb(pca, features, grid):
    # Match the official DINOv3 notebook: whiten 3 PCs, then sigmoid(2*x).
    return expit(2 * pca.transform(features)).reshape(grid, grid, 3)


def draw_pages(output_dir, maps, inputs, masks, labels, title, stem,
               masked):
    column_titles = ['DVS input', 'Event activity'] + [title for _, title in MODELS]
    for page in range(2):
        fig, axes = plt.subplots(5, 6, figsize=(17, 15))
        for row in range(5):
            index = page * 5 + row
            axes[row, 0].imshow(inputs[index], interpolation='nearest')
            axes[row, 1].imshow(masks[index], cmap='gray', vmin=0, vmax=1,
                                interpolation='nearest')
            for col, (short, _title) in enumerate(MODELS, start=2):
                image = maps[short][index].copy()
                if masked:
                    image[~masks[index]] = 0
                axes[row, col].imshow(image, interpolation='nearest',
                                       vmin=0, vmax=1)
            for col, ax in enumerate(axes[row]):
                ax.axis('off')
                if row == 0:
                    ax.set_title(column_titles[col], fontsize=10)
            axes[row, 0].text(-.08, .5, CIFAR_CLASSES[labels[index]],
                              transform=axes[row, 0].transAxes,
                              va='center', ha='right', fontsize=10)
        subtitle = ('Event-activity Otsu mask (proxy, not DINOv3 classifier)'
                    if masked else 'No foreground mask')
        fig.suptitle('DINO-style RGB PCA of teacher patch spikes | '
                     f'CIFAR10-DVS | {subtitle}\n'
                     'PCA fit separately per model on the same selected event patches; '
                     '8x8 patch grid, colors not comparable across columns',
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output_dir / f'{stem}_{page+1:02d}.png', dpi=220,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    loaded = {}
    labels = paths = frames = activity = None
    for short, _title in MODELS:
        path = args.input_dir / f'{short}_cifar10dvs_seed42_patchmaps.npz'
        with np.load(path, allow_pickle=False) as contents:
            loaded[short] = {key: contents[key].copy()
                             for key in contents.files}
        current = loaded[short]
        if labels is None:
            labels, paths = current['labels'], current['record_paths']
            frames, activity = current['input_frames'], current['patch_activity']
        elif any(not np.array_equal(current[key], expected)
                 for key, expected in (('labels', labels),
                                       ('record_paths', paths),
                                       ('input_frames', frames),
                                       ('patch_activity', activity))):
            raise ValueError('All models must use identical DVS recordings and frames')
    if not np.array_equal(labels, np.arange(10)):
        raise ValueError('Expected one median-activity recording per class')
    grid = activity.shape[-1]
    if activity.shape != (10, grid, grid):
        raise ValueError('Unexpected event patch grid')
    masks = np.stack([foreground_mask(row) for row in activity])
    inputs = [event_input_rgb(row) for row in frames]
    maps = {}
    fitted = {}
    metadata = {}
    for short, title in MODELS:
        features = loaded[short]['final_patch']
        if features.shape[:2] != (10, grid * grid):
            raise ValueError('Patch features and DVS grid do not align')
        pca = fit_rgb_pca(features, masks)
        fitted[short] = pca
        maps[short] = [project_rgb(pca, row, grid) for row in features]
        metadata[short] = {
            'title': title,
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'foreground_spatial_feature_std': float(np.mean([
                row[mask.reshape(-1)].std(0).mean()
                for row, mask in zip(features, masks)])),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_pages(args.output_dir, maps, inputs, masks, labels,
               'all patches', 'patch_rgb_pca_all', False)
    draw_pages(args.output_dir, maps, inputs, masks, labels,
               'event support', 'patch_rgb_pca_event_mask', True)

    # A fixed PCA basis per model across depths reveals whether blocks change
    # spatial features. It is fit on final foreground patches as above.
    chosen = (0, 3, 7)
    columns = [('DVS input', None)] + [
        (f'{title}\n{layer}', (short, layer))
        for short, title in (MODELS[0], MODELS[2]) for layer in LAYERS]
    fig, axes = plt.subplots(len(chosen), len(columns),
                             figsize=(2.5 * len(columns), 3 * len(chosen)))
    for row, index in enumerate(chosen):
        for col, (title, key) in enumerate(columns):
            ax = axes[row, col]
            if key is None:
                ax.imshow(inputs[index])
            else:
                short, layer = key
                image = project_rgb(fitted[short], loaded[short][layer][index],
                                    grid)
                image[~masks[index]] = 0
                ax.imshow(image, interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(title.replace('_', ' '), fontsize=9)
            if col == 0:
                ax.text(-.08, .5, CIFAR_CLASSES[labels[index]],
                        transform=ax.transAxes, va='center', ha='right',
                        fontsize=10)
    fig.suptitle('Patch RGB-PCA across spiking layers | '
                 'each model uses its own final-layer PCA basis', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.output_dir / 'patch_rgb_pca_layer_progression.png',
                dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    report = {
        'source': 'train-internal CIFAR10-DVS validation; official test untouched',
        'selection': 'median input event mass per class, fixed before model visualization',
        'selection_record_paths': paths.tolist(),
        'grid': [grid, grid], 'patch_size': frames.shape[-1] // grid,
        'pca': '3 components, whitening, sigmoid(2*projection), fitted per model on activity-selected patches',
        'foreground': 'Otsu threshold on log1p DVS patch activity; unsupervised proxy, not official DINOv3 classifier',
        'foreground_patches_per_record': masks.sum((1, 2)).tolist(),
        'models': metadata,
    }
    (args.output_dir / 'patch_rgb_pca_metadata.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'files': [
        'patch_rgb_pca_all_01.png', 'patch_rgb_pca_all_02.png',
        'patch_rgb_pca_event_mask_01.png',
        'patch_rgb_pca_event_mask_02.png',
        'patch_rgb_pca_layer_progression.png',
        'patch_rgb_pca_metadata.json']}))


if __name__ == '__main__':
    main()
