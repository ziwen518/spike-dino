"""Plot fixed-cohort PCA of Spike-DINO student spiking-layer features."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from spike_dino.benchmark_manifest import CIFAR_CLASSES


MODELS = (
    ('base3ep', 'Full-spatial baseline'),
    ('balanced3ep', 'Temporal + balanced'),
    ('stemglobal3ep', 'Temporal + stem guard'),
    ('gram3ep', 'Temporal + SNN Gram'),
)
CLS_LAYERS = ('block_0_cls', 'block_3_cls', 'block_6_cls',
              'block_9_cls', 'block_11_cls', 'final_cls')
PATCH_LAYERS = ('token_patch_mean', 'block_3_patch_mean',
                'block_11_patch_mean', 'final_patch_mean')


def pca_metrics(features, labels):
    x = features.astype(np.float64)
    centered = x - x.mean(0)
    pca = PCA(n_components=2, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(x)
    covariance = centered.T @ centered
    singular = np.sqrt(np.clip(np.linalg.eigvalsh(covariance), 0, None))
    singular_sum = singular.sum()
    if singular_sum:
        probabilities = singular / singular_sum
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    else:
        effective_rank = 0.
    total = float((centered ** 2).sum())
    between = sum(
        int((labels == label).sum()) *
        float(((x[labels == label].mean(0) - x.mean(0)) ** 2).sum())
        for label in np.unique(labels))
    metrics = {
        'pca_2d_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'pc1_explained_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_explained_variance': float(pca.explained_variance_ratio_[1]),
        'mean_channel_std': float(x.std(0).mean()),
        'spectral_effective_rank': effective_rank,
        'between_class_scatter_fraction': float(between / total) if total else 0.,
    }
    return scores, metrics


def draw_panel(ax, scores, labels, title, metrics, colors):
    for label in range(10):
        selection = labels == label
        ax.scatter(scores[selection, 0], scores[selection, 1],
                   s=5, alpha=.35, color=colors[label], edgecolors='none',
                   rasterized=True)
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_xlabel(f"PC1 {metrics['pc1_explained_variance']:.1%}", fontsize=8)
    ax.set_ylabel(f"PC2 {metrics['pc2_explained_variance']:.1%}", fontsize=8)
    ax.text(.02, .98,
            f"rank {metrics['spectral_effective_rank']:.0f}  "
            f"class {metrics['between_class_scatter_fraction']:.1%}\n"
            f"mean std {metrics['mean_channel_std']:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(facecolor='white', alpha=.82, edgecolor='none'))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.12)


def legend(fig, colors):
    handles = [Line2D([0], [0], marker='o', linestyle='',
                      color=colors[index], label=name, markersize=5)
               for index, name in enumerate(CIFAR_CLASSES)]
    fig.legend(handles=handles, loc='lower center', ncol=10,
               frameon=False, fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    loaded = {}
    labels = paths = None
    for short, _title in MODELS:
        path = args.input_dir / f'{short}_cifar10dvs_seed42_v100_layers.npz'
        with np.load(path, allow_pickle=False) as contents:
            loaded[short] = {key: contents[key].copy()
                             for key in contents.files}
        if labels is None:
            labels = loaded[short]['labels']
            paths = loaded[short]['record_paths']
        elif (not np.array_equal(labels, loaded[short]['labels']) or
              not np.array_equal(paths, loaded[short]['record_paths'])):
            raise ValueError('PCA inputs must use identical labeled recordings')
    if len(labels) != 1000 or sorted(np.bincount(labels).tolist()) != [100] * 10:
        raise ValueError('Expected 100 recordings from each of 10 classes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap('tab10').colors
    scores = {}
    metrics = {}
    for short, _title in MODELS:
        scores[short] = {}
        metrics[short] = {}
        for layer in dict.fromkeys((*CLS_LAYERS, *PATCH_LAYERS)):
            scores[short][layer], metrics[short][layer] = pca_metrics(
                loaded[short][layer], labels)

    def save_grid(filename, layers, title, size_per_panel=(4.2, 3.1)):
        fig, axes = plt.subplots(len(MODELS), len(layers),
                                 figsize=(len(layers) * size_per_panel[0],
                                          len(MODELS) * size_per_panel[1]),
                                 squeeze=False)
        for row, (short, model_title) in enumerate(MODELS):
            for col, layer in enumerate(layers):
                draw_panel(axes[row, col], scores[short][layer], labels,
                           f'{model_title}\n{layer.replace("_", " ")}',
                           metrics[short][layer], colors)
        fig.suptitle(title + '\nIndependent PCA per panel; compare metrics, not axis orientation',
                     fontsize=14, y=1.01)
        legend(fig, colors)
        fig.tight_layout(rect=(0, .045, 1, .99))
        fig.savefig(args.output_dir / filename, dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)

    save_grid('pca_cls_layers.png', CLS_LAYERS,
              'CIFAR10-DVS student CLS spiking layers (train-internal validation)')
    save_grid('pca_patch_layers.png', PATCH_LAYERS,
              'CIFAR10-DVS student patch spiking layers (train-internal validation)')
    for filename, layer, title in (
            ('pca_final_cls.png', 'final_cls', 'Final CLS spikes'),
            ('pca_final_patch.png', 'final_patch_mean',
             'Final mean patch spikes')):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for ax, (short, model_title) in zip(axes.flat, MODELS):
            draw_panel(ax, scores[short][layer], labels, model_title,
                       metrics[short][layer], colors)
        fig.suptitle(title + ': identical CIFAR10-DVS recordings\n'
                     'Independent PCA per panel; compare metrics, not axis orientation',
                     fontsize=14)
        legend(fig, colors)
        fig.tight_layout(rect=(0, .05, 1, .94))
        fig.savefig(args.output_dir / filename, dpi=200,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    layer_change = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for short, title in MODELS:
        layer_change[short] = {}
        for ax, kind in zip(axes, ('cls', 'patch_mean')):
            anchor = loaded[short][f'block_0_{kind}']
            depths = (3, 6, 9, 11)
            changes = [float(np.abs(
                loaded[short][f'block_{depth}_{kind}'] - anchor).mean())
                       for depth in depths]
            unchanged = [float(np.mean(
                loaded[short][f'block_{depth}_{kind}'] == anchor))
                         for depth in depths]
            layer_change[short][kind] = {
                'depths': list(depths), 'mean_absolute_change': changes,
                'exact_equal_fraction': unchanged}
            ax.plot(depths, changes, marker='o', label=title)
            ax.set_title(kind.replace('_', ' ').upper())
            ax.set_xlabel('Block index (relative to block 0 spikes)')
            ax.set_ylabel('Mean absolute rate change')
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.grid(alpha=.2)
    axes[1].legend(loc='upper left', fontsize=8)
    fig.suptitle('How much do the 12 spiking blocks change the representation?')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'spike_layer_change.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    report = {
        'dataset': 'CIFAR10-DVS', 'cohort_seed': 42,
        'records': len(labels), 'per_class': 100,
        'record_source': 'unused labeled subset of pretraining train partition',
        'official_test_accessed': False,
        'feature_interface': 'time-averaged binary spikes',
        'pca_basis': 'fitted independently per model and layer',
        'models': {short: value for short, value in metrics.items()},
        'layer_change_from_block0': layer_change,
    }
    (args.output_dir / 'pca_metrics.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'records': len(labels), 'files': [
        'pca_cls_layers.png', 'pca_patch_layers.png',
        'pca_final_cls.png', 'pca_final_patch.png',
        'spike_layer_change.png', 'pca_metrics.json']}))


if __name__ == '__main__':
    main()
