"""Read-only class coherence of SNN patch and iBOT bottleneck features.

Class labels are used only after feature extraction on a fixed cohort from the
pretraining train partition. No official test files or labels enter training.
"""

import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.diagnose_v3_representation import model_from_checkpoint
from spike_dino.evaluate import FixedEventDataset
from spike_dino.evaluate_v3 import cohorts


def pair_metrics(features, labels):
    """Cosines and one-neighbor labels at the SAME spatial patch position."""
    normalized = F.normalize(features.float(), dim=-1)
    similarity = torch.einsum('bpc,dpc->pbd', normalized, normalized)
    count = len(labels)
    diagonal = torch.eye(count, dtype=torch.bool)
    same = labels[:, None] == labels[None, :]
    same = same & ~diagonal
    different = ~same & ~diagonal
    neighbour = similarity.masked_fill(diagonal[None], -2).argmax(-1)
    predicted = labels[neighbour]
    same_count = same.sum(-1).clamp_min(1)
    different_count = different.sum(-1).clamp_min(1)
    per_record_gap = (
        ((similarity * same[None]).sum(-1) / same_count[None]) -
        ((similarity * different[None]).sum(-1) /
         different_count[None])).mean(0)
    return {
        'same_class_cosine': float(similarity[:, same].mean()),
        'different_class_cosine': float(similarity[:, different].mean()),
        'same_minus_different_cosine': float(
            similarity[:, same].mean() - similarity[:, different].mean()),
        'same_position_one_neighbor_class_accuracy': float(
            (predicted == labels[None]).float().mean()),
        'mean_feature_std_across_records': float(
            features.float().std(0, unbiased=False).mean()),
        'mean_spike_rate': float(features.float().mean()),
        'per_record_semantic_gap': [float(value) for value in per_record_gap],
    }


def pooled_one_neighbor(features, labels):
    pooled = F.normalize(features.float().mean(1), dim=-1)
    similarity = pooled @ pooled.T
    similarity.fill_diagonal_(-2)
    neighbours = similarity.argmax(-1)
    return float((labels[neighbours] == labels).float().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    _probe, validation = cohorts(
        args.dataset_root, seed=42, train_limit=512,
        validation_per_class=100,
        manifest_path=config['manifest_path'])
    selected = []
    for label in range(10):
        selected.extend([record for record in validation
                         if record['label'] == label][:args.per_class])
    loader = DataLoader(
        FixedEventDataset(args.dataset_root, selected,
                          config['time_steps'], config['size']),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.teacher.backbone
    head = model.teacher.ibot_head
    backbone_features, bottleneck_features, labels = [], [], []
    with torch.no_grad():
        for frames, targets in loader:
            events = frames.transpose(0, 1).contiguous().to(args.device)
            functional.reset_net(backbone)
            _cls, patch = backbone(events)
            functional.reset_net(backbone)
            t, b, positions, dim = patch.shape
            flattened = patch.reshape(t, b * positions, dim)
            functional.reset_net(head)
            hidden = head.lif1(head.bn1(head.fc1(flattened)))
            if head.fc_mid is not None:
                hidden = head.lif_mid(head.bn_mid(head.fc_mid(hidden)))
            bottleneck = head.lif2(head.bn2(head.fc2(hidden)))
            functional.reset_net(head)
            backbone_features.append(patch.float().mean(0).cpu())
            bottleneck_features.append(
                bottleneck.float().mean(0).reshape(
                    b, positions, head.bottleneck).cpu())
            labels.append(targets.cpu())
    backbone_features = torch.cat(backbone_features)
    bottleneck_features = torch.cat(bottleneck_features)
    labels = torch.cat(labels)
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_steps': payload['steps'],
        'records': len(labels),
        'per_class': args.per_class,
        'patch_positions': backbone_features.shape[1],
        'validation_source': 'fixed labeled cohort within pretraining train partition',
        'official_test_accessed': False,
        'backbone_patch': pair_metrics(backbone_features, labels),
        'ibot_bottleneck': pair_metrics(bottleneck_features, labels),
        'backbone_patch_pooled_one_neighbor_class_accuracy':
            pooled_one_neighbor(backbone_features, labels),
        'ibot_bottleneck_pooled_one_neighbor_class_accuracy':
            pooled_one_neighbor(bottleneck_features, labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
