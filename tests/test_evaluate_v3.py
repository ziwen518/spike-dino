import torch
from torch.utils.data import DataLoader, TensorDataset

from spike_dino.dinov3 import SpikingDinoV3Prototype
from spike_dino.evaluate_v3 import batch_context_difference, extract, feature_stats


torch.set_num_threads(2)


def test_frozen_cls_extraction_and_batch_context():
    torch.manual_seed(4)
    model = SpikingDinoV3Prototype(image_size=32, patch_size=8,
                                  dim=48, depth=2, heads=4,
                                  head_hidden=96, bottleneck=24,
                                  prototypes=32,
                                  norm_policy='layer_norm')
    frames = torch.rand(4, 4, 2, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(frames, labels), batch_size=2)
    backbone = model.teacher.backbone
    features, actual_labels = extract(backbone, loader, 'cpu')
    assert features.shape == (4, 48)
    torch.testing.assert_close(actual_labels, labels)
    assert feature_stats(features)['unique_feature_vectors'] > 1
    difference = batch_context_difference(backbone, frames[:2].transpose(0, 1),
                                          'cpu', 'eval')
    assert difference == 0
    patch_features, _ = extract(backbone, loader, 'cpu',
                                feature_source='patch_mean')
    combined_features, _ = extract(backbone, loader, 'cpu',
                                   feature_source='concat')
    assert patch_features.shape == (4, 48)
    assert combined_features.shape == (4, 96)
    torch.testing.assert_close(combined_features[:, :48], features)
    torch.testing.assert_close(combined_features[:, 48:], patch_features)
    assert batch_context_difference(backbone, frames[:2].transpose(0, 1),
                                    'cpu', 'eval', 'patch_mean') == 0
