"""Read-only event information audit through the SNN patch/CLS stem.

Three passes over one deterministic unlabeled training batch: two real global
views and a zero-event input. Reports sample separation and event sensitivity
at convolution, normalization, position addition, LIF, and CLS pooling.
"""

import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from scripts.diagnose_v3_representation import (matched_cosine,
                                                 model_from_checkpoint)
from spike_dino.data import ManifestEventDataset
from spike_dino.pretrain_v3 import FixedEpochBatchSampler


def as_patches(tensor, time_steps, batch_size):
    """Return feature current in [T,B,P,C] for spatial stages."""
    if tensor.ndim == 4:
        _, channels, height, width = tensor.shape
        return tensor.reshape(time_steps, batch_size, channels,
                              height * width).permute(0, 1, 3, 2)
    if tensor.ndim == 3:
        return tensor[:, :, None, :]
    return tensor


def summarize(first, second, zero, *, binary=False):
    first = first.float()
    second = second.float()
    zero = zero.float()
    rate_a = first.mean((0, 2))
    rate_b = second.mean((0, 2))
    spatial_a = first.mean(0).flatten(1)
    spatial_b = second.mean(0).flatten(1)
    result = {
        'feature_shape': list(first.shape),
        'rate_view_cosine': matched_cosine(rate_a, rate_b),
        'spatial_view_cosine': matched_cosine(spatial_a, spatial_b),
        'rate_sample_std': float(rate_a.std(0, unbiased=False).mean()),
        'zero_event_full_l1': float((first - zero).abs().mean()),
        'zero_event_rate_l1': float((rate_a - zero.mean((0, 2))).abs().mean()),
        'real_mean': float(first.mean()),
        'zero_mean': float(zero.mean()),
    }
    if binary:
        result['binary_values_only'] = bool(torch.all((first == 0) | (first == 1)))
        result['spike_rate'] = float(first.mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.teacher.backbone
    if backbone.cls_aggregation != 'spike_pool':
        parser.error('CLS pooling stages require cls_aggregation=spike_pool')
    dataset = ManifestEventDataset(
        config['dataset_root'], config['manifest_path'],
        seed=config['seed'], time_steps=config['time_steps'],
        size=config['size'], local_crops=0,
        global_crop_scale=config['global_crop_scale'],
        local_crop_scale=config['local_crop_scale'],
        global_temporal_crop_range=config['global_time_range'],
        local_temporal_crop_range=config['local_time_range'],
        global_view_policy='shared_core')
    sampler = FixedEpochBatchSampler(len(dataset), args.batch_size,
                                     config['seed'], 0, 1)
    batch = next(iter(DataLoader(
        dataset, batch_sampler=sampler, num_workers=0,
        generator=torch.Generator().manual_seed(config['seed'] + 1))))
    views = [value.permute(1, 0, 2, 3, 4).to(args.device) for value in batch[:2]]
    t, b = views[0].shape[:2]
    captured = {}
    current = ['0']
    handles = []

    def hook(name, selector, before=False):
        def record(_module, inputs, output=None):
            value = inputs[0] if before else output
            captured[current[0]][name] = selector(value).detach()
        return record

    handles.append(backbone.patch_embed.register_forward_hook(
        hook('patch_conv', lambda value: as_patches(value, t, b))))
    handles.append(backbone.patch_norm.register_forward_hook(
        hook('patch_norm', lambda value: as_patches(value, t, b))))
    handles.append(backbone.token_spike.register_forward_pre_hook(
        hook('position_plus_patch', lambda value: value[:, :, 1:], before=True)))
    handles.append(backbone.token_spike.register_forward_hook(
        hook('token_spikes', lambda value: value[:, :, 1:])))
    handles.append(backbone.cls_pool_norm.register_forward_hook(
        hook('cls_pool_norm', lambda value: as_patches(value, t, b))))
    handles.append(backbone.cls_pool_spike.register_forward_hook(
        hook('cls_pool_spikes', lambda value: as_patches(value, t, b))))
    with torch.no_grad():
        for label, events in (('0', views[0]), ('1', views[1]),
                              ('zero', torch.zeros_like(views[0]))):
            current[0] = label
            captured[label] = {}
            functional.reset_net(backbone)
            cls, _patch = backbone(events)
            functional.reset_net(backbone)
            captured[label]['final_cls_spikes'] = as_patches(cls.detach(), t, b)
    for handle in handles:
        handle.remove()
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_steps': payload['steps'],
        'records': b,
        'training_split_only': True,
        'input_spatial_view_cosine': matched_cosine(
            views[0].float().mean(0).flatten(1),
            views[1].float().mean(0).flatten(1)),
        'stages': {},
    }
    for name in captured['0']:
        result['stages'][name] = summarize(
            captured['0'][name], captured['1'][name], captured['zero'][name],
            binary=name in ('token_spikes', 'cls_pool_spikes',
                            'final_cls_spikes'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
