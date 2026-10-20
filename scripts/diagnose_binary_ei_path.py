"""Measure whether binary excitation-inhibition blocks actually update DVS spikes."""
import argparse
import json
from pathlib import Path

import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from spike_dino.data import ManifestEventDataset
from spike_dino.dinov3 import _replace_bn_with_layer_norm
from spike_dino.pretrain_v3 import FixedEpochBatchSampler
from spike_dino.spiking_vit import SpikingVisionTransformer
from scripts.diagnose_v3_representation import model_from_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location='cpu',
                             weights_only=True)
        model = model_from_checkpoint(payload).teacher.backbone
        model = model.to(args.device).eval()
    else:
        model = SpikingVisionTransformer(
            image_size=config['size'], patch_size=config['patch_size'],
            dim=config['dim'], depth=config['depth'], heads=config['heads'],
            attention_variant=config['attention_variant'],
            cls_aggregation=config['cls_aggregation'],
            residual_policy=config['residual_policy'],
            residual_gradient_policy=config['residual_gradient_policy'])
        if config['norm_policy'] == 'layer_norm':
            _replace_bn_with_layer_norm(model)
        model = model.to(args.device).train()
    dataset = ManifestEventDataset(
        config['dataset_root'], config['manifest_path'], seed=config['seed'],
        time_steps=config['time_steps'], size=config['size'],
        local_crops=0, local_size=config['local_size'],
        global_crop_scale=config['global_crop_scale'],
        local_crop_scale=config['local_crop_scale'],
        global_temporal_crop_range=config['global_time_range'],
        local_temporal_crop_range=config['local_time_range'],
        global_view_policy=config['global_view_policy'])
    sampler = FixedEpochBatchSampler(len(dataset), config['batch_size'],
                                     config['seed'], 0, 1)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    views = next(iter(loader))
    frames = views[0].permute(1, 0, 2, 3, 4).to(args.device)
    current = {}
    results = []
    handles = []

    def before(_module, inputs, index):
        current[index] = {'input': inputs[0].detach()}

    def after(_module, _inputs, output, index):
        current[index]['output'] = output.detach()

    def branch(_module, _inputs, output, index, name):
        current.setdefault(index, {})[name] = output.detach()

    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_pre_hook(
            lambda module, inputs, index=index: before(module, inputs, index)))
        handles.append(block.register_forward_hook(
            lambda module, inputs, output, index=index:
            after(module, inputs, output, index)))
        handles.append(block.attn.proj.register_forward_hook(
            lambda module, inputs, output, index=index:
            branch(module, inputs, output, index, 'attention')))
        handles.append(block.fc2.register_forward_hook(
            lambda module, inputs, output, index=index:
            branch(module, inputs, output, index, 'inhibition')))
    try:
        with torch.no_grad():
            functional.reset_net(model)
            cls, patch = model(frames)
            functional.reset_net(model)
        for index in range(len(model.blocks)):
            state = current[index]
            incoming = state['input']
            outgoing = state['output']
            results.append({
                'block': index,
                'input_spike_rate': float(incoming.float().mean()),
                'attention_spike_rate': float(state['attention'].float().mean()),
                'inhibition_spike_rate': float(state['inhibition'].float().mean()),
                'output_spike_rate': float(outgoing.float().mean()),
                'changed_fraction': float((outgoing != incoming).float().mean()),
                'cls_changed_fraction': float((outgoing[:, :, 0] !=
                                               incoming[:, :, 0]).float().mean()),
                'patch_changed_fraction': float((outgoing[:, :, 1:] !=
                                                 incoming[:, :, 1:]).float().mean()),
                'binary_output': bool(torch.all((outgoing == 0) | (outgoing == 1))),
            })
    finally:
        for handle in handles:
            handle.remove()
        functional.reset_net(model)
    report = {
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'dataset': config['dataset'],
        'seed': config['seed'], 'batch_size': config['batch_size'],
        'time_steps': config['time_steps'],
        'final_cls_spike_rate': float(cls.float().mean()),
        'final_patch_spike_rate': float(patch.float().mean()),
        'blocks': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
