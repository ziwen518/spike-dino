"""Read-only proposed-current audit on frozen Q/K/V from real event views.

Alternative currents are evaluated separately at each old block, never
chained into a new network. This is a local operator diagnostic, not accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional

from scripts.diagnose_v3_representation import (matched_cosine,
                                                 model_from_checkpoint)
from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import ManifestEventDataset
from spike_dino.evidence_block import (
    energy_matched_residual, normalized_spike_evidence, rms_current)
from spike_dino.spiking_vit import _lif


def features(value):
    return {'cls': value[:, :, 0].mean(0),
            'patch': value[:, :, 1:].mean((0, 2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--records', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    manifest = load_manifest(args.manifest_path)
    if config.get('manifest_sha256') != manifest['sha256']:
        parser.error('Manifest does not match checkpoint')
    model = model_from_checkpoint(payload).to(args.device).eval()
    backbone = model.teacher.backbone
    dataset = ManifestEventDataset(
        args.dataset_root, args.manifest_path, seed=config['seed'],
        time_steps=config['time_steps'], size=config['size'], local_crops=0,
        global_crop_scale=(.32, 1.), global_temporal_crop_range=(.5, 1.),
        global_view_policy='shared_core')
    indices = np.linspace(0, len(manifest['train']) - 1, args.records, dtype=int)
    records = [dataset[(int(index), 0)] for index in indices]
    views = [torch.stack([record[j] for record in records], 1) for j in range(2)]
    observed = []
    for view in views:
        t, b = view.shape[:2]
        mass = F.avg_pool2d(view.abs().sum(2).flatten(0, 1).unsqueeze(1),
                            config['patch_size'], config['patch_size'])
        support = mass.reshape(t, b, -1) > 0
        observed.append(torch.cat((torch.zeros(t, b, 1, dtype=torch.bool),
                                    support), dim=2))
    captures = [{}, {}]
    current = [0]
    handles = []
    for index, block in enumerate(backbone.blocks):
        def carrier_hook(_module, inputs, index=index):
            captures[current[0]].setdefault(index, {})['carrier'] = (
                inputs[0].detach().float().cpu())
        handles.append(block.register_forward_pre_hook(carrier_hook))
        for name in ('q', 'k', 'v'):
            def hook(_module, _inputs, output, index=index, name=name):
                captures[current[0]].setdefault(index, {})[name] = (
                    output.detach().float().cpu())
            handles.append(getattr(block.attn, name).register_forward_hook(hook))
    with torch.no_grad():
        for j, view in enumerate(views):
            current[0] = j
            functional.reset_net(backbone)
            backbone(view.to(args.device))
            functional.reset_net(backbone)
    for handle in handles:
        handle.remove()
    weights = [block.attn.proj.linear.weight.detach().cpu()
               for block in backbone.blocks]
    heads = config['heads']
    model.cpu()
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    rows = []
    with torch.no_grad():
        for index, weight in enumerate(weights):
            alternatives = []
            activity = []
            for j in range(2):
                values = captures[j][index]
                t, b, n, dim = values['q'].shape
                split = lambda x: x.reshape(t, b, n, heads, -1).permute(0, 1, 3, 2, 4)
                evidence = normalized_spike_evidence(
                    split(values['q']), split(values['k']), split(values['v']),
                    observed[j])
                merge = lambda x: x.transpose(2, 3).reshape(t, b, n, dim)
                contrast = merge(evidence.contrast)
                normalized = rms_current(contrast)
                on, off = _lif(threshold=.5), _lif(threshold=.5)
                positive, negative = on(F.relu(normalized)), off(F.relu(-normalized))
                signed = positive - negative
                projected = rms_current(F.linear(positive, weight) -
                                        F.linear(negative, weight))
                projected = projected * (evidence.coincidence_mass > 0).any(2)
                updated = energy_matched_residual(
                    values['carrier'], projected, len(weights))
                attended = merge(evidence.attended)
                channel_centered = attended - attended.mean(-1, keepdim=True)
                centered_norm = rms_current(channel_centered)
                stable_on, stable_off = _lif(threshold=.5), _lif(threshold=.5)
                stable_positive = stable_on(F.relu(centered_norm))
                stable_negative = stable_off(F.relu(-centered_norm))
                stable_signed = stable_positive - stable_negative
                stable_projected = rms_current(
                    F.linear(stable_positive, weight) -
                    F.linear(stable_negative, weight))
                stable_projected = stable_projected * (
                    evidence.coincidence_mass > 0).any(2)
                stable_updated = energy_matched_residual(
                    values['carrier'], stable_projected, len(weights))
                alternatives.append({
                    'attended': features(attended),
                    'contrast': features(contrast),
                    'bipolar': features(signed),
                    'projected_current': features(projected),
                    'carrier_before': features(values['carrier']),
                    'carrier_after': features(updated),
                    'channel_centered_attended': features(channel_centered),
                    'channel_centered_bipolar': features(stable_signed),
                    'channel_centered_projected': features(stable_projected),
                    'channel_centered_carrier': features(stable_updated),
                })
                activity.append({'on_rate': float(positive.mean()),
                                 'off_rate': float(negative.mean()),
                                 'cls_evidence_fraction': float(
                                     (evidence.coincidence_mass[:, :, :, 0] > 0).float().mean())})
            entry = {'block': index, 'activity': activity}
            for stage in alternatives[0]:
                for token in ('cls', 'patch'):
                    entry[f'{stage}_{token}'] = matched_cosine(
                        alternatives[0][stage][token], alternatives[1][stage][token])
            rows.append(entry)
    result = {'checkpoint': str(args.checkpoint), 'checkpoint_steps': payload['steps'],
              'records': args.records, 'training_split_only': True,
              'scope': 'independent current replacement on frozen old QKV; not a chained new network',
              'observed_patch_fraction': [float(o[:, :, 1:].float().mean()) for o in observed],
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
