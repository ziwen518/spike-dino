"""Official-split downstream evaluation of a Spike-DINO v7 checkpoint.

    python -m scripts.eval_v7_downstream --checkpoint runs/<arm>/epoch_300.pt \
        --mode lp --shots full 1 9 45 --protocol spikeclr --report-test \
        --output runs/_lp_v7_<arm>.json

Why this file exists.  ``scripts/eval_v6_downstream.py`` rebuilds a
``SpikeDinoV6`` from the checkpoint config and calls ``load_state_dict(...,
strict=True)``; a v7 checkpoint's state dict holds ``student`` and ``teacher``
modules built from ``SpikeDinoV7``, whose key set differs, so that entry point
refuses a v7 checkpoint.  ``scripts/eval_c10_v4.py``'s own ``load_encoder``
builds a v4 through ``sdino_v4.model_from_config`` and cannot restore a v7
either.  Without this module the project can measure v7 only on the internal
8000-fit / 1000-dev cohort, while the SpikeCLR-LP 56 % target the work is
accounted against is measured on the 10 % test split of the same 90/10
partition -- a comparison the goal requires and that no existing entry point
can produce.

What is and is NOT changed.  This module only LOADS a v7 model and hands
``model.teacher`` to the unmodified ``run_lp`` / ``run_ft`` from
``scripts.eval_c10_v4``: same cohorts, same probe cache, same
``AugmentedRecords``, same ``Classifier``, same LR grid, same
``--protocol spikeclr`` definition (BN calibrated on the whole train set, the
linear classifier fitted on all 9000 labelled train records with the LR chosen
on a train-only 100/class holdout, then the 1000 test records read once).  The
feature is ``encoder.representation(...)['concat']``, i.e. exactly the
"concat" readout that every in-run probe in this project reports as
"concat linear", so the official number and the dev number are two cohorts of
one measurement and not two measurements.

The deployed ``spike_dino/sdino_v7.py`` and ``scripts/train_sdino_v7.py`` are
NOT touched by this file, so adding it cannot change the arithmetic of an arm
that is already running, and source-hash pairing of the running arms is
unaffected.
"""
import argparse
import json
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
PHYSICAL_GPU = os.environ.get('SDINO_GPU', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU

import time
from pathlib import Path

import numpy as np
import torch

from scripts.eval_c10_v4 import (CACHE, MANIFEST, load_encoder as load_reference,
                                 run_ft, run_lp)

# Every keyword below is a real ``SpikeDinoV7.__init__`` parameter in the
# deployed model file.  The list is written out explicitly rather than splatted
# from the config because the config is a *record* (81 descriptive keys,
# including human-readable strings such as head_input and rate_centering),
# while the constructor takes a small typed subset.  An unknown key silently
# ignored would rebuild a different model and strict-load would then blame the
# checkpoint, so the mapping is explicit and the two non-scalar entries
# (rate_centering, head_input) are decoded rather than passed through.
_MODEL_FLOAT_KEYS = (
    'student_temp', 'center_momentum', 'temporal_weight', 'reliability_floor',
    'rate_center_momentum', 'proto_frame_weight', 'koleo_distance_floor',
    'balance_temp', 'bottleneck_var_gamma',
    'cssp_amp_floor', 'cssp_align_floor', 'cssp_rank_floor', 'cssp_energy_floor',
)
_MODEL_WEIGHT_KEYS = (
    'koleo_weight', 'proto_balance_weight',
    'bottleneck_var_weight', 'bottleneck_decorr_weight',
    'cssp_amp_weight', 'cssp_align_weight', 'cssp_shared_weight',
    'cssp_signal_weight',
)


def model_kwargs_from_config(config):
    """Map a recorded v7 pretrain config onto ``SpikeDinoV7`` keyword arguments.

    Pure function, no torch state: the unit test calls it directly.
    """
    kwargs = {}
    for key in _MODEL_FLOAT_KEYS:
        kwargs[key] = float(config[key]) if key in config else 0.
    for key in _MODEL_WEIGHT_KEYS:
        kwargs[key] = float(config.get(key, 0.))
    kwargs['student_temp'] = float(config.get('student_temp', .1))
    kwargs['center_momentum'] = float(config.get('center_momentum', .9))
    kwargs['temporal_weight'] = float(config.get('lambda_temporal', 1.))
    kwargs['reliability_floor'] = float(config.get('reliability_floor', .25))
    kwargs['rate_center_momentum'] = float(config.get('rate_center_momentum', .99))
    kwargs['balance_temp'] = float(config.get('balance_temp', .1))
    kwargs['bottleneck_var_gamma'] = float(config.get('bottleneck_var_gamma', .15))
    # Floors default to the model's own defaults, not to zero: cssp_rank_floor
    # has a default of 1. and the constructor refuses a non-positive value,
    # while cssp_signal_weight > 0 with cssp_energy_floor == 0 is refused.  A
    # config that carries the floors always wins; the fallbacks exist only so a
    # pre-CSSP arm (whose config predates the keys) still rebuilds.
    kwargs['cssp_rank_floor'] = float(config.get('cssp_rank_floor', 1.))

    kwargs['proto_count'] = int(config.get('prototype_count', config.get('prototypes', 256)))
    kwargs['bottleneck'] = int(config.get('bottleneck', 256))
    kwargs['center_rate'] = config.get('rate_centering', 'ema_origin') == 'ema_origin'
    kwargs['arcsin_sqrt'] = bool(config.get('arcsin_sqrt', False))
    kwargs['proto_orthonormal'] = bool(config.get('proto_orthonormal', False))
    kwargs['cssp_grad_log'] = bool(config.get('cssp_grad_log', False))
    kwargs['teacher_bn_mode'] = config.get('teacher_bn_mode', 'train')
    # Two descriptive config strings that the constructor takes as enums.  The
    # head_input entry in the config is prose ('layer-4 GAP spikes ++ ...'),
    # which is NOT a legal constructor value; every arm in this project so far
    # pretrained with the default 'spikes' and the model file raises for
    # 'current' on SEW anyway (sdino_v4.py:59-60).
    kwargs['head_input'] = 'spikes'
    kwargs['embedding_source'] = config.get('embedding_source', 'head')
    if kwargs['embedding_source'] not in ('head', 'backbone'):
        raise ValueError('Unrecognised embedding_source %r' % (kwargs['embedding_source'],))
    kwargs['backbone'] = config.get('backbone_arch', 'sew_resnet18')
    if kwargs['backbone'] != 'sew_resnet18':
        raise ValueError('This entry point is for the SEW backbone, got %r' % (kwargs['backbone'],))
    kwargs['image_size'] = int(config.get('image_size', 224))
    if kwargs['image_size'] != 128:
        raise ValueError('Checkpoint was not pretrained at CIFAR10-DVS geometry')
    return kwargs


def branch_view(encoder, branch):
    """Expose one branch of ``representation()`` under the key ``run_lp`` reads.

    ``run_lp`` hard-codes ``representation(...)['concat']`` and the SEW concat
    readout is **768-d** (layer-4 GAP 512 ++ layer-3 GAP 256), while the
    SpikeCLR-LP reference is a **512 x 10** head (recorded here 2026-11-12 from
    the paper's protocol, not measured).  On the in-run dev probes the concat
    branch reads 1.10-2.40 points ABOVE the 512-d ``cls`` branch on the same
    checkpoint, so a concat LP number is not a like-for-like substitute for the
    512-d reference.  Renaming the branch keeps ``run_lp`` itself untouched --
    same cohorts, same BN calibration, same LR grid -- so an official read with
    ``--branch cls`` differs from the shipped protocol in exactly one thing:
    which slice of the readout is fitted.

    Instance-level rebinding, not a wrapper module: ``calibrate`` calls
    ``encoder.modules()`` / ``encoder(events)``, and a wrapper would change the
    module tree the probe walks.  ``'concat'`` returns the encoder unmodified.
    """
    if branch == 'concat':
        return encoder
    inner = encoder.representation

    def _view(events, _inner=inner, _key=branch):
        rolled = _inner(events)
        if _key not in rolled:
            raise KeyError('readout %r not produced by representation(); has %s'
                           % (_key, sorted(rolled)))
        return {'concat': rolled[_key]}

    encoder.representation = _view
    return encoder


def load_encoder_v7(checkpoint, seed=42, backbone='sew_resnet18'):
    if checkpoint == 'random':
        return load_reference('random', seed, backbone)
    from spike_dino.sdino_v7 import SpikeDinoV7
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    if config.get('backbone_arch', 'vit') != 'sew_resnet18':
        raise ValueError("This entry point is for the SEW backbone, got "
                         "%r" % (config.get('backbone_arch'),))
    model = SpikeDinoV7(**model_kwargs_from_config(config))
    state = payload['model']
    # Refuse rather than silently probe a half-loaded encoder.  The one
    # objective-only surplus this family has ever produced is v5's
    # ``logit_scale``; v7 records nothing of that kind, so any surplus here is
    # an architecture mismatch and must fail loudly.
    surplus = sorted(set(state) - set(model.state_dict()))
    if surplus:
        raise RuntimeError('Checkpoint holds weights this model cannot hold: %s'
                           % (surplus,))
    missing = sorted(set(model.state_dict()) - set(state))
    if missing:
        raise RuntimeError('Model expects weights the checkpoint does not hold: %s'
                           % (missing,))
    model.load_state_dict(state, strict=True)
    meta = {'source': str(checkpoint), 'step': payload['step'],
            'pretrain_config': config}
    return model.teacher, meta


def main(args):
    if args.output.exists():
        raise RuntimeError('Output exists; inspect rather than overwrite')
    from spike_dino import probe_v4
    from spike_dino.benchmark_manifest import load_manifest
    probe_v4.set_cache(CACHE)
    manifest = load_manifest(MANIFEST, 'cifar10dvs')
    labels = torch.as_tensor(np.load(CACHE / 'labels.npy'))
    encoder, meta = load_encoder_v7(args.checkpoint, args.init_seed, args.backbone)
    encoder = branch_view(encoder, args.branch).cuda()
    report = {
        'protocol': 'CIFAR10-DVS seed42 90/10; cohorts(validation 100/class from '
                    'train); test evaluated once with validation-selected setting',
        'loader': 'spike_dino.sdino_v7 reconstruction; encoder = model.teacher',
        'mode': args.mode, 'shots': args.shots, 'seeds': args.seeds,
        'feature_branch': args.branch, 'head_branch': args.head_branch,
        'report_test': args.report_test, 'physical_gpu': PHYSICAL_GPU,
        **{k: v for k, v in meta.items() if k != 'pretrain_config'},
        'args': {k: str(v) for k, v in vars(args).items()}}
    started = time.time()
    if args.mode == 'lp':
        report['lp_protocol'] = args.protocol
        report['results'] = run_lp(encoder, manifest, labels, args.shots, args.seeds,
                                   args.report_test, args.protocol)
    else:
        report['results'] = run_ft(encoder, manifest, args.shots, args.seeds,
                                   args.report_test, args)
    report['seconds'] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help="path or 'random'")
    parser.add_argument('--mode', choices=('lp', 'ft'), default='lp')
    parser.add_argument('--shots', nargs='+', default=['full'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--backbone', choices=('vit', 'sew_resnet18'), default='sew_resnet18')
    parser.add_argument('--protocol', choices=('cohort', 'spikeclr'), default='spikeclr')
    parser.add_argument('--branch', choices=('concat', 'cls', 'gap'), default='concat',
                        help='which representation() slice the probe fits; concat is the '
                             'shipped 768-d protocol, cls is the 512-d readout that matches '
                             'the SpikeCLR-LP 512x10 head')
    parser.add_argument('--head-branch', choices=('concat', 'cls', 'gap'), default='concat',
                        help="which slice the FT classifier head reads (mode ft only); "
                             "cls builds nn.Linear(512, 10) on z_bar, the SpikeCLR-FT head")
    parser.add_argument('--report-test', action='store_true')
    parser.add_argument('--init-seed', type=int, default=42)
    parser.add_argument('--ft-epochs', type=int, default=100)
    parser.add_argument('--fewshot-steps', type=int, default=600)
    parser.add_argument('--ft-lr', type=float, default=1e-3)
    parser.add_argument('--ft-encoder-lr', type=float, default=0.,
                        help='encoder LR for FT; 0 keeps the historical frozen-encoder FT')
    parser.add_argument('--ft-weight-decay', type=float, default=.05)
    parser.add_argument('--ft-label-smoothing', type=float, default=0.,
                        help='label smoothing for FT (the paper reported value)')
    parser.add_argument('--ft-freeze-bn', action='store_true',
                        help='keep BN running stats fixed during FT')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--evaluations', type=int, default=25)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
