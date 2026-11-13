"""Spike-DINO v7 pretraining: the DINO objective on a constrained sphere.

v6 closed the decoupled-GMM line with a measured negative -- the distillation
half added nothing at matched schedule (instboth e150 60.40 against instonly
60.43) and the project's best readout came from the arm with ``--dino-weight
0``, i.e. without the DINO objective at all.  v7 goes back to DINO: a standard
learnable cosine prototype table, the DC of the spike rate removed by an EMA
origin so the features actually live on a sphere, and optional geometric
constraints on the table and the features (``spike_dino/sdino_v7.py``).

Everything else -- views, reliability weighting, EMA Teacher, Teacher
temperature warmup, Teacher centring, the frozen-probe protocol and the stop
rule -- is the v5/v6 recipe, so every probe stays comparable with the existing
arms.

Usage (server):
  python -m scripts.train_sdino_v7 --output runs/sdino_v7_<tag> --epochs 30 \\
      --dataset cifar10dvs --backbone sew_resnet18 --batch-size 64 --precision bf16 \\
      --teacher-bn-mode train --probe-at-start --probe-every 5 --probe-dense-until 5
"""
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
# Physical GPU (PCI bus order). Default 0; SDINO_GPU=1 only with user approval.
PHYSICAL_GPU = os.environ.get('SDINO_GPU', '0')
if PHYSICAL_GPU not in ('0', '1'):
    raise SystemExit('SDINO_GPU must be 0 or 1')
os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_generalization_reference import atomic
from scripts.train_sdino_v4 import (DATASETS, early_stop_decision, finite_tensors,
                                    gpu_idle, verify_physical_gpu)
from spike_dino import probe_v4
from spike_dino.data import SENSOR_SIZES


def flat_norm(grads):
    """L2 norm of a list of per-parameter gradients, None entries as zero.

    ``allow_unused=True`` yields None for parameters the objective does not
    touch, which is normal here (the temporal views and the CSSP branch reach
    different parts of the head); treating them as zero is what makes the norm
    comparable with ``clip_grad_norm_``.
    """
    total = 0.
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return total ** .5


def flat_cosine(left, right, norm_left, norm_right):
    """Cosine between two per-parameter gradient lists, measured elementwise."""
    dot = 0.
    for a, b in zip(left, right):
        if a is not None and b is not None:
            dot += float((a.detach() * b.detach()).sum())
    return dot / max(norm_left * norm_right, 1e-12)
from spike_dino.pretrain_v3 import FixedEpochBatchSampler
from spike_dino.probe_v4 import load_split, run_probe, summary_line
from spike_dino.sdino_v4 import cosine_schedule, teacher_temperature
from spike_dino.sdino_v7 import SpikeDinoV7
from spike_dino.views_v4 import MulticropEventDataset, support_iou
from spike_dino.views_v5 import TemporalDinoViews

if torch.cuda.is_initialized():
    raise RuntimeError('CUDA initialized before GPU selection')
os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU

# scripts.train_generalization_reference pins CUDA_VISIBLE_DEVICES at import time
SOURCES = ['spike_dino/sdino_v7.py', 'spike_dino/sdino_v4.py', 'spike_dino/views_v5.py',
           'spike_dino/views_v4.py', 'spike_dino/fused_lif.py', 'spike_dino/probe_v4.py',
           'spike_dino/spiking_vit.py', 'spike_dino/dinov3.py', 'spike_dino/data.py',
           'spike_dino/pretrain_v3.py', 'spike_dino/sdino_sew.py',
           'scripts/train_sdino_v7.py']


def source_hashes():
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCES}


def build_config(args):
    return {
        'method': 'Spike-DINO v7 (DINO on a constrained sphere)',
        'dataset': args.dataset, 'dataset_note': (
            'mini_nimagenet parts1+2 unlabeled' if args.dataset == 'mini_nimagenet'
            else 'CIFAR10-DVS seed42 90/10 train split, labels unused; test never read'),
        'manifest': DATASETS[args.dataset]['manifest'], 'seed': args.seed,
        'time_steps': 4, 'render_size': DATASETS[args.dataset]['render'],
        'global_size': DATASETS[args.dataset]['global_size'],
        'image_size': DATASETS[args.dataset]['global_size'],
        'probe_cache': DATASETS[args.dataset]['probe_cache'],
        'views': '2 global + 2 temporal; all four at the global resolution',
        'global_scale': list(args.global_scale), 'global_time': list(args.global_time),
        'temporal_crops': 2, 'temporal_time': list(args.temporal_time),
        'temporal_scale': list(args.temporal_scale),
        'temporal_event_mass_min': args.temporal_mass_min,
        'thinning': [.8, 1.], 'flip': .5, 'polarity_jitter': [.8, .3],
        'event_support_gate_min_relative_mass': .5,
        'backbone_arch': args.backbone,
        'backbone': ('SEW-ResNet-18 (spikingjelly, 2-ch input, identity residual)'
                     if args.backbone == 'sew_resnet18' else
                     'SPS-BN 12x384 6-head spikformer-count LayerScale.1'),
        'head_input': ('layer-4 GAP spikes ++ LIF(BN(layer-4 GAP spikes)) (1024)'
                       if args.backbone == 'sew_resnet18' else
                       'binary CLS ++ BN+LIF(mean patch current) spikes'),
        'representation': ('layer-4 GAP rate ++ layer-3 GAP rate' if args.backbone == 'sew_resnet18'
                           else 'CLS rate ++ GAP patch rate'),
        'bn_policy': ('student train batch stats; teacher calibrated running stats in eval'
                      if args.teacher_bn_mode == 'running_eval' else
                      'student and teacher train batch stats; precise-BN evaluation'),
        'teacher_bn_mode': args.teacher_bn_mode,
        'embedding_source': args.embedding_source,
        'readout': ('cosine(feature, prototype) / student_temp, feature = L2-normalised '
                    '(rate - ema_origin) with an optional arcsin(sqrt(.)) transform; '
                    'prototypes are a trainable parameter, EMA-copied to the Teacher'),
        'prototype_count': args.proto_count,
        'rate_centering': 'ema_origin' if args.center_rate else 'none',
        'rate_center_momentum': args.rate_center_momentum,
        'arcsin_sqrt': bool(args.arcsin_sqrt),
        'proto_frame_weight': args.proto_frame_weight,
        'proto_orthonormal': bool(args.proto_orthonormal),
        'koleo_weight': args.koleo_weight,
        # 0 keeps the stock DINOv2 term bit-identical; a positive value shifts
        # the log onto ``d + floor``, which bounds the near-duplicate tail that
        # produced grad_norm 167.58 in the one previous KoLeo arm.  See
        # ``sdino_v7.koleo_hinge``.
        'koleo_distance_floor': args.koleo_distance_floor,
        'koleo_auto_weight': bool(args.koleo_auto_weight),
        'koleo_probe_target': args.koleo_probe_target,
        'koleo_probe_cap': args.koleo_probe_cap,
        'proto_balance_weight': args.proto_balance_weight,
        'balance_temp': args.balance_temp,
        'bottleneck_var_weight': args.bottleneck_var_weight,
        'bottleneck_decorr_weight': args.bottleneck_decorr_weight,
        'bottleneck_var_gamma': args.bottleneck_var_gamma,
        # CSSP: three one-sided floors on the cross-view shared spectrum of the
        # two global views.  All weights default to 0, so every arm written
        # before this option is bit-identical (the loss term is not even built).
        'cssp_amp_weight': args.cssp_amp_weight,
        'cssp_align_weight': args.cssp_align_weight,
        'cssp_shared_weight': args.cssp_shared_weight,
        'cssp_amp_floor': args.cssp_amp_floor,
        'cssp_align_floor': args.cssp_align_floor,
        'cssp_rank_floor': args.cssp_rank_floor,
        'cssp_signal_weight': args.cssp_signal_weight,
        'cssp_energy_floor': args.cssp_energy_floor,
        # ``cssp_shared_weight`` above is the COMMAND-LINE value.  With
        # --cssp-auto-weight the weight actually used is calibrated at run time
        # from the Encoder gradient ratio and is recorded in status.json under
        # 'cssp_grad_probe'.  A resumed arm reads the calibrated value back.
        'cssp_auto_weight': bool(args.cssp_auto_weight),
        'cssp_probe_target': args.cssp_probe_target,
        'cssp_signal_probe_target': args.cssp_signal_probe_target,
        'cssp_probe_cap': args.cssp_probe_cap,
        'save_at_steps': list(args.save_at_steps),
        'objective': ('L_global-DINO + lambda * L_reliable-temporal-DINO'
                      + (f' + {args.proto_frame_weight:g} * frame_potential(prototypes)'
                         if args.proto_frame_weight else '')
                      + (' + Stiefel retraction of the prototype table each step'
                         if args.proto_orthonormal else '')
                      + (f' + {args.koleo_weight:g} * KoLeo(student global features'
                         f', distance floor {args.koleo_distance_floor:g})'
                         if args.koleo_weight else '')
                      + (f' + {args.proto_balance_weight:g} * H_marginal(student assignment)'
                         if args.proto_balance_weight else '')
                      + (f' + CSSP(lambda_a {args.cssp_amp_weight:g}, lambda_b '
                         f'{args.cssp_align_weight:g}, lambda_c '
                         f'{args.cssp_shared_weight:g}, lambda_e '
                         f'{args.cssp_signal_weight:g}; floors {args.cssp_amp_floor:g}/'
                         f'{args.cssp_align_floor:g}/{args.cssp_rank_floor:g}/'
                         f'{args.cssp_energy_floor:g}) on the '
                         'cross-view shared spectrum of the two student global views'
                         if (args.cssp_amp_weight or args.cssp_align_weight
                             or args.cssp_shared_weight
                             or args.cssp_signal_weight) else '')),
        'lambda_temporal': args.lambda_temporal,
        'reliability': ('w_i = floor + (1-floor)*clamp(cos(teacher_global0, teacher_global1),0,1) '
                        'on the centred sphere features; temporal terms weighted, global not'),
        'reliability_floor': args.reliability_floor,
        'prototypes': args.proto_count, 'head_hidden': 2048, 'bottleneck': 256,
        'student_temp': args.student_temp,
        'teacher_temp': [*args.teacher_temp, args.teacher_temp_warmup_epochs],
        'center_momentum': .9, 'teacher_momentum': [.996, 1.],
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'base_lr': args.base_lr, 'lr': args.base_lr * args.batch_size / 256,
        'min_lr': 1e-6, 'warmup_epochs': args.warmup_epochs, 'weight_decay': [.04, .4],
        'gradient_clip': 3., 'precision': args.precision,
        'early_stop': {'gate_epoch': args.gate_epoch,
                       'gate': 'concat linear>=base+8, kNN>=base+5, rank>=base/2',
                       'after_gate': '2 probes without +1 linear and CLS correlation rank<10'},
        'official_validation_read': False,
    }


def parameter_groups(model):
    """Weight decay on matrices, none on biases and norms.

    v7 has no learnable logit scale by design (a scale shared by Student and
    Teacher can shrink both softmaxes together, lowering the loss without
    moving any feature); the prototype table is an ordinary parameter and
    decays like the rest of the matrices, which the retraction absorbs.
    """
    decay, no_decay = [], []
    for name, parameter in model.student.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0.}]


def main(args):
    output = args.output
    if args.resume is None and output.exists():
        raise RuntimeError('Output exists; inspect rather than overwrite')
    gpu_idle()
    gpu_check = verify_physical_gpu()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'probes').mkdir(exist_ok=True)
    cfg = build_config(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    autocast_dtype = torch.bfloat16 if args.precision == 'bf16' else None

    transform = TemporalDinoViews(
        SENSOR_SIZES[args.dataset], render_size=cfg['render_size'],
        global_size=cfg['global_size'], global_scale=tuple(args.global_scale),
        global_time=tuple(args.global_time), temporal_time=tuple(args.temporal_time),
        temporal_scale=tuple(args.temporal_scale),
        temporal_mass_min=args.temporal_mass_min)
    dataset = MulticropEventDataset(Path(DATASETS[args.dataset]['root']),
                                    Path(DATASETS[args.dataset]['manifest']), transform,
                                    seed=args.seed)
    cfg['train_records'] = len(dataset)
    probe_v4.set_cache(cfg['probe_cache'])
    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    stop_step = min(total_steps, args.max_steps) if args.max_steps else total_steps

    model = SpikeDinoV7(proto_count=args.proto_count,
                        student_temp=args.student_temp,
                        teacher_bn_mode=args.teacher_bn_mode,
                        image_size=cfg['image_size'],
                        center_rate=args.center_rate,
                        rate_center_momentum=args.rate_center_momentum,
                        arcsin_sqrt=args.arcsin_sqrt,
                        proto_frame_weight=args.proto_frame_weight,
                        proto_orthonormal=args.proto_orthonormal,
                        koleo_weight=args.koleo_weight,
                        koleo_distance_floor=args.koleo_distance_floor,
                        proto_balance_weight=args.proto_balance_weight,
                        balance_temp=args.balance_temp,
                        bottleneck_var_weight=args.bottleneck_var_weight,
                        bottleneck_var_gamma=args.bottleneck_var_gamma,
                        bottleneck_decorr_weight=args.bottleneck_decorr_weight,
                        cssp_amp_weight=args.cssp_amp_weight,
                        cssp_align_weight=args.cssp_align_weight,
                        cssp_shared_weight=args.cssp_shared_weight,
                        cssp_amp_floor=args.cssp_amp_floor,
                        cssp_align_floor=args.cssp_align_floor,
                        cssp_rank_floor=args.cssp_rank_floor,
                        cssp_signal_weight=args.cssp_signal_weight,
                        cssp_energy_floor=args.cssp_energy_floor,
                        cssp_grad_log=args.cssp_grad_log,
                        temporal_weight=args.lambda_temporal,
                        reliability_floor=args.reliability_floor,
                        embedding_source=args.embedding_source,
                        head_input=args.head_input,
                        backbone=args.backbone).cuda().train()
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=cfg['lr'], weight_decay=.04)
    lr_schedule = cosine_schedule(cfg['lr'], 1e-6, total_steps,
                                  args.warmup_epochs * steps_per_epoch, 0.)
    wd_schedule = cosine_schedule(.04, .4, total_steps)
    momentum_schedule = cosine_schedule(.996, 1., total_steps)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        recorded_keys = checkpoint['config']
        missing_keys = []
        for key in ('batch_size', 'epochs', 'base_lr', 'precision', 'teacher_bn_mode',
                    'seed', 'teacher_temp', 'warmup_epochs', 'weight_decay',
                    'early_stop', 'rate_centering', 'rate_center_momentum',
                    'arcsin_sqrt', 'prototype_count', 'proto_frame_weight',
                    'proto_orthonormal', 'koleo_weight', 'koleo_distance_floor',
                    'proto_balance_weight',
                    'balance_temp', 'bottleneck_var_weight', 'bottleneck_decorr_weight',
                    'bottleneck_var_gamma', 'global_scale', 'global_time',
                    'cssp_amp_weight', 'cssp_align_weight', 'cssp_shared_weight',
                    'cssp_amp_floor', 'cssp_align_floor', 'cssp_rank_floor',
                    'cssp_signal_weight', 'cssp_energy_floor',
                    'temporal_time', 'temporal_scale', 'temporal_event_mass_min',
                    'embedding_source', 'head_input', 'dataset', 'backbone_arch',
                    'lambda_temporal', 'reliability_floor'):
            if 'config' not in checkpoint:
                raise ValueError('Checkpoint carries no config')
            # Only keys the checkpoint actually recorded can be compared; a
            # checkpoint written before an option existed cannot disagree about
            # it, and that option's default is the behaviour it trained under.
            if key not in recorded_keys:
                missing_keys.append(key)
            elif checkpoint['config'][key] != cfg[key]:
                raise ValueError(f'Resume config differs: {key}')
        if missing_keys:
            print('[resume] checkpoint predates these options, keeping current '
                  'defaults:', sorted(set(missing_keys)), flush=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    initial_teacher_calibration = None
    if args.teacher_bn_mode == 'running_eval' and start_step == 0:
        from scripts.train_sdino_v4 import calibrate_initial_teacher
        initial_teacher_calibration = calibrate_initial_teacher(
            model, dataset, args.batch_size, args.seed)
    split = load_split()
    sampler = FixedEpochBatchSampler(len(dataset), args.batch_size, args.seed,
                                     start_step, stop_step)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
    status_path = output / 'status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    source_hash = source_hashes()
    state = {'state': 'running', 'config': cfg, 'source_sha256': source_hash,
             'pid': os.getpid(), 'physical_gpu': PHYSICAL_GPU, 'gpu_check': gpu_check,
             'started': time.time(), 'start_step': start_step,
             'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps,
             'stop_step': stop_step, 'step': start_step,
             'probes': previous.get('probes', []),
             'resumed_from': str(args.resume) if args.resume else None,
             'student_parameters': sum(p.numel() for p in model.student.parameters()),
             'fused_lif_modules': model.student.fused_lif_modules}
    state['initial_teacher_calibration'] = initial_teacher_calibration
    def write_status():
        """Persist status.json without letting a transient reader kill the run.

        On Windows ``os.replace`` fails with WinError 5 while another process
        holds the destination open, and the shared ``atomic`` helper retries for
        five seconds before re-raising.  status.json is a convenience record --
        the probe JSONs, metrics.jsonl and the checkpoints are the durable
        artefacts -- so a persistent lock must cost one snapshot rather than a
        300-epoch run.  Measured 2026-11-10: the v7 stiefel arm died at step
        2800 this way, and its ``finally`` write moments later succeeded, i.e.
        the lock was transient and the run was not otherwise unhealthy.
        """
        try:
            atomic(status_path, state)
        except PermissionError as error:
            print('[status] write skipped after retries: %r' % (error,), flush=True)

    write_status()
    metrics_file = open(output / 'metrics.jsonl', 'a', encoding='utf-8')

    def save(step, path):
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'step': step, 'epoch': step / steps_per_epoch, 'config': cfg,
                   'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                   'source_sha256': source_hash}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def probe(epoch_label):
        started = time.time()
        model.eval()
        result = run_probe(model.teacher, split, batch_size=64, workers=0,
                           predictions_path=output / 'probes' / f'{epoch_label}_predictions.npz')
        model.train()
        result.update(epoch=epoch_label, seconds=time.time() - started)
        atomic(output / 'probes' / f'{epoch_label}.json', result)
        state['probes'].append({'epoch': epoch_label, 'summary': summary_line(result),
                                'concat_knn': result['concat']['knn']['acc1'],
                                'concat_linear': result['concat']['linear']['acc1_mean'],
                                'concat_rank': result['concat']['feature']['effective_rank'],
                                'cls_correlation_rank': result['cls']['feature'][
                                    'correlation_effective_rank']})
        write_status()
        torch.cuda.empty_cache()

    # Encoder-space split-gradient probe (see the training loop).  Kept here so
    # a resume does not re-calibrate a weight that was already fixed.
    # Every key appended to below MUST be declared here.  'shared' was added
    # when the calibration was switched from the WEIGHTED objective to the
    # unweighted hinge (otherwise the update is an oscillator between 1 and
    # lambda*) and this initialiser was not updated with it, so the first step
    # whose CSSP hinge was non-zero died with KeyError('shared') -- after the
    # e0 probe had already been written.  Pinned by
    # tests/test_sdino_v7.py::test_trainer_grad_probe_accumulator_defines_every_key_it_appends.
    # 'done' counts every probed step (diagnostic only, it gates nothing).
    # 'done_shared' / 'done_signal' are PER-TERM budgets.  A single shared
    # budget starved the second term in round 2: L_signal is live from step 1
    # and L_shared only from step 30, so all 20 probes were spent before the
    # shared hinge ever fired, its list stayed empty, its weight remained at
    # the 0.01 pre-calibration seed and the arm silently tested L_signal with
    # the rank constraint 39x weaker than round 1's calibrated 0.3925.  Caught
    # at step 140 of 2800, before any gate, from the arm's own metrics.jsonl
    # (cssp_shared_weight never leaving 0.01 while cssp_shared > 0 emitted
    # rows).  --cssp-grad-probe N is therefore the budget PER calibrated TERM.
    # 'done_koleo' is a third per-term budget, for the same reason as
    # 'done_shared' / 'done_signal': --koleo-grad-probe N is the budget PER
    # calibrated term, and KoLeo engages on every step (unlike a hinge, which is
    # zero whenever its floor is cleared), so sharing one counter with the CSSP
    # hinges would starve them instead.  'koleo_cos' is the DINO-vs-KoLeo
    # gradient cosine, kept separate from 'cos' (DINO-vs-CSSP) so that adding
    # this term cannot change what an existing CSSP arm records.
    grad_probe = {'done': 0, 'done_shared': 0, 'done_signal': 0, 'done_koleo': 0,
                  'dino': [], 'cssp': [], 'shared': [], 'signal': [],
                  'koleo': [], 'koleo_cos': [], 'cos': [], 'steps': []}
    if start_step and (args.cssp_auto_weight or args.koleo_auto_weight):
        raise SystemExit('--cssp-auto-weight / --koleo-auto-weight cannot be combined '
                         'with --resume: the weight of a resumed arm must be read from '
                         'its config, not re-calibrated mid-run')

    try:
        if args.probe_at_start and start_step == 0:
            probe('epoch_000')
        iterator = iter(loader)
        wait_total = compute_total = 0.
        tick = time.time()
        for step in range(start_step, stop_step):
            batch = next(iterator)
            views = [view.permute(1, 0, 2, 3, 4).contiguous().cuda(non_blocking=True)
                     for view in batch]
            if len(views) != 4:
                raise RuntimeError('Unexpected number of views (v7 needs 2 global + 2 temporal)')
            data_ready = time.time()
            wait_total += data_ready - tick
            if step == start_step:
                iou = support_iou(views[0].transpose(0, 1), views[1].transpose(0, 1))
                state['first_batch_global_output_support_iou'] = float(iou.mean())
            for index, group in enumerate(optimizer.param_groups):
                group['lr'] = lr_schedule[step]
                if index == 0:
                    group['weight_decay'] = wd_schedule[step]
            temp = teacher_temperature(step, steps_per_epoch,
                                       args.teacher_temp_warmup_epochs,
                                       args.teacher_temp[0], args.teacher_temp[1])
            loss, metrics = model(views, temp, autocast_dtype)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
            # ---- CSSP split gradient on the Encoder ------------------------
            # The user's requirement is stated on the *Encoder* gradient, not
            # on the loss values: KoLeo at weight .1 was accepted by any loss
            # ratio and still put ||g|| = 167.58 through a model whose total
            # pre-clip norm is 2-3.  ``grad_norm`` below measures the SUM of
            # both objectives and can never say which half moved the model, so
            # the two are separated here while the graph is alive.
            #
            # Only steps on which one of the TWO CALIBRATED hinges is actually
            # NON-ZERO are probed: at a zero hinge the gradient is exactly zero
            # and the ratio is undefined.  This is also the check the user asked
            # for -- the shared-rank hinge is a ratio of spectral energies, and
            # at an exactly rank-1 C_AB it is positive with vanishing gradient,
            # so a calibration that divides by ||g_CSSP|| must be able to refuse.
            #
            # CORRECTION after round 1: the gate used to test the sum of all
            # three hinges, so the very first training batch qualified through
            # L_amp alone (a cold-start trace artefact, see AGENTS.md) and spent
            # one of the 20 probes on a step whose shared gradient was exactly
            # zero.  The guards are not calibrated and must therefore not open
            # the gate; only the terms whose weight the probe sets may do that.
            shared_live = float(metrics['cssp_shared']) > 0.
            signal_live = float(metrics['cssp_signal']) > 0.
            # ``!= 0`` rather than ``> 0``: KoLeo is -log(d), which is not bounded
            # below, so its sign is not a liveness test.  With --koleo-weight 0
            # the term is an exact zero and this is False, which is what keeps
            # every CSSP arm bit-identical.
            koleo_live = float(metrics['proto_koleo']) != 0.
            # Each calibrated term gets its OWN budget, so a term that engages
            # early cannot starve one that engages later (see the initialiser).
            probe_shared = (shared_live
                            and grad_probe['done_shared'] < args.cssp_grad_probe)
            probe_signal = (signal_live
                            and grad_probe['done_signal'] < args.cssp_grad_probe)
            probe_koleo = (koleo_live and args.koleo_weight > 0
                           and grad_probe['done_koleo'] < args.koleo_grad_probe)
            if ((probe_shared or probe_signal or probe_koleo)
                    and model.loss_parts is not None):
                dino_part, cssp_part, shared_part, signal_part, koleo_part = model.loss_parts
                trainable = [p for p in model.student.parameters() if p.requires_grad]
                grad_d = torch.autograd.grad(dino_part, trainable, retain_graph=True,
                                             allow_unused=True)
                # Each auxiliary part is guarded on ``requires_grad``.  A part is
                # a plain zero when its family carries no weight (a KoLeo-only
                # arm has no CSSP graph and vice versa) and autograd refuses a
                # tensor with no graph.  For an arm whose parts DO have a graph
                # the guard is a no-op, so the four CSSP calls below keep their
                # order, their retain_graph and therefore their exact numbers.
                grad_c = grad_s = grad_e = grad_k = None
                if cssp_part.requires_grad:
                    grad_c = torch.autograd.grad(cssp_part, trainable, retain_graph=True,
                                                 allow_unused=True)
                if shared_part.requires_grad:
                    grad_s = torch.autograd.grad(shared_part, trainable, retain_graph=True,
                                                 allow_unused=True)
                if signal_part.requires_grad:
                    grad_e = torch.autograd.grad(signal_part, trainable, retain_graph=True,
                                                 allow_unused=True)
                if koleo_part.requires_grad:
                    grad_k = torch.autograd.grad(koleo_part, trainable, retain_graph=True,
                                                 allow_unused=True)
                norm_d = flat_norm(grad_d)
                norm_c = flat_norm(grad_c) if grad_c is not None else 0.
                norm_s = flat_norm(grad_s) if grad_s is not None else 0.
                norm_e = flat_norm(grad_e) if grad_e is not None else 0.
                norm_k = flat_norm(grad_k) if grad_k is not None else 0.
                grad_probe['dino'].append(norm_d)
                grad_probe['cssp'].append(norm_c)
                # The two calibrated hinges are accumulated SEPARATELY and only
                # on the probes where that hinge was actually live, so a probe
                # that fired for the other term cannot drag the divisor toward
                # zero and blow the weight up through the cap.  norm_d is
                # appended on every probe and the ratio is taken between MEANS
                # (`target * mean g_D / mean g_h`), which is well defined even
                # when the two lists have different lengths; with equal lengths
                # it is algebraically the sum-over-sum round 1 used.
                if probe_shared:
                    grad_probe['shared'].append(norm_s)
                    grad_probe['done_shared'] += 1
                if probe_signal:
                    grad_probe['signal'].append(norm_e)
                    grad_probe['done_signal'] += 1
                if probe_koleo:
                    grad_probe['koleo'].append(norm_k)
                    grad_probe['done_koleo'] += 1
                # ``cos`` is the DINO-vs-CSSP cosine and is left exactly as it was
                # for every arm that ran before this term existed (a CSSP arm
                # always has grad_c); ``koleo_cos`` is a separate list so that
                # adding KoLeo cannot change a recorded CSSP number.
                if grad_c is not None:
                    grad_probe['cos'].append(flat_cosine(grad_d, grad_c, norm_d, norm_c))
                if grad_k is not None:
                    grad_probe['koleo_cos'].append(
                        flat_cosine(grad_d, grad_k, norm_d, norm_k))
                grad_probe['steps'].append(step + 1)
                grad_probe['done'] += 1
                mean_d = sum(grad_probe['dino']) / len(grad_probe['dino'])
                # Calibrated on the UNWEIGHTED hinge, so the ratio does not
                # depend on the weight being set and cannot feed back on itself.
                if args.cssp_auto_weight and grad_probe['shared']:
                    mean_s = sum(grad_probe['shared']) / len(grad_probe['shared'])
                    if mean_s > 1e-9:
                        weight = args.cssp_probe_target * mean_d / mean_s
                        if weight * mean_s > args.cssp_probe_cap:
                            weight = args.cssp_probe_cap / mean_s
                        model.cssp_shared_weight = weight
                    else:
                        state.setdefault('cssp_probe_degenerate', []).append(
                            {'step': step + 1, 'term': 'shared', 'g': mean_s})
                # The energy hinge is calibrated independently: it has its own
                # units (a null-corrected cross-covariance energy, not a ratio),
                # so one multiplier cannot serve both.  Its own target keeps the
                # SUM of the two calibrated terms inside the agreed band.
                if (args.cssp_auto_weight and args.cssp_signal_weight > 0
                        and grad_probe['signal']):
                    mean_e = sum(grad_probe['signal']) / len(grad_probe['signal'])
                    if mean_e > 1e-9:
                        weight_e = args.cssp_signal_probe_target * mean_d / mean_e
                        if weight_e * mean_e > args.cssp_probe_cap:
                            weight_e = args.cssp_probe_cap / mean_e
                        model.cssp_signal_weight = weight_e
                    else:
                        state.setdefault('cssp_probe_degenerate', []).append(
                            {'step': step + 1, 'term': 'signal', 'g': mean_e})
                # KoLeo is calibrated the same way and for the same reason: the
                # one previous KoLeo arm used DINOv2's published .1 and put a
                # pre-clip gradient of 167.58 through a model whose total is 2-3.
                # The ratio here is weight-free because ``koleo`` is the
                # UNWEIGHTED hinge, so one measurement sets the weight instead of
                # the oscillating feedback a weighted objective would give.
                if (args.koleo_auto_weight and args.koleo_weight > 0
                        and grad_probe['koleo']):
                    mean_k = sum(grad_probe['koleo']) / len(grad_probe['koleo'])
                    if mean_k > 1e-9:
                        weight_k = args.koleo_probe_target * mean_d / mean_k
                        if weight_k * mean_k > args.koleo_probe_cap:
                            weight_k = args.koleo_probe_cap / mean_k
                        model.koleo_weight = weight_k
                    else:
                        state.setdefault('cssp_probe_degenerate', []).append(
                            {'step': step + 1, 'term': 'koleo', 'g': mean_k})
                mean_s = (sum(grad_probe['shared']) / len(grad_probe['shared'])
                          if grad_probe['shared'] else 0.)
                mean_e = (sum(grad_probe['signal']) / len(grad_probe['signal'])
                          if grad_probe['signal'] else 0.)
                mean_k = (sum(grad_probe['koleo']) / len(grad_probe['koleo'])
                          if grad_probe['koleo'] else 0.)
                state['koleo_grad_probe'] = {
                    'steps': grad_probe['steps'], 'g_dino_encoder': grad_probe['dino'],
                    'g_koleo_unweighted': grad_probe['koleo'],
                    'cosine': grad_probe['koleo_cos'],
                    'weight': model.koleo_weight,
                    'ratio': model.koleo_weight * mean_k / (mean_d + 1e-12)}
                state['cssp_grad_probe'] = {
                    'steps': grad_probe['steps'], 'g_dino_encoder': grad_probe['dino'],
                    'g_cssp_encoder': grad_probe['cssp'],
                    'g_shared_unweighted': grad_probe['shared'],
                    'g_signal_unweighted': grad_probe['signal'],
                    'cosine': grad_probe['cos'],
                    'weight': model.cssp_shared_weight,
                    'weight_signal': model.cssp_signal_weight,
                    'ratio': model.cssp_shared_weight * mean_s / (mean_d + 1e-12),
                    'ratio_signal': (model.cssp_signal_weight * mean_e
                                     / (mean_d + 1e-12))}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.,
                                                       error_if_nonfinite=True)
            optimizer.step()
            # Hard frame constraint, then the EMA copy: the Teacher must see
            # the table the Student is actually scoring against.
            model.project_prototypes()
            model.update_teacher(momentum_schedule[step])
            torch.cuda.synchronize()
            tick = time.time()
            compute_total += tick - data_ready
            completed = step + 1
            if completed % args.log_every == 0 or completed == start_step + 1:
                row = {'step': completed, 'epoch': completed / steps_per_epoch,
                       **{k: float(v) for k, v in metrics.items()},
                       **model.prototype_health(),
                       'grad_norm': float(grad_norm), 'lr': lr_schedule[step],
                       'teacher_temp': temp, 'momentum': momentum_schedule[step],
                       'data_wait_s_per_step': wait_total / (completed - start_step),
                       'compute_s_per_step': compute_total / (completed - start_step),
                       'peak_mib': round(torch.cuda.max_memory_allocated() / 2 ** 20)}
                metrics_file.write(json.dumps(row) + '\n')
                metrics_file.flush()
                state.update(step=completed, heartbeat=time.time(), latest=row)
                write_status()
            if completed % args.checkpoint_every == 0:
                save(completed, output / 'last.pt')
                state['last_checkpoint_step'] = completed
            if args.save_at_steps and completed in args.save_at_steps:
                # Exact-step snapshots, because the Student spectrum collapses
                # inside the first 140 steps and every epoch-boundary artefact
                # is already past it.  Written from the same live model as
                # last.pt, so resuming from one is identical to resuming from
                # last.pt at that step.
                save(completed, output / f'step_{completed:05d}.pt')
                state.setdefault('extra_checkpoints', []).append(completed)
                write_status()
            if completed % steps_per_epoch == 0:
                finished_epoch = completed // steps_per_epoch
                if not finite_tensors(model) or not all(
                        bool(torch.isfinite(t).all()) for s in optimizer.state.values()
                        for t in s.values() if torch.is_tensor(t)):
                    raise FloatingPointError('Nonfinite model or optimizer state')
                save(completed, output / 'last.pt')
                state['last_checkpoint_step'] = completed
                if (finished_epoch % args.probe_every == 0 or finished_epoch <= args.probe_dense_until
                        or completed == total_steps):
                    label = f'epoch_{finished_epoch:03d}'
                    shutil.copyfile(output / 'last.pt', output / f'{label}.pt')
                    probe(label)
                    decision = early_stop_decision(state['probes'], finished_epoch, args.gate_epoch)
                    state.setdefault('early_stop_checks', []).append(
                        {'epoch': finished_epoch, **decision})
                    write_status()
                    if decision['stop']:
                        state['state'] = 'early_stopped'
                        state['early_stop_reason'] = decision['reason']
                        break
        if state['state'] != 'early_stopped':
            save(stop_step, output / 'last.pt')
            state['last_checkpoint_step'] = stop_step
            state['state'] = 'completed' if stop_step == total_steps else 'stopped_at_max_steps'
    except BaseException as error:
        state['state'] = 'failed'
        state['error'] = repr(error)
        raise
    finally:
        state['finished'] = time.time()
        write_status()
        metrics_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--temporal-time', type=float, nargs=2, default=(.25, .6))
    parser.add_argument('--temporal-scale', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--temporal-mass-min', type=float, default=.25)
    parser.add_argument('--lambda-temporal', type=float, default=1.)
    parser.add_argument('--reliability-floor', type=float, default=.25)
    parser.add_argument('--precision', choices=('fp32', 'tf32', 'bf16'), default='bf16')
    # 'running_eval' normalises the Teacher with the Student's EMA running
    # statistics while the Student uses batch statistics; measured on this
    # backbone (v6c, step 140) that kills the Teacher's bottleneck (firing rate
    # 0.0078 = 2/256, cross-sample std exactly 0) and with it the target.  v7
    # therefore defaults to 'train' -- the setting every v6 arm that worked used.
    parser.add_argument('--teacher-bn-mode', choices=('train', 'running_eval'),
                        default='train')
    parser.add_argument('--base-lr', type=float, default=8e-4)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--teacher-temp-warmup-epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--probe-every', type=int, default=5)
    parser.add_argument('--probe-at-start', action='store_true')
    parser.add_argument('--probe-dense-until', type=int, default=10)
    parser.add_argument('--gate-epoch', type=int, default=10)
    parser.add_argument('--embedding-source', choices=('head', 'backbone', 'current'),
                        default='head')
    parser.add_argument('--proto-count', type=int, default=256)
    # The sphere, not the loss, is the object v7 constrains.
    parser.add_argument('--center-rate', dest='center_rate', action='store_true', default=True,
                        help='remove an EMA origin from the spike rate before normalising')
    parser.add_argument('--no-center-rate', dest='center_rate', action='store_false',
                        help='ablation: cosine on the raw non-negative rate (v5 geometry)')
    parser.add_argument('--rate-center-momentum', type=float, default=.99)
    parser.add_argument('--arcsin-sqrt', action='store_true',
                        help='variance-stabilising arcsin(sqrt(rate)) before centring')
    parser.add_argument('--proto-frame-weight', type=float, default=0.,
                        help='weight of the prototype frame potential (spread on the sphere)')
    parser.add_argument('--proto-orthonormal', action='store_true',
                        help='hard Stiefel retraction of the prototype table each step')
    parser.add_argument('--koleo-weight', type=float, default=0.)
    # DINOv2's KoLeo is -log(d) on the nearest-neighbour distance, which is
    # unbounded at d -> 0: the one previous KoLeo arm on this backbone (weight
    # .1, published value) measured pre-clip grad_norm 167.58 against a total of
    # 2-3 and was stopped at e4.71 on budget, so its effect was never tested.
    # A positive floor SHIFTS the log to -log(d + floor), which bounds both the
    # value (-log floor) and the gradient (1/floor) while keeping a finite
    # repulsion everywhere.  It must be a shift, not a clamp: clamp_min makes
    # the gradient exactly ZERO on the closest pairs, which is the opposite of
    # what the term is for.  Calibrated on the measured mean NN distance of the
    # one run that logged it, exp(-1.1417) = 0.319.
    parser.add_argument('--koleo-distance-floor', type=float, default=0.,
                        help='shift delta in -log(d + delta); 0 keeps stock DINOv2')
    parser.add_argument('--koleo-auto-weight', action='store_true',
                        help='set --koleo-weight from the measured Encoder '
                             'gradient ratio instead of the command line')
    parser.add_argument('--koleo-grad-probe', type=int, default=0,
                        help='measure the UNWEIGHTED KoLeo gradient on the '
                             'ENCODER for the first N steps (its own budget)')
    parser.add_argument('--koleo-probe-target', type=float, default=.25,
                        help='target ||lambda_k g_KoLeo|| / ||g_DINO|| on the Encoder')
    parser.add_argument('--koleo-probe-cap', type=float, default=1.,
                        help='absolute cap on ||lambda_k g_KoLeo||, in the same '
                             'units as the pre-clip grad_norm')
    parser.add_argument('--proto-balance-weight', type=float, default=0.)
    parser.add_argument('--balance-temp', type=float, default=.1)
    parser.add_argument('--bottleneck-var-weight', type=float, default=0.)
    parser.add_argument('--bottleneck-decorr-weight', type=float, default=0.)
    parser.add_argument('--bottleneck-var-gamma', type=float, default=.15)
    # CSSP -- three one-sided floors on the cross-view shared spectrum of the
    # two student global views.  Defaults are inert: w=0 skips the term entire.
    parser.add_argument('--cssp-amp-weight', type=float, default=0.)
    parser.add_argument('--cssp-align-weight', type=float, default=0.)
    parser.add_argument('--cssp-shared-weight', type=float, default=0.)
    parser.add_argument('--cssp-amp-floor', type=float, default=0.,
                        help='V_min: floor on tr(C) of each global view')
    parser.add_argument('--cssp-align-floor', type=float, default=0.,
                        help='A_min: floor on the paired-view alignment cosine')
    parser.add_argument('--cssp-rank-floor', type=float, default=1.,
                        help='R_min: floor on the shared cross-view effective rank')
    parser.add_argument('--cssp-signal-weight', type=float, default=0.,
                        help='lambda_e: weight of the shared-ENERGY floor.  The '
                             'rank hinge is scale-free, so raising it can be paid '
                             'for by flattening the shared spectrum; this term '
                             'prices that.  Off by default.')
    parser.add_argument('--cssp-energy-floor', type=float, default=0.,
                        help='E_min: floor on shared_energy_corrected (the '
                             'cross-view covariance energy above its 1/(N-1) '
                             'permutation null).  Required when lambda_e > 0.')
    # Split-gradient instrumentation.  All three default off, so every arm
    # written before them is bit-identical.
    parser.add_argument('--cssp-grad-log', action='store_true',
                        help='every step, log the CSSP and DINO gradients taken '
                             'apart at the two global features (cheap)')
    parser.add_argument('--cssp-grad-probe', type=int, default=0,
                        help='measure the two gradients on the ENCODER for the '
                             'first N steps whose CSSP hinge is non-zero')
    parser.add_argument('--cssp-auto-weight', action='store_true',
                        help='set --cssp-shared-weight from the measured Encoder '
                             'gradient ratio instead of the command line')
    parser.add_argument('--cssp-probe-target', type=float, default=.25,
                        help='target ||lambda g_CSSP|| / ||g_DINO|| on the Encoder')
    parser.add_argument('--cssp-signal-probe-target', type=float, default=.15,
                        help='target ||lambda_e g_signal|| / ||g_DINO|| on the '
                             'Encoder.  Calibrated separately from the rank term '
                             'because the two hinges have different units; the '
                             'two targets sum to the budget for the whole CSSP.')
    parser.add_argument('--cssp-probe-cap', type=float, default=1.,
                        help='absolute cap on ||lambda g_CSSP||, in the same units '
                             'as the pre-clip grad_norm')
    parser.add_argument('--teacher-temp', type=float, nargs=2, default=(.04, .07))
    parser.add_argument('--student-temp', type=float, default=.1)
    parser.add_argument('--head-input', choices=('spikes', 'current'), default='spikes')
    parser.add_argument('--backbone', choices=('vit', 'sew_resnet18'), default='vit')
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='cifar10dvs')
    parser.add_argument('--global-scale', type=float, nargs=2, default=(.35, 1.))
    parser.add_argument('--global-time', type=float, nargs=2, default=(.6, 1.))
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--save-at-steps', type=int, nargs='+', default=[],
                        help='extra last.pt snapshots at these exact steps '
                             '(step_NNNNN.pt); the collapse this project '
                             'studies completes inside the first epoch, so '
                             'epoch-boundary checkpoints cannot see it')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--resume', type=Path)
    main(parser.parse_args())
