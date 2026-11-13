"""Spike-DINO v4 pretraining: all-spiking ViT teacher/student, DINO recipe.

Usage (server, GPU0 only):
  python -m scripts.train_sdino_v4 --output runs/sdino_v4_<tag> --epochs 10 \
      --batch-size 64 --precision bf16 --probe-every 1 --probe-at-start
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
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

import subprocess

from scripts.train_generalization_reference import atomic
from spike_dino.data import SENSOR_SIZES
from spike_dino.pretrain_v3 import FixedEpochBatchSampler
from spike_dino import probe_v4
from spike_dino.probe_v4 import load_split, run_probe, summary_line
from spike_dino.sdino_v4 import SpikeDinoV4, cosine_schedule, teacher_temperature
from spike_dino.views_v4 import IndependentMulticropViews, MulticropEventDataset, support_iou

# scripts.train_generalization_reference pins CUDA_VISIBLE_DEVICES at import
# time; CUDA is still uninitialized here, so restore the selected GPU.
if torch.cuda.is_initialized():
    raise RuntimeError('CUDA initialized before GPU selection')
os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU

DATA_ROOT = Path('dataset/mini_nimagenet')
MANIFEST = DATA_ROOT / 'metadata/parts1_2_manifest.json'
# Per-dataset pretraining data and view geometry. SPS tokenizes at /16, so
# CIFAR10-DVS (128x128 sensor) uses 128px globals (8x8 tokens), 64px locals.
DATASETS = {
    'mini_nimagenet': dict(root='dataset/mini_nimagenet',
                           manifest='dataset/mini_nimagenet/metadata/parts1_2_manifest.json',
                           render=320, global_size=224, local_size=96,
                           probe_cache='runs/probe_cache_v4'),
    'cifar10dvs': dict(root='dataset/CIFAR10DVS', manifest='configs/splits/cifar10dvs_seed42.json',
                       render=128, global_size=128, local_size=64,
                       probe_cache='runs/probe_cache_c10_v1'),
}
SOURCES = ['spike_dino/sdino_v4.py', 'spike_dino/views_v4.py', 'spike_dino/fused_lif.py',
           'spike_dino/probe_v4.py', 'spike_dino/spiking_vit.py', 'spike_dino/dinov3.py',
           'spike_dino/data.py', 'scripts/train_sdino_v4.py', 'spike_dino/pretrain_v3.py',
           'spike_dino/sdino_sew.py']


def gpu_used_mib(index):
    return int(subprocess.check_output(
        ['nvidia-smi', '-i', str(index), '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], text=True).strip())


def gpu_idle(index=PHYSICAL_GPU, limit_mib=2300):
    used = gpu_used_mib(index)
    if used > limit_mib:
        raise RuntimeError(f'GPU{index} occupied: {used}MiB')


def verify_physical_gpu(index=PHYSICAL_GPU, probe_mib=2048):
    """Allocate on cuda:0 and confirm the memory lands on the selected physical GPU."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(index):
        raise RuntimeError('CUDA_VISIBLE_DEVICES does not match the selected GPU')
    before = gpu_used_mib(index)
    block = torch.zeros(probe_mib * 1024 * 256, dtype=torch.float32, device='cuda')
    time.sleep(1)
    after = gpu_used_mib(index)
    del block
    torch.cuda.empty_cache()
    if after - before < probe_mib // 2:
        raise RuntimeError(f'Allocation did not land on physical GPU{index}: {before}->{after}MiB')
    return {'physical_gpu': index, 'probe_mib': probe_mib, 'before_mib': before, 'after_mib': after}


def source_hashes():
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCES}


def build_config(args):
    return {
        'method': 'Spike-DINO v4', 'dataset': args.dataset, 'dataset_note': (
            'mini_nimagenet parts1+2 unlabeled' if args.dataset == 'mini_nimagenet'
            else 'CIFAR10-DVS seed42 90/10 train split, labels unused; test never read'),
        'manifest': DATASETS[args.dataset]['manifest'], 'seed': args.seed,
        'time_steps': 4, 'render_size': DATASETS[args.dataset]['render'],
        'global_size': DATASETS[args.dataset]['global_size'],
        'local_size': DATASETS[args.dataset]['local_size'],
        'image_size': DATASETS[args.dataset]['global_size'],
        'probe_cache': DATASETS[args.dataset]['probe_cache'],
        'local_crops': args.local_crops, 'global_scale': list(args.global_scale), 'local_scale': [.05, .35],
        'global_time': list(args.global_time), 'local_time': list(args.local_time),
        'embedding_source': args.embedding_source,
        'analog': args.analog,
        'head_input': args.head_input,
        'thinning': [.8, 1.], 'flip': .5, 'polarity_jitter': [.8, .3],
        'event_support_gate_min_relative_mass': .5,
        'backbone_arch': args.backbone,
        'backbone': ('SEW-ResNet-18 (spikingjelly, 2-ch input, identity residual)'
                     if args.backbone == 'sew_resnet18' else
                     'SPS-BN 12x384 6-head spikformer-count LayerScale.1 (supervised dev 49.15)'),
        'head_input': ('layer-4 GAP spikes ++ LIF(BN(layer-4 GAP spikes)) (1024)'
                       if args.backbone == 'sew_resnet18' else
                       'binary CLS ++ BN+LIF(mean patch current) spikes'),
        'representation': ('layer-4 GAP rate ++ layer-3 GAP rate' if args.backbone == 'sew_resnet18'
                           else 'CLS rate ++ GAP patch rate'),
        'bn_policy': ('student train batch stats; teacher calibrated running stats in eval'
                      if args.teacher_bn_mode == 'running_eval' else
                      'student and teacher train batch stats; precise-BN evaluation'),
        'teacher_bn_mode': args.teacher_bn_mode,
        'koleo_weight': args.koleo_weight,
        'rate_centering': args.rate_centering,
        'objective': args.objective,
        'moco_queue': args.moco_queue,
        'moco_temp': args.moco_temp,
        'moco_weight': args.moco_weight,
        'bottleneck_var_weight': args.bottleneck_var_weight,
        'bottleneck_var_gamma': args.bottleneck_var_gamma,
        'bottleneck_decorr_weight': args.bottleneck_decorr_weight,
        'bottleneck_regularizer': ('Student global-crop 256-d bottleneck rate: '
                                   'w_var*mean relu(gamma-std_j) + w_decorr*mean offdiag corr^2'),
        'rate_center_momentum': args.rate_center_momentum,
        'readout': ('cos(normalize(bottleneck rate - EMA rate center), prototypes); '
                    'center updated from Student global crops only, copied to Teacher'
                    if args.rate_centering else 'cos(normalize(bottleneck rate), prototypes)'),
        'koleo_features': 'L2-normalized CLS rate ++ GAP patch rate per Student global crop',
        'prototypes': 4096, 'head_hidden': 2048, 'bottleneck': 256,
        'student_temp': .1, 'teacher_temp': [.04, .07, args.teacher_temp_warmup_epochs],
        'center_momentum': .9, 'teacher_momentum': [.996, 1.],
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'base_lr': args.base_lr, 'lr': args.base_lr * args.batch_size / 256,
        'min_lr': 1e-6, 'warmup_epochs': args.warmup_epochs, 'weight_decay': [.04, .4],
        'gradient_clip': 3., 'freeze_prototypes_epochs': 1, 'precision': args.precision,
        'early_stop': {'gate_epoch': args.gate_epoch, 'gate': 'concat linear>=base+8, kNN>=base+5, rank>=base/2',
                       'after_gate': '2 probes without +1 linear and CLS correlation rank<10'},
        'student_running_stats': ('global pass only' if args.teacher_bn_mode == 'running_eval' else 'all passes'),
        'fused_lif': 'TorchScript FusedLIF (spikes equal to SpikingJelly; grad diff ~4e-7)', 'view_batching': 'same-resolution crops share one forward', 'official_validation_read': False,
    }


def parameter_groups(model):
    decay, no_decay = [], []
    for name, parameter in model.student.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0.}]


def early_stop_decision(probes, epoch, gate_epoch):
    """Pre-registered representation stop rule (fixed before the v4b run).

    Baseline = epoch_000 (random-init teacher, same probe). At gate_epoch the
    concat readout must reach linear >= baseline+8 points, kNN >= baseline+5
    points and keep effective rank >= half the baseline rank. After the gate,
    stop when the two latest probes both fail to beat the best earlier concat
    linear by >= 1 point while the CLS correlation rank is below 10.
    """
    baseline = next((p for p in probes if p['epoch'] == 'epoch_000'), None)
    latest = probes[-1]
    if baseline is None:
        return {'stop': False, 'reason': 'no epoch_000 baseline probe'}
    if epoch == gate_epoch:
        failures = []
        if latest['concat_linear'] < baseline['concat_linear'] + .08:
            failures.append('linear < baseline+8')
        if latest['concat_knn'] < baseline['concat_knn'] + .05:
            failures.append('kNN < baseline+5')
        if latest['concat_rank'] < .5 * baseline['concat_rank']:
            failures.append('rank < half baseline')
        return {'stop': bool(failures), 'reason': ('gate failed: ' + ', '.join(failures)
                                                   if failures else 'gate passed')}
    if epoch > gate_epoch and len(probes) >= 4:
        earlier_best = max(p['concat_linear'] for p in probes[:-2])
        stalled = all(p['concat_linear'] < earlier_best + .01 for p in probes[-2:])
        low_rank = (latest['cls_correlation_rank'] or 0) < 10
        if stalled and low_rank:
            return {'stop': True, 'reason': 'two probes without +1 linear and CLS correlation rank < 10'}
    return {'stop': False, 'reason': 'continue'}


def finite_tensors(module):
    return all(bool(torch.isfinite(t).all()) for t in module.state_dict().values()
               if torch.is_tensor(t) and t.is_floating_point())


@torch.no_grad()
def calibrate_initial_teacher(model, dataset, batch_size, seed, batches=16):
    """Fixed unlabeled BN initialization for an eval-mode EMA Teacher."""
    norms = [module for module in model.teacher.modules()
             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms or any(not module.track_running_stats for module in norms):
        raise RuntimeError('Running-eval Teacher requires running-stat BN')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
    model.teacher.train()
    sampler = FixedEpochBatchSampler(len(dataset), batch_size, seed + 100003,
                                     0, batches)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4,
                        pin_memory=True,
                        generator=torch.Generator().manual_seed(seed + 100004))
    cached = []
    for views in loader:
        globals_ = torch.cat([view.permute(1, 0, 2, 3, 4).contiguous().cuda(
            non_blocking=True) for view in views[:2]], 1)
        cached.append(globals_.cpu())
        functional.reset_net(model.teacher)
        model.teacher(globals_)
        functional.reset_net(model.teacher)
    for module in norms:
        module.momentum = .1
    model.train()  # Teacher -> eval with the calibrated running statistics
    rate_center = None
    if model.teacher.rate_centering:
        # Initialize the bottleneck rate center from the calibrated eval-mode
        # Teacher on the same unlabeled global crops (no labels, no update).
        rates = []
        for globals_ in cached:
            functional.reset_net(model.teacher)
            rates.append(model.teacher(globals_.cuda(non_blocking=True))['bottleneck_rate'])
            functional.reset_net(model.teacher)
        model.teacher.rate_center.copy_(torch.cat(rates).mean(0))
        rate_center = {'mean': float(model.teacher.rate_center.mean()),
                       'min': float(model.teacher.rate_center.min()),
                       'max': float(model.teacher.rate_center.max())}
    for student_buffer, teacher_buffer in zip(model.student.buffers(),
                                              model.teacher.buffers()):
        student_buffer.copy_(teacher_buffer)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    return {'batches': batches, 'records': batches * batch_size,
            'labels_used': False, 'teacher_eval': not model.teacher.training,
            'initial_rate_center': rate_center,
            'student_teacher_buffers_equal': all(torch.equal(a, b) for a, b
                in zip(model.student.buffers(), model.teacher.buffers()))}


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

    transform = IndependentMulticropViews(
        SENSOR_SIZES[args.dataset], local_crops=args.local_crops,
        render_size=cfg['render_size'], global_size=cfg['global_size'],
        local_size=cfg['local_size'],
        local_time=tuple(args.local_time), global_scale=tuple(args.global_scale),
        global_time=tuple(args.global_time))
    dataset = MulticropEventDataset(Path(DATASETS[args.dataset]['root']),
                                    Path(DATASETS[args.dataset]['manifest']), transform,
                                    seed=args.seed)
    cfg['train_records'] = len(dataset)
    probe_v4.set_cache(cfg['probe_cache'])
    steps_per_epoch = len(dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    stop_step = min(total_steps, args.max_steps) if args.max_steps else total_steps

    model = SpikeDinoV4(prototypes=4096, teacher_bn_mode=args.teacher_bn_mode,
                        image_size=cfg['image_size'],
                        koleo_weight=args.koleo_weight,
                        rate_centering=args.rate_centering,
                        rate_center_momentum=args.rate_center_momentum,
                        bottleneck_var_weight=args.bottleneck_var_weight,
                        bottleneck_var_gamma=args.bottleneck_var_gamma,
                        bottleneck_decorr_weight=args.bottleneck_decorr_weight,
                        objective=args.objective, moco_queue=args.moco_queue,
                        moco_temp=args.moco_temp, moco_weight=args.moco_weight,
                        embedding_source=args.embedding_source,
                        analog=args.analog, head_input=args.head_input,
                        backbone=args.backbone).cuda().train()
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=cfg['lr'], weight_decay=.04)
    lr_schedule = cosine_schedule(cfg['lr'], 1e-6, total_steps,
                                  args.warmup_epochs * steps_per_epoch, 0.)
    wd_schedule = cosine_schedule(.04, .4, total_steps)
    momentum_schedule = cosine_schedule(.996, 1., total_steps)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('batch_size', 'epochs', 'local_crops', 'base_lr', 'precision',
                    'teacher_bn_mode', 'local_time', 'seed', 'teacher_temp', 'warmup_epochs',
                    'weight_decay', 'student_running_stats', 'early_stop', 'koleo_weight',
                    'rate_centering', 'rate_center_momentum', 'bottleneck_var_weight',
                    'bottleneck_var_gamma', 'bottleneck_decorr_weight', 'objective',
                    'moco_queue', 'moco_temp', 'moco_weight', 'global_scale', 'global_time',
                    'embedding_source', 'analog', 'head_input', 'dataset',
                    'backbone_arch'):
            # Keys added after a run started default to that run's behaviour.
            legacy = {'koleo_weight': 0., 'rate_centering': False, 'rate_center_momentum': .9,
                      'bottleneck_var_weight': 0., 'bottleneck_var_gamma': .15,
                      'bottleneck_decorr_weight': 0., 'objective': 'dino',
                      'moco_queue': 8192, 'moco_temp': .2, 'moco_weight': 1.,
                      'global_scale': [.35, 1.], 'global_time': [.6, 1.],
                      'embedding_source': 'head', 'analog': 'none', 'head_input': 'spikes',
                      'dataset': 'mini_nimagenet', 'backbone_arch': 'vit'}
            if checkpoint['config'].get(key, legacy.get(key)) != cfg[key]:
                raise ValueError(f'Resume config differs: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    initial_teacher_calibration = (
        calibrate_initial_teacher(model, dataset, args.batch_size, args.seed)
        if args.teacher_bn_mode == 'running_eval' and start_step == 0 else None)
    split = load_split()
    sampler = FixedEpochBatchSampler(len(dataset), args.batch_size, args.seed,
                                     start_step, stop_step)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
    status_path = output / 'status.json'
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    # Hash the sources once, at run start: a mid-run edit of an unrelated file
    # must not relabel later checkpoints with code they were not trained with.
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
    atomic(status_path, state)
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
        atomic(status_path, state)
        torch.cuda.empty_cache()

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
            if len(views) != 2 + args.local_crops:
                raise RuntimeError('Unexpected number of views')
            data_ready = time.time()
            wait_total += data_ready - tick
            if step == start_step:
                iou = support_iou(views[0].transpose(0, 1), views[1].transpose(0, 1))
                state['first_batch_global_output_support_iou'] = float(iou.mean())
            epoch = step // steps_per_epoch
            for index, group in enumerate(optimizer.param_groups):
                group['lr'] = lr_schedule[step]
                if index == 0:
                    group['weight_decay'] = wd_schedule[step]
            temp = teacher_temperature(step, steps_per_epoch, args.teacher_temp_warmup_epochs)
            loss, metrics = model(views, temp, autocast_dtype)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < cfg['freeze_prototypes_epochs']:
                model.student.head.prototypes.grad = None
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3.,
                                                       error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(momentum_schedule[step])
            torch.cuda.synchronize()
            tick = time.time()
            compute_total += tick - data_ready
            completed = step + 1
            if completed % args.log_every == 0 or completed == start_step + 1:
                row = {'step': completed, 'epoch': completed / steps_per_epoch,
                       **{k: float(v) for k, v in metrics.items()},
                       'grad_norm': float(grad_norm), 'lr': lr_schedule[step],
                       'teacher_temp': temp, 'momentum': momentum_schedule[step],
                       'data_wait_s_per_step': wait_total / (completed - start_step),
                       'compute_s_per_step': compute_total / (completed - start_step),
                       'peak_mib': round(torch.cuda.max_memory_allocated() / 2 ** 20)}
                metrics_file.write(json.dumps(row) + '\n')
                metrics_file.flush()
                state.update(step=completed, heartbeat=time.time(), latest=row)
                atomic(status_path, state)
            if completed % args.checkpoint_every == 0:
                save(completed, output / 'last.pt')
                state['last_checkpoint_step'] = completed
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
                    atomic(status_path, state)
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
        atomic(status_path, state)
        metrics_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--local-crops', type=int, default=6)
    parser.add_argument('--local-time', type=float, nargs=2, default=(.6, 1.))
    parser.add_argument('--precision', choices=('fp32', 'tf32', 'bf16'), default='bf16')
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
    parser.add_argument('--koleo-weight', type=float, default=0.)
    parser.add_argument('--rate-centering', action='store_true')
    parser.add_argument('--rate-center-momentum', type=float, default=.9)
    parser.add_argument('--objective', choices=('dino', 'moco', 'dino_moco'), default='dino')
    parser.add_argument('--moco-weight', type=float, default=1.)
    parser.add_argument('--embedding-source', choices=('head', 'backbone', 'current'), default='head')
    parser.add_argument('--head-input', choices=('spikes', 'current'), default='spikes')
    parser.add_argument('--backbone', choices=('vit', 'sew_resnet18'), default='vit')
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='mini_nimagenet')
    parser.add_argument('--analog', choices=('none', 'count', 'softmax', 'spiking_softmax_attention'),
                        default='none')
    parser.add_argument('--global-scale', type=float, nargs=2, default=(.35, 1.))
    parser.add_argument('--global-time', type=float, nargs=2, default=(.6, 1.))
    parser.add_argument('--moco-queue', type=int, default=8192)
    parser.add_argument('--moco-temp', type=float, default=.2)
    parser.add_argument('--bottleneck-var-weight', type=float, default=0.)
    parser.add_argument('--bottleneck-var-gamma', type=float, default=.15)
    parser.add_argument('--bottleneck-decorr-weight', type=float, default=0.)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--resume', type=Path)
    main(parser.parse_args())
