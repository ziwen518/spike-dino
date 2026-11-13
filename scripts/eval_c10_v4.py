"""CIFAR10-DVS downstream evaluation of a Spike-DINO v4 encoder (SpikeCLR protocol).

Split: configs/splits/cifar10dvs_seed42.json (90/10). Labeled cohorts from
spike_dino.downstream.cohorts(train, shot, seed): 100/class of train are held
out as validation (selection only); 'full' uses the remaining 8000 train
records, k-shot uses k/class (same records as the SpikeCLR few-shot runs,
seeds 42/1042/2042). The 1000 test records are evaluated once per run with the
validation-selected setting/checkpoint and are never used for selection.

Modes
* lp: frozen encoder (Teacher), BN calibrated label-free on all train records,
  features = CLS ++ GAP spike rate on whole recordings (probe cache); linear
  classifier on standardized features, lr chosen on validation.
* ft: end-to-end fine-tuning of the encoder + linear head on the cohort with
  one augmented global view per sample (pretraining view family, no locals);
  identical recipe for random init (supervised-from-scratch) and pretrained.
"""
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
PHYSICAL_GPU = os.environ.get('SDINO_GPU', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional

from spike_dino import probe_v4
from spike_dino.benchmark_manifest import load_manifest
from spike_dino.data import SENSOR_SIZES, read_event_record
from spike_dino.downstream import cohorts
from spike_dino.sdino_v4 import model_from_config
from spike_dino.views_v4 import IndependentMulticropViews

os.environ['CUDA_VISIBLE_DEVICES'] = PHYSICAL_GPU
MANIFEST = Path('configs/splits/cifar10dvs_seed42.json')
DATA_ROOT = Path('dataset/CIFAR10DVS')
CACHE = Path('runs/probe_cache_c10_v1')
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)


def load_encoder(checkpoint, seed=42, backbone='vit'):
    if checkpoint == 'random':
        torch.manual_seed(seed)
        model = model_from_config({'teacher_bn_mode': 'running_eval', 'image_size': 128,
                                   'backbone_arch': backbone})
        return model.teacher, {'source': f'random_init_seed{seed}', 'backbone': backbone}
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['config'].get('image_size', 224) != 128:
        raise ValueError('Checkpoint was not pretrained at CIFAR10-DVS geometry')
    model = model_from_config(payload['config'])
    state = payload['model']
    # Spike-DINO v5 adds one learnable prototype-logit scale. It multiplies the
    # objective's logits and is read by nothing else, so a frozen-encoder probe
    # loses nothing by not restoring it -- but it is a real weight, and a strict
    # load would refuse the whole checkpoint long after the pretraining run had
    # finished. Drop exactly that key; refuse any other surplus so a genuine
    # architecture mismatch still fails loudly instead of probing a half-loaded
    # encoder.
    objective_only = {'logit_scale'}
    surplus = sorted(set(state) - set(model.state_dict()))
    unknown = [key for key in surplus if key not in objective_only]
    if unknown:
        raise RuntimeError(f'Checkpoint holds weights this model cannot hold: {unknown}')
    if surplus:
        state = {key: value for key, value in state.items() if key not in objective_only}
    model.load_state_dict(state)
    meta = {'source': str(checkpoint), 'step': payload['step'],
            'pretrain_config': payload['config']}
    if surplus:
        meta['not_restored_objective_only'] = surplus
    return model.teacher, meta


def rows_for(manifest, records):
    position = {row['path']: i for i, row in enumerate(manifest['train'])}
    return [position[r['path']] for r in records]


# ---------------------------------------------------------------- linear probe
@torch.no_grad()
def cache_features(encoder, rows, batch=128):
    encoder.eval()
    out = []
    loader = DataLoader(probe_v4.ProbeCacheViews(rows, CACHE), batch_size=batch, num_workers=0)
    for events, _, _ in loader:
        functional.reset_net(encoder)
        out.append(encoder.representation(events.transpose(0, 1).contiguous().cuda())['concat'].float().cpu())
    functional.reset_net(encoder)
    return torch.cat(out)


def train_linear(x, y, lr, seed, steps=1500):
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], 10).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    x, y = x.cuda(), y.cuda()
    g = torch.Generator(device='cuda').manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(y), (min(256, len(y)),), generator=g, device='cuda')
        opt.zero_grad()
        F.cross_entropy(head(x[idx]), y[idx]).backward()
        opt.step()
        sched.step()
    return head


def accuracy(head, x, y):
    with torch.no_grad():
        return float((head(x.cuda()).argmax(-1).cpu() == y).float().mean())


def run_lp(encoder, manifest, labels, shots, seeds, report_test, protocol='cohort'):
    """protocol='cohort' (default): the historical 8000 fit / 1000 dev split.

    protocol='spikeclr': the SpikeCLR LP definition for shot 'full' - the linear
    classifier is fitted on all 9000 labelled train records and the 1000 test
    records are read once (only with --report-test). The learning rate is still
    chosen on a train-only 100/class holdout, so no test information reaches
    selection. Few-shot shots are identical in both protocols.
    """
    probe_v4.calibrate(encoder, list(range(len(manifest['train']))), batch_size=64)
    all_rows = list(range(len(manifest['train']) + len(manifest['test'])))
    feats = cache_features(encoder, all_rows)
    test_rows = list(range(len(manifest['train']), len(all_rows)))
    results = {}
    for shot in shots:
        if protocol == 'spikeclr' and shot == 'full':
            runs = []
            seed = seeds[0]
            train_rec, val_rec = cohorts(manifest, 'full', seed)
            tr, va = rows_for(manifest, train_rec), rows_for(manifest, val_rec)
            fit_rows = tr + va                      # all 9000 labelled train records
            mu, sd = feats[fit_rows].mean(0), feats[fit_rows].std(0).clamp_min(1e-6)
            z = (feats - mu) / sd
            val = {lr: accuracy(train_linear(z[tr], labels[tr], lr, seed), z[va], labels[va])
                   for lr in LR_GRID}
            best = max(val, key=val.get)
            head = train_linear(z[fit_rows], labels[fit_rows], best, seed)
            run = {'seed': seed, 'train_records': len(fit_rows), 'lr_selected_on': 'train_holdout',
                   'val_by_lr': val, 'selected_lr': best, 'val_acc1': val[best]}
            if report_test:
                run['test_acc1'] = accuracy(head, z[test_rows], labels[test_rows])
            runs.append(run)
            results[shot] = summarize(runs)
            print('lp(protocol=spikeclr)', shot,
                  {k: v for k, v in results[shot].items() if k != 'runs'}, flush=True)
            continue
        runs = []
        for seed in (seeds if shot != 'full' else seeds[:1]):
            train_rec, val_rec = cohorts(manifest, shot, seed)
            tr, va = rows_for(manifest, train_rec), rows_for(manifest, val_rec)
            mu, sd = feats[tr].mean(0), feats[tr].std(0).clamp_min(1e-6) if len(tr) > 1 else torch.ones(feats.shape[1])
            z = (feats - mu) / sd
            ytr, yva, yte = labels[tr], labels[va], labels[test_rows]
            val = {lr: accuracy(train_linear(z[tr], ytr, lr, seed), z[va], yva) for lr in LR_GRID}
            best = max(val, key=val.get)
            head = train_linear(z[tr], ytr, best, seed)
            run = {'seed': seed, 'train_records': len(tr), 'val_by_lr': val, 'selected_lr': best,
                   'val_acc1': val[best]}
            if report_test:
                run['test_acc1'] = accuracy(head, z[test_rows], yte)
            runs.append(run)
        results[shot] = summarize(runs)
        print('lp', shot, {k: v for k, v in results[shot].items() if k != 'runs'}, flush=True)
    return results


# ------------------------------------------------------------------ fine-tune
class AugmentedRecords(Dataset):
    def __init__(self, manifest, rows, seed):
        self.records = [manifest['train'][r] for r in rows]
        self.seed = seed
        self.epoch = 0
        # A per-visit counter, not the epoch, drives the augmentation seed.
        # DataLoader workers hold a pickled copy of this dataset, so an epoch
        # written by the parent never reaches them; with persistent workers that
        # would freeze every view at epoch 0. The counter advances inside each
        # worker instead and keeps the augmentation fresh for every visit.
        self.visits = 0
        self.transform = IndependentMulticropViews(
            SENSOR_SIZES['cifar10dvs'], render_size=128, global_size=128, local_size=64,
            local_crops=0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        events = read_event_record(DATA_ROOT / 'raw' / rec['path'], 'cifar10dvs')
        self.visits += 1
        view = self.transform(events, (self.seed * 1_000_003 + i * 1000
                                       + self.visits * 10_000_019) % 2**31)[0]
        return view, rec['label']


class Classifier(nn.Module):
    """End-to-end classifier over one encoder readout, one linear head.

    ``head_branch='concat'`` is the historical protocol -- layer-4 GAP spikes
    (512) ++ layer-3 GAP spikes (256) = 768-d, ``nn.Linear(768, 10)`` -- and is
    bit-identical to every FT arm this project has run.  ``'cls'`` is the 512-d
    ``z_bar``: spatial GAP inside each timestep ([T,N,512]) then the temporal
    mean ([N,512]).  ``'gap'`` is the 256-d layer-3 readout.  The SpikeCLR
    paper's supervised-FT head is a single ``nn.Linear(512, 10)`` on that
    512-d feature, so ``'cls'`` is the strictly comparable column -- the user
    fixed this definition on 2026-11-12 (FT head = LP head = one Linear(512,10),
    whole backbone forward, every layer unfrozen).

    Why the non-concat features are computed HERE rather than by reusing a
    model-side helper:

    * ``Encoder.classifier_features`` / ``sew_classifier_features`` are
      hard-wired to the 768-d concat, and ``spike_dino/sdino_sew.py`` and
      ``spike_dino/sdino_v4.py`` are both in ``scripts/train_sdino_v7.py``'s
      ``SOURCES`` list, i.e. their bytes are part of every future arm's
      recorded ``source_sha256``.  Editing them would silently unpair every
      future arm from Arm A/F for a downstream-protocol change.
    * ``Encoder.representation`` produces the right slices but is decorated
      ``@torch.no_grad()``; reusing it would make "FT" a head-only fit on
      frozen features -- exactly the defect that voided every FT arm before
      2026-11-10.

    So the branch is built from ``Encoder.encode``, which is grad-enabled and
    is the same call ``sew_classifier_features`` makes; only the pooling after
    it differs.  ``'cls'`` is therefore the identical tensor to
    ``sew_representation``'s ``cls`` key, and ``run_lp --branch cls`` (via
    ``branch_view``) fits the same feature from the frozen side.
    """

    BRANCHES = ('concat', 'cls', 'gap')

    def __init__(self, encoder, head_branch='concat'):
        super().__init__()
        if head_branch not in self.BRANCHES:
            raise ValueError('Unknown head branch %r; expected one of %s'
                             % (head_branch, self.BRANCHES))
        self.encoder = encoder
        self.head_branch = head_branch
        if head_branch == 'concat':
            width = encoder.classifier_dim
        elif head_branch == 'cls':
            width = encoder.dim
        else:
            width = encoder.backbone.mid_dim
        self.head = nn.Linear(width, 10)

    def features(self, x):
        if self.head_branch == 'concat':
            return self.encoder.classifier_features(x)
        if self.head_branch == 'cls':
            # z_bar: [T,N,512] per-timestep spatial GAP -> mean over T.
            cls, _patch, _pooled = self.encoder.encode(x)
            return cls.float().mean(0)
        backbone = self.encoder.backbone
        keep = backbone.keep_output_current
        backbone.keep_output_current = True
        try:
            _cls, _patch, _pooled = self.encoder.encode(x)
            mid, _current_fine, _current_mid = backbone.aux
        finally:
            backbone.aux = None
            backbone.output_current = None
            backbone.keep_output_current = keep
        return mid.float().mean((0, 3, 4))

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, rows, batch=128):
    model.eval()
    loader = DataLoader(probe_v4.ProbeCacheViews(rows, CACHE), batch_size=batch, num_workers=0)
    correct = total = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for events, y, _ in loader:
            functional.reset_net(model)
            pred = model(events.transpose(0, 1).contiguous().cuda()).argmax(-1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    functional.reset_net(model)
    return correct / total


def run_ft(encoder0, manifest, shots, seeds, report_test, args):
    test_rows = list(range(len(manifest['train']), len(manifest['train']) + len(manifest['test'])))
    head_branch = getattr(args, 'head_branch', 'concat')
    results = {}
    for shot in shots:
        runs = []
        for seed in (seeds if shot != 'full' else seeds[:1]):
            train_rec, val_rec = cohorts(manifest, shot, seed)
            tr, va = rows_for(manifest, train_rec), rows_for(manifest, val_rec)
            torch.manual_seed(seed)
            model = Classifier(copy.deepcopy(encoder0), head_branch).cuda()
            # Both loaders hand back `model.teacher`, and v6 builds the Teacher as
            # `deepcopy(student).requires_grad_(False)` (sdino_v6.py:493).  A deepcopy
            # keeps that flag, an AdamW parameter that never receives `.grad` is never
            # updated, so before this line EVERY fine-tuning arm in this project (both
            # entry points, pretrained and random-init, all three LR/BN recipes) was a
            # head-only fit on frozen features -- measured 2026-11-10: 0 of 74 encoder
            # tensors received a gradient, and the saved best states are bit-identical
            # to epoch_300.pt in every weight, differing only in BN running statistics
            # (tmp/_check_ft_grads.py, tmp/_ckpt_diff_cpu.py).  The frozen path is `--mode
            # lp`, so FT always enables gradients; the numbers printed below make the
            # distinction visible in every log.
            model.encoder.requires_grad_(True)
            trainable = sum(p.requires_grad for p in model.encoder.parameters())
            print('ft encoder trainable: %d of %d tensors'
                  % (trainable, len(list(model.encoder.parameters()))), flush=True)
            data = AugmentedRecords(manifest, tr, seed)
            batch = min(args.batch, len(tr))
            steps_per_epoch = max(1, len(tr) // batch)
            total = args.ft_epochs * steps_per_epoch if shot == 'full' else args.fewshot_steps
            warmup = max(1, total // 20)
            # Discriminative fine-tuning is opt-in. With --ft-encoder-lr absent
            # (or equal to --ft-lr) the model is one parameter group and the
            # optimiser is exactly the original one over model.parameters().
            encoder_lr = float(getattr(args, 'ft_encoder_lr', 0.) or 0.)
            weight_decay = float(getattr(args, 'ft_weight_decay', .05))
            if encoder_lr and encoder_lr != args.ft_lr:
                groups = [{'params': list(model.encoder.parameters()), 'lr': encoder_lr},
                          {'params': list(model.head.parameters()), 'lr': args.ft_lr}]
            else:
                groups = [{'params': list(model.parameters()), 'lr': args.ft_lr}]
            opt = torch.optim.AdamW(groups, lr=args.ft_lr, weight_decay=weight_decay)
            base_lrs = [group['lr'] for group in groups]
            lr_at = lambda s: (s / warmup if s < warmup else
                               .5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
            # Opt-in BN freeze for fine-tuning. Measured 2026-11-10 on the
            # v6 e300 checkpoint: during FT the encoder's BN running statistics
            # drift 13-44 % while the encoder weights move only ~2 %, so the
            # classifier head is fitted on batch-statistic features and then
            # scored on running-statistic features that no longer match -- the
            # same batch-versus-running mismatch that collapsed the v6c
            # teacher. With --ft-freeze-bn every BN layer stays in eval mode after
            # a clean-cache calibration (see below), so the head sees the same
            # features at train and at eval time. Default stays off, so every
            # earlier FT arm is bit-identical.
            freeze_bn = bool(getattr(args, 'ft_freeze_bn', False))
            bn_modules = [m for m in model.modules()
                          if getattr(m, 'running_mean', None) is not None]
            if freeze_bn:
                # The statistics frozen must be the ones the reported readouts
                # use. `run_lp` always recalibrates BN on the clean probe cache
                # (`probe_v4.calibrate`) before extracting features, so the
                # checkpoint's own running statistics -- accumulated over the
                # pretraining AUGMENTED views -- are not what the frozen linear
                # probe (62.97 on this checkpoint) was scored with. Freezing the
                # pretraining statistics would change the normalisation of the
                # baseline side of the comparison and make a null unreadable.
                # Calibrate on the training rows of the clean cache first, then
                # hold: features at train and at eval time are then produced by
                # the same function AND the eval side matches `run_lp` exactly.
                calibration = probe_v4.calibrate(model.encoder, tr, batch_size=64)
                print('ft bn calibrated before freeze %s' % json.dumps(calibration), flush=True)
            evals = max(1, total // args.evaluations)
            best, best_state, history, step, epoch = -1., None, [], 0, 0
            started = time.time()
            # One loader for the whole run. Rebuilding it every epoch respawns
            # every worker, and on this server process creation costs ~6 s each
            # (measured 2026-11-10: 16 workers took >2 min to come up and the
            # spawn stalled at 8), which dwarfs the 125 steps of an epoch. With
            # persistent workers the pool is built once; the augmentation stays
            # fresh because AugmentedRecords derives its seed from a per-visit
            # counter that lives inside each worker.
            loader = DataLoader(data, batch_size=batch, shuffle=True, drop_last=True,
                                num_workers=args.workers,
                                persistent_workers=bool(args.workers),
                                generator=torch.Generator().manual_seed(seed * 7919))
            while step < total:
                data.epoch = epoch
                for x, y in loader:
                    model.train()
                    if freeze_bn:
                        for module in bn_modules:
                            module.eval()
                    for gparam, base_lr in zip(opt.param_groups, base_lrs):
                        gparam['lr'] = base_lr * lr_at(step)
                    functional.reset_net(model)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(
                            model(x.transpose(0, 1).contiguous().cuda()).float(), y.cuda(),
                            label_smoothing=float(getattr(args, 'ft_label_smoothing', 0.) or 0.))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    step += 1
                    if step % evals == 0 or step == total:
                        acc = evaluate(model, va)
                        # Output only: a 100-epoch full-label fine-tune runs for
                        # hours and previously reported nothing until it ended,
                        # so a long run could only be judged after the fact.
                        print(f'ft-step {step}/{total} loss {float(loss):.4f} '
                              f'val {acc * 100:.2f}', flush=True)
                        history.append({'step': step, 'loss': float(loss), 'val_acc1': acc})
                        if acc > best:
                            best, best_state = acc, copy.deepcopy(model.state_dict())
                    if step >= total:
                        break
                epoch += 1
            run = {'seed': seed, 'train_records': len(tr), 'steps': total, 'val_acc1': best,
                   'head_branch': head_branch,
                   'history': history, 'seconds': time.time() - started}
            if report_test:
                model.load_state_dict(best_state)
                run['test_acc1'] = evaluate(model, test_rows)
            runs.append(run)
            print('ft', shot, seed, {k: v for k, v in run.items() if k != 'history'}, flush=True)
            # Persist the best-dev weights so the fine-tuned model can be
            # re-scored later (in particular with precise-BN calibration, the
            # way the frozen probe is scored) without a 100-epoch retrain.
            # Output only: the metrics above are unchanged.
            if best_state is not None:
                stem = Path(str(args.output)).with_suffix('')
                torch.save(best_state, '%s_%s_best.pt' % (stem, shot))
        results[shot] = summarize(runs)
    return results


def summarize(runs):
    out = {'runs': runs}
    for key in ('val_acc1', 'test_acc1'):
        vals = [r[key] for r in runs if key in r]
        if vals:
            out[key + '_mean'] = float(np.mean(vals))
            out[key + '_std'] = float(np.std(vals))
    return out


def main(args):
    if args.output.exists():
        raise RuntimeError('Output exists; inspect rather than overwrite')
    probe_v4.set_cache(CACHE)
    manifest = load_manifest(MANIFEST, 'cifar10dvs')
    labels = torch.as_tensor(np.load(CACHE / 'labels.npy'))
    encoder, meta = load_encoder(args.checkpoint, args.init_seed, args.backbone)
    encoder = encoder.cuda()
    report = {'protocol': 'CIFAR10-DVS seed42 90/10; cohorts(validation 100/class from train); '
                          'test evaluated once with validation-selected setting',
              'mode': args.mode, 'shots': args.shots, 'seeds': args.seeds, 'report_test': args.report_test,
              'physical_gpu': PHYSICAL_GPU, **{k: v for k, v in meta.items() if k != 'pretrain_config'},
              'args': {k: str(v) for k, v in vars(args).items()}}
    started = time.time()
    if args.mode == 'lp':
        report['lp_protocol'] = args.protocol
        report['results'] = run_lp(encoder, manifest, labels, args.shots, args.seeds,
                                   args.report_test, args.protocol)
    else:
        report['results'] = run_ft(encoder, manifest, args.shots, args.seeds, args.report_test, args)
    report['seconds'] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help="path or 'random'")
    parser.add_argument('--mode', choices=('lp', 'ft'), required=True)
    parser.add_argument('--shots', nargs='+', default=['full', '1', '9', '45'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1042, 2042])
    parser.add_argument('--backbone', choices=('vit', 'sew_resnet18'), default='vit',
                        help='random-init reference architecture; checkpoints carry their own')
    parser.add_argument('--report-test', action='store_true')
    parser.add_argument('--protocol', choices=('cohort', 'spikeclr'), default='cohort',
                        help="'spikeclr': LP on all 9000 train records, test read once")
    parser.add_argument('--init-seed', type=int, default=42)
    parser.add_argument('--ft-epochs', type=int, default=100)
    parser.add_argument('--fewshot-steps', type=int, default=600)
    parser.add_argument('--ft-lr', type=float, default=1e-3)
    parser.add_argument('--ft-encoder-lr', type=float, default=0.,
                        help='0 (default) keeps one global --ft-lr parameter group')
    parser.add_argument('--ft-weight-decay', type=float, default=.05)
    parser.add_argument('--ft-label-smoothing', type=float, default=0.,
                        help='label smoothing for the fine-tune cross-entropy;'
                             ' 0 (default) keeps every earlier FT arm bit-identical')
    parser.add_argument('--ft-freeze-bn', action='store_true',
                        help='calibrate BN on the clean-cache training records,'
                             ' then keep every BN in eval mode during fine-tuning'
                             ' (default off: bit-identical to earlier FT arms)')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--evaluations', type=int, default=25)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
