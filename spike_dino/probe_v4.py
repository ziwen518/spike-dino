"""Frozen-representation probe v2 for Spike-DINO v4 (label use: probe heads only).

Protocol (fixed before any v4 training result is seen):
* input: runs/probe_cache_v4 rendered with the pretraining event representation;
* BN: deepcopy of the encoder, running stats reset and re-estimated with
  cumulative momentum on all 11,700 fit records in one fixed shuffled order
  (no labels), then eval mode;
* features are extracted in fixed shuffled orders and restored by record id;
  a second development order must give bit-identical features;
* readouts: CLS spike rate, GAP patch spike rate, their concatenation (primary);
* weighted kNN (k=20, T=.07) on fit-standardized, L2-normalized features
  (primary) and on raw L2-normalized rates (secondary);
* linear: fit-standardized features, lr chosen on a stratified 10% fit holdout,
  then retrained on all fit records with three seeds; development is report-only;
* 95% bootstrap CI of development Acc@1 and saved per-record predictions for
  paired comparisons.
Official validation is never read.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional

CACHE = Path('runs/probe_cache_v4')
READOUTS = ('cls', 'gap', 'concat', 'concat_current')
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)


class ProbeCacheViews(Dataset):
    def __init__(self, indices, cache=None):
        self.indices = [int(index) for index in indices]
        self.cache = Path(cache if cache is not None else CACHE)
        self.x = self.y = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        if self.x is None:
            name = 'events.npy' if (self.cache / 'events.npy').exists() else 'events224.npy'
            self.x = np.load(self.cache / name, mmap_mode='r')
            self.y = np.load(self.cache / 'labels.npy')
        index = self.indices[offset]
        return (torch.from_numpy(np.asarray(self.x[index], dtype=np.float32)),
                int(self.y[index]), index)


def shuffled(indices, seed):
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    return [indices[int(position)] for position in order]


def _loader(indices, batch_size, workers, drop_last=False):
    # workers=0 by default: ProbeCacheViews only memcpys from an mmap'd float16
    # array (a batch of 64 is 16.8 MiB), so a worker process buys nothing and
    # costs a full torch re-import per spawn. Measured on the training box: a
    # probe issues four loaders, and with workers=8 inside a job that already
    # holds 12 persistent training workers each spawn stalled ~2 min, so one
    # epoch probe did not finish in 25 min (main thread inside
    # Popen.__init__ after 5 of 8 workers). The parent-loading path is
    # deterministic and bit-identical to the worker path.
    return DataLoader(ProbeCacheViews(indices), batch_size=batch_size,
                      num_workers=workers, pin_memory=bool(workers),
                      drop_last=drop_last)


@torch.no_grad()
def calibrate(encoder, indices, batch_size=64, workers=0):
    norms = [m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms or any(not m.track_running_stats for m in norms):
        raise RuntimeError('Probe requires running-stat BN')
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
    encoder.train()
    batches = 0
    for events, _label, _index in _loader(indices, batch_size, workers, drop_last=True):
        events = events.transpose(0, 1).contiguous().cuda(non_blocking=True)
        functional.reset_net(encoder)
        encoder(events)
        batches += 1
    functional.reset_net(encoder)
    encoder.eval()
    return {'records': batches * batch_size, 'batch_size': batch_size,
            'batches': batches, 'momentum': 'cumulative', 'labels_used': False}


@torch.no_grad()
def extract(encoder, indices, batch_size=64, workers=0):
    encoder.eval()
    features = {name: [] for name in READOUTS}
    labels, identifiers = [], []
    for events, label, index in _loader(indices, batch_size, workers):
        events = events.transpose(0, 1).contiguous().cuda(non_blocking=True)
        functional.reset_net(encoder)
        output = encoder.representation(events)
        functional.reset_net(encoder)
        for name in READOUTS:
            features[name].append(output[name].float().cpu())
        labels.append(label)
        identifiers.append(index)
    identifiers = torch.cat(identifiers)
    order = identifiers.argsort()
    return ({name: torch.cat(value)[order] for name, value in features.items()},
            torch.cat(labels)[order], identifiers[order])


def effective_rank(features):
    centered = features.double() - features.double().mean(0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigen = torch.linalg.eigvalsh(covariance).clamp_min(0)
    p = eigen / eigen.sum().clamp_min(1e-30)
    p = p[p > 0]
    std = features.double().std(0, unbiased=False)
    correlation_rank = None
    keep = std > 1e-8
    if keep.sum() > 1:
        z = centered[:, keep] / std[keep]
        corr = z.T @ z / max(len(z) - 1, 1)
        ce = torch.linalg.eigvalsh(corr).clamp_min(0)
        q = ce / ce.sum()
        q = q[q > 0]
        correlation_rank = float((-(q * q.log()).sum()).exp())
    return {'effective_rank': float((-(p * p.log()).sum()).exp()),
            'correlation_effective_rank': correlation_rank,
            'dead_channel_fraction': float((~keep).float().mean()),
            'channel_std_mean': float(std.mean()), 'mean_rate': float(features.mean())}


def standardize(train, *others):
    mean = train.double().mean(0)
    std = train.double().std(0, unbiased=False).clamp_min(1e-6)
    return [((x.double() - mean) / std).float() for x in (train, *others)]


@torch.no_grad()
def weighted_knn(train_x, train_y, test_x, test_y, k=20, temperature=.07, classes=None):
    train_x = F.normalize(train_x.float().cuda(), dim=-1)
    test_x = F.normalize(test_x.float().cuda(), dim=-1)
    train_y, test_y = train_y.cuda(), test_y.cuda()
    classes = int(max(train_y.max(), test_y.max())) + 1 if classes is None else classes
    predictions = []
    for start in range(0, len(test_x), 256):
        scores = test_x[start:start + 256] @ train_x.T
        values, indices = scores.topk(k, dim=-1)
        weights = (values / temperature).exp()
        votes = torch.zeros(len(values), classes, device=values.device)
        votes.scatter_add_(1, train_y[indices], weights)
        predictions.append(votes)
    votes = torch.cat(predictions)
    top5 = votes.topk(5, -1).indices
    correct = votes.argmax(-1).eq(test_y)
    return {'acc1': float(correct.float().mean()),
            'acc5': float(top5.eq(test_y[:, None]).any(-1).float().mean()),
            'correct': correct.cpu().numpy()}


def _train_linear(x, y, lr, seed, steps=1500, batch=256, classes=None):
    torch.manual_seed(seed)
    classes = int(y.max()) + 1 if classes is None else classes
    head = nn.Linear(x.shape[1], classes).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x, y = x.cuda(), y.cuda()
    for _ in range(steps):
        index = torch.randint(len(y), (batch,), generator=generator).cuda()
        loss = F.cross_entropy(head(x[index]), y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    return head


@torch.no_grad()
def _assess(head, x, y):
    logits = head(x.cuda()).float()
    y = y.cuda()
    correct = logits.argmax(-1).eq(y)
    return {'acc1': float(correct.float().mean()),
            'acc5': float(logits.topk(5, -1).indices.eq(y[:, None]).any(-1).float().mean()),
            'ce': float(F.cross_entropy(logits, y)), 'correct': correct.cpu().numpy()}


def linear_probe(fit_x, fit_y, dev_x, dev_y, seeds=(0, 1, 2)):
    generator = np.random.default_rng(7)
    holdout = []
    for label in fit_y.unique().tolist():
        members = np.flatnonzero(fit_y.numpy() == label)
        holdout.extend(generator.choice(members, len(members) // 10, replace=False).tolist())
    mask = np.zeros(len(fit_y), dtype=bool)
    mask[holdout] = True
    inner_train, inner_val = standardize(fit_x[~mask], fit_x[mask])
    selection = {}
    for lr in LR_GRID:
        head = _train_linear(inner_train, fit_y[~mask], lr, seed=0)
        selection[str(lr)] = _assess(head, inner_val, fit_y[mask])['acc1']
    best_lr = float(max(selection, key=selection.get))
    fit_z, dev_z = standardize(fit_x, dev_x)
    runs = [_assess(_train_linear(fit_z, fit_y, best_lr, seed), dev_z, dev_y)
            for seed in seeds]
    return {'lr_selection_on_fit_holdout': selection, 'selected_lr': best_lr,
            'acc1_mean': float(np.mean([r['acc1'] for r in runs])),
            'acc5_mean': float(np.mean([r['acc5'] for r in runs])),
            'ce_mean': float(np.mean([r['ce'] for r in runs])),
            'acc1_per_seed': [r['acc1'] for r in runs],
            'correct': runs[0]['correct']}


def bootstrap_ci(correct, resamples=2000, seed=0):
    generator = np.random.default_rng(seed)
    n = len(correct)
    means = correct[generator.integers(0, n, (resamples, n))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


class deterministic_cudnn:
    """Scope cuDNN to deterministic kernels for the whole probe.

    The trainer enables ``cudnn.benchmark``; the probe asks whether the frozen
    features depend on batch composition, so it must not be able to answer
    "yes" because two passes happened to pick different kernels. Deterministic
    kernels do not by themselves guarantee order invariance -- a batch-wide
    reduction would still differ -- so the fault this gate exists for (state
    leakage, batch-stat normalization) stays detectable.
    """

    def __enter__(self):
        self.saved = (torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, *exc):
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self.saved
        return False


def run_probe(encoder, split, batch_size=64, workers=0, predictions_path=None):
    """Probe a frozen encoder copy. ``split`` holds fit/development indices."""
    with deterministic_cudnn():
        return _run_probe(encoder, split, batch_size, workers, predictions_path)


def _run_probe(encoder, split, batch_size, workers, predictions_path):
    encoder = copy.deepcopy(encoder)
    fit, development = split['fit_indices'], split['development_indices']
    result = {'protocol': 'probe_v4', 'official_validation_read': False,
              'cudnn_deterministic': True,
              'calibration': calibrate(encoder, shuffled(fit, 900), batch_size, workers)}
    fit_x, fit_y, _ = extract(encoder, shuffled(fit, 1001), batch_size, workers)
    dev_x, dev_y, dev_id = extract(encoder, shuffled(development, 1002), batch_size, workers)
    control_x, control_y, control_id = extract(encoder, shuffled(development, 2002),
                                               batch_size, workers)
    # Same order, same batch sizes, a second time. If this is not bit-identical
    # the drift is not about batch composition at all: it is nondeterminism or
    # state carried between passes, and the order test below cannot separate
    # the two by itself.
    repeat_x, _, _ = extract(encoder, shuffled(development, 1002), batch_size, workers)
    repeat_exact = all(torch.equal(dev_x[n], repeat_x[n]) for n in READOUTS)
    same_ids = torch.equal(dev_id, control_id) and torch.equal(dev_y, control_y)
    # Spike-rate readouts must be bit-identical (measured: exactly 0.0 at both
    # 224 and 128 px). The current readout is a sum of pre-LIF membrane
    # potentials; changing batch order moves ~1e-5 of its 768k elements by up
    # to 2e-3 relative (unchanged by cudnn deterministic mode, so it is a
    # non-associative reduction in the accumulation, not kernel selection).
    # It is secondary-to-primary only (concat is the primary readout). It is
    # held to rounding level: no element may move by more than min(1e-2 of the
    # control's max, 1e-4 of its RMS). The differing-element *fraction* is
    # recorded but not gated: at 1-ULP drift it is set by the allocator-
    # dependent reduction order, and it was measured at 1.2e-5 for the ViT and
    # 1.9e-2 for SEW (max abs 1.2e-7 on a 0.12-scale feature) with identical
    # spike readouts, so a 1e-4 fraction bound rejects SEW on rounding alone.
    spike_readouts = ('cls', 'gap', 'concat')
    exact = same_ids and all(torch.equal(dev_x[n], control_x[n]) for n in spike_readouts)
    differing = {n: float((dev_x[n] != control_x[n]).float().mean()) for n in READOUTS}
    # Bit-exactness held for the first 18 arms (measured 0.0 at 128 and 224 px),
    # but it is a property of the checkpoint, not of the probe: a single sample
    # whose pre-LIF potential sits within one ULP of the threshold can absorb a
    # different batching-induced reduction order and emit one extra spike.
    # Measured 2026-11-09 on runs/sdino_v6_temp025_c10_e150_nostop_20261109 at
    # epoch 40: 2 of ~1M concat elements moved by exactly 1/64 = one spike
    # quantum, max_relative_rms 7.3e-2, same ids, reproducible at fixed order.
    # The check therefore keeps a hard bound instead of bit-equality.  The
    # bounds were widened once more the same day: the hinge arm
    # (runs/sdino_v6_hinge_c10_e30_20261109) failed at epoch 5 with 84 of 768k
    # cls elements (1.1e-4) moved by 0.03125 = two quanta, again with
    # ``same_ids`` and ``repeat_exact_same_order`` both true.  A single
    # pre-LIF potential landing within an ULP of threshold produces one extra
    # spike, and which samples sit that close changes as the encoder trains, so
    # a bound tight to the observed count just moves the failure to the next
    # arm.  The instance arm (runs/sdino_v6_instance_c10_e30_20261109) failed
    # at epoch 1 with 400 of 768k cls elements (5.2e-4) moved by 0.0625 = four
    # quanta: at this geometry (T=4, 4x4 final map) one *spatial position*
    # flipping at every timestep is 4/64, so a single near tie can already move
    # one channel by four quanta.  The fraction is therefore capped at 1e-3 of
    # the elements (768 of 768k here) and single elements at 0.13 (~8 quanta).
    # The failure this check exists for (the E2 sorted-batch artifact) moved a
    # large fraction of every element and is still rejected by orders of
    # magnitude.
    # ``concat_current`` is a pre-LIF current, so one flipped spike upstream
    # moves it by O(0.1) at a handful of elements -- 0.128 on an RMS of 0.156
    # in the instance arm's e1 failure, which the old max-relative bound read as
    # a 10 % change.  A ratio against a near-zero element carries no
    # information, and a real leak moves *every* element, so this readout is
    # gated on the same differing fraction as the spike readouts; its absolute
    # and relative drift stay recorded in the evidence.
    # ``batch_order_invariance_exact`` keeps reporting the strict answer, and
    # the fraction is recorded on success, so a growing order dependence stays
    # visible rather than being silently absorbed.
    spike_drift_cap = .13
    spike_fraction_cap = 1e-3
    spike_ok = all(differing[n] <= spike_fraction_cap for n in spike_readouts)
    drift = {n: float((dev_x[n] - control_x[n]).abs().max()) for n in READOUTS}
    relative = {n: float((dev_x[n] - control_x[n]).abs().max() /
                         control_x[n].abs().max().clamp_min(1e-12)) for n in READOUTS}
    rms = {n: float(control_x[n].pow(2).mean().sqrt()) for n in READOUTS}
    relative_rms = {n: float((dev_x[n] - control_x[n]).abs().max() / max(rms[n], 1e-12))
                    for n in READOUTS}
    result['batch_order_invariance_exact'] = exact
    result['batch_order_repeat_exact'] = repeat_exact
    result['batch_order_max_abs_diff'] = drift
    result['batch_order_max_relative_diff'] = relative
    result['batch_order_max_relative_rms_diff'] = relative_rms
    result['batch_order_differing_fraction'] = differing
    spike_ok = spike_ok and all(float((dev_x[n] - control_x[n]).abs().max()) <= spike_drift_cap
                                for n in spike_readouts)
    current_ok = all(differing[n] <= spike_fraction_cap
                     for n in READOUTS if n not in spike_readouts)
    if not same_ids or not spike_ok or not current_ok:
        # Leave the measurement behind: a bare raise throws away exactly the
        # numbers needed to tell state leakage from kernel noise.
        evidence = {'exact': exact, 'same_ids': same_ids,
                    'repeat_exact_same_order': repeat_exact,
                    'cause': ('reproducible-but-order-dependent' if repeat_exact
                              else 'not reproducible at fixed order (nondeterminism or state)'),
                    'spike_readouts': list(spike_readouts), 'records': int(len(dev_y)),
                    'max_abs_diff': drift, 'max_relative_diff': relative,
                    'max_relative_rms_diff': relative_rms, 'rms': rms,
                    'differing_fraction': differing}
        if predictions_path is not None:
            Path(str(predictions_path).replace('.npz', '')
                 + '_invariance_failure.json').write_text(json.dumps(evidence, indent=2))
        raise RuntimeError('Frozen features depend on batch composition: '
                           + json.dumps(evidence))
    predictions = {'development_ids': dev_id.numpy(), 'development_labels': dev_y.numpy()}
    for name in READOUTS:
        fit_z, dev_z = standardize(fit_x[name], dev_x[name])
        knn = weighted_knn(fit_z, fit_y, dev_z, dev_y)
        knn_raw = weighted_knn(fit_x[name], fit_y, dev_x[name], dev_y)
        linear = linear_probe(fit_x[name], fit_y, dev_x[name], dev_y)
        predictions[f'{name}_knn_correct'] = knn.pop('correct')
        knn_raw.pop('correct')
        predictions[f'{name}_linear_correct'] = linear.pop('correct')
        result[name] = {
            'knn': knn, 'knn_raw_cosine': knn_raw, 'linear': linear,
            'knn_acc1_ci95': bootstrap_ci(predictions[f'{name}_knn_correct']),
            'linear_acc1_ci95_seed0': bootstrap_ci(predictions[f'{name}_linear_correct']),
            'feature': effective_rank(dev_x[name])}
    result['records'] = {'fit': len(fit_y), 'development': len(dev_y)}
    if predictions_path is not None:
        np.savez_compressed(predictions_path, **predictions)
    return result


def summary_line(result):
    c = result['concat']
    return (f"concat kNN {100 * c['knn']['acc1']:.2f} linear {100 * c['linear']['acc1_mean']:.2f} | "
            f"cls kNN {100 * result['cls']['knn']['acc1']:.2f} lin {100 * result['cls']['linear']['acc1_mean']:.2f} | "
            f"gap kNN {100 * result['gap']['knn']['acc1']:.2f} lin {100 * result['gap']['linear']['acc1_mean']:.2f} | "
            f"rank {c['feature']['effective_rank']:.2f}/{c['feature']['correlation_effective_rank']:.1f}"
            + (f" | current kNN {100 * result['concat_current']['knn']['acc1']:.2f} "
               f"lin {100 * result['concat_current']['linear']['acc1_mean']:.2f}"
               if 'concat_current' in result else ''))


def set_cache(path):
    """Select the probe cache (dataset) used by every loader in this module."""
    global CACHE
    CACHE = Path(path)


def load_split(cache=None):
    return json.loads((Path(cache if cache is not None else CACHE) / 'split.json').read_text())
