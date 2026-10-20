"""Bounded synthetic smoke runs, or explicitly supplied pre-binned NPZ views."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .core import Experiment, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dino', 'spike_dino', 'spikeclr', 'temporal_ntxent'], default='dino')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--views-npz', type=Path, help='view0/view1 arrays [N,T,2,H,W]; training split only')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--time-steps', type=int, default=4)
    parser.add_argument('--width', type=float, default=.125)
    parser.add_argument('--size', type=int, default=48)
    parser.add_argument('--local-crops', type=int, default=0)
    parser.add_argument('--output-target-rate', type=float, default=None)
    parser.add_argument('--input-coding', choices=['occupancy', 'bernoulli'], default='occupancy')
    parser.add_argument('--teacher-input-coding', choices=['occupancy','bernoulli'])
    parser.add_argument('--population', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.time_steps < 1 or args.width < 1/64 or args.size < 16:
        parser.error('Require positive steps/time, batch >=2, width >=1/64, size >=16')
    if args.local_crops < 0 or (args.local_crops and (args.method != 'dino' or not args.synthetic)):
        parser.error('Local crops currently supported only by synthetic DINO smoke tests')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty; use a new run directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = Experiment(args.method, width=args.width, hidden=256,
                       bottleneck=64, prototypes=128,
                       output_target_rate=args.output_target_rate,
                       input_coding=args.input_coding,
                       population=args.population,
                       teacher_input_coding=args.teacher_input_coding).to(args.device).train()
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-3, weight_decay=1e-4)
    arrays = None
    if args.views_npz:
        with np.load(args.views_npz, allow_pickle=False) as archive:
            arrays = [torch.tensor(archive[key], dtype=torch.float32) for key in ['view0', 'view1']]
        if any(x.ndim != 5 or x.shape[2] != 2 for x in arrays) or arrays[0].shape[:2] != arrays[1].shape[:2]:
            parser.error('NPZ views need matching N,T and shape [N,T,2,H,W]')
        if len(arrays[0]) < args.batch_size or any(not torch.isfinite(x).all() for x in arrays):
            parser.error('Insufficient samples or nonfinite input')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'purpose': 'engineering_smoke_not_benchmark', 'hidden':256,
                   'bottleneck':64, 'prototypes':128, 'lr':1e-3, 'weight_decay':1e-4,
                   'torch':str(torch.__version__), 'teacher_bn':'batch_stats_train_copy_student_buffers_after_step'})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for step in range(args.steps):
        if arrays is None:
            base = torch.rand(args.time_steps, args.batch_size, 2, args.size, args.size)
            base *= (torch.rand_like(base) < .2)
            views = [base.clone(), base.roll(1, -1)]
            for _ in range(args.local_crops):
                views.append(base[..., :args.size//2, :args.size//2].clone())
        else:
            indices = torch.randperm(len(arrays[0]))[:args.batch_size]
            views = [x[indices].transpose(0, 1).contiguous() for x in arrays]
        views = [x.to(args.device) for x in views]
        optimizer.zero_grad(set_to_none=True)
        loss, features, teacher_logits = model(views)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3., error_if_nonfinite=True)
        metrics = {'step':step, 'loss':loss.item(), 'gradient_norm':float(grad_norm),
                   **diagnostics(features, teacher_logits)}
        if args.method == 'spike_dino':
            metrics.update(model.student.last_rates)
            metrics['prototype_threshold'] = model.student.head.prototype_threshold.item()
        if teacher_logits:
            p = torch.cat([model.criterion.teacher_distribution(z) for z in teacher_logits])
            entropy = lambda q: -(q*q.clamp_min(1e-12).log()).sum(-1).mean().item()
            metrics.update(teacher_sample_entropy=entropy(p), teacher_batch_entropy=entropy(p.mean(0)),
                           prototype_usage=p.argmax(-1).unique().numel()/p.shape[-1])
        optimizer.step()
        model.after_optimizer_step(teacher_logits)
        with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics), flush=True)
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'config':config,
                'step':args.steps, 'torch_rng':torch.get_rng_state()}, args.output/'checkpoint.pt')


if __name__ == '__main__':
    main()
