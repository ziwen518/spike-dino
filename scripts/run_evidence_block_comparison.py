"""Finite GPU0 experiment: count-operator reference versus evidence block.

Run on the training server from the repository root. Children run sequentially;
each terminates before the next stage so no trainer retains GPU memory.
No official test examples are accessed by these internal frozen probes.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = root / 'runs/evidence_block_comparison_20261020'
    output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    status = {'started': time.time(), 'state': 'running', 'completed': []}

    def save():
        (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

    def run(name, arguments):
        status['stage'] = name
        save()
        print('START', name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', *arguments], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed'].append(name)
        save()
        print('DONE', name, flush=True)

    common = ['-m', 'spike_dino.pretrain_v3', '--dataset-root', 'dataset/CIFAR10DVS',
              '--manifest-path', 'configs/splits/cifar10dvs_seed42.json',
              '--device', 'cuda:0', '--size', '128', '--patch-size', '16',
              '--local-size', '64', '--global-crop-scale', '0.32', '1.0',
              '--global-time-range', '0.5', '1.0', '--global-view-policy', 'shared_core',
              '--time-steps', '4', '--batch-size', '16', '--local-crops', '0',
              '--depth', '12', '--dim', '384', '--heads', '6',
              '--cls-aggregation', 'spike_pool', '--norm-policy', 'layer_norm',
              '--head-hidden', '2048', '--head-layers', '3',
              '--prototype-readout', 'cosine', '--ibot-reduction', 'sample',
              '--bottleneck', '256', '--prototypes', '65536',
              '--target-policy', 'sinkhorn', '--mask-policy', 'fixed_block',
              '--seed', '42', '--lr', '0.0001', '--workers', '4',
              '--steps', '281', '--checkpoint-every', '50']
    arms = [
        ('count_reference', 'v3_count_reference_seed42_281_20261020',
         ['--block-variant', 'standard', '--attention-variant', 'spikformer_count',
          '--residual-policy', 'layer_scale']),
        ('evidence_block', 'v3_evidence_block_seed42_281_20261020',
         ['--block-variant', 'evidence']),
    ]
    try:
        save()
        memory = subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits'], text=True).strip()
        if int(memory) > 2000:
            raise RuntimeError('GPU0 already has an unexpected allocation: ' + memory)
        old = 'runs/v3_sinkhorn_sharedcore_layerscale_epoch1_b16_20261015/eval_step_000281.pt'
        if not Path(old).is_file():
            raise FileNotFoundError(old)
        for name, folder, options in arms:
            run(name + '_train', [*common, '--output', 'runs/' + folder, *options])
            for source in ('cls', 'patch_mean'):
                run(name + '_' + source, [
                    '-m', 'spike_dino.evaluate_v3', '--dataset-root', 'dataset/CIFAR10DVS',
                    '--checkpoint', 'runs/' + folder + '/eval_step_000281.pt',
                    '--output', str(output / (name + '_' + source + '.json')),
                    '--device', 'cuda:0', '--workers', '4', '--modes', 'eval',
                    '--train-limit', '512', '--validation-per-class', '100',
                    '--cohort-seed', '42', '--probe-seed', '42', '--feature-source', source])
        for source in ('cls', 'patch_mean'):
            run('old_baseline_' + source, [
                '-m', 'spike_dino.evaluate_v3', '--dataset-root', 'dataset/CIFAR10DVS',
                '--checkpoint', old, '--output', str(output / ('old_baseline_' + source + '.json')),
                '--device', 'cuda:0', '--workers', '4', '--modes', 'eval',
                '--train-limit', '512', '--validation-per-class', '100',
                '--cohort-seed', '42', '--probe-seed', '42', '--feature-source', source])
        status['state'] = 'completed'
    except BaseException as error:
        status['state'] = 'failed'
        status['error'] = repr(error)
        raise
    finally:
        status['finished'] = time.time()
        status['gpu_after'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'],
            text=True).strip()
        save()


if __name__ == '__main__':
    main()
