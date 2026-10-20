"""Merge only checkpoint-consistent metrics after a detached run resumes."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(
        encoding='utf-8').splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--resumed', type=Path, required=True)
    p.add_argument('--resume-step', type=int, required=True)
    p.add_argument('--expected-final', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    first = read_rows(args.initial)
    second = read_rows(args.resumed)
    if len(first) < args.resume_step or [row['step'] for row in first[:
            args.resume_step]] != list(range(1, args.resume_step + 1)):
        raise ValueError('Initial log is not contiguous through the safe checkpoint')
    if [row['step'] for row in second] != list(
            range(args.resume_step + 1, args.expected_final + 1)):
        raise ValueError('Resumed log is not a contiguous continuation')
    merged = first[:args.resume_step] + second
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in merged),
                           encoding='utf-8')
    print(json.dumps({'rows': len(merged),
                      'discarded_uncheckpointed_rows': len(first) -
                      args.resume_step,
                      'resume_step': args.resume_step,
                      'final_step': merged[-1]['step']}))


if __name__ == '__main__':
    main()
