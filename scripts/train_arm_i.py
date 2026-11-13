"""Run Arm I exactly: the Arm F payload with spherical-DC removal switched OFF.

Arm I differs from Arm F by one CLI flag, so `scripts/train_sdino_v7.py` -- the
Arm I source file, hash-pinned at 51372 B D1FBD0ABD3D9707D -- still defaults to
`center_rate=True` (DC subtraction ON, i.e. Arm F).  Passing `--no-center-rate`
is what makes a run Arm I.

This wrapper appends that flag unless you already chose a centering mode
yourself.  It touches no source file, so the pinned hashes stay valid.

    python scripts/train_arm_i.py --output runs/arm_i_repro --epochs 300 ...

Equivalent to writing `--no-center-rate` on the `train_sdino_v7.py` command
line by hand.  Confirm after launch: the run's `recorded_config` must read
`rate_centering: none`, and `metrics.jsonl` must show `feature_center_norm`
exactly 0.0 on every row.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if '--no-center-rate' not in sys.argv and '--center-rate' not in sys.argv:
    sys.argv.append('--no-center-rate')

runpy.run_module('scripts.train_sdino_v7', run_name='__main__')
