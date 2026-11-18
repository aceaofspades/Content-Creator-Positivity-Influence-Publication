#%%
"""
00_run_all.py
=============
Discovers and runs all analysis scripts in this directory in filename order.

New scripts are picked up automatically — no changes needed here. The numeric
prefix convention (01_, 02_, 05b_, 06c_, etc.) sorts lexicographically into
the correct dependency order.

Run from the project root or from within 7_analysis/ — the script resolves
the project root automatically and passes it as cwd to each subprocess so
that `from analysis.shared...` imports work regardless of where you launch from.

A failed script is logged and skipped; the rest continue running.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent          # 7_analysis/
PROJECT_ROOT = HERE.parent

sys.path.insert(0, str(PROJECT_ROOT))
from analysis.shared.game_config import get_config as _get_config
GAME_LABEL = _get_config()['game_label']

# ── Discover scripts (sorted by name = dependency order via numeric prefix) ────
SCRIPTS = sorted(
    p for p in HERE.glob('*.py')
    if p.name != Path(__file__).name          # exclude this runner
)

# ── Child environment ─────────────────────────────────────────────────────────
# The analysis scripts print box-drawing separators (U+2500). On Windows a
# redirected/piped stdout defaults to cp1252, which cannot encode them, so a
# script can die on a print() *after* its figures are written but *before* its
# tables are — 03 lost 03_hierarchical_regression.csv exactly this way. Forcing
# UTF-8 on the children makes redirected runs behave like the interactive
# console. Agg keeps plt.show() from blocking on a headless/batch run.
CHILD_ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
CHILD_ENV.setdefault('MPLBACKEND', 'Agg')

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_all():
    total  = len(SCRIPTS)
    passed = []
    failed = []

    print(f'Game: {GAME_LABEL}')
    print(f'Found {total} scripts. Running from {PROJECT_ROOT}\n')
    wall_start = time.time()

    for path in SCRIPTS:
        t0 = time.time()
        print(f'  RUN   {path.name} ...', end='', flush=True)
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=PROJECT_ROOT,
            env=CHILD_ENV,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f'  OK  ({elapsed:.0f}s)')
            passed.append(path.name)
        else:
            print(f'  FAILED  ({elapsed:.0f}s)')
            failed.append(path.name)

    total_elapsed = time.time() - wall_start
    print(f'\n{"="*60}')
    print(f'Done in {total_elapsed:.0f}s  |  {len(passed)} passed, {len(failed)} failed')
    if failed:
        print('\nFailed scripts:')
        for name in failed:
            print(f'  - {name}')


if __name__ == '__main__':
    run_all()
