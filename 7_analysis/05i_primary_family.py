#%%
"""
05i_primary_family.py
=====================
Family-wise view of the primary H1 test, plus a season-phase control.

1. Primary cross-correlation family (20 cells = 5 games x 2 community targets x
   2 granularities). This script handles ONE game (the active ANALYSIS_GAME) and
   writes its 4 cells; the cross-game Holm adjustment is computed by
   `combine_primary_family()` once all five games have run (it reads the five
   per-game tables).
   Per cell, two statistics on the unweighted creator vs community overall
   sentiment series (daily deseasonalized, +/-28 d; weekly means, +/-4 wk):
     - gate:        max |r| across the whole lag window (the statistic every
                    lead/lag test in the paper is gated on)
     - directional: max r across creator-leading lags 1..L (the H1 prediction:
                    a positive correlation with creators first)
   Null: N_PERM circular shifts of the creator series with its gaps kept in place
   (the 05b null model). p = (b + 1) / (N + 1), b = null draws >= observed.

2. Pre-specified daily Granger family (10 cells = 5 games x 2 targets), read from
   05_granger_results.csv. Each block's p is its minimum over lag orders,
   Bonferroni-corrected for the number of orders estimated.

3. Season-phase control for the weekly H1 test. Two series that both follow a
   ~10-12-week season cycle correlate at a fixed lag through cycle phase alone,
   and dropping release weeks does not remove that. Each weekly series is
   residualized on weeks-since-last-release dummies (0..SEASON_PHASE_CAP-1, the
   rest pooled) before the gated xcorr, at +/-4 and +/-12 weeks.

Outputs (per game):
  tables/05i_primary_family.csv     -- 4 xcorr cells (+ adjusted p once combined)
  tables/05i_granger_family.csv     -- 2 Granger cells
  tables/05i_season_phase.csv       -- season-phase-residualized weekly H1 cells
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.game_config import get_config
from analysis.shared.lagcorr import normalised_xcorr
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    load_releases, ensure_output_dirs, TABLES_DIR, GAME_OF_FOCUS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow,
)

N_PERM           = 10_000
RANDOM_SEED      = 42
MAX_LAG_DAYS     = 28
MAX_LAG_WEEKS    = 4
WIDE_LAG_WEEKS   = 12
SEASON_PHASE_CAP = 16     # weeks-since-release dummies 0..15; later weeks pooled
FIELD            = 'overall__score'
COMMENT_FIELD    = 'sentiment_score'
GAMES            = ['destiny2', 'warframe', 'csgo', 'dota2', 'deadbydaylight']

#%%
def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed (05b null)."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def _stats(cc: pd.Series) -> tuple[float, float]:
    """(max |r| over the window, max r over creator-leading lags)."""
    lead = cc[cc.index > 0]
    return float(cc.abs().max()), (float(lead.max()) if not lead.empty else np.nan)


def permutation_cell(x: pd.Series, y: pd.Series, max_lag: int, rng) -> dict | None:
    cc = normalised_xcorr(x, y, max_lag)
    if cc.empty:
        return None
    peak_lag = int(cc.abs().idxmax())
    obs_gate, obs_dir = _stats(cc)
    n = int(x.notna().sum())
    # A circular shift of n observed values has only n-1 distinct non-zero
    # offsets. When that is fewer than N_PERM (every weekly series), enumerate
    # them all for an exact p; otherwise draw N_PERM at random.
    shifts = range(1, n) if n - 1 <= N_PERM else rng.integers(1, max(n, 2), N_PERM)
    b_gate = b_dir = n_ok = 0
    for sh in shifts:
        cc_n = normalised_xcorr(shift_values_only(x, int(sh)), y, max_lag)
        if cc_n.empty:
            continue
        g, d = _stats(cc_n)
        n_ok   += 1
        b_gate += g >= obs_gate
        b_dir  += d >= obs_dir
    return {
        'peak_lag':        peak_lag,
        'peak_r':          round(float(cc.loc[peak_lag]), 4),
        'gate_stat':       round(obs_gate, 4),
        'p_gate':          (b_gate + 1) / (n_ok + 1),
        'lead_lag':        int(cc[cc.index > 0].idxmax()),
        'dir_stat':        round(obs_dir, 4),
        'p_directional':   (b_dir + 1) / (n_ok + 1),
        'n_perm':          n_ok,
        'exact_enumeration': n - 1 <= N_PERM,
        'n_joint':         int((x.notna() & y.notna()).sum()),
    }


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (index-aligned)."""
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def season_phase_residualize(s: pd.Series, release_starts: np.ndarray) -> pd.Series:
    """Residualize a weekly series on weeks-since-last-release dummies."""
    wk = s.index.values.astype('datetime64[D]').astype(int)
    since = np.full(len(wk), SEASON_PHASE_CAP, dtype=int)
    for i, w in enumerate(wk):
        prior = release_starts[release_starts <= w]
        if prior.size:
            since[i] = min((w - prior.max()) // 7, SEASON_PHASE_CAP)
    phase = pd.Series(since, index=s.index)
    means = s.groupby(phase).transform('mean')
    return s - means


#%%
def run_game() -> None:
    ensure_output_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    yt, rd, cm = load_youtube_creators(), load_reddit_submissions(), load_comments_master()
    c_agg = aggregate_creator_daily(yt, [FIELD])[f'{FIELD}_unweighted']
    targets = {
        'submissions': aggregate_reddit_daily(rd, [FIELD])[f'{FIELD}_unweighted'],
        'comments':    aggregate_comments_daily(cm, [COMMENT_FIELD])[f'{COMMENT_FIELD}_unweighted'],
    }
    rel = pd.to_datetime(load_releases()['start_date']).dropna().values.astype('datetime64[D]').astype(int)

    # Date ranges follow the headline tests: submissions use the creator x
    # submissions overlap (05b), comments the creator x submissions x comments
    # overlap (05's comment panels).
    s_agg = targets['submissions']
    three_way = (max(c_agg.index.min(), s_agg.index.min(), targets['comments'].index.min()),
                 min(c_agg.index.max(), s_agg.index.max(), targets['comments'].index.max()))
    ranges = {'submissions': (max(c_agg.index.min(), s_agg.index.min()),
                              min(c_agg.index.max(), s_agg.index.max())),
              'comments': three_way}

    cells, phase_cells = [], []
    for target, t_agg in targets.items():
        start, end = ranges[target]
        days  = pd.date_range(start, end, freq='D')
        c_d, t_d = deseasonalize_dow(c_agg.reindex(days)), deseasonalize_dow(t_agg.reindex(days))
        # Weekly bins as 05/05b build them: resample labels, which keep the final
        # partial week.
        c_w = c_agg.reindex(days).resample('W-MON').mean()
        t_w = t_agg.reindex(days).resample('W-MON').mean()

        for gran, x, y, L in [('daily', c_d, t_d, MAX_LAG_DAYS), ('weekly', c_w, t_w, MAX_LAG_WEEKS)]:
            res = permutation_cell(x, y, L, rng)
            if res:
                cells.append({'game': GAME_OF_FOCUS, 'target': target, 'granularity': gran, **res})
                print(f'{target:11s} {gran:6s} peak {res["peak_lag"]:+d} r={res["peak_r"]:+.3f} '
                      f'p_gate={res["p_gate"]:.4f} | lead max r={res["dir_stat"]:+.3f} '
                      f'@{res["lead_lag"]:+d} p_dir={res["p_directional"]:.4f}')

        x_ph, y_ph = season_phase_residualize(c_w, rel), season_phase_residualize(t_w, rel)
        for L in (MAX_LAG_WEEKS, WIDE_LAG_WEEKS):
            res = permutation_cell(x_ph, y_ph, L, rng)
            if res:
                phase_cells.append({'game': GAME_OF_FOCUS, 'target': target,
                                    'treatment': 'season_phase_residualized', 'max_lag_weeks': L, **res})
                print(f'  season-phase {target:11s} +/-{L:2d}wk peak {res["peak_lag"]:+d} '
                      f'r={res["peak_r"]:+.3f} p_gate={res["p_gate"]:.4f} p_dir={res["p_directional"]:.4f}')

    pd.DataFrame(cells).to_csv(TABLES_DIR / '05i_primary_family.csv', index=False)
    pd.DataFrame(phase_cells).to_csv(TABLES_DIR / '05i_season_phase.csv', index=False)

    gr = pd.read_csv(TABLES_DIR / '05_granger_results.csv')
    g_rows = []
    for target in ('submissions', 'comments'):
        g = gr[gr['target'] == target]
        if g.empty:
            continue
        best = g.loc[g['p_value'].idxmin()]
        g_rows.append({'game': GAME_OF_FOCUS, 'target': target, 'n_orders': len(g),
                       'min_p': float(best['p_value']), 'order_at_min': int(best['lag']),
                       'p_block': min(1.0, float(best['p_value']) * len(g)),
                       'sign_at_min': best.get('sign', '')})
    pd.DataFrame(g_rows).to_csv(TABLES_DIR / '05i_granger_family.csv', index=False)
    print(f'Saved 05i tables to {TABLES_DIR}')


def combine_primary_family(project_root: Path) -> pd.DataFrame:
    """Holm-adjust the 20 xcorr cells and 10 Granger cells across games and within game."""
    def collect(name):
        frames = []
        for slug in GAMES:
            f = project_root / 'analysis' / slug / 'outputs' / 'tables' / name
            if f.exists():
                frames.append(pd.read_csv(f).assign(slug=slug))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    x = collect('05i_primary_family.csv')
    for col in ('p_gate', 'p_directional'):
        x[f'{col}_holm_all'] = holm(x[col])
        x[f'{col}_holm_game'] = x.groupby('slug')[col].transform(holm)
    g = collect('05i_granger_family.csv')
    if not g.empty:
        g['p_block_holm_all'] = holm(g['p_block'])
        g['p_block_holm_game'] = g.groupby('slug')['p_block'].transform(holm)
    out = project_root / 'analysis' / 'cross_game'
    out.mkdir(exist_ok=True)
    x.to_csv(out / '05i_primary_family_all.csv', index=False)
    g.to_csv(out / '05i_granger_family_all.csv', index=False)
    print(f'Combined {len(x)} xcorr cells and {len(g)} Granger cells -> {out}')
    return x


#%%
if __name__ == '__main__':
    if '--combine' in sys.argv:
        combine_primary_family(Path(__file__).resolve().parents[1])
    else:
        run_game()
