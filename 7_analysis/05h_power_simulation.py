#%%
"""
05h_power_simulation.py
=======================
How large a creator -> community lead could the H1 test actually detect?

The permutation tests (05b) control FALSE positives: how often noise looks like a
lead. They say nothing about how often a REAL lead would be caught. This script
measures that directly, per game, by planting leads of known size into the real
data and running the exact H1 decision rule on them.

Procedure (per game x target x granularity x lag L x effect size rho):
  1. Donor = the game's real creator series, circularly shifted by a random amount
     with NaN positions kept fixed (shift_values_only). It has the real series'
     autocorrelation and missing-data pattern but no real relationship to Reddit.
  2. Planted series = real community series + beta * donor_{t-L} (donor centred;
     periods with no creator data add nothing -- no video, no influence).
     beta is chosen so the planted lead has Pearson correlation rho between
     donor_{t-L} and the planted series over jointly observed periods.
  3. Run the H1 rule exactly as 05/05b do: normalised xcorr over +-28 days (daily,
     DOW-deseasonalised) or +-4 weeks (weekly means); peak = max |r|; DETECTED if the
     peak clears the circular-shift noise gate (95th pct of the null max|r|) AND sits
     at a creator-leading lag (> 0) with positive r.
  4. Repeat N_REPS times with fresh donor shifts; power = share detected.

The noise-gate threshold is computed once per game x target x granularity from
N_NULL = 1000 circular shifts of the real creator series against the real community
series -- exactly 05b's null for the overall peak |r| -- and applied to every
replicate (rather than re-running 1000 permutations inside each replicate). The
threshold depends on the two series' autocorrelation, which planting a lead of
rho <= 0.3 barely changes; the rho = 0 cells report the false-positive rate as a check.

Outputs:
  tables/05h_power_simulation.csv  -- power per cell (+ exact-lag power, achieved rho)
  tables/05h_mde.csv               -- smallest rho detected with >= 80% power, per lag
  figures/05h_power_curves.png     -- power vs rho, one line per planted lag
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, show_figure,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow,
)
from analysis.shared.lagcorr import normalised_xcorr
from analysis.shared.game_config import get_config

warnings.filterwarnings('ignore')

RHOS        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
LAGS        = {'daily': [1, 3, 7, 14], 'weekly': [1, 2, 3, 4]}
MAX_LAG     = {'daily': 28, 'weekly': 4}
LAG_TOL     = {'daily': 2, 'weekly': 0}     # "exact lag" tolerance
N_REPS      = 200
N_NULL      = 1000
POWER_GOAL  = 0.80
RNG         = np.random.default_rng(20261018)
GAME_LABEL  = get_config()['game_label']


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed (05b's null model)."""
    out = series.copy()
    valid = series.notna()
    vals = series[valid].to_numpy()
    out[valid] = np.roll(vals, shift % max(len(vals), 1))
    return out


def plant(target: pd.Series, donor: pd.Series, lag: int, rho: float) -> pd.Series:
    """target_t + beta * donor_{t-lag}, beta set so corr(donor_{t-lag}, planted_t) ~= rho."""
    if rho == 0:
        return target
    d_lag = donor.shift(lag)
    joint = d_lag.notna() & target.notna()
    sd_d, sd_t = d_lag[joint].std(), target[joint].std()
    beta = rho * sd_t / (sd_d * np.sqrt(1 - rho ** 2))
    add = (d_lag - d_lag[joint].mean()).fillna(0.0)
    return target + beta * add


def achieved_rho(target: pd.Series, donor: pd.Series, lag: int) -> float:
    pair = pd.concat([donor.shift(lag), target], axis=1).dropna()
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 10 else np.nan


def peak(x: pd.Series, y: pd.Series, max_lag: int):
    cc = normalised_xcorr(x, y, max_lag)
    if cc.empty:
        return None, None
    k = int(cc.abs().idxmax())
    return k, float(cc.loc[k])


def null_threshold(creator: pd.Series, target: pd.Series, max_lag: int) -> float:
    """95th pct of max|r| over circular shifts of the creator series (05b's null)."""
    donor, planted = creator, target
    n_valid = int(donor.notna().sum())
    vals = []
    for _ in range(N_NULL):
        s = int(RNG.integers(1, max(n_valid, 2)))
        cc = normalised_xcorr(shift_values_only(donor, s), planted, max_lag)
        if not cc.empty:
            vals.append(float(cc.abs().max()))
    return float(np.percentile(vals, 95)) if vals else np.nan


#%%
ensure_output_dirs()
yt, rd, cm = load_youtube_creators(), load_reddit_submissions(), load_comments_master()
c_agg = aggregate_creator_daily(yt, ['overall__score'])
r_agg = aggregate_reddit_daily(rd, ['overall__score'])
m_agg = aggregate_comments_daily(cm, ['sentiment_score'])

# Same common range and series construction as script 05
start = max(c_agg.index.min(), r_agg.index.min(), m_agg.index.min())
end   = min(c_agg.index.max(), r_agg.index.max(), m_agg.index.max())
dates = pd.date_range(start, end, freq='D')
c_d = c_agg['overall__score_unweighted'].reindex(dates)
series = {
    'daily': {
        'creator':     deseasonalize_dow(c_d),
        'submissions': deseasonalize_dow(r_agg['overall__score_unweighted'].reindex(dates)),
        'comments':    deseasonalize_dow(m_agg['sentiment_score_unweighted'].reindex(dates)),
    },
    'weekly': {
        'creator':     c_d.resample('W-MON').mean(),
        'submissions': r_agg['overall__score_unweighted'].reindex(dates).resample('W-MON').mean(),
        'comments':    m_agg['sentiment_score_unweighted'].reindex(dates).resample('W-MON').mean(),
    },
}

#%%
records = []
for gran in ['daily', 'weekly']:
    creator = series[gran]['creator']
    n_valid = int(creator.notna().sum())
    for tgt in ['submissions', 'comments']:
        target = series[gran][tgt]
        thr = null_threshold(creator, target, MAX_LAG[gran])   # one gate per series pair
        for L in LAGS[gran]:
            for rho in RHOS:
                hits = exact = 0
                rhos_got = []
                for _ in range(N_REPS):
                    donor = shift_values_only(creator, int(RNG.integers(1, n_valid)))
                    planted = plant(target, donor, L, rho)
                    k, r = peak(donor, planted, MAX_LAG[gran])
                    if k is None:
                        continue
                    if rho > 0:
                        rhos_got.append(achieved_rho(planted, donor, L))
                    det = (abs(r) > thr) and (k > 0) and (r > 0)
                    hits += det
                    exact += det and abs(k - L) <= LAG_TOL[gran]
                records.append({
                    'game': GAME_LABEL, 'granularity': gran, 'target': tgt,
                    'planted_lag': L, 'rho': rho,
                    'achieved_rho_mean': round(float(np.nanmean(rhos_got)), 4) if rhos_got else 0.0,
                    'power_h1': round(hits / N_REPS, 3),
                    'power_exact_lag': round(exact / N_REPS, 3),
                    'noise_threshold': round(thr, 4),
                    'n_reps': N_REPS,
                })
            print(f'  {gran:6s} {tgt:11s} lag {L:2d}: ' + '  '.join(
                f"rho={r['rho']:.2f}:{r['power_h1']:.2f}" for r in records[-len(RHOS):]))

power_df = pd.DataFrame(records)
power_df.to_csv(TABLES_DIR / '05h_power_simulation.csv', index=False)

#%%
# Minimum detectable effect: smallest rho with power >= 80% (linear interpolation)
mde_rows = []
for (gran, tgt, L), g in power_df[power_df.rho > 0].groupby(['granularity', 'target', 'planted_lag']):
    g = g.sort_values('rho')
    mde = np.nan
    prev_rho, prev_pow = 0.0, float(power_df[(power_df.granularity == gran) & (power_df.target == tgt)
                                             & (power_df.planted_lag == L) & (power_df.rho == 0)].power_h1.iloc[0])
    for _, row in g.iterrows():
        if row.power_h1 >= POWER_GOAL:
            span = row.power_h1 - prev_pow
            mde = prev_rho + (POWER_GOAL - prev_pow) / span * (row.rho - prev_rho) if span > 0 else row.rho
            break
        prev_rho, prev_pow = row.rho, row.power_h1
    fp = power_df[(power_df.granularity == gran) & (power_df.target == tgt)
                  & (power_df.planted_lag == L) & (power_df.rho == 0)].power_h1.iloc[0]
    mde_rows.append({'game': GAME_LABEL, 'granularity': gran, 'target': tgt, 'planted_lag': L,
                     'mde_rho_80pct': round(float(mde), 3) if not np.isnan(mde) else np.nan,
                     'false_positive_rate_rho0': fp,
                     'power_at_rho_0.10': float(g[g.rho == 0.10].power_h1.iloc[0]),
                     'power_at_rho_0.20': float(g[g.rho == 0.20].power_h1.iloc[0])})
mde_df = pd.DataFrame(mde_rows)
mde_df.to_csv(TABLES_DIR / '05h_mde.csv', index=False)
print('\n-- Minimum detectable effect (rho at 80% power; NaN = not reached by rho=0.30) --')
print(mde_df.to_string(index=False))

#%%
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
for i, gran in enumerate(['daily', 'weekly']):
    for j, tgt in enumerate(['submissions', 'comments']):
        ax = axes[i, j]
        sub = power_df[(power_df.granularity == gran) & (power_df.target == tgt)]
        for L, g in sub.groupby('planted_lag'):
            unit = 'd' if gran == 'daily' else 'wk'
            ax.plot(g.rho, g.power_h1, marker='o', lw=1.8, label=f'planted lead +{L}{unit}')
        ax.axhline(POWER_GOAL, color='grey', ls='--', lw=1)
        ax.axhline(0.05, color='grey', ls=':', lw=1)
        ax.set_title(f'{gran.capitalize()} — creator → {tgt}', fontsize=11)
        ax.set_xlabel('Planted lead size (Pearson ρ)')
        ax.set_ylabel('Share detected by the H1 rule')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
fig.suptitle(f'{GAME_LABEL}: power of the H1 test to detect a planted creator lead\n'
             f'({N_REPS} replicates per point; dashed = 80% power, dotted = 5%)', fontsize=13)
plt.tight_layout()
out = FIGURES_DIR / '05h_power_curves.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
show_figure()
