#%%
"""
14_xcorr_ci.py
---------------
Confidence intervals on the peak cross-correlations from scripts 04, 05, 06.

Null findings need to demonstrate power: "p > 0.05" does not tell a reviewer
how large an effect the data can rule out. This script computes 95% CIs for
each key xcorr finding using circular block bootstrap, then reports
max_effect_ruled_out_95pct — the upper CI bound, which is the largest effect
consistent with the data at 95% confidence.

Method: circular block bootstrap (1 000 iterations). Block length = max(7,
sqrt(n_eff)) to capture weekly autocorrelation structure. For each test case the
CI is computed at TWO lags: a PRE-SPECIFIED lag (fixed before looking at the
data — the unbiased headline) and the SELECTED peak lag (max|r| over the scanned
window, whose |r| is selection-inflated). Reporting both makes the inflation
visible instead of quoting only the cherry-picked peak.

Test cases (pre-specified lag in brackets):
  Script 05 — creator overall__score → Reddit overall__score        [lag 0]
    daily ±28d (unweighted) and weekly ±4w (unweighted)
  Script 06 — authority-stratified weekly xcorr vs. Reddit submissions [lag 0]
    High / Mid / Low authority tiers (terciles cut over CREATORS, per script 06)
  Script 04 — creator engagement fields, weekly ±4w vs. player counts  [lag +1]
    still_playing / burned_out / would_recommend

Interpretation of the upper bound: at 95% confidence the data are consistent
with effects no larger than it — i.e. we rule out anything above it. It is not
a claim of certainty about a point value.

Outputs:
  figures/14_xcorr_ci.png      — forest plot: pre-specified (solid) + peak (hollow) CI per test
  tables/14_xcorr_ci.csv       — prespec_* and peak_* CIs + max_effect_ruled_out per test
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.lagcorr import aligned_pair
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_player_counts,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, CREATOR_ENGAGEMENT_COLS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, deseasonalize_dow,
)

N_BOOT       = 1000
RANDOM_SEED  = 42
MAX_LAG_DAYS = 28
MAX_LAG_WEEKS = 4

# One generator for the whole run. Seeding *inside* the bootstrap function
# restarted the same stream on every call, so cases with equal n drew identical
# resamples and cases with different n shared a common origin -- the per-case
# CIs stayed individually valid but were correlated across cases. Creating the
# generator once here and threading it through means each case consumes a fresh,
# non-overlapping slice of one reproducible stream -> cases are independent.
RNG = np.random.default_rng(RANDOM_SEED)

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
pc = load_player_counts(source='total').set_index('date').sort_index()['players']

#%%
# ── Aggregate and align ───────────────────────────────────────────────────────
c_agg = aggregate_creator_daily(yt, ['overall__score'] + CREATOR_ENGAGEMENT_COLS)
r_agg = aggregate_reddit_daily(rd, ['overall__score'])

start = max(c_agg.index.min(), r_agg.index.min())
end   = min(c_agg.index.max(), r_agg.index.max())
dates = pd.date_range(start, end, freq='D')
c_agg = c_agg.reindex(dates)
r_agg = r_agg.reindex(dates)

# Daily deseasonalised overall for script 05 cases
c_daily_uw  = deseasonalize_dow(c_agg['overall__score_unweighted'])
r_daily_uw  = deseasonalize_dow(r_agg['overall__score_unweighted'])

# Weekly overall for script 05 / 06 cases
c_weekly_uw = c_agg['overall__score_unweighted'].resample('W-MON').mean()
r_weekly_uw = r_agg['overall__score_unweighted'].resample('W-MON').mean()

# Weekly player counts for script 04 cases
pc_weekly = pc.resample('W-MON').mean()

#%%
def get_aligned_pair(x: pd.Series, y: pd.Series, lag: int):
    """
    Return (xa, ya) = (x_t, y_{t+lag}) over periods where both are observed.
    Positive lag = x leads y, lags in calendar periods.

    Delegates to analysis/shared/lagcorr.aligned_pair. The old version dropped NaN
    rows and then offset by position (gap-collapsing) AND had the sign reversed:
    its "+lag" paired x_{t+lag} with y_t, i.e. y leading. Before 2026-10-18 every
    non-zero-lag CI in this table (the +1-week engagement cases and the
    selected-peak cases) was therefore computed at the mirrored lag.
    """
    xa, ya = aligned_pair(x, y, lag)
    if len(xa) < abs(lag) + 10:
        return np.array([]), np.array([])
    return xa, ya


def block_bootstrap_ci(xa: np.ndarray, ya: np.ndarray,
                        rng: np.random.Generator,
                        n_boot: int = N_BOOT,
                        ci_level: float = 0.95) -> dict:
    """
    Circular block bootstrap on Pearson correlation. Block length is
    max(7, sqrt(n)) to preserve at least weekly autocorrelation structure.
    `rng` is supplied by the caller (a single module-level generator) so that
    successive test cases draw from one continuous, non-repeating stream.
    Returns observed_r, ci_lower, ci_upper, n_obs, block_len.
    """
    n = len(xa)
    if n < 10:
        return {}
    block_len = max(7, int(np.sqrt(n)))
    observed_r = float(np.corrcoef(xa, ya)[0, 1])

    boot_r = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start_i = rng.integers(0, n)
            block = np.arange(start_i, start_i + block_len) % n
            indices.extend(block.tolist())
        idx = np.array(indices[:n])
        xa_b, ya_b = xa[idx], ya[idx]
        std_x, std_y = xa_b.std(), ya_b.std()
        if std_x == 0 or std_y == 0:
            continue
        boot_r.append(float(np.corrcoef(xa_b, ya_b)[0, 1]))

    boot_r = np.array(boot_r)
    alpha  = 1 - ci_level
    ci_lo  = float(np.percentile(boot_r, 100 * alpha / 2))
    ci_hi  = float(np.percentile(boot_r, 100 * (1 - alpha / 2)))

    return {
        'observed_r':              round(observed_r, 4),
        'ci_lower':                round(ci_lo, 4),
        'ci_upper':                round(ci_hi, 4),
        'n_obs':                   n,
        'block_len':               block_len,
        'max_effect_ruled_out_95pct': round(ci_hi, 4),
    }


#%%
# ── Define test cases ─────────────────────────────────────────────────────────
# Each case: (label, source_series, target_series, peak_lag, group)
# peak_lag is the observed peak lag from the relevant script's output.
# If a CSV exists we read from it; otherwise we use the known values from
# the analysis already completed.

def _load_peak_lag(csv_path: Path, key_col: str, key_val, lag_col: str):
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        row = df[df[key_col] == key_val]
        if not row.empty:
            return int(row[lag_col].iloc[0])
    return None

# Script 05 peaks (from 05_field_xcorr_summary.csv for daily; known from run)
# Note: use an explicit None check, not `or -1` -- a legitimate peak lag of 0
# is falsy and would be silently overwritten by the fallback.
_dp          = _load_peak_lag(TABLES_DIR / '05_field_xcorr_summary.csv',
                               'field', 'overall__score', 'peak_lag')
daily_peak   = _dp if _dp is not None else -1
weekly_peak  = 0   # known from script 05 weekly run

# Script 06 authority peaks (from 06_tier_xcorr_summary.csv)
tier_peaks = {}
tier_csv = TABLES_DIR / '06_tier_xcorr_summary.csv'
if tier_csv.exists():
    t = pd.read_csv(tier_csv)
    t = t[(t['granularity'] == 'weekly') & (t['target'] == 'submissions') &
          (t['split'] == 'authority')]
    for _, row in t.iterrows():
        tier_peaks[row['tier']] = int(row['peak_lag'])

# Script 04 peaks (engagement fields at lag +1 from 04_engagement_xcorr_extended.csv)
eng_csv = TABLES_DIR / '04_engagement_xcorr_extended.csv'
eng_peaks = {}
if eng_csv.exists():
    e = pd.read_csv(eng_csv)
    e = e[(e['granularity'] == 'weekly') & (e['weighting'] == 'unweighted')]
    for _, row in e.iterrows():
        eng_peaks[row['field']] = int(row['peak_lag'])

print(f'Loaded peak lags — daily: {daily_peak}, weekly: {weekly_peak}')
print(f'Authority tiers: {tier_peaks}')
print(f'Engagement:      {eng_peaks}')

#%%
# ── Script 06: authority tercile data ─────────────────────────────────────────
# Reproduce tier splits to get per-tier series (mirrors CORRECTED script 06).
# Authority = per-CREATOR total view count, terciles cut over CREATORS (one
# value per creator), then mapped back onto each creator's videos. The earlier
# version here cut terciles over video ROWS (groupby.transform('sum') + quantile),
# which weighted the boundaries by output volume and conflated authority with
# posting frequency -- the exact bug fixed in scripts 06/06c. That contaminated
# construction put different creators in each tier than script 06, so the CIs
# were computed on tiers whose peak lags they then read from 06's CSV.
creator_total_views   = yt.groupby('creator')['view_count'].sum().fillna(0)
creator_authority_tier = pd.qcut(creator_total_views, 3, labels=['Low', 'Mid', 'High'])
yt_sorted = yt.copy()
yt_sorted['authority_tier'] = yt_sorted['creator'].map(creator_authority_tier).astype('object')

tier_weekly = {}
for tier in ['High', 'Mid', 'Low']:
    sub = yt_sorted[yt_sorted['authority_tier'] == tier]
    c_agg_t = aggregate_creator_daily(sub, ['overall__score'])
    c_w_t = c_agg_t['overall__score_unweighted'].resample('W-MON').mean()
    tier_weekly[tier] = c_w_t

#%%
# ── Run bootstrap for all test cases ─────────────────────────────────────────
# Each case reports the CI at TWO lags:
#   • prespec_lag — a lag fixed BEFORE looking at the data. This is the unbiased
#     headline bound. lag 0 (contemporaneous co-movement) for the 05/06 tests;
#     lag +1 (the claimed one-step creator lead) for the 04 engagement tests.
#   • peak_lag — the lag selected as max|r| over the scanned window. Its |r| is
#     the maximum of many correlated estimates, so it is biased UPWARD
#     (selection / winner's-curse); the CI around it does not undo that. Kept
#     alongside the pre-specified CI so the inflation gap is visible.
records = []

def _ci_at_lag(x: pd.Series, y: pd.Series, lag: int) -> dict:
    """Block-bootstrap CI at a single lag; empty dict if too few aligned pairs."""
    xa, ya = get_aligned_pair(x, y, lag)
    if len(xa) < 10:
        return {}
    return block_bootstrap_ci(xa, ya, RNG)

def _run_case(label: str, group: str, x: pd.Series, y: pd.Series,
              peak_lag: int, prespec_lag: int):
    peak = _ci_at_lag(x, y, peak_lag)
    pre  = _ci_at_lag(x, y, prespec_lag)
    if not peak and not pre:
        print(f'  Skipping {label}: insufficient aligned pairs')
        return
    rec = {'label': label, 'group': group}
    if pre:
        rec.update({
            'prespec_lag':                        prespec_lag,
            'prespec_r':                          pre['observed_r'],
            'prespec_ci_lower':                   pre['ci_lower'],
            'prespec_ci_upper':                   pre['ci_upper'],
            'prespec_max_effect_ruled_out_95pct': pre['ci_upper'],
        })
    if peak:
        rec.update({
            'peak_lag':                        peak_lag,
            'peak_r':                          peak['observed_r'],
            'peak_ci_lower':                   peak['ci_lower'],
            'peak_ci_upper':                   peak['ci_upper'],
            'peak_max_effect_ruled_out_95pct': peak['ci_upper'],
        })
    anchor = pre or peak
    rec.update({'n_obs': anchor['n_obs'], 'block_len': anchor['block_len']})
    records.append(rec)
    _fmt = lambda d: (f'{d["observed_r"]:+.4f} [{d["ci_lower"]:+.4f}, {d["ci_upper"]:+.4f}]'
                      if d else 'n/a')
    print(f'  {label:<52s}\n'
          f'      pre (lag {prespec_lag:+d}): r={_fmt(pre)}\n'
          f'      peak(lag {peak_lag:+d}): r={_fmt(peak)}')

print('\n── Script 05: overall sentiment xcorr ──────────────────────────────')
_run_case('05 overall (daily, unweighted, deseasonalised)',
          'Script 05 — primary xcorr', c_daily_uw, r_daily_uw,
          peak_lag=daily_peak, prespec_lag=0)

_run_case('05 overall (weekly, unweighted)',
          'Script 05 — primary xcorr', c_weekly_uw, r_weekly_uw,
          peak_lag=weekly_peak, prespec_lag=0)

print('\n── Script 06: authority tier weekly xcorr ───────────────────────────')
for tier in ['High', 'Mid', 'Low']:
    lag = tier_peaks.get(tier, 0)
    _run_case(f'06 authority {tier} (weekly vs submissions)',
              'Script 06 — authority tiers',
              tier_weekly.get(tier, pd.Series(dtype=float)),
              r_weekly_uw, peak_lag=lag, prespec_lag=0)

print('\n── Script 04: engagement fields, weekly vs player counts ────────────')
for field in CREATOR_ENGAGEMENT_COLS:
    col = f'{field}_unweighted'
    if col not in c_agg.columns:
        continue
    c_w_eng = c_agg[col].resample('W-MON').mean()
    lag = eng_peaks.get(field, 1)
    short_name = field.replace('creator__are_they_', '').replace('creator__would_they_', '')
    _run_case(f'04 engagement: {short_name} (weekly vs player counts)',
              'Script 04 — engagement fields',
              c_w_eng, pc_weekly, peak_lag=lag, prespec_lag=1)

#%%
# ── Table ─────────────────────────────────────────────────────────────────────
ci_df = pd.DataFrame(records)
col_order = ['label', 'group',
             'prespec_lag', 'prespec_r', 'prespec_ci_lower', 'prespec_ci_upper',
             'prespec_max_effect_ruled_out_95pct',
             'peak_lag', 'peak_r', 'peak_ci_lower', 'peak_ci_upper',
             'peak_max_effect_ruled_out_95pct',
             'n_obs', 'block_len']
ci_df = ci_df[[c for c in col_order if c in ci_df.columns]]
out_tbl = TABLES_DIR / '14_xcorr_ci.csv'
ci_df.to_csv(out_tbl, index=False)
print(f'\nSaved: {out_tbl}')

#%%
# ── Forest plot ───────────────────────────────────────────────────────────────
group_colors = {
    'Script 05 — primary xcorr':    '#3498db',
    'Script 06 — authority tiers':  '#e67e22',
    'Script 04 — engagement fields':'#2ecc71',
}

n_cases = len(ci_df)
fig, ax = plt.subplots(figsize=(11, max(5, n_cases * 0.7)))

y_pos = np.arange(n_cases)

for i, (_, row) in enumerate(ci_df.iterrows()):
    color = group_colors.get(row['group'], '#95a5a6')
    # Pre-specified-lag CI (unbiased headline) — solid, filled, offset up
    if pd.notna(row.get('prespec_r')):
        ax.plot([row['prespec_ci_lower'], row['prespec_ci_upper']],
                [i + 0.16, i + 0.16], color=color, lw=2.6, solid_capstyle='round')
        ax.scatter(row['prespec_r'], i + 0.16, color=color, s=70, zorder=5,
                   edgecolor='white', linewidth=0.6)
    # Selected-peak-lag CI (selection-inflated) — faded, hollow, offset down
    if pd.notna(row.get('peak_r')):
        ax.plot([row['peak_ci_lower'], row['peak_ci_upper']],
                [i - 0.16, i - 0.16], color=color, lw=1.8, alpha=0.45,
                solid_capstyle='round')
        ax.scatter(row['peak_r'], i - 0.16, facecolor='none', edgecolor=color,
                   s=55, zorder=5, alpha=0.9)

ax.axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(ci_df['label'], fontsize=8)
ax.set_xlabel('Pearson correlation coefficient (95% CI)', fontsize=11)
ax.set_title(
    'Cross-correlation confidence intervals — circular block bootstrap\n'
    '(n=1 000; block length = max(7, √n); solid = pre-specified lag, hollow = selected peak)',
    fontsize=12,
)
ax.grid(True, alpha=0.25, axis='x')

# Legends — one for source group, one for lag basis
from matplotlib.lines import Line2D
group_handles = [Line2D([0], [0], color=c, lw=3, label=g)
                 for g, c in group_colors.items()]
style_handles = [
    Line2D([0], [0], color='#555', lw=2.6, marker='o', markersize=8,
           markerfacecolor='#555', label='Pre-specified lag (unbiased)'),
    Line2D([0], [0], color='#555', lw=1.8, alpha=0.6, marker='o', markersize=8,
           markerfacecolor='none', markeredgecolor='#555',
           label='Selected peak lag (inflated)'),
]
leg1 = ax.legend(handles=group_handles, fontsize=8, loc='lower right', title='Source')
ax.add_artist(leg1)
ax.legend(handles=style_handles, fontsize=8, loc='upper right', title='Lag basis')

plt.tight_layout()
out_fig = FIGURES_DIR / '14_xcorr_ci.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')
plt.show()

#%%
# ── Summary: what effects can we rule out? ────────────────────────────────────
# Reported at the PRE-SPECIFIED lag: at 95% confidence the data are consistent
# with effects no larger than the upper bound (i.e. rule out anything above it).
print('\n── Effects ruled out at 95% confidence (pre-specified lag) ───────────')
for _, row in ci_df.iterrows():
    if pd.notna(row.get('prespec_r')):
        print(f'  {row["label"][:55]:<55s}  '
              f'r={row["prespec_r"]:+.3f}  rule out > {row["prespec_max_effect_ruled_out_95pct"]:+.3f}')
