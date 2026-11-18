#%%
"""
05b_permutation_baseline.py
----------------------------
Permutation baseline for the creator→Reddit lead/lag analyses in script 05.

Addresses: are the headline statistics from 05 distinguishable from chance?
Given the primary result is a null, these tests confirm that the null has
adequate power — the observed statistics are not unusual relative to a
shuffled null, consistent with no systematic lead in either direction.

Three statistics tested at daily granularity (±28 days), with weekly
equivalents (±4 weeks) for A and B:

  A. Peak absolute cross-correlation of overall__score
     — is the observed max(|r|) across ±28 days unusual?
  B. Median peak lag across all topic fields
     — does the observed -2 days sit in the bulk of the null?
  C. Fraction of topic fields with a positive peak lag
     — is the creator-leads proportion higher or lower than chance?

Null model: circular-shift of non-NaN creator values (preserves the
creator series' marginal distribution and its own autocorrelation; only
destroys the creator→Reddit timing relationship). 1000 permutations.

Outputs:
  figures/05b_permutation_daily.png   — null distributions for A, B, C
  figures/05b_permutation_weekly.png  — weekly equivalents for A and B
  tables/05b_permutation_results.csv  — observed stats + p-values
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily,
    deseasonalize_dow,
)

MAX_LAG_DAYS      = 28
MAX_LAG_WEEKS     = 4
MIN_OVERLAP       = 90
MIN_OVERLAP_WEEKS = 20
N_PERM            = 1000
RANDOM_SEED       = 42

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
topic_cols = get_topic_cols(yt)
all_fields = ['overall__score'] + topic_cols

creator_agg = aggregate_creator_daily(yt, all_fields)
reddit_agg  = aggregate_reddit_daily(rd, all_fields)

start = max(creator_agg.index.min(), reddit_agg.index.min())
end   = min(creator_agg.index.max(), reddit_agg.index.max())
dates = pd.date_range(start, end, freq='D')
creator_agg = creator_agg.reindex(dates)
reddit_agg  = reddit_agg.reindex(dates)

#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def plot_null_dist(ax, null_vals, obs_val, title, xlabel, p_val):
    ax.hist(null_vals, bins=40, color='#95a5a6', alpha=0.75,
            edgecolor='white', label='Null (1 000 perm.)')
    ax.axvline(obs_val, color='#e74c3c', lw=2, linestyle='--',
               label=f'Observed = {obs_val:.3f}\np = {p_val:.3f}')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


#%%
# ── Pre-compute deseasonalised daily series for all fields ────────────────────
creator_deseas = {}
reddit_deseas  = {}

for field in all_fields:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col in creator_agg.columns:
        creator_deseas[field] = deseasonalize_dow(creator_agg[c_col].reindex(dates))
    if r_col in reddit_agg.columns:
        reddit_deseas[field]  = deseasonalize_dow(reddit_agg[r_col].reindex(dates))

#%%
# ── Observed statistics (daily) ───────────────────────────────────────────────
overall_cc = normalised_xcorr(
    creator_deseas.get('overall__score', pd.Series(dtype=float)),
    reddit_deseas.get('overall__score', pd.Series(dtype=float)),
    MAX_LAG_DAYS,
)
obs_peak_r = float(overall_cc.abs().max()) if not overall_cc.empty else np.nan

field_peak_lags = {}
for field in topic_cols:
    if field not in creator_deseas or field not in reddit_deseas:
        continue
    both = pd.concat([creator_deseas[field], reddit_deseas[field]], axis=1).dropna()
    if len(both) < MIN_OVERLAP:
        continue
    cc = normalised_xcorr(creator_deseas[field], reddit_deseas[field], MAX_LAG_DAYS)
    if not cc.empty:
        field_peak_lags[field] = int(cc.abs().idxmax())

obs_median_lag  = float(np.median(list(field_peak_lags.values()))) if field_peak_lags else np.nan
obs_frac_pos    = float(np.mean([v > 0 for v in field_peak_lags.values()])) if field_peak_lags else np.nan
n_fields_daily  = len(field_peak_lags)

print(f'Observed (daily, n_fields={n_fields_daily}):')
print(f'  overall peak |r| = {obs_peak_r:.4f}')
print(f'  median peak lag  = {obs_median_lag:.1f} days')
print(f'  frac positive lag = {obs_frac_pos:.3f}')

#%%
# ── Permutation loop (daily) ──────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)
null_peak_r   = []
null_med_lag  = []
null_frac_pos = []

for i in range(N_PERM):
    shift = np.random.randint(1, len(dates))

    # Test A: overall peak |r|
    if 'overall__score' in creator_deseas and 'overall__score' in reddit_deseas:
        c_shifted = shift_values_only(creator_deseas['overall__score'], shift)
        cc = normalised_xcorr(c_shifted, reddit_deseas['overall__score'], MAX_LAG_DAYS)
        null_peak_r.append(float(cc.abs().max()) if not cc.empty else np.nan)

    # Tests B+C: field-level median lag and fraction positive
    perm_lags = []
    for field in field_peak_lags:
        c_shifted = shift_values_only(creator_deseas[field], shift)
        cc = normalised_xcorr(c_shifted, reddit_deseas[field], MAX_LAG_DAYS)
        if not cc.empty:
            perm_lags.append(int(cc.abs().idxmax()))

    if perm_lags:
        null_med_lag.append(float(np.median(perm_lags)))
        null_frac_pos.append(float(np.mean([v > 0 for v in perm_lags])))

    if (i + 1) % 200 == 0:
        print(f'  Daily permutation {i + 1}/{N_PERM}')

null_peak_r   = np.array([x for x in null_peak_r if not np.isnan(x)])
null_med_lag  = np.array(null_med_lag)
null_frac_pos = np.array(null_frac_pos)

p_peak_r  = float(np.mean(null_peak_r  >= obs_peak_r))
p_med_lag = float(np.mean(np.abs(null_med_lag) >= abs(obs_median_lag)))
p_frac_pos = float(np.mean(null_frac_pos >= obs_frac_pos))

print(f'\nPermutation p-values (daily):')
print(f'  A  peak |r| overall:        p={p_peak_r:.4f}')
print(f'  B  median peak lag fields:  p={p_med_lag:.4f}')
print(f'  C  frac positive lag:       p={p_frac_pos:.4f}')

#%%
# ── Figure: daily null distributions ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_null_dist(axes[0], null_peak_r, obs_peak_r,
               'A: Overall peak |r|  (daily ±28d)',
               'Peak absolute cross-correlation', p_peak_r)

plot_null_dist(axes[1], null_med_lag, obs_median_lag,
               'B: Median peak lag — all topic fields  (daily)',
               'Median peak lag (days)', p_med_lag)

plot_null_dist(axes[2], null_frac_pos, obs_frac_pos,
               'C: Fraction of fields with positive peak lag  (daily)',
               'Fraction of fields (peak lag > 0)', p_frac_pos)

fig.suptitle(
    'Creator→Reddit: observed statistics vs. circular-shift null  (1 000 permutations, daily)',
    fontsize=13,
)
plt.tight_layout()
out_daily = FIGURES_DIR / '05b_permutation_daily.png'
plt.savefig(out_daily, dpi=150, bbox_inches='tight')
print(f'Saved: {out_daily}')
plt.show()

#%%
# ── Weekly equivalents ────────────────────────────────────────────────────────
creator_weekly = creator_agg.resample('W-MON').mean()
reddit_weekly  = reddit_agg.resample('W-MON').mean()

start_w = max(creator_weekly.index.min(), reddit_weekly.index.min())
end_w   = min(creator_weekly.index.max(), reddit_weekly.index.max())
weeks   = pd.date_range(start_w, end_w, freq='W-MON')
creator_weekly = creator_weekly.reindex(weeks)
reddit_weekly  = reddit_weekly.reindex(weeks)

overall_col = 'overall__score_unweighted'

if overall_col in creator_weekly.columns and overall_col in reddit_weekly.columns:
    cc_w = normalised_xcorr(creator_weekly[overall_col], reddit_weekly[overall_col], MAX_LAG_WEEKS)
    obs_peak_r_w = float(cc_w.abs().max()) if not cc_w.empty else np.nan
else:
    obs_peak_r_w = np.nan

field_peak_lags_w = {}
for field in topic_cols:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col not in creator_weekly.columns or r_col not in reddit_weekly.columns:
        continue
    both = pd.concat([creator_weekly[c_col], reddit_weekly[r_col]], axis=1).dropna()
    if len(both) < MIN_OVERLAP_WEEKS:
        continue
    cc = normalised_xcorr(creator_weekly[c_col], reddit_weekly[r_col], MAX_LAG_WEEKS)
    if not cc.empty:
        field_peak_lags_w[field] = int(cc.abs().idxmax())

obs_med_lag_w  = float(np.median(list(field_peak_lags_w.values()))) if field_peak_lags_w else np.nan
obs_frac_pos_w = float(np.mean([v > 0 for v in field_peak_lags_w.values()])) if field_peak_lags_w else np.nan
n_fields_weekly = len(field_peak_lags_w)

print(f'\nObserved (weekly, n_fields={n_fields_weekly}):')
print(f'  overall peak |r| = {obs_peak_r_w:.4f}')
print(f'  median peak lag  = {obs_med_lag_w:.2f} weeks')
print(f'  frac positive lag = {obs_frac_pos_w:.3f}')

#%%
np.random.seed(RANDOM_SEED)
null_peak_r_w   = []
null_med_lag_w  = []
null_frac_pos_w = []

for i in range(N_PERM):
    shift = np.random.randint(1, len(weeks))

    # Test A weekly: overall peak |r|
    if overall_col in creator_weekly.columns and overall_col in reddit_weekly.columns:
        c_shifted = shift_values_only(creator_weekly[overall_col], shift)
        cc = normalised_xcorr(c_shifted, reddit_weekly[overall_col], MAX_LAG_WEEKS)
        null_peak_r_w.append(float(cc.abs().max()) if not cc.empty else np.nan)

    # Tests B+C weekly
    perm_lags_w = []
    for field in field_peak_lags_w:
        c_col = f'{field}_unweighted'
        r_col = f'{field}_unweighted'
        if c_col not in creator_weekly.columns:
            continue
        c_shifted = shift_values_only(creator_weekly[c_col], shift)
        cc = normalised_xcorr(c_shifted, reddit_weekly[r_col], MAX_LAG_WEEKS)
        if not cc.empty:
            perm_lags_w.append(int(cc.abs().idxmax()))

    if perm_lags_w:
        null_med_lag_w.append(float(np.median(perm_lags_w)))
        null_frac_pos_w.append(float(np.mean([v > 0 for v in perm_lags_w])))

    if (i + 1) % 200 == 0:
        print(f'  Weekly permutation {i + 1}/{N_PERM}')

null_peak_r_w   = np.array([x for x in null_peak_r_w if not np.isnan(x)])
null_med_lag_w  = np.array(null_med_lag_w)
null_frac_pos_w = np.array(null_frac_pos_w)

p_peak_r_w   = float(np.mean(null_peak_r_w   >= obs_peak_r_w))
p_med_lag_w  = float(np.mean(np.abs(null_med_lag_w) >= abs(obs_med_lag_w)))
p_frac_pos_w = float(np.mean(null_frac_pos_w >= obs_frac_pos_w))

print(f'\nPermutation p-values (weekly):')
print(f'  A  peak |r| overall:        p={p_peak_r_w:.4f}')
print(f'  B  median peak lag fields:  p={p_med_lag_w:.4f}')
print(f'  C  frac positive lag:       p={p_frac_pos_w:.4f}')

#%%
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

plot_null_dist(axes2[0], null_peak_r_w, obs_peak_r_w,
               'A: Overall peak |r|  (weekly ±4w)',
               'Peak absolute cross-correlation', p_peak_r_w)

plot_null_dist(axes2[1], null_med_lag_w, obs_med_lag_w,
               'B: Median peak lag — all topic fields  (weekly)',
               'Median peak lag (weeks)', p_med_lag_w)

fig2.suptitle(
    'Creator→Reddit (weekly): observed vs. circular-shift null  (1 000 permutations)',
    fontsize=13,
)
plt.tight_layout()
out_weekly = FIGURES_DIR / '05b_permutation_weekly.png'
plt.savefig(out_weekly, dpi=150, bbox_inches='tight')
print(f'Saved: {out_weekly}')
plt.show()

#%%
# ── Save results table ────────────────────────────────────────────────────────
results = pd.DataFrame([
    {'granularity': 'daily',  'test': 'A_peak_abs_r_overall',
     'observed': obs_peak_r,    'null_mean': null_peak_r.mean(),
     'null_sd':  null_peak_r.std(),   'p_value': p_peak_r},
    {'granularity': 'daily',  'test': 'B_median_peak_lag_fields',
     'observed': obs_median_lag,  'null_mean': null_med_lag.mean(),
     'null_sd':  null_med_lag.std(),  'p_value': p_med_lag},
    {'granularity': 'daily',  'test': 'C_frac_fields_positive_lag',
     'observed': obs_frac_pos,    'null_mean': null_frac_pos.mean(),
     'null_sd':  null_frac_pos.std(), 'p_value': p_frac_pos},
    {'granularity': 'weekly', 'test': 'A_peak_abs_r_overall',
     'observed': obs_peak_r_w,  'null_mean': null_peak_r_w.mean(),
     'null_sd':  null_peak_r_w.std(), 'p_value': p_peak_r_w},
    {'granularity': 'weekly', 'test': 'B_median_peak_lag_fields',
     'observed': obs_med_lag_w, 'null_mean': null_med_lag_w.mean(),
     'null_sd':  null_med_lag_w.std(), 'p_value': p_med_lag_w},
    {'granularity': 'weekly', 'test': 'C_frac_fields_positive_lag',
     'observed': obs_frac_pos_w, 'null_mean': null_frac_pos_w.mean(),
     'null_sd':  null_frac_pos_w.std(), 'p_value': p_frac_pos_w},
])

out_tbl = TABLES_DIR / '05b_permutation_results.csv'
results.to_csv(out_tbl, index=False)
print(f'\nSaved: {out_tbl}')
print(results.to_string(index=False))
