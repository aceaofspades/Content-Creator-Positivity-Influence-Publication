#%%
"""
06c_creator_subset_robustness.py
----------------------------------
Tests whether the creator→Reddit lead/lag results from script 05 are
sensitive to which creators happen to be in the dataset.

Addresses the concern that one unusually negative, unusually large, or
unusually small creator could be driving or masking a real signal.

Three leave-one-group-out subsets (defined at the creator level):
  1. Drop most-negative: bottom quintile by mean overall__score
  2. Drop largest: top tercile by total portfolio view count
  3. Drop smallest: bottom tercile by total portfolio view count

For each subset: aggregate overall__score daily (unweighted), deseasonalize
(DOW dummies), and run normalized cross-correlation vs. Reddit at:
  - Daily ±28 days
  - Weekly ±4 weeks

Comparison baseline: full-creator-set result (matching script 05 unweighted).

NOTE: The loaded master parquets are never modified. All subset operations
work on in-memory filtered copies created locally in this script.

Outputs:
  figures/06c_subset_xcorr_daily.png   — xcorr curves: full set + 3 subsets (daily)
  figures/06c_subset_xcorr_weekly.png  — same at weekly granularity
  tables/06c_subset_robustness.csv     — peak lag + peak corr per subset × granularity
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
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily,
    deseasonalize_dow,
)

MAX_LAG_DAYS      = 28
MAX_LAG_WEEKS     = 4
FIELD             = 'overall__score'
NEG_QUINTILE      = 0.20   # bottom quintile by mean overall__score
LARGE_UPPER       = 2 / 3  # top tercile by total portfolio views
SMALL_LOWER       = 1 / 3  # bottom tercile by total portfolio views

# Circular-shift noise gate (mirrors scripts 04 / 05b / 05g): a peak is only
# reported as real if its |r| exceeds the 95th-pctile of a max-|r| null built by
# circularly shifting the creator series. Applied to EVERY subset/tier peak so
# no tier is implicitly assumed real -- the bare abs().idxmax() pitfall fixed in 04.
N_PERM            = 1000
NULL_PCTILE       = 95
RANDOM_SEED       = 42

SUBSET_COLORS = {
    'Full set':      '#2c3e50',
    'Drop negative': '#27ae60',
    'Drop largest':  '#2980b9',
    'Drop smallest': '#e67e22',
}
SUBSET_STYLES = {
    'Full set':      dict(lw=2.5, alpha=0.9, linestyle='-'),
    'Drop negative': dict(lw=1.8, alpha=0.85, linestyle='--'),
    'Drop largest':  dict(lw=1.8, alpha=0.85, linestyle='-.'),
    'Drop smallest': dict(lw=1.8, alpha=0.85, linestyle=':'),
}

#%%
ensure_output_dirs()

# ── Load data — originals never modified ─────────────────────────────────────
yt_full = load_youtube_creators()   # .copy() guaranteed by loader
rd      = load_reddit_submissions()

# ── Identify creator-level groupings ─────────────────────────────────────────
creator_mean_score  = yt_full.groupby('creator')[FIELD].mean()
creator_total_views = yt_full.groupby('creator')['view_count'].sum().fillna(0)

neg_threshold   = creator_mean_score.quantile(NEG_QUINTILE)
large_threshold = creator_total_views.quantile(LARGE_UPPER)
small_threshold = creator_total_views.quantile(SMALL_LOWER)

drop_negative = set(creator_mean_score[creator_mean_score <= neg_threshold].index)
drop_largest  = set(creator_total_views[creator_total_views >= large_threshold].index)
drop_smallest = set(creator_total_views[creator_total_views <= small_threshold].index)

print(f'Total creators: {yt_full["creator"].nunique()}')
print(f'Drop negative  ({NEG_QUINTILE*100:.0f}th pctile, score ≤ {neg_threshold:.3f}): '
      f'{len(drop_negative)} creators, '
      f'{yt_full[yt_full["creator"].isin(drop_negative)].shape[0]:,} videos removed')
print(f'Drop largest   (top tercile, views ≥ {large_threshold:,.0f}): '
      f'{len(drop_largest)} creators, '
      f'{yt_full[yt_full["creator"].isin(drop_largest)].shape[0]:,} videos removed')
print(f'Drop smallest  (bot tercile, views ≤ {small_threshold:,.0f}): '
      f'{len(drop_smallest)} creators, '
      f'{yt_full[yt_full["creator"].isin(drop_smallest)].shape[0]:,} videos removed')

# ── Build in-script filtered copies (masters remain untouched) ────────────────
subsets = {
    'Full set':      yt_full,
    'Drop negative': yt_full[~yt_full['creator'].isin(drop_negative)].copy(),
    'Drop largest':  yt_full[~yt_full['creator'].isin(drop_largest)].copy(),
    'Drop smallest': yt_full[~yt_full['creator'].isin(drop_smallest)].copy(),
}
for name, df in subsets.items():
    print(f'  {name}: {df["creator"].nunique()} creators, {len(df):,} videos')

#%%
# ── Aggregate Reddit once (unchanged across subsets) ─────────────────────────
reddit_agg = aggregate_reddit_daily(rd, [FIELD])

# Date bounds from full creator set
creator_agg_full = aggregate_creator_daily(yt_full, [FIELD])
start = max(creator_agg_full.index.min(), reddit_agg.index.min())
end   = min(creator_agg_full.index.max(), reddit_agg.index.max())
dates = pd.date_range(start, end, freq='D')
reddit_agg = reddit_agg.reindex(dates)

r_daily  = deseasonalize_dow(reddit_agg[f'{FIELD}_unweighted'].reindex(dates))
r_weekly = reddit_agg[f'{FIELD}_unweighted'].resample('W-MON').mean()

#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circularly shift only the non-NaN values, preserving NaN positions.
    Destroys cross-series timing while keeping the series' own autocorrelation."""
    result = series.copy()
    valid = series.notna()
    vals = series[valid].values
    if len(vals):
        result[valid] = np.roll(vals, shift % len(vals))
    return result


def gated_peak(x: pd.Series, y: pd.Series, max_lag: int,
               n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """Hypothesis-blind global max-|r| peak, gated by a circular-shift null.

    x is the creator series (shuffled in the null); y is the fixed target. The
    null draws max-|r| ACROSS the lag window on each shuffle, so the gate
    corrects for the multiple-lag search behind abs().idxmax(). Returns
    (peak_lag, peak_corr, noise_threshold, p_value, significant) or None.
    """
    cc = normalised_xcorr(x, y, max_lag)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])
    rng = np.random.default_rng(seed)
    n_valid = int(x.notna().sum())
    null_max = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n_valid, 2)))
        cc_n = normalised_xcorr(shift_values_only(x, shift), y, max_lag)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
    if not null_max:
        return peak_lag, peak_corr, float('nan'), float('nan'), False
    null_max = np.asarray(null_max)
    threshold = float(np.percentile(null_max, NULL_PCTILE))
    p_value   = float(np.mean(null_max >= abs(peak_corr)))
    return peak_lag, peak_corr, threshold, p_value, bool(abs(peak_corr) > threshold)


#%%
# ── Compute xcorr for each subset ────────────────────────────────────────────
records = []
daily_cc  = {}
weekly_cc = {}

for name, yt_sub in subsets.items():
    # Daily
    agg = aggregate_creator_daily(yt_sub, [FIELD])
    agg = agg.reindex(dates)
    c_d = deseasonalize_dow(agg[f'{FIELD}_unweighted'])
    cc_d = normalised_xcorr(c_d, r_daily, MAX_LAG_DAYS)
    daily_cc[name] = cc_d

    # Weekly
    agg_w = agg.resample('W-MON').mean()
    c_w   = agg_w[f'{FIELD}_unweighted']
    cc_w  = normalised_xcorr(c_w, r_weekly, MAX_LAG_WEEKS)
    weekly_cc[name] = cc_w

    for gran, cc, cx, cy, mlag in [
        ('daily',  cc_d, c_d, r_daily,  MAX_LAG_DAYS),
        ('weekly', cc_w, c_w, r_weekly, MAX_LAG_WEEKS),
    ]:
        if cc.empty:
            continue
        peak_lag, peak_corr, thr, p, sig = gated_peak(cx, cy, mlag)
        unit = 'days' if gran == 'daily' else 'weeks'
        records.append({
            'subset':      name,
            'granularity': gran,
            'n_creators':  yt_sub['creator'].nunique(),
            'n_videos':    len(yt_sub),
            'peak_lag':    peak_lag,
            'peak_corr':   round(peak_corr, 4),
            'noise_threshold': round(thr, 4),
            'p_value':     round(p, 4),
            'significant': sig,
        })
        flag = '' if sig else '  (within noise)'
        print(f'  {name:<18} {gran:<8} peak_lag={peak_lag:+3d}{unit[0]}  '
              f'peak_corr={peak_corr:+.4f}  p={p:.3f}{flag}')

results_df = pd.DataFrame(records)
results_df.to_csv(TABLES_DIR / '06c_subset_robustness.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "06c_subset_robustness.csv"}')

#%%
# ── Figure: daily ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

for name, cc in daily_cc.items():
    if cc.empty:
        continue
    ax.plot(cc.index, cc.values,
            color=SUBSET_COLORS[name],
            label=f'{name}  (n={subsets[name]["creator"].nunique()} creators)',
            **SUBSET_STYLES[name])

ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax.axhline(0, color='gray', lw=0.6)
ax.set_xlabel('Lag (days)  [+ = creator leads Reddit]', fontsize=11)
ax.set_ylabel('Cross-correlation', fontsize=11)
ax.set_title(
    'Creator→Reddit overall__score: creator subset robustness check  (daily ±28d)\n'
    'Deseasonalized; unweighted; drop groups defined at creator level',
    fontsize=12,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
out_d = FIGURES_DIR / '06c_subset_xcorr_daily.png'
plt.savefig(out_d, dpi=150, bbox_inches='tight')
print(f'Saved: {out_d}')
plt.show()

#%%
# ── Figure: weekly ────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

for name, cc in weekly_cc.items():
    if cc.empty:
        continue
    ax2.plot(cc.index, cc.values,
             color=SUBSET_COLORS[name],
             label=f'{name}  (n={subsets[name]["creator"].nunique()} creators)',
             **SUBSET_STYLES[name])

ax2.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax2.axhline(0, color='gray', lw=0.6)
ax2.set_xlabel('Lag (weeks)  [+ = creator leads Reddit]', fontsize=11)
ax2.set_ylabel('Cross-correlation', fontsize=11)
ax2.set_title(
    'Creator→Reddit overall__score: creator subset robustness check  (weekly ±4w)',
    fontsize=12,
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)
plt.tight_layout()
out_w = FIGURES_DIR / '06c_subset_xcorr_weekly.png'
plt.savefig(out_w, dpi=150, bbox_inches='tight')
print(f'Saved: {out_w}')
plt.show()

#%%
# ── Print summary comparison ───────────────────────────────────────────────────
print('\n── Summary ─────────────────────────────────────────────────────────────')
print(results_df.to_string(index=False))
print('\nInterpretation guide:')
print('  If peak_lag and peak_corr are stable across subsets, the result is')
print('  not driven by any single creator tier. Large shifts suggest the')
print('  excluded group was pulling the aggregate signal.')

#%%
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Authority-tier gradient sensitivity
# ─────────────────────────────────────────────────────────────────────────────
# Script 06 binned creators into authority terciles using total portfolio view
# count and found an inverted gradient at weekly granularity: Low-authority
# creators showed the strongest co-movement with Reddit (r≈0.167, lag=0) while
# High-authority tiers were near zero or negative — the opposite of the elite-
# influence prediction. This section tests whether that gradient survives three
# alternative authority operationalizations:
#
#   1. Total portfolio views      — original script 06 definition (reference)
#   2. Mean views per video       — neutralizes the output-volume advantage of
#                                   prolific creators; ~authority per unit effort
#   3. Peak views                 — maximum single-video reach; captures headline-
#                                   moment authority independent of frequency
#   4. Recency-corrected views    — mean(view_count / days_since_publish); adjusts
#                                   for the known accumulation bias where older
#                                   videos have more time to grow view counts
#
# If the Low > High gradient holds across all four operationalizations, the
# inverted result is not an artifact of how "authority" is defined.
#
# Additional outputs:
#   figures/06c_authority_sensitivity.png   — 2×2 grid; weekly xcorr by tier
#   tables/06c_authority_sensitivity.csv    — peak lag + peak corr per metric × tier

TIER_COLORS = {'Low': '#27ae60', 'Mid': '#f39c12', 'High': '#c0392b'}
TIER_STYLES = {'Low': '-',       'Mid': '--',      'High': ':'}

# ── Compute recency-corrected view rate ───────────────────────────────────────
pub_dates = pd.to_datetime(yt_full['published_at'], utc=True).dt.tz_convert(None)
reference_date = pub_dates.max()
days_since_pub = (reference_date - pub_dates).dt.days.clip(lower=1)

views_per_day = pd.Series(
    yt_full['view_count'].fillna(0).values / days_since_pub.values,
    index=yt_full.index,
)
creator_vpd = views_per_day.groupby(yt_full['creator']).mean().fillna(0)

# ── Four authority metrics (per-creator Series) ───────────────────────────────
authority_metrics = {
    'Total portfolio views\n(script 06 definition)':
        yt_full.groupby('creator')['view_count'].sum().fillna(0),
    'Mean views per video\n(output-volume neutral)':
        yt_full.groupby('creator')['view_count'].mean().fillna(0),
    'Peak views\n(headline-moment authority)':
        yt_full.groupby('creator')['view_count'].max().fillna(0),
    'Recency-corrected\n(views/day, corrects accumulation bias)':
        creator_vpd,
}

#%%
# ── Compute tier xcorr for each authority metric ──────────────────────────────
auth_records = []
auth_gate = {}   # (metric_name, tier) -> (peak_lag, peak_corr, significant) for the figure

for metric_name, metric_vals in authority_metrics.items():
    low_thresh  = metric_vals.quantile(1 / 3)
    high_thresh = metric_vals.quantile(2 / 3)
    tier_assignment = pd.cut(
        metric_vals,
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=['Low', 'Mid', 'High'],
    )
    print(f'\n{metric_name.replace(chr(10), " ")}')
    for tier_label in ['Low', 'Mid', 'High']:
        tier_creators = set(tier_assignment[tier_assignment == tier_label].index)
        yt_tier = yt_full[yt_full['creator'].isin(tier_creators)].copy()

        agg_tier = aggregate_creator_daily(yt_tier, [FIELD]).reindex(dates)
        c_w = agg_tier[f'{FIELD}_unweighted'].resample('W-MON').mean()
        cc_w = normalised_xcorr(c_w, r_weekly, MAX_LAG_WEEKS)

        if cc_w.empty:
            continue
        peak_lag, peak_corr, thr, p, sig = gated_peak(c_w, r_weekly, MAX_LAG_WEEKS)
        auth_gate[(metric_name, tier_label)] = (peak_lag, peak_corr, sig)
        flag = '' if sig else '  (within noise)'
        print(f'  {tier_label:<4}  n={len(tier_creators):2d}  '
              f'peak_lag={peak_lag:+2d}w  peak_corr={peak_corr:+.4f}  '
              f'p={p:.3f}{flag}')
        auth_records.append({
            'authority_metric': metric_name.replace('\n', ' '),
            'tier':             tier_label,
            'n_creators':       len(tier_creators),
            'peak_lag_weeks':   peak_lag,
            'peak_corr':        round(peak_corr, 4),
            'noise_threshold':  round(thr, 4),
            'p_value':          round(p, 4),
            'significant':      sig,
        })

auth_df = pd.DataFrame(auth_records)
auth_df.to_csv(TABLES_DIR / '06c_authority_sensitivity.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "06c_authority_sensitivity.csv"}')

#%%
# ── Figure: 2×2 grid, one panel per authority metric ─────────────────────────
fig_auth, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for ax, (metric_name, metric_vals) in zip(axes.flatten(), authority_metrics.items()):
    low_thresh  = metric_vals.quantile(1 / 3)
    high_thresh = metric_vals.quantile(2 / 3)
    tier_assignment = pd.cut(
        metric_vals,
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=['Low', 'Mid', 'High'],
    )
    for tier_label in ['Low', 'Mid', 'High']:
        tier_creators = set(tier_assignment[tier_assignment == tier_label].index)
        yt_tier = yt_full[yt_full['creator'].isin(tier_creators)].copy()

        agg_tier = aggregate_creator_daily(yt_tier, [FIELD]).reindex(dates)
        c_w = agg_tier[f'{FIELD}_unweighted'].resample('W-MON').mean()
        cc_w = normalised_xcorr(c_w, r_weekly, MAX_LAG_WEEKS)

        if cc_w.empty:
            continue
        gate = auth_gate.get((metric_name, tier_label))
        sig = gate[2] if gate else False
        ax.plot(cc_w.index, cc_w.values,
                color=TIER_COLORS[tier_label],
                linestyle=TIER_STYLES[tier_label],
                lw=1.8, alpha=0.85,
                label=f'{tier_label}  (n={len(tier_creators)}){"" if sig else " (ns)"}')
        if gate is not None:
            p_lag, p_corr, p_sig = gate
            ax.scatter([p_lag], [p_corr],
                       s=75 if p_sig else 45,
                       facecolor=TIER_COLORS[tier_label] if p_sig else 'none',
                       edgecolor=TIER_COLORS[tier_label],
                       marker='o', linewidths=1.6, zorder=5)

    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(metric_name, fontsize=10)
    ax.set_xlabel('Lag (weeks)  [+ = creator leads Reddit]', fontsize=9)
    ax.set_ylabel('Cross-correlation', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

fig_auth.suptitle(
    'Authority-tier gradient: sensitivity to authority operationalization\n'
    'Weekly ±4 weeks, unweighted, overall__score vs. Reddit submissions\n'
    'Dot at each peak: filled = clears 95th-pctile noise gate, hollow = within noise',
    fontsize=11,
)
plt.tight_layout()
out_auth = FIGURES_DIR / '06c_authority_sensitivity.png'
plt.savefig(out_auth, dpi=150, bbox_inches='tight')
print(f'Saved: {out_auth}')
plt.show()

#%%
# ── Gated gradient diagnostic ──────────────────────────────────────────────────
# A tier only counts toward a "gradient" if its peak clears the circular-shift
# noise gate. Ordering the Low/Mid/High of ungated abs-max peaks would dress up
# noise as structure -- and crucially it would single out the Mid tier as "the
# noisy one" while implicitly trusting Low/High picked the same blind way. The
# gate is applied to all three so the comparison is symmetric.
print('\n-- Authority sensitivity: gated gradient diagnostic --')
print('   (* = peak clears 95th-pctile circular-shift noise gate; r at selected lag)')
n_metrics_with_gradient = 0
for metric_name in auth_df['authority_metric'].unique():
    sub = auth_df[auth_df['authority_metric'] == metric_name].set_index('tier')
    cells, sig_tiers = [], []
    for t in ['Low', 'Mid', 'High']:
        if t in sub.index:
            r = float(sub.loc[t, 'peak_corr'])
            s = bool(sub.loc[t, 'significant'])
            cells.append(f'{t}={r:+.3f}{"*" if s else " "}')
            if s:
                sig_tiers.append((t, r))
        else:
            cells.append(f'{t}=  n/a ')
    if len(sig_tiers) >= 2:
        n_metrics_with_gradient += 1
        ordering = ' < '.join(t for t, _ in sorted(sig_tiers, key=lambda kv: kv[1]))
        verdict = f'{len(sig_tiers)} tiers clear gate: {ordering}'
    elif len(sig_tiers) == 1:
        verdict = f'only {sig_tiers[0][0]} clears gate -> no gradient'
    else:
        verdict = 'no tier clears gate -> no gradient'
    print(f'  {metric_name.split("(")[0].strip():<26}  {"  ".join(cells)}   -> {verdict}')

print()
n_metrics = auth_df['authority_metric'].nunique()
if n_metrics_with_gradient == 0:
    print('  No operationalization yields >=2 authority tiers above the noise gate.')
    print('  There is no statistically distinguishable authority gradient -- the')
    print('  Low/Mid/High differences are within-noise abs-max picks. This still')
    print('  refutes the elite-influence prediction (no High-tier lead), but via')
    print('  ABSENCE of a gradient, not an inverted one.')
else:
    print(f'  {n_metrics_with_gradient} of {n_metrics} operationalizations have >=2 '
          'tiers above the gate; inspect the')
    print('  ordering above. Only gated tiers should be cited for any gradient claim.')

# %%
