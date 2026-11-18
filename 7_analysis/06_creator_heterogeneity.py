#%%
"""
06_creator_heterogeneity.py
----------------------------
Tests whether the creator→Reddit lead/lag effect scales with creator reach
(view-count tier) and authority (creator-size tier).

Elite influence theory predicts effects should be stronger for high-reach
and high-authority creators. If either split shows scaling, that supports
the theory; null results are reported honestly.

Method:
  - Reach tier: per-video view_count tercile (reach is a per-video property, so
    terciles over video rows are correct).
  - Authority tier: per-CREATOR total view count, terciles cut over CREATORS,
    then mapped back onto each creator's videos. (Earlier versions cut terciles
    over video rows via transform('sum') + qcut, which weighted boundaries by
    output volume and conflated authority with posting frequency. Fixed to match
    06c Section 2.)
  - Every tier peak is gated with a circular-shift noise test (1000 perms,
    95th-pctile max-|r| null; mirrors 06c / 04 / 05b). A peak is only called real
    if it clears the gate; otherwise it is tagged (ns) and drawn with a hollow
    marker. Without the gate, abs().idxmax() reports a peak for every tier whether
    or not it is distinguishable from noise.
  - Run cross-correlation of each tier's sentiment vs. Reddit overall__score
  - Compare peak lag and peak correlation across tiers
  - Daily (±28 days) and weekly (±4 weeks) granularities — weekly replicates
    the pattern from script 04 where aggregation revealed signal hidden at daily level

Outputs:
  figures/06a_tier_xcorr.png         — daily corr curves by reach + authority tier
                                       (filled/hollow peak dot = clears/within noise gate)
  figures/06b_weekly_tier_xcorr.png  — weekly corr curves by reach + authority tier
  tables/06_tier_xcorr_summary.csv   — peak lag + corr + noise_threshold/p_value/
                                       significant for each tier × granularity × target
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow,
)

MAX_LAG_DAYS  = 28
MAX_LAG_WEEKS = 4
SENTIMENT_FIELD = 'overall__score'
N_TIERS = 3
TIER_LABELS = {1: 'Low', 2: 'Mid', 3: 'High'}
TIER_COLORS = {1: '#95a5a6', 2: '#f39c12', 3: '#e74c3c'}

# Circular-shift noise gate (mirrors 06c / 04 / 05b): a tier peak is only called
# real if its |r| exceeds the 95th-pctile of a max-|r| null built by circularly
# shifting the creator series. Without it, abs().idxmax() reports a peak for every
# tier regardless of whether it is distinguishable from noise.
N_PERM      = 1000
NULL_PCTILE = 95
RANDOM_SEED = 42

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
cm = load_comments_master()

# ── Compute tier assignments ──────────────────────────────────────────────────
# Reach tier: per-video view_count tercile
yt = yt.copy()
yt['reach_tier'] = pd.qcut(yt['view_count'].fillna(0), N_TIERS,
                            labels=[1, 2, 3], duplicates='drop').astype(float)

# Authority tier: creator-LEVEL total view count, with terciles cut over
# CREATORS (not video rows). Cutting over video rows -- the old transform('sum')
# + qcut approach -- weights the terciles by output volume: a prolific creator's
# many rows drag the boundaries, so "High authority" became "videos belonging to
# high-OUTPUT creators," conflating authority with posting frequency. Build one
# value per creator, tercile those, then map each creator's tier back onto their
# videos. This matches 06c Section 2's correct per-creator construction.
creator_total_views = yt.groupby('creator')['view_count'].sum().fillna(0)
creator_authority_tier = pd.qcut(creator_total_views, N_TIERS,
                                  labels=[1, 2, 3], duplicates='drop')
yt['authority_tier'] = yt['creator'].map(creator_authority_tier).astype(float)

print('Creators per authority tier (terciles cut over creators):')
print(creator_authority_tier.value_counts().sort_index())
print('\nVideos per reach tier (terciles over videos -- correct: reach is per-video):')
print(yt['reach_tier'].value_counts().sort_index())
print('\nVideos per authority tier (unbalanced by design -- prolific creators '
      'contribute more rows to their own tier):')
print(yt['authority_tier'].value_counts().sort_index())

#%%
reddit_agg  = aggregate_reddit_daily(rd, [SENTIMENT_FIELD])
comment_agg = aggregate_comments_daily(cm, ['sentiment_score'])

r_base        = deseasonalize_dow(reddit_agg[f'{SENTIMENT_FIELD}_unweighted'])
r_base_weekly = reddit_agg[f'{SENTIMENT_FIELD}_unweighted'].resample('W-MON').mean()

cm_base        = deseasonalize_dow(comment_agg['sentiment_score_unweighted'])
cm_base_weekly = comment_agg['sentiment_score_unweighted'].resample('W-MON').mean()

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

    Mirrors 06c / 04 / 05b. x is the creator series (shuffled in the null); y is
    the fixed target. The null draws max-|r| ACROSS the lag window on each shuffle,
    correcting for the multiple-lag search behind abs().idxmax(). Returns
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
records = []

def plot_tier_xcorr(tier_col: str, ax: plt.Axes, split_label: str,
                    ref_series: pd.Series, max_lag: int,
                    granularity: str = 'daily', weekly: bool = False,
                    target_label: str = 'submissions') -> None:
    for tier in sorted(yt[tier_col].dropna().unique()):
        subset = yt[yt[tier_col] == tier]
        agg    = aggregate_creator_daily(subset, [SENTIMENT_FIELD])
        if agg.empty:
            continue
        c_s = agg[f'{SENTIMENT_FIELD}_unweighted']
        if weekly:
            c_s = c_s.resample('W-MON').mean()
        else:
            c_s = deseasonalize_dow(c_s)
        cc = normalised_xcorr(c_s, ref_series, max_lag)
        if cc.empty:
            continue
        gp = gated_peak(c_s, ref_series, max_lag)
        if gp is None:
            continue
        peak_lag, peak_corr, thr, p_val, sig = gp
        label_str = (f'{TIER_LABELS[int(tier)]} (n={len(subset):,} videos)'
                     f'{"" if sig else " (ns)"}')
        ax.plot(cc.index, cc.values, color=TIER_COLORS[int(tier)],
                lw=1.8, alpha=0.85, label=label_str)
        ax.scatter([peak_lag], [peak_corr],
                   s=75 if sig else 45,
                   facecolor=TIER_COLORS[int(tier)] if sig else 'none',
                   edgecolor=TIER_COLORS[int(tier)],
                   marker='o', linewidths=1.6, zorder=5)
        records.append({
            'granularity':  granularity,
            'target':       target_label,
            'split':        split_label,
            'tier':         TIER_LABELS[int(tier)],
            'n_videos':     len(subset),
            'peak_lag':     peak_lag,
            'peak_corr':    round(float(peak_corr), 4),
            'noise_threshold': round(thr, 4),
            'p_value':      round(p_val, 4),
            'significant':  sig,
        })

    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_ylabel('Cross-correlation')
    unit = 'weeks' if weekly else 'days'
    ax.set_xlabel(f'Lag ({unit})  [+ = creator leads Reddit]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


# ── Figure 06a: daily ─────────────────────────────────────────────────────────
# 2×2: top row = creator vs submissions, bottom row = creator vs comments
fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)

plot_tier_xcorr('reach_tier',     axes[0, 0], 'reach',     r_base,  MAX_LAG_DAYS,
                target_label='submissions')
axes[0, 0].set_title('vs Submissions — by reach tier (per-video view count)', fontsize=11)

plot_tier_xcorr('authority_tier', axes[0, 1], 'authority', r_base,  MAX_LAG_DAYS,
                target_label='submissions')
axes[0, 1].set_title('vs Submissions — by authority tier (creator total views)', fontsize=11)

plot_tier_xcorr('reach_tier',     axes[1, 0], 'reach',     cm_base, MAX_LAG_DAYS,
                target_label='comments')
axes[1, 0].set_title('vs Comments — by reach tier (per-video view count)', fontsize=11)

plot_tier_xcorr('authority_tier', axes[1, 1], 'authority', cm_base, MAX_LAG_DAYS,
                target_label='comments')
axes[1, 1].set_title('vs Comments — by authority tier (creator total views)', fontsize=11)

# Figure titles live in the paper captions, not in the image.
plt.tight_layout()

out_a = FIGURES_DIR / '06a_tier_xcorr.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

#%%
# ── Figure 06b: weekly ────────────────────────────────────────────────────────
# DOW effects average out in weekly means; no deseasonalization applied.
# 2×2: top row = creator vs submissions, bottom row = creator vs comments
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11), sharey=True)

plot_tier_xcorr('reach_tier',     axes2[0, 0], 'reach',     r_base_weekly,  MAX_LAG_WEEKS,
                granularity='weekly', weekly=True, target_label='submissions')
axes2[0, 0].set_title('vs Submissions — by reach tier (per-video view count)', fontsize=11)

plot_tier_xcorr('authority_tier', axes2[0, 1], 'authority', r_base_weekly,  MAX_LAG_WEEKS,
                granularity='weekly', weekly=True, target_label='submissions')
axes2[0, 1].set_title('vs Submissions — by authority tier (creator total views)', fontsize=11)

plot_tier_xcorr('reach_tier',     axes2[1, 0], 'reach',     cm_base_weekly, MAX_LAG_WEEKS,
                granularity='weekly', weekly=True, target_label='comments')
axes2[1, 0].set_title('vs Comments — by reach tier (per-video view count)', fontsize=11)

plot_tier_xcorr('authority_tier', axes2[1, 1], 'authority', cm_base_weekly, MAX_LAG_WEEKS,
                granularity='weekly', weekly=True, target_label='comments')
axes2[1, 1].set_title('vs Comments — by authority tier (creator total views)', fontsize=11)

# Figure titles live in the paper captions, not in the image.
plt.tight_layout()

out_b = FIGURES_DIR / '06b_weekly_tier_xcorr.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')
plt.show()

#%%
tier_df = pd.DataFrame(records).sort_values(['granularity', 'split', 'tier'])
print(tier_df.to_string(index=False))
tier_df.to_csv(TABLES_DIR / '06_tier_xcorr_summary.csv', index=False)

# ── Limitation note ───────────────────────────────────────────────────────────
print(
    '\nNote: view_count reflects accumulated views at time of data collection, '
    'not contemporary reach at time of upload. Recent videos are systematically '
    'undervalued by this metric. Flag this in the methods/limitations section.'
)
