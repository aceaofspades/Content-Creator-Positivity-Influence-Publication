#%%
"""
05g_patch_split.py
-------------------
Tests whether the negative-lag cluster on Sandbox & Balance and
Progression & Economy fields (Reddit leads creators by ~22–26 days in
script 05) is concentrated in patch-adjacent periods.

Mechanism being tested: patch notes trigger immediate Reddit reactions;
creators synthesise and respond weeks later. If this production-lag
mechanism drives the cluster, it should be present near season launches
(the largest balance patches) and absent outside them.

Patch proxy: season start dates from the releases calendar. Mid-season
hotfixes are not individually tracked — the season-start proxy captures
the largest events, making this test conservative (contamination from
untracked mid-season patches would work against finding a difference).

Two window widths are pre-specified and both reported:
  narrow: ±21 days of each season start date
  wide:   ±28 days of each season start date
Both are wider than the ~22-26 day cluster on purpose: a window narrower than the
lag being tested cannot contain that lag, so the patch arm would be blind to the
very effect it is meant to detect (see Method note).

Method note (bug fix): the cross-correlation is computed date-aware and WITHIN
contiguous kept blocks. The old version NaN-ed off-subset days, dropped them, and
correlated by array position — which pairs values across the removed gaps as if
they were adjacent, corrupting the lag axis. within_block_xcorr instead shifts on
the full daily index and only pairs creator_t with reddit_{t+h} when both fall in
the same contiguous kept block, so 'lag h' always means h real days.

Interpretation guide:
  If cluster disappears in non-patch weeks → production-lag mechanism supported.
  If cluster persists in non-patch weeks → alternative explanation needed.
  If neither subset shows the cluster → pooled result may be a seasonal artifact.

Outputs:
  figures/05g_patch_split.png   — 2×2 grid: category lag distributions,
                                   rows = window width, cols = patch vs non-patch
  tables/05g_patch_split.csv    — per-category peak lag stats by subset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_releases,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, deseasonalize_dow,
)
from analysis.shared.parsers import TOPIC_CATEGORIES

MAX_LAG_DAYS     = 28
MIN_OVERLAP      = 30   # reduced — patch-adjacent subsets are sparse
MIN_PAIRS        = 30   # minimum within-block aligned pairs to report a lag
# Patch windows must be WIDER than the negative-lag cluster (~22-26 days), or the
# patch arm structurally cannot contain that lag (you can't measure a 22-day lag
# inside a 15-day block). With seasons ~91 days apart, both widths below keep the
# patch AND non-patch blocks wide enough to reach the cluster.
NARROW_DAYS      = 21
WIDE_DAYS        = 28
TARGET_CATS      = {'Sandbox & Balance', 'Progression & Economy'}

# Noise gate: a 'mean lag' is meaningless without the |r| behind it. Report per-cell
# magnitude (mean |peak r|, mean |contemporaneous r|) and a within-block circular-
# shift significance test, so a cluster built on ~zero correlations is revealed as
# noise rather than dressed up as a patch effect.
N_PERM_GATE      = 200
NULL_PCTILE      = 95
RANDOM_SEED      = 42

TOPIC_CATEGORY = {f'topic__{k}': v for k, v in TOPIC_CATEGORIES.items()}

#%%
ensure_output_dirs()

yt       = load_youtube_creators()
rd       = load_reddit_submissions()
releases = load_releases()
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
def build_patch_mask(dates: pd.DatetimeIndex, releases: pd.DataFrame,
                     window_days: int) -> pd.Series:
    """True for dates within window_days of any season start."""
    mask = pd.Series(False, index=dates)
    for start_dt in releases['start_date'].dropna():
        delta = (dates - start_dt).days
        mask |= (np.abs(delta) <= window_days)
    return mask


patch_mask_narrow = build_patch_mask(dates, releases, NARROW_DAYS)
patch_mask_wide   = build_patch_mask(dates, releases, WIDE_DAYS)

n_narrow_patch = patch_mask_narrow.sum()
n_wide_patch   = patch_mask_wide.sum()
print(f'Patch-adjacent days: narrow={n_narrow_patch} ({n_narrow_patch/len(dates):.1%}), '
      f'wide={n_wide_patch} ({n_wide_patch/len(dates):.1%})')

#%%
def _prep_subset(x: pd.Series, y: pd.Series, keep: pd.Series):
    """Numpy arrays on the full daily index: x/y NaN off-subset; integer block id
    (-1 off-subset; equal ids == same contiguous kept run)."""
    block = (keep != keep.shift()).cumsum()
    block = block.where(keep, -1).to_numpy()
    xv = x.where(keep).to_numpy(dtype=float)
    yv = y.where(keep).to_numpy(dtype=float)
    return xv, yv, block


def _block_xcorr_arr(xv, yv, block, max_lag, min_pairs):
    """Date-aware within-block Pearson xcorr on arrays -> {lag: r}. +lag = x leads y.
    A pair (t, t+h) counts only when both days are kept, both have data, and they
    sit in the SAME contiguous kept block — so no pair straddles a removed gap and
    'lag h' always means h real days."""
    n = len(xv)
    out = {}
    for h in range(-max_lag, max_lag + 1):
        if h >= 0:
            a, b, ba, bb = xv[:n - h], yv[h:], block[:n - h], block[h:]
        else:
            a, b, ba, bb = xv[-h:], yv[:n + h], block[-h:], block[:n + h]
        m = (ba == bb) & (ba >= 0) & ~np.isnan(a) & ~np.isnan(b)
        if m.sum() >= min_pairs:
            aa, bb2 = a[m], b[m]
            sa, sb = aa.std(), bb2.std()
            if sa > 0 and sb > 0:
                out[h] = float(((aa - aa.mean()) * (bb2 - bb2.mean())).mean() / (sa * sb))
    return out


def within_block_xcorr(x: pd.Series, y: pd.Series, keep: pd.Series,
                       max_lag: int = MAX_LAG_DAYS, min_pairs: int = MIN_PAIRS) -> pd.Series:
    """Date-aware within-block cross-correlation (see _block_xcorr_arr). Fixes the
    gap-collapsing bug; a block narrower than |h| simply yields no value at lag h."""
    xv, yv, block = _prep_subset(x, y, keep)
    return pd.Series(_block_xcorr_arr(xv, yv, block, max_lag, min_pairs), dtype=float)


def gated_within_block(x: pd.Series, y: pd.Series, keep: pd.Series,
                       max_lag: int = MAX_LAG_DAYS, min_pairs: int = MIN_PAIRS,
                       n_perm: int = N_PERM_GATE, seed: int = RANDOM_SEED):
    """
    within_block_xcorr + the magnitude and significance behind the peak lag, so a
    'mean lag' is never read without the |r| supporting it.

    Noise gate: circular-shift the kept creator values (destroys creator->Reddit
    timing, preserves the creator marginal), recompute the within-block max |r|
    n_perm times -> null. The peak is 'significant' only if |peak r| exceeds the
    NULL_PCTILE of that null. Returns peak_lag, peak_corr, corr_at_0, p_value,
    significant — or None if no lag has enough within-block pairs.
    """
    xv, yv, block = _prep_subset(x, y, keep)
    cc = _block_xcorr_arr(xv, yv, block, max_lag, min_pairs)
    if not cc:
        return None
    peak_lag  = max(cc, key=lambda h: abs(cc[h]))
    peak_corr = cc[peak_lag]

    kept = np.where(block >= 0)[0]
    vals = xv[kept]
    rng  = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        xv2 = xv.copy()
        xv2[kept] = np.roll(vals, int(rng.integers(1, max(len(vals), 2))))
        cc2 = _block_xcorr_arr(xv2, yv, block, max_lag, min_pairs)
        if cc2:
            null.append(max(abs(v) for v in cc2.values()))
    null = np.asarray(null)
    if null.size:
        thr = float(np.percentile(null, NULL_PCTILE))
        p   = float(np.mean(null >= abs(peak_corr)))
        sig = bool(abs(peak_corr) > thr)
    else:
        thr, p, sig = np.nan, np.nan, False
    return {
        'peak_lag':    peak_lag,
        'peak_corr':   round(peak_corr, 4),
        'corr_at_0':   round(cc[0], 4) if 0 in cc else np.nan,
        'noise_thr':   round(thr, 4) if not np.isnan(thr) else np.nan,
        'p_value':     round(p, 4) if not np.isnan(p) else np.nan,
        'significant': sig,
    }


#%%
# Deseasonalise all fields once
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
# Fields in the target categories
target_fields = [
    f for f in topic_cols
    if TOPIC_CATEGORY.get(f'topic__{f.replace("topic__", "")}', '') in TARGET_CATS
    or TOPIC_CATEGORY.get(f, '') in TARGET_CATS
]
# normalise keys
target_fields = [f for f in all_fields
                 if TOPIC_CATEGORY.get(f, '') in TARGET_CATS]

print(f'\nTarget fields ({len(target_fields)}):')
for f in target_fields:
    print(f'  {f}  [{TOPIC_CATEGORY.get(f, "?")}]')

#%%
# Run splits
subsets = {
    ('narrow', 'patch'):     patch_mask_narrow,
    ('narrow', 'non_patch'): ~patch_mask_narrow,
    ('wide',   'patch'):     patch_mask_wide,
    ('wide',   'non_patch'): ~patch_mask_wide,
}

records = []

for (window_label, subset_label), mask in subsets.items():
    n_days = mask.sum()
    for field in all_fields:
        if field not in creator_deseas or field not in reddit_deseas:
            continue
        cat = TOPIC_CATEGORY.get(field, 'Overall' if field == 'overall__score' else 'Other')
        both = pd.concat([creator_deseas[field], reddit_deseas[field]], axis=1)
        n_overlap = int(both[mask].dropna().shape[0])
        if n_overlap < MIN_OVERLAP:          # skip sparse fields before the gate
            continue
        res = gated_within_block(creator_deseas[field], reddit_deseas[field], mask)
        if res is None:
            continue
        c0 = res['corr_at_0']
        records.append({
            'window':      window_label,
            'subset':      subset_label,
            'field':       field,
            'category':    cat,
            'peak_lag':    res['peak_lag'],
            'peak_corr':   res['peak_corr'],
            'abs_peak_r':  abs(res['peak_corr']),
            'corr_at_0':   c0,
            'abs_corr0':   abs(c0) if c0 == c0 else np.nan,   # c0==c0 is False for NaN
            'p_value':     res['p_value'],
            'significant': res['significant'],
            'n_overlap':   n_overlap,
        })

results_df = pd.DataFrame(records)

#%%
# Category-level summary — now with the magnitude and significance behind the lag
cat_summary = (
    results_df.groupby(['window', 'subset', 'category'])
    .agg(mean_lag=('peak_lag', 'mean'),
         median_lag=('peak_lag', 'median'),
         mean_abs_r=('abs_peak_r', 'mean'),
         mean_abs_r0=('abs_corr0', 'mean'),
         n_fields=('field', 'count'),
         n_sig=('significant', 'sum'))
    .round(3)
    .reset_index()
)

out_tbl = TABLES_DIR / '05g_patch_split.csv'
cat_summary.to_csv(out_tbl, index=False)
print(f'Saved: {out_tbl}')
print('\nCategory-level summary (lag + |r| + significance):')
print(cat_summary.to_string(index=False))

#%%
# ── Cell-level magnitude/significance for the cluster (target categories) ─────
# The decisive question: is there a real correlation behind the lags in EITHER
# column? If |r| sits at the noise floor in both patch and non-patch — and few/no
# fields clear the gate — the cluster is noise, a cleaner conclusion than "patches
# explain it". mean_abs_r is selection-inflated (max over the window); mean_abs_r0
# is the honest contemporaneous magnitude; n_sig is how many of n_fields beat the
# within-block circular-shift null.
tgt = results_df[results_df['category'].isin(TARGET_CATS)]
cell = (
    tgt.groupby(['window', 'subset'])
    .agg(n_fields=('field', 'count'),
         n_sig=('significant', 'sum'),
         mean_n_overlap=('n_overlap', 'mean'),
         mean_lag=('peak_lag', 'mean'),
         mean_abs_r=('abs_peak_r', 'mean'),
         mean_abs_r0=('abs_corr0', 'mean'))
    .round(3)
    .reset_index()
)
cell['n_days'] = [int(subsets[(w, s)].sum()) for w, s in zip(cell['window'], cell['subset'])]
cell = cell[['window', 'subset', 'n_days', 'n_fields', 'n_sig',
             'mean_n_overlap', 'mean_lag', 'mean_abs_r', 'mean_abs_r0']]
out_cell = TABLES_DIR / '05g_cluster_cell_summary.csv'
cell.to_csv(out_cell, index=False)
print(f'\nSaved: {out_cell}')
print('\nCluster cells (Sandbox & Balance + Progression & Economy) -- '
      'is there |r| behind the lag?')
print(cell.to_string(index=False))

#%%
# ── Figure: 2×2 grid of category boxplots ────────────────────────────────────
window_labels  = ['narrow', 'wide']
subset_labels  = ['patch', 'non_patch']
panel_titles   = {
    ('narrow', 'patch'):     f'Narrow (±{NARROW_DAYS}d): patch-adjacent',
    ('narrow', 'non_patch'): f'Narrow (±{NARROW_DAYS}d): non-patch',
    ('wide',   'patch'):     f'Wide (±{WIDE_DAYS}d): patch-adjacent',
    ('wide',   'non_patch'): f'Wide (±{WIDE_DAYS}d): non-patch',
}

all_cats = sorted(results_df['category'].unique())

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

for row_i, wl in enumerate(window_labels):
    for col_i, sl in enumerate(subset_labels):
        ax = axes[row_i][col_i]
        sub = results_df[(results_df['window'] == wl) & (results_df['subset'] == sl)]

        cats_present = [c for c in all_cats if c in sub['category'].values]
        data = [sub[sub['category'] == c]['peak_lag'].values for c in cats_present]
        colors_by_cat = ['#e74c3c' if c in TARGET_CATS else '#95a5a6' for c in cats_present]

        if data:
            bp = ax.boxplot(data, vert=False, patch_artist=True,
                            medianprops=dict(color='black', lw=2))
            for patch, col in zip(bp['boxes'], colors_by_cat):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)
            ax.set_yticks(range(1, len(cats_present) + 1))
            ax.set_yticklabels(cats_present, fontsize=8)

        ax.axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.6)
        ax.set_xlabel('Peak lag (days)  [+ = creator leads]', fontsize=9)
        ax.set_title(panel_titles[(wl, sl)], fontsize=10)
        ax.grid(True, alpha=0.25, axis='x')

        n_days_sub = subsets[(wl, sl)].sum()
        ax.text(0.98, 0.02, f'n_days={n_days_sub}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='gray')

fig.suptitle(
    'Creator→Reddit peak lag: patch-adjacent vs. non-patch periods\n'
    '(red = Sandbox & Balance / Progression & Economy — the negative-lag cluster)',
    fontsize=12,
)
plt.tight_layout()
out_fig = FIGURES_DIR / '05g_patch_split.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')
plt.show()
