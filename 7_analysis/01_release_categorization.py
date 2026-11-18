#%%
"""
01_release_categorization.py
-----------------------------
Computes three continuous per-release metrics relative to each game's own median:

  Revival   = launch-week peak / prior-year mean daily players
  Retention = % of launch-week peak still playing at week W_RETENTION post-launch
  Floor     = stabilised player count (weeks W_FLOOR_START–W_FLOOR_END) / prior-year mean

Baseline definition
  "Prior release year" = all days belonging to the immediately preceding year_name
  group in the releases calendar. For Year N, baseline = mean (and median) of all
  daily player counts in Year N-1 (from first day of that year's first season to
  the day before Year N begins).
  Fallback for early seasons with no prior release year: all available pre-launch
  data.  Cutover: the first season whose prior year has at least MIN_PRIOR_YEAR_DAYS
  days of data switches to the standard method.

Robustness check
  Both mean and median prior-year baselines are computed. If top/bottom-3 revival
  rankings differ between the two, both are reported in the baseline reference table.

Outputs:
  tables/01_release_metrics.csv          — one row per release, all metrics + bucket
  tables/01_baseline_reference.csv       — prior-year mean/median + peak revival per season
  figures/01_release_metrics.png         — scatter of revival vs. retention, sized by floor
  figures/01_raw_player_counts.png       — supplementary: raw daily counts, windows shaded
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_player_counts, load_releases, ensure_output_dirs,
    available_player_count_sources, GAME_OF_FOCUS,
    FIGURES_DIR, TABLES_DIR,
)

# Only D2 has all-platform counts; the rest fall back to Steam-only, so the
# figure must not claim 'all platforms' for them. Resolves to the identical
# string for D2, leaving that figure unchanged.
PC_SCOPE = ('all platforms' if 'total' in available_player_count_sources()
            else 'Steam only')

# ── Parameters ────────────────────────────────────────────────────────────────
LAUNCH_WEEK_DAYS   = 7    # days after launch counted as "launch week"
W_RETENTION        = 6    # weeks post-launch for retention measurement
W_FLOOR_START      = 8    # weeks post-launch to begin floor window
W_FLOOR_END        = 12   # weeks post-launch to end floor window
MIN_PRE_WEEKS      = 4    # minimum pre-release data required to include a release
MIN_PRIOR_YEAR_DAYS = 90  # minimum days of data in prior year to use standard baseline

N_HIGHLIGHT = 3   # top and bottom N for baseline reference table

BUCKET_COLORS = {
    'successful':        '#2ecc71',
    'nuanced':           '#f39c12',
    'failure':           '#e74c3c',
    'insufficient_data': '#cccccc',
}

#%%
ensure_output_dirs()

releases = load_releases()
pc       = load_player_counts(source='total').set_index('date').sort_index()['players']

#%%
# ── Build year-name → date range map ─────────────────────────────────────────
# For each year_name, record [first_season_start, day_before_next_year_start].
year_order = (
    releases.dropna(subset=['start_date'])
    .drop_duplicates('year_name', keep='first')
    .sort_values('start_date')[['year_name', 'start_date']]
    .reset_index(drop=True)
)

year_ranges = {}
for i, row in year_order.iterrows():
    yr   = row['year_name']
    y_start = row['start_date']
    if i + 1 < len(year_order):
        y_end = year_order.loc[i + 1, 'start_date'] - pd.Timedelta(days=1)
    else:
        y_end = pc.index.max()
    year_ranges[yr] = (y_start, y_end)

# Map each year_name to the name of the previous year
year_names_ordered = list(year_order['year_name'])
prior_year_map = {
    yr: year_names_ordered[i - 1] if i > 0 else None
    for i, yr in enumerate(year_names_ordered)
}



def compute_prior_year_baseline(t0: pd.Timestamp, year_name: str):
    """
    Return (mean, median, source) of daily player counts for the prior release year.
    Falls back to all available pre-launch data if no prior year or insufficient days.
    """
    prior_yr = prior_year_map.get(year_name)
    if prior_yr and prior_yr in year_ranges:
        py_start, py_end = year_ranges[prior_yr]
        prior_data = pc.loc[py_start:py_end].dropna()
        if len(prior_data) >= MIN_PRIOR_YEAR_DAYS:
            return prior_data.mean(), prior_data.median(), 'prior_year'

    # Fallback: all data before launch
    fallback = pc.loc[:t0 - pd.Timedelta(days=1)].dropna()
    if len(fallback) < MIN_PRE_WEEKS * 7:
        return np.nan, np.nan, 'insufficient'
    return fallback.mean(), fallback.median(), 'fallback_all_prior'


#%%
rows = []
for _, rel in releases.iterrows():
    t0        = rel['start_date']
    year_name = rel['year_name']
    if pd.isna(t0):
        continue

    bl_mean, bl_median, bl_source = compute_prior_year_baseline(t0, year_name)

    if pd.isna(bl_mean) or bl_mean == 0:
        rows.append({
            'season_number':  rel['season_number'],
            'season_name':    rel['season_name'],
            'year_name':      year_name,
            'start_date':     t0,
            'baseline_mean':  np.nan,
            'baseline_median': np.nan,
            'baseline_source': bl_source,
            'revival':        np.nan,
            'revival_median': np.nan,
            'retention':      np.nan,
            'floor':          np.nan,
        })
        continue

    # Revival
    launch_end       = t0 + pd.Timedelta(days=LAUNCH_WEEK_DAYS - 1)
    launch_week_data = pc.loc[t0:launch_end]
    if launch_week_data.empty:
        rows.append({
            'season_number':  rel['season_number'],
            'season_name':    rel['season_name'],
            'year_name':      year_name,
            'start_date':     t0,
            'baseline_mean':  bl_mean,
            'baseline_median': bl_median,
            'baseline_source': bl_source,
            'revival':        np.nan,
            'revival_median': np.nan,
            'retention':      np.nan,
            'floor':          np.nan,
        })
        continue

    peak          = launch_week_data.max()
    revival_mean  = peak / bl_mean
    revival_med   = peak / bl_median

    # Retention: avg ±3 days around W_RETENTION, as fraction of peak
    ret_date   = t0 + pd.Timedelta(weeks=W_RETENTION)
    ret_window = pc.loc[ret_date - pd.Timedelta(days=3) : ret_date + pd.Timedelta(days=3)]
    retention  = (ret_window.mean() / peak) if (not ret_window.empty and peak > 0) else np.nan

    # Floor: mean of stabilised period, normalised to prior-year mean
    floor_data = pc.loc[
        t0 + pd.Timedelta(weeks=W_FLOOR_START) :
        t0 + pd.Timedelta(weeks=W_FLOOR_END)
    ]
    floor = floor_data.mean() / bl_mean if not floor_data.empty else np.nan

    rows.append({
        'season_number':   rel['season_number'],
        'season_name':     rel['season_name'],
        'year_name':       year_name,
        'start_date':      t0,
        'baseline_mean':   round(bl_mean, 0),
        'baseline_median': round(bl_median, 0),
        'baseline_source': bl_source,
        'revival':         revival_mean,
        'revival_median':  revival_med,
        'retention':       retention,
        'floor':           floor,
    })

df = pd.DataFrame(rows)

standard_seasons = df[df['baseline_source'] == 'prior_year'].sort_values('start_date')
if not standard_seasons.empty:
    print(f'First season using prior-year baseline: {standard_seasons.iloc[0]["season_name"]}')
print(f'Seasons using fallback baseline: '
      f'{(df["baseline_source"] == "fallback_all_prior").sum()}')

#%%
# ── Robustness check: do rankings differ between mean and median baselines? ───
valid = df.dropna(subset=['revival', 'revival_median'])
rank_mean   = valid.set_index('season_name')['revival'].rank(ascending=False)
rank_median = valid.set_index('season_name')['revival_median'].rank(ascending=False)

top3_mean   = set(rank_mean.nsmallest(N_HIGHLIGHT).index)
top3_median = set(rank_median.nsmallest(N_HIGHLIGHT).index)
bot3_mean   = set(rank_mean.nlargest(N_HIGHLIGHT).index)
bot3_median = set(rank_median.nlargest(N_HIGHLIGHT).index)

top_stable = top3_mean == top3_median
bot_stable = bot3_mean == bot3_median
rankings_stable = top_stable and bot_stable
print(f'\nRobustness check — top-{N_HIGHLIGHT} stable: {top_stable}  |  bottom-{N_HIGHLIGHT} stable: {bot_stable}')
if not top_stable:
    print(f'  Top — mean-baseline:   {top3_mean}')
    print(f'  Top — median-baseline: {top3_median}')
if not bot_stable:
    print(f'  Bot — mean-baseline:   {bot3_mean}')
    print(f'  Bot — median-baseline: {bot3_median}')

#%%
# ── Narrative buckets (mean-baseline revival) ─────────────────────────────────
revival_med_val   = df['revival'].median()
retention_med_val = df['retention'].median()

def assign_bucket(row):
    if pd.isna(row['revival']) or pd.isna(row['retention']):
        return 'insufficient_data'
    above_rev = row['revival']   >= revival_med_val
    above_ret = row['retention'] >= retention_med_val
    if above_rev and above_ret:
        return 'successful'
    if not above_rev and not above_ret:
        return 'failure'
    return 'nuanced'

df['bucket'] = df.apply(assign_bucket, axis=1)

df.to_csv(TABLES_DIR / '01_release_metrics.csv', index=False)
print(df[['season_name', 'baseline_source', 'revival', 'revival_median',
          'retention', 'floor', 'bucket']].to_string(index=False))
print(f'\nRevival median (mean-baseline):   {revival_med_val:.3f}')
print(f'Retention median:                 {retention_med_val:.3f}')

#%%
# ── Baseline reference table ──────────────────────────────────────────────────
# Top and bottom N by revival, plus any seasons where mean/median baseline diverge
highlight = set(rank_mean.nsmallest(N_HIGHLIGHT).index) | set(rank_mean.nlargest(N_HIGHLIGHT).index)
ref = (
    df[df['season_name'].isin(highlight)]
    .sort_values('revival', ascending=False)
    [['season_name', 'year_name', 'baseline_source',
      'baseline_mean', 'baseline_median', 'revival', 'revival_median', 'retention']]
)
ref.to_csv(TABLES_DIR / '01_baseline_reference.csv', index=False)
print(f'\nBaseline reference table (top/bottom {N_HIGHLIGHT}):')
print(ref.to_string(index=False))

#%%
# ── Figure 1: revival vs. retention scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))

valid_plot = df.dropna(subset=['revival', 'retention'])
floor_norm = valid_plot['floor'].fillna(valid_plot['floor'].median())
floor_scaled = ((floor_norm / floor_norm.median()).clip(0.2, 5) * 80).clip(40, 400)

for _, row in valid_plot.iterrows():
    color = BUCKET_COLORS.get(row['bucket'], '#cccccc')
    ax.scatter(row['revival'], row['retention'],
               s=floor_scaled.loc[row.name], color=color, alpha=0.8,
               edgecolors='white', linewidths=0.5, zorder=3)
    ax.annotate(row['season_name'], (row['revival'], row['retention']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points', color='#333333')

ax.axvline(revival_med_val,   color='gray', lw=1, linestyle='--', alpha=0.6,
           label=f'Revival median ({revival_med_val:.2f}×)')
ax.axhline(retention_med_val, color='gray', lw=1, linestyle=':',  alpha=0.6,
           label=f'Retention median ({retention_med_val:.1%})')

# Quadrant labels
ax.text(revival_med_val * 1.02, retention_med_val * 1.02, 'Successful',
        color='#2ecc71', fontsize=9, fontstyle='italic', va='bottom')
ax.text(revival_med_val * 0.5,  retention_med_val * 1.02, 'Nuanced\n(held players, low hype)',
        color='#f39c12', fontsize=8, fontstyle='italic', va='bottom', ha='center')
ax.text(revival_med_val * 1.02, retention_med_val * 0.5,  'Nuanced\n(big launch, poor hold)',
        color='#f39c12', fontsize=8, fontstyle='italic', va='center')
ax.text(revival_med_val * 0.5,  retention_med_val * 0.5,  'Failure',
        color='#e74c3c', fontsize=9, fontstyle='italic', va='center', ha='center')

patches = [mpatches.Patch(color=v, label=k.capitalize())
           for k, v in BUCKET_COLORS.items() if k != 'insufficient_data']
ax.legend(handles=patches + ax.get_legend_handles_labels()[0][-2:],
          loc='lower right', fontsize=9)

ax.set_xlabel(
    f'Revival  (launch-week peak / prior-year mean daily players)', fontsize=11)
ax.set_ylabel(
    f'Retention  (% of peak still playing at week {W_RETENTION})', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.set_title(
    'Release categorisation: revival vs. retention\n'
    '(bubble size = floor lift; baseline = prior release-year mean daily players)',
    fontsize=13)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = FIGURES_DIR / '01_release_metrics.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
plt.show()

#%%
# ── Supplementary figure: raw daily player counts with launch windows shaded ──
fig2, ax2 = plt.subplots(figsize=(18, 5))

ax2.plot(pc.index, pc.values, color='#2c3e50', lw=1, alpha=0.85)

# Shade ±1 week around each launch
for _, rel in releases.iterrows():
    t0 = rel['start_date']
    if pd.isna(t0):
        continue
    ax2.axvspan(t0 - pd.Timedelta(weeks=1), t0 + pd.Timedelta(weeks=1),
                alpha=0.15, color='#3498db', zorder=0)
    ax2.axvline(t0, color='#2980b9', lw=0.5, alpha=0.4, zorder=1)

ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel(f'Daily players ({PC_SCOPE})', fontsize=11)
ax2.set_title(
    f'{GAME_OF_FOCUS} — raw daily player counts ({PC_SCOPE})\n'
    'Shaded bands: ±1 week around each season/expansion launch',
    fontsize=13)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1e3:.0f}K'))
ax2.grid(True, alpha=0.2)
plt.tight_layout()

out2 = FIGURES_DIR / '01_raw_player_counts.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
plt.show()
