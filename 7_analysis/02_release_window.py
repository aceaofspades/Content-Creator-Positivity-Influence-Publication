#%%
"""
02_release_window.py
--------------------
T-6 to T+6 week event study around each Destiny 2 season/expansion launch.

Run 01_release_categorization.py first — revival metrics are re-derived here
using the corrected metric, so ranking may differ from that script's output.

Metric: log ratio — log(P_t / baseline), displayed as multiplicative index
  (baseline = 1.0, doubling = 2.0, halving = 0.5) on a log y-axis.
  Symmetric, additive over time, standard for event studies.

Baseline: 6-week mean centered on the same calendar date ONE YEAR PRIOR.
  Avoids the end-of-cycle trough that contaminates T-6→T-1 baselines.
  Seasons without 1-year lookback data (too early in D2's history) fall back
  to T-10→T-7 weeks (mid-previous-season, well before the dying tail).

Sensitivity analysis: varies baseline window width (2/4/6/8 wk) and source
  (year-prior vs. pre-launch) and reports Spearman rank correlation of revival
  rankings across all variants — establishes that top/bottom seasons are stable.

Outputs:
  figures/02a_release_window_expansions.png
  figures/02b_release_window_top_bottom.png
  tables/02_sensitivity_rank_correlation.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_player_counts, load_releases, load_youtube_creators,
    load_reddit_submissions, ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import aggregate_creator_daily, aggregate_reddit_daily

# ── Parameters ────────────────────────────────────────────────────────────────
WEEKS_BEFORE    = 6
WEEKS_AFTER     = 6
MIN_PRE_WEEKS   = 4
SENTIMENT_FIELD = 'overall__score'
N_LABEL         = 3   # top and bottom N to highlight in Figure 2

# Primary baseline: 6-week window centred on same date 1 year prior
BASELINE_WEEKS_PRIMARY = 6

# Fallback baseline (used when year-prior data unavailable):
# T-10 to T-7 weeks — mid-previous-season, well clear of the dying tail
FALLBACK_START_WEEKS = 10
FALLBACK_END_WEEKS   = 7

ALPHA_FOCUS = 0.90
ALPHA_FADE  = 0.12
LW_FOCUS    = 1.8
LW_FADE     = 0.7

# Display only: D2 seasons whose year-prior baseline is not comparable with the
# others (Undying's predates the 2019-10-01 Steam/free-to-play move; Outlaw's falls
# on the launch weeks). Drawn dotted on the player panel; metrics are unchanged.
DOTTED_BASELINE_SEASONS = {'Season of the Undying', 'Season of the Outlaw'}

METRICS_CSV          = TABLES_DIR / '01_release_metrics.csv'
METRICS_CSV_FALLBACK = TABLES_DIR / '02_release_metrics.csv'

#%%
ensure_output_dirs()

releases = load_releases()
pc       = load_player_counts(source='total').set_index('date').sort_index()['players']
yt       = load_youtube_creators()
rd       = load_reddit_submissions()

expansion_seasons = set(
    releases.dropna(subset=['start_date'])
    .groupby('year_name', sort=False)['season_name']
    .first()
)

creator_agg  = aggregate_creator_daily(yt, [SENTIMENT_FIELD])
reddit_agg   = aggregate_reddit_daily(rd, [SENTIMENT_FIELD])
creator_sent = creator_agg[f'{SENTIMENT_FIELD}_unweighted']
reddit_sent  = reddit_agg[f'{SENTIMENT_FIELD}_unweighted']

#%%
def compute_baseline(t0: pd.Timestamp, weeks: int, use_year_prior: bool) -> float:
    """
    Compute the reference baseline player count for a given launch date.

    Year-prior mode: 6-week window centred on t0 - 1 year.
    Fallback mode: T-FALLBACK_START_WEEKS to T-FALLBACK_END_WEEKS.
    Returns NaN if insufficient data.
    """
    if use_year_prior:
        half   = pd.Timedelta(weeks=weeks // 2)
        centre = t0 - pd.DateOffset(years=1)
        window = pc.loc[centre - half : centre + half]
    else:
        window = pc.loc[
            t0 - pd.Timedelta(weeks=FALLBACK_START_WEEKS) :
            t0 - pd.Timedelta(weeks=FALLBACK_END_WEEKS)
        ]
    return window.mean() if window.notna().sum() >= 7 else np.nan


def build_log_index(t0: pd.Timestamp, baseline: float) -> pd.Series:
    """
    Build the T-WEEKS_BEFORE to T+WEEKS_AFTER window as a multiplicative
    index on a log scale: baseline = 1.0, doubling = 2.0, halving = 0.5.
    """
    start  = t0 - pd.Timedelta(weeks=WEEKS_BEFORE)
    end    = t0 + pd.Timedelta(weeks=WEEKS_AFTER)
    window = pc.reindex(pd.date_range(start, end, freq='D'))
    idx    = window / baseline           # multiplicative index
    idx.index = (idx.index - t0).days
    return idx


def build_sentiment_window(t0: pd.Timestamp, series: pd.Series) -> pd.Series:
    start  = t0 - pd.Timedelta(weeks=WEEKS_BEFORE)
    end    = t0 + pd.Timedelta(weeks=WEEKS_AFTER)
    window = series.reindex(pd.date_range(start, end, freq='D'))
    window.index = (window.index - t0).days
    return window.rolling(7, center=True, min_periods=3).mean()


#%%
# ── Build windows with corrected metric ──────────────────────────────────────
windows = []
for _, row in releases.iterrows():
    t0   = row['start_date']
    name = row['season_name']
    if pd.isna(t0):
        continue

    # Try year-prior baseline first; fall back if insufficient data
    baseline = compute_baseline(t0, BASELINE_WEEKS_PRIMARY, use_year_prior=True)
    baseline_source = 'year_prior'
    if pd.isna(baseline):
        baseline = compute_baseline(t0, BASELINE_WEEKS_PRIMARY, use_year_prior=False)
        baseline_source = 'fallback_T-10_to_T-7'
    if pd.isna(baseline) or baseline == 0:
        continue

    log_idx = build_log_index(t0, baseline)
    pre_days = log_idx.loc[log_idx.index < 0].dropna()
    if len(pre_days) < MIN_PRE_WEEKS * 7:
        continue

    # Revival = peak multiplicative index in launch week (days 0-6)
    launch_week = log_idx.loc[(log_idx.index >= 0) & (log_idx.index <= 6)]
    revival = launch_week.max() if not launch_week.dropna().empty else np.nan

    windows.append({
        'name':            name,
        'year_name':       row['year_name'],
        'is_expansion':    name in expansion_seasons,
        'revival':         revival,
        'baseline_source': baseline_source,
        'log_idx':         log_idx,
        'c_win':           build_sentiment_window(t0, creator_sent),
        'r_win':           build_sentiment_window(t0, reddit_sent),
    })

print(f'{len(windows)} seasons  |  '
      f'{sum(w["is_expansion"] for w in windows)} expansion launches  |  '
      f'{sum(w["baseline_source"] == "fallback_T-10_to_T-7" for w in windows)} used fallback baseline')

#%%
# ── Per-season color palette ──────────────────────────────────────────────────
tab20  = [plt.cm.tab20(i / 20)  for i in range(20)]
tab20b = [plt.cm.tab20b(i / 20) for i in range(20)]
palette = tab20 + tab20b
# Cycle the palette: games with longer season calendars than Destiny 2 (Dota 2 has
# 52) would otherwise run off the end of the 40-colour list. Only the highlighted
# seasons carry a legend, so repeated colours among the faded background lines
# are not ambiguous.
season_color = {w['name']: palette[i % len(palette)] for i, w in enumerate(windows)}

#%%
def _draw_figure(focused_names: set, title: str, out_path: Path) -> None:
    fig, (ax_pc, ax_sent) = plt.subplots(1, 2, figsize=(18, 7))

    # Background lines — uniform gray, no legend entry
    for w in windows:
        if w['name'] in focused_names:
            continue
        ax_pc.plot(w['log_idx'].index, w['log_idx'].values,
                   color='#999999', lw=LW_FADE, alpha=ALPHA_FADE, zorder=1)
        ax_sent.plot(w['c_win'].index, w['c_win'].values,
                     color='#999999', lw=LW_FADE, alpha=ALPHA_FADE, zorder=1)
        ax_sent.plot(w['r_win'].index, w['r_win'].values,
                     color='#999999', lw=LW_FADE * 0.7, alpha=ALPHA_FADE * 0.7,
                     zorder=1, linestyle='dashed')

    # Focused lines — unique color, appear in legend
    legend_handles = []
    for w in windows:
        if w['name'] not in focused_names:
            continue
        color = season_color[w['name']]
        dotted = w['name'] in DOTTED_BASELINE_SEASONS
        ax_pc.plot(w['log_idx'].index, w['log_idx'].values,
                   color=color, lw=LW_FOCUS, alpha=ALPHA_FOCUS, zorder=3,
                   linestyle=':' if dotted else '-')
        ax_sent.plot(w['c_win'].index, w['c_win'].values,
                     color=color, lw=LW_FOCUS, alpha=ALPHA_FOCUS, zorder=3)
        ax_sent.plot(w['r_win'].index, w['r_win'].values,
                     color=color, lw=LW_FOCUS * 0.7, alpha=ALPHA_FOCUS * 0.7,
                     zorder=3, linestyle='dashed')
        rev_str = f'  ({w["revival"]:.2f}×)' if not pd.isna(w['revival']) else ''
        legend_handles.append(
            Line2D([0], [0], color=color, lw=LW_FOCUS,
                   linestyle=':' if dotted else '-',
                   label=f'{w["name"]}{rev_str}'
                         + (', baseline not comparable' if dotted else ''))
        )

    legend_handles += [
        Line2D([0], [0], color='#999999', lw=LW_FADE, alpha=ALPHA_FADE * 3,
               label='Other seasons (context)'),
        Line2D([0], [0], color='black', lw=1.4, linestyle='--', alpha=0.7,
               label='T = 0  (launch day)'),
    ]
    # The line-style key applies only to the sentiment panel.
    style_key = Line2D([0], [0], color='black', lw=0, alpha=0,
                       label='Solid = creators  |  dashed = Reddit')

    for ax in (ax_pc, ax_sent):
        ax.axvline(0, color='black', lw=1.4, linestyle='--', alpha=0.7)
        ax.axhline(1, color='gray', lw=0.7, linestyle=':')   # baseline = 1.0
        ax.set_xlabel('Days relative to season launch')
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_handles + ([style_key] if ax is ax_sent else []),
                  loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0,
                  fontsize=9, frameon=True)

    # Log y-axis: explicit ticks at meaningful multipliers across the data range
    ax_pc.set_yscale('log')
    yticks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    ax_pc.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax_pc.yaxis.set_minor_locator(ticker.NullLocator())
    ax_pc.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: '1× (baseline)' if y == 1.0 else f'{y:g}×'
    ))
    ax_pc.set_title('Player count: multiplicative index vs. year-prior baseline\n'
                    '(log scale — 2.0× = doubled, 0.5× = halved)', fontsize=11)
    ax_pc.set_ylabel('Player count relative to baseline (log scale)')

    ax_sent.set_title('Overall sentiment (7-day rolling mean)', fontsize=11)
    ax_sent.set_ylabel('Overall score (−2 to +2)')
    ax_sent.set_ylim(-2.2, 2.2)

    # Figure titles live in the paper captions, not in the image.
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


#%%
# ── Figure 1: expansion launches ─────────────────────────────────────────────
expansion_names = {w['name'] for w in windows if w['is_expansion']}
_draw_figure(
    focused_names=expansion_names,
    title='Release-window event study — expansion launches highlighted\n'
          '(baseline = 6-week mean, same calendar period 1 year prior)',
    out_path=FIGURES_DIR / '02a_release_window_expansions.png',
)

#%%
# ── Figure 2: successful / failure seasons from 01_release_metrics.csv ───────
# Buckets are assigned by 01_release_categorization.py using revival AND retention
# vs. their game-wide medians, so the selection here is consistent with that script.
metrics_path = METRICS_CSV if METRICS_CSV.exists() else METRICS_CSV_FALLBACK
if not metrics_path.exists():
    raise FileNotFoundError(
        'Run 01_release_categorization.py first to generate release metrics.')

metrics_df  = pd.read_csv(metrics_path)
top_names   = set(metrics_df.loc[metrics_df['bucket'] == 'successful', 'season_name'])
bottom_names = set(metrics_df.loc[metrics_df['bucket'] == 'failure',   'season_name'])

# Restrict to seasons that have event-study windows
windowed = {w['name'] for w in windows}
top_names    &= windowed
bottom_names &= windowed

print(f'\nSuccessful seasons (revival >= median AND retention >= median):')
for name in sorted(top_names):
    row = metrics_df[metrics_df['season_name'] == name].iloc[0]
    print(f'  {name:40s}  revival = {row["revival"]:.3f}×  retention = {row["retention"]:.1%}')
print(f'Failure seasons (revival < median AND retention < median):')
for name in sorted(bottom_names):
    row = metrics_df[metrics_df['season_name'] == name].iloc[0]
    print(f'  {name:40s}  revival = {row["revival"]:.3f}×  retention = {row["retention"]:.1%}')

_draw_figure(
    focused_names=top_names | bottom_names,
    title='Release-window event study — successful vs. failure seasons highlighted\n'
          '(buckets: revival AND retention vs. game-wide medians)',
    out_path=FIGURES_DIR / '02b_release_window_top_bottom.png',
)

#%%
# ── Sensitivity analysis: rank stability across baseline variants ─────────────
# Vary baseline source (year-prior vs. pre-launch) and window width (2/4/6/8 wk).
# Report Spearman rank correlation of revival rankings vs. primary specification.
print('\nBaseline sensitivity analysis...')

def revival_under_spec(weeks: int, use_year_prior: bool) -> dict:
    out = {}
    for _, row in releases.iterrows():
        t0   = row['start_date']
        name = row['season_name']
        if pd.isna(t0):
            continue
        b = compute_baseline(t0, weeks, use_year_prior=use_year_prior)
        if pd.isna(b) or b == 0:
            # For pre-launch specs, also try the pre-launch window directly
            if not use_year_prior:
                half = weeks // 2
                b = pc.loc[t0 - pd.Timedelta(weeks=weeks) : t0 - pd.Timedelta(weeks=1)].mean()
            if pd.isna(b) or b == 0:
                continue
        lw_start = t0
        lw_end   = t0 + pd.Timedelta(days=6)
        peak = pc.loc[lw_start:lw_end].max()
        if pd.notna(peak) and b > 0:
            out[name] = peak / b
    return out

primary = {w['name']: w['revival'] for w in windows if not pd.isna(w['revival'])}
primary_series = pd.Series(primary)

sensitivity_records = []
for weeks in [2, 4, 6, 8]:
    for use_yp, label in [(True, 'year_prior'), (False, 'pre_launch')]:
        spec_label = f'{label}_{weeks}wk'
        alt = pd.Series(revival_under_spec(weeks, use_yp))
        common = primary_series.index.intersection(alt.index)
        if len(common) < 5:
            continue
        rho, p = spearmanr(primary_series.loc[common], alt.loc[common])
        sensitivity_records.append({
            'spec':         spec_label,
            'baseline_src': label,
            'window_weeks': weeks,
            'n_seasons':    len(common),
            'spearman_rho': round(rho, 3),
            'p_value':      round(p, 4),
        })

sens_df = pd.DataFrame(sensitivity_records)
print(sens_df.to_string(index=False))
sens_df.to_csv(TABLES_DIR / '02_sensitivity_rank_correlation.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "02_sensitivity_rank_correlation.csv"}')
print('High Spearman rho (>0.8) across all specs = rankings are robust to baseline choice.')
