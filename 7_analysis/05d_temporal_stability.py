#%%
"""
05d_temporal_stability.py
--------------------------
Tests whether the creator→Reddit lead/lag null from script 05 is stable
across expansion eras, or whether it masks opposing relationships in different
periods of the game's lifecycle.

Approach:
  Deseasonalise both series across the full date range (DOW dummies, same method
  as script 05). Then, for each major expansion era (year_number 1–7 in
  releases.csv), subset the deseasonalised series and compute the normalised
  cross-correlation of creator overall__score vs. Reddit overall__score at:
    - daily  ±28 days
    - weekly ±4  weeks  (raw weekly means; DOW effects average out at this scale)

  Records peak lag and peak |r| per era. Each era peak is noise-gated with a
  circular-shift null (1000 perms, 95th-pctile max-|r|; mirrors 04/05b/06) so a
  bare abs().idxmax() peak in a short era window is not mistaken for a real lead.
  If all eras cluster near lag=0 with low |r| (and few clear the gate), the null
  from script 05 is structurally stable across the game's lifecycle. If one era
  shows a strongly positive or negative lag AND clears the gate, that period
  warrants a note in the paper.

Year 8 (Year of Prophecy) is excluded — data too recent and incomplete.

Outputs:
  figures/05d_temporal_stability.png  — dot plot, two panels (daily / weekly)
                                        dot size ∝ |r|; colour = lag direction;
                                        filled = clears noise gate, hollow = ns
  tables/05d_temporal_stability.csv   — era, era_start, era_end, n_days, n_weeks,
                                        daily/weekly peak_lag, peak_corr,
                                        noise_threshold, p_value, significant
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_releases,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily,
    deseasonalize_dow, resample_weekly,
)
from analysis.shared.game_config import get_config

MAX_LAG_DAYS  = 28
MAX_LAG_WEEKS = 4
MIN_OVERLAP   = 60   # minimum overlapping non-NaN days for xcorr to be reliable

#%%
ensure_output_dirs()

yt       = load_youtube_creators()
rd       = load_reddit_submissions()
releases = load_releases()

# Aggregate to daily — both series, unweighted only (primary result from script 05)
creator_agg = aggregate_creator_daily(yt, ['overall__score'])
reddit_agg  = aggregate_reddit_daily(rd,  ['overall__score'])

# Deseasonalise across the FULL date range — DOW coefficients estimated more
# reliably on the full dataset than per-era subsets.
full_dates = pd.date_range(
    max(creator_agg.index.min(), reddit_agg.index.min()),
    min(creator_agg.index.max(), reddit_agg.index.max()),
    freq='D',
)
c_full_ds = deseasonalize_dow(
    creator_agg['overall__score_unweighted'].reindex(full_dates)
)
r_full_ds = deseasonalize_dow(
    reddit_agg['overall__score_unweighted'].reindex(full_dates)
)

# Weekly series (no deseasonalisation needed at weekly granularity)
creator_weekly = resample_weekly(creator_agg)
reddit_weekly  = resample_weekly(reddit_agg)

# ── Expansion era windows ─────────────────────────────────────────────────────
# Destiny 2 caps at year_number 7 (Year 8 was incomplete when this analysis was
# specified); that cap lives in game_config so it does not silently truncate
# games with longer calendars — Warframe and Dota 2 run to 13/14 years and were
# being cut off at 2019 by the old hard-coded `<= 7`.
_ERA_MAX_YEAR = get_config().get('era_max_year_number')
if _ERA_MAX_YEAR is None:
    era_map = releases.copy()
    print(f'Eras: all {releases["year_number"].nunique()} years (no cap configured).')
else:
    era_map = releases[releases['year_number'] <= _ERA_MAX_YEAR].copy()
    print(f'Eras: capped at year_number <= {_ERA_MAX_YEAR} '
          f'({era_map["year_number"].nunique()} of {releases["year_number"].nunique()} years).')
era_windows = (
    era_map.groupby(['year_number', 'year_name'])
    .agg(era_start=('start_date', 'min'), era_end=('end_date', 'max'))
    .reset_index()
    .sort_values('year_number')
    .reset_index(drop=True)
)

# ── Cross-correlation helper ──────────────────────────────────────────────────
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402

N_PERM      = 1000   # circular-shift permutations for the peak noise gate
NULL_PCTILE = 95
RANDOM_SEED = 42


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circularly shift only the non-NaN values, preserving NaN positions."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    if len(vals):
        result[valid] = np.roll(vals, shift % len(vals))
    return result


def gated_peak(x: pd.Series, y: pd.Series, max_lag: int,
               n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> tuple:
    """Global max-|r| peak gated by a circular-shift null (mirrors 04 / 05b / 06).

    A bare abs().idxmax() peak is only real if its |r| beats the 95th percentile
    of the null max-|r| across the lag window (corrects for the multiple-lag
    search). Returns (peak_lag, peak_corr, noise_threshold, p_value, significant);
    lags/corrs are np.nan and significant is False when the xcorr can't be computed.
    """
    cc = normalised_xcorr(x, y, max_lag)
    if cc.empty:
        return np.nan, np.nan, np.nan, np.nan, False
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])
    rng = np.random.default_rng(seed)
    n_valid = int(x.notna().sum())
    null_max = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n_valid, 2)))
        cc_n  = normalised_xcorr(shift_values_only(x, shift), y, max_lag)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
    if not null_max:
        return peak_lag, peak_corr, np.nan, np.nan, False
    null_max = np.asarray(null_max)
    thr = float(np.percentile(null_max, NULL_PCTILE))
    p   = float(np.mean(null_max >= abs(peak_corr)))
    return peak_lag, peak_corr, thr, p, bool(abs(peak_corr) > thr)

#%%
records = []

for _, era in era_windows.iterrows():
    name  = era['year_name']
    start = era['era_start']
    end   = era['era_end']

    # ── Daily xcorr (deseasonalised full-series residuals, era window) ────────
    c_d = c_full_ds.loc[start:end]
    r_d = r_full_ds.loc[start:end]
    n_days = int(pd.concat([c_d, r_d], axis=1).dropna().shape[0])

    if n_days < MIN_OVERLAP:
        print(f'{name}: only {n_days} overlapping days — xcorr skipped (below {MIN_OVERLAP} minimum).')
        daily_lag, daily_corr, daily_thr, daily_p, daily_sig = np.nan, np.nan, np.nan, np.nan, False
    else:
        daily_lag, daily_corr, daily_thr, daily_p, daily_sig = gated_peak(c_d, r_d, MAX_LAG_DAYS)

    # ── Weekly xcorr (raw weekly means, era window) ───────────────────────────
    c_w_mask = (creator_weekly.index >= start) & (creator_weekly.index < end)
    r_w_mask = (reddit_weekly.index  >= start) & (reddit_weekly.index  < end)
    c_w = creator_weekly.loc[c_w_mask, 'overall__score_unweighted']
    r_w = reddit_weekly.loc[r_w_mask,  'overall__score_unweighted']
    n_weeks = int(pd.concat([c_w, r_w], axis=1).dropna().shape[0])

    weekly_lag, weekly_corr, weekly_thr, weekly_p, weekly_sig = gated_peak(c_w, r_w, MAX_LAG_WEEKS)

    d_tag = '' if daily_sig  else ' ns'
    w_tag = '' if weekly_sig else ' ns'
    print(
        f'{str(name):20s}  n_days={n_days:4d}  '
        f'daily_lag={daily_lag:+4}d (r={daily_corr:+.3f}{d_tag})  '
        f'weekly_lag={weekly_lag:+3}w (r={weekly_corr:+.3f}{w_tag})'
    )

    records.append({
        'year_number':     int(era['year_number']),
        'era':             name,
        'era_start':       start.date(),
        'era_end':         end.date(),
        'n_days':          n_days,
        'n_weeks':         n_weeks,
        'daily_peak_lag':  daily_lag,
        'daily_peak_corr': daily_corr,
        'daily_noise_threshold':  daily_thr,
        'daily_p_value':          daily_p,
        'daily_significant':      daily_sig,
        'weekly_peak_lag': weekly_lag,
        'weekly_peak_corr': weekly_corr,
        'weekly_noise_threshold': weekly_thr,
        'weekly_p_value':         weekly_p,
        'weekly_significant':     weekly_sig,
    })

df_out = pd.DataFrame(records)
df_out.to_csv(TABLES_DIR / '05d_temporal_stability.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "05d_temporal_stability.csv"}')

#%%
# ── Figure: dot plot, daily and weekly panels ─────────────────────────────────
def dot_colors(lags):
    return ['#e74c3c' if (not np.isnan(l) and l > 0)
            else '#3498db' if (not np.isnan(l) and l < 0)
            else '#95a5a6' for l in lags]

def dot_sizes(corrs, base=80, scale=400):
    return [base + abs(c) * scale if not np.isnan(c) else base for c in corrs]

era_labels = [f"Y{r['year_number']}\n{r['era']}" for _, r in df_out.iterrows()]
x = np.arange(len(df_out))

fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(15, 5))

# ── Daily panel ───────────────────────────────────────────────────────────────
d_lags  = df_out['daily_peak_lag'].values
d_corrs = df_out['daily_peak_corr'].values
d_sigs  = df_out['daily_significant'].values

# Filled = peak clears the circular-shift noise gate; hollow = within noise (ns).
d_edge = dot_colors(d_lags)
d_face = [col if s else 'none' for col, s in zip(d_edge, d_sigs)]
ax_d.scatter(x, d_lags, facecolors=d_face, edgecolors=d_edge,
             s=dot_sizes(d_corrs), linewidths=1.6, zorder=3)
ax_d.axhline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax_d.set_xticks(x)
ax_d.set_xticklabels(era_labels, fontsize=8)
ax_d.set_ylabel('Peak lag (days)  [+ = creator leads Reddit]', fontsize=10)
ax_d.set_title('Daily ±28 days\n(deseasonalised)', fontsize=11)
ax_d.grid(True, axis='y', alpha=0.25)
for i, (lag, corr) in enumerate(zip(d_lags, d_corrs)):
    if not np.isnan(lag):
        ax_d.annotate(f'r={corr:+.2f}', (i, lag),
                      textcoords='offset points', xytext=(0, 9),
                      ha='center', fontsize=7.5)

# ── Weekly panel ──────────────────────────────────────────────────────────────
w_lags  = df_out['weekly_peak_lag'].values
w_corrs = df_out['weekly_peak_corr'].values
w_sigs  = df_out['weekly_significant'].values

w_edge = dot_colors(w_lags)
w_face = [col if s else 'none' for col, s in zip(w_edge, w_sigs)]
ax_w.scatter(x, w_lags, facecolors=w_face, edgecolors=w_edge,
             s=dot_sizes(w_corrs), linewidths=1.6, zorder=3)
ax_w.axhline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax_w.set_xticks(x)
ax_w.set_xticklabels(era_labels, fontsize=8)
ax_w.set_ylabel('Peak lag (weeks)  [+ = creator leads Reddit]', fontsize=10)
ax_w.set_title('Weekly ±4 weeks\n(raw weekly means)', fontsize=11)
ax_w.grid(True, axis='y', alpha=0.25)
for i, (lag, corr) in enumerate(zip(w_lags, w_corrs)):
    if not np.isnan(lag):
        ax_w.annotate(f'r={corr:+.2f}', (i, lag),
                      textcoords='offset points', xytext=(0, 9),
                      ha='center', fontsize=7.5)

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=9, label='Creator leads Reddit (lag > 0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
           markersize=9, label='Reddit leads creator (lag < 0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
           markersize=9, label='Concurrent (lag = 0)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
           markersize=9, label='Filled = clears noise gate'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#555555', markersize=9, label='Hollow = within noise (ns)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    'Temporal stability: creator → Reddit overall__score peak lag by expansion era\n'
    '(unweighted creator, unweighted Reddit; dot size ∝ |r|; '
    'filled = clears circular-shift noise gate, hollow = within noise)',
    fontsize=12,
)
plt.tight_layout()
out = FIGURES_DIR / '05d_temporal_stability.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
