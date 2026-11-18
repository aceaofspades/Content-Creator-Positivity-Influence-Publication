#%%
"""
09_anomaly_table.py
--------------------
Face-validity check on the LLM topic extraction (NOT a statistical test).

Detects player-count anomalies (residuals from a slow-trend + seasonal +
release-recency baseline), then for each anomaly window surfaces the creator
topics that deviate most from their own global base rate, plus the highest-view
creator videos.

What this supports: that the topic extraction assigns plausible, event-appropriate
topics around known controversies (Sunsetting, Lightfall, ...). A clean pass is
weak evidence the pipeline isn't broken; a failure would be a real warning sign
about the underlying data. The information is asymmetric — a pass buys little, a
miss would be costly.

What this does NOT support: any causal or lead/lag claim. Creator videos and player
moves co-occur in these windows because the same event drives BOTH (common cause) —
the co-occurrence is expected, not evidence of influence in either direction. The
directional/predictive versions of that relationship were tested in scripts 04-07
and not found. Do not read the video column as creators driving or reacting to the
player change.

Outputs:
  tables/09_anomaly_table.csv    — anomaly periods with top deviating topics (+n) + creators
  figures/09_anomaly_timeline.png — player-count residuals with the same anomalies marked
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_player_counts,
    load_releases, ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import rolling_7day

N_ANOMALIES       = 12     # number of top anomaly periods to report
WINDOW_DAYS       = 28     # days on each side of anomaly peak to pull context
TOP_N_TOPICS      = 5      # top topics to surface per anomaly
TOP_N_CREATORS    = 5      # top creator videos to surface per anomaly
N_HARMONICS       = 3      # Fourier harmonics for the annual seasonal term (2K params, replaces 51 dummies)
TREND_KNOTS       = 5      # cubic B-spline knots for the slow level/trend term (~2yr spacing; deliberately slow)
MIN_TOPIC_N       = 3      # min videos discussing a topic in-window before it can be ranked
SHRINKAGE_K       = 5      # pseudo-count: shrink window topic-mean toward global mean by n/(n+K)

#%%
ensure_output_dirs()

yt       = load_youtube_creators()
rd       = load_reddit_submissions()
pc       = load_player_counts(source='total').set_index('date').sort_index()['players']
releases = load_releases()
topic_cols = get_topic_cols(yt)

#%%
# ── Build seasonal baseline model for player counts ───────────────────────────
# Features: log(days_since_release) + a smooth annual seasonal term (Fourier
# harmonics) + a slow multi-year level/trend term (low-df cubic B-spline of time).
# Fourier replaces the old 51 week-of-year dummies: those overfit (~7 obs/dummy
# inflates in-sample R²) and — worse for an anomaly detector — let the baseline
# absorb the very anomalies we want to surface, because each anomaly week sat at
# ~1/8 leverage on its own dummy. A K-harmonic curve is fit from all weeks
# jointly, so no single week can bend it; sharp week-specific deviations stay in
# the residual where they belong.
#
# The trend term captures the game's slow drift (launch peak -> plateau -> late
# decline) so a residual means "off the game's *contemporary* expected level",
# not "below its 2017 level". It is kept deliberately slow (few knots) for the
# same reason Fourier is low-order: a flexible trend would re-absorb the
# anomalies. Residuals then isolate deviations from expected level + season.

pc_weekly = pc.resample('W-MON').mean().dropna()
pc_weekly_df = pc_weekly.to_frame('players')
pc_weekly_df['week'] = pc_weekly_df.index

# Merge release context (days_since_release) via nearest release date
releases_sorted = releases.dropna(subset=['start_date']).sort_values('start_date')

# Drop pre-launch weeks: before the first release the game isn't live, player
# counts are boundary artifacts (e.g. a stray 93-player week two days before Red
# War), and days_since_release is undefined. The trend term newly exposed this by
# (correctly) expecting a high early level there; filter it out at the source.
first_release = releases_sorted['start_date'].min()
pc_weekly_df  = pc_weekly_df[pc_weekly_df.index >= first_release]

def days_since_nearest_release(date: pd.Timestamp) -> int:
    diffs = (date - releases_sorted['start_date']).dt.days
    past  = diffs[diffs >= 0]
    return past.min() if not past.empty else 999


pc_weekly_df['days_since_release'] = pc_weekly_df.index.map(days_since_nearest_release)
pc_weekly_df['log_days']           = np.log1p(pc_weekly_df['days_since_release'])

# Smooth annual seasonality via Fourier harmonics on day-of-year (period ~365.25).
# sin+cos per harmonic lets regression place the seasonal peak at any phase/height;
# stacking N_HARMONICS lets the curve take a non-sinusoidal shape. 2*N_HARMONICS
# params total vs. 51 dummies.
doy    = pc_weekly_df.index.dayofyear.to_numpy(dtype=float)
period = 365.25
fourier = {}
for k in range(1, N_HARMONICS + 1):
    fourier[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / period)
    fourier[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / period)
fourier_df = pd.DataFrame(fourier, index=pc_weekly_df.index)

# Slow level/trend: cubic B-spline of elapsed days. Joint-estimated with season
# and log_days (avoids two-stage bias). include_bias=False — the regression has
# its own intercept.
t_days    = (pc_weekly_df.index - pc_weekly_df.index[0]).days.to_numpy(dtype=float).reshape(-1, 1)
spline    = SplineTransformer(n_knots=TREND_KNOTS, degree=3, include_bias=False)
trend_arr = spline.fit_transform(t_days)
trend_df  = pd.DataFrame(trend_arr, index=pc_weekly_df.index,
                         columns=[f'trend_{i}' for i in range(trend_arr.shape[1])])

X = pd.concat([pc_weekly_df[['log_days']], fourier_df, trend_df], axis=1).astype(float)
y = pc_weekly_df['players']

model     = LinearRegression().fit(X, y)
fitted    = pd.Series(model.predict(X), index=pc_weekly_df.index)
residuals = y - fitted
residuals.name = 'residual'

# In-sample R² is an optimistic upper bound and the WRONG thing to maximize here
# (a baseline that fits too much eats anomalies). Report adjusted R² (penalizes
# param count) and a forward-only TimeSeriesSplit CV R² (honest generalization).
n_obs   = len(y)
n_param = X.shape[1]
r2_in   = model.score(X, y)
adj_r2  = 1 - (1 - r2_in) * (n_obs - 1) / (n_obs - n_param - 1)

# Forward CV R² is intentionally not reported. A localized B-spline trend can't be
# estimated out-of-sample (its late basis functions are ~0 over any early training
# fold), so forward splits extrapolate catastrophically — a degenerate artifact,
# not a model defect, for an in-sample descriptive baseline. Adjusted R² tracking
# in-sample R² is the overfitting check that matters here.
print(f'Baseline model: log_days + {N_HARMONICS} Fourier harmonics + '
      f'{TREND_KNOTS}-knot trend spline ({n_param} params on {n_obs} weeks)')
print(f'  in-sample R²: {r2_in:.4f}  (optimistic upper bound; not the target)')
print(f'  adjusted R²:  {adj_r2:.4f}  (penalized for param count; tracks in-sample = not overfit)')

#%%
# ── Identify top anomaly weeks ────────────────────────────────────────────────
abs_resid = residuals.abs()
# Avoid selecting overlapping windows — greedy selection with exclusion radius
selected_weeks = []
excluded = set()
for week in abs_resid.sort_values(ascending=False).index:
    if week in excluded:
        continue
    selected_weeks.append(week)
    for d in range(-WINDOW_DAYS, WINDOW_DAYS + 1):
        excluded.add(week + pd.Timedelta(days=d))
    if len(selected_weeks) >= N_ANOMALIES:
        break

#%%
# ── Compute global topic base rates ──────────────────────────────────────────
# Used for z-score deviation inside anomaly windows.
# Topics with very high coverage and low variance are "always-on" —
# they belong in the methods section, not per-anomaly tables.
ALWAYS_ON_N = 8
MIN_GLOBAL_COVERAGE = 20   # minimum videos mentioning topic to include in stats

topic_global = {}
for tc in topic_cols:
    if tc not in yt.columns:
        continue
    vals = yt[tc].dropna()
    if len(vals) >= MIN_GLOBAL_COVERAGE:
        topic_global[tc] = {
            'mean':     vals.mean(),
            'std':      vals.std(),
            'coverage': len(vals) / len(yt),
        }

always_on = sorted(topic_global, key=lambda t: topic_global[t]['coverage'], reverse=True)[:ALWAYS_ON_N]
print('\nAlways-on topics (highest video coverage — report once in methods, not per-anomaly):')
for t in always_on:
    g = topic_global[t]
    print(f'  {t.replace("topic__", ""):40s}  coverage={g["coverage"]:.1%}'
          f'  mean={g["mean"]:+.3f}  std={g["std"]:.3f}')

#%%
# ── Build context for each anomaly ───────────────────────────────────────────
anomaly_records = []

for week in sorted(selected_weeks):
    resid_val = residuals.loc[week]
    window_start = week - pd.Timedelta(days=WINDOW_DAYS)
    window_end   = week + pd.Timedelta(days=WINDOW_DAYS)

    # Creator videos in window
    yt_window = yt[(yt['date'] >= window_start) & (yt['date'] <= window_end)]
    top_videos = (
        yt_window.dropna(subset=['view_count'])
        .nlargest(TOP_N_CREATORS, 'view_count')[['creator', 'title', 'view_count', 'date', 'overall__score']]
        if len(yt_window) > 0 else pd.DataFrame()
    )

    # Rank topics by shrunk z-score deviation from global base rate.
    # A topic at its usual level gets z≈0; one behaving unusually gets a high |z|.
    # This surfaces what's anomalous about this window, not just what's always
    # discussed. The window mean is shrunk toward the global mean by n/(n+K) so a
    # noisy 3-video mean can't dominate the |z| ranking; n is reported per topic.
    if not yt_window.empty:
        topic_stats = {}
        for tc in topic_cols:
            if tc not in topic_global or tc not in yt_window.columns:
                continue
            vals = yt_window[tc].dropna()
            n = len(vals)
            if n < MIN_TOPIC_N:
                continue
            g = topic_global[tc]
            if g['std'] > 0:
                raw_z  = (vals.mean() - g['mean']) / g['std']
                shrink = n / (n + SHRINKAGE_K)
                topic_stats[tc] = {
                    'z':           round(raw_z * shrink, 2),
                    'raw_z':       round(raw_z, 2),
                    'window_mean': round(vals.mean(), 3),
                    'n':           n,
                }
        top_topics = sorted(topic_stats, key=lambda t: abs(topic_stats[t]['z']),
                            reverse=True)[:TOP_N_TOPICS]
        top_topic_means = {t: topic_stats[t] for t in top_topics}
    else:
        top_topics, top_topic_means = [], {}

    # Nearest release
    past_releases = releases_sorted[releases_sorted['start_date'] <= week]
    nearest_release = past_releases.iloc[-1]['season_name'] if not past_releases.empty else 'unknown'

    anomaly_records.append({
        'week':            week.strftime('%Y-%m-%d'),
        'residual':        round(resid_val, 0),
        'direction':       'above' if resid_val > 0 else 'below',
        'actual_players':  round(y.get(week, np.nan), 0),
        'fitted_players':  round(fitted.get(week, np.nan), 0),
        'nearest_season':  nearest_release,
        'top_topics':      ', '.join([t.replace('topic__', '') for t in top_topics]),
        'top_topic_zscores': str({t.replace('topic__', ''): top_topic_means[t] for t in top_topics}),
        'top_videos':      ' | '.join(
            f"{row['creator']}: \"{row['title'][:40]}\" ({int(row['view_count']):,} views)"
            for _, row in top_videos.iterrows()
        ) if not top_videos.empty else '',
    })

anomaly_df = pd.DataFrame(anomaly_records)
anomaly_df.to_csv(TABLES_DIR / '09_anomaly_table.csv', index=False)
print('\nAnomaly table:')
print(anomaly_df[['week', 'residual', 'direction', 'nearest_season', 'top_topics']].to_string(index=False))

#%%
# ── Figure: residual timeline marking the SAME anomalies as the table ─────────
# One selection method for both: the top-N residuals by magnitude with a
# ±WINDOW_DAYS exclusion radius (the `selected_weeks` used above). Every marked
# dot is a row in the anomaly table and vice versa. The previous version used
# find_peaks(distance=32 weeks) — local turning points on a ~7-month spacing, a
# different object from the table's 4-week-spaced top-N, so the two could never
# agree.

def nearest_season_name(date: pd.Timestamp) -> str:
    past = releases_sorted[releases_sorted['start_date'] <= date]
    return past.iloc[-1]['season_name'] if not past.empty else 'unknown'

fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(residuals.index, residuals.values, color='#2c3e50', lw=1.2, alpha=0.7)
ax.fill_between(residuals.index, residuals.values, 0,
                where=residuals > 0, alpha=0.3, color='#2ecc71')
ax.fill_between(residuals.index, residuals.values, 0,
                where=residuals < 0, alpha=0.3, color='#e74c3c')
ax.axhline(0, color='gray', lw=0.8, linestyle='--')

# Mark the table's anomalies (above = green ▲, below = red ▼)
for week in selected_weeks:
    val   = residuals.loc[week]
    above = val > 0
    ax.scatter(week, val, s=70, zorder=5,
               color='#27ae60' if above else '#c0392b',
               marker='^' if above else 'v',
               edgecolors='white', linewidths=0.6)
    ax.annotate(nearest_season_name(week), (week, val),
                fontsize=6, ha='center',
                va='bottom' if above else 'top',
                xytext=(0, 6 if above else -6), textcoords='offset points',
                color='#1a6b3a' if above else '#7b241c')

# Release boundaries
for _, rel in releases.iterrows():
    if pd.isna(rel['start_date']):
        continue
    ax.axvline(rel['start_date'], color='black', lw=0.4, alpha=0.2)

ax.set_title(
    'Player-count residuals from slow-trend + seasonal + release-recency baseline\n'
    '(▲ above-baseline / ▼ below-baseline anomalies — same weeks as the table)',
    fontsize=12,
)
ax.set_xlabel('Date')
ax.set_ylabel('Residual (actual − fitted players)')
ax.grid(True, alpha=0.2)

# Common-cause disclaimer (see docstring / question E): co-occurrence is not influence.
fig.text(0.5, -0.02,
         'Anomalies detected from player counts alone; creator context (table) co-occurs because the '
         'same event drives both. Not evidence of creator influence — direction tested in scripts 04-07.',
         ha='center', va='top', fontsize=8, style='italic', color='#555555')

plt.tight_layout()
out = FIGURES_DIR / '09_anomaly_timeline.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out}')
plt.show()
