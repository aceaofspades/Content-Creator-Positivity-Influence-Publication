#%%
"""
13_confidence_filter.py
------------------------
Self-consistency check: are the main findings driven by the LLM outputs the
model *itself* flagged as low-confidence (meta__confidence_score <= 0)?

Scope and limits — read before citing this script:
  - This is a NARROW self-consistency check, NOT the primary defense of
    measurement validity. It only asks whether dropping the model's own
    low-confidence rows moves the headline statistics. It is:
      * circular — it trusts the LLM's self-reported confidence as the quality
        signal, so it cannot catch confident-but-wrong labels (undetected
        sarcasm, systematic miscalibration); and
      * accuracy-blind — "findings survive" only rules out that uncertain rows
        are steering results; it says nothing about whether the surviving rows
        are correct.
  - The PRIMARY defense against the "LLM-generated slop" objection is the
    human-vs-model agreement report from 6_data_validation.py (external ground
    truth on a stratified test set). Cross-genre calibration of the
    creator/submission gap is handled in 08_sentiment_divergence.py. Cite those
    for validity; cite this only for self-consistency.
  - The exclusion rate matters: a small rate (YouTube here, ~6%) makes survival
    near-guaranteed and therefore weak evidence; the meaningful arm is Reddit,
    where a much larger share (~38%) is dropped. Rates are printed and persisted
    to the summary table.

Tests run under both conditions (include all / exclude confidence_score <= 0).
The comparison is "materially unchanged?" (peak lag + r side by side), NOT a
significance test on the difference — the two datasets are nested (filtered is a
subset of all), so a naive two-sample test would be invalid.
  1. Primary xcorr — creator overall__score -> Reddit overall__score,
     unweighted, daily +/-28 days and weekly +/-4 weeks (matches script 05)
  2. Creator engagement weekly xcorr — 3 fields, +/-4 weeks (matches script 04)
  3. Late-season M3 regression — creator late-season sentiment -> next-season
     revival, controlling for revival autocorrelation (matches script 03)

Peak reporting: each xcorr peak is the global max-|r| lag, but is labelled a
real peak only when |r| clears a circular-shift noise gate (N_PERM permutations;
must exceed the 95th percentile of the null max-|r| across the lag window, which
searches the same lags the observation did). Sub-threshold peaks are tagged
not-significant. Same null model as 04 / 05b / 06.

Outputs:
  figures/13_confidence_filter_xcorr.png  — before/after xcorr overlay
                                            (filled peak dot = clears noise gate)
  tables/13_confidence_filter_summary.csv — before/after stats, exclusion rates,
                                            per-peak gate results
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.shared.loaders as _loaders
from analysis.shared.loaders import (
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
    CREATOR_ENGAGEMENT_COLS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, deseasonalize_dow,
)

MAX_LAG_DAYS  = 28
MAX_LAG_WEEKS = 4

N_PERM      = 1000   # circular-shift permutations for the noise gate
NULL_PCTILE = 95     # a peak counts only if |r| exceeds this pctile of the null
RANDOM_SEED = 42

#%%
ensure_output_dirs()

# ── Load data under both conditions ──────────────────────────────────────────
_loaders.EXCLUDE_LOW_CONFIDENCE = False
yt_all = _loaders.load_youtube_creators()
rd_all = _loaders.load_reddit_submissions()

_loaders.EXCLUDE_LOW_CONFIDENCE = True
yt_filt = _loaders.load_youtube_creators()
rd_filt = _loaders.load_reddit_submissions()

_loaders.EXCLUDE_LOW_CONFIDENCE = False   # reset to default

n_yt_all,  n_yt_filt  = len(yt_all),  len(yt_filt)
n_rd_all,  n_rd_filt  = len(rd_all),  len(rd_filt)
pct_yt_excl = (n_yt_all - n_yt_filt) / n_yt_all * 100
pct_rd_excl = (n_rd_all - n_rd_filt) / n_rd_all * 100

print(f'YouTube creators: {n_yt_all} total -> {n_yt_filt} after filter '
      f'({pct_yt_excl:.1f}% excluded)')
print(f'Reddit subs:      {n_rd_all} total -> {n_rd_filt} after filter '
      f'({pct_rd_excl:.1f}% excluded)')

datasets = {
    'all':      (yt_all,  rd_all),
    'filtered': (yt_filt, rd_filt),
}

#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed.
    Matches the null model in 05b_permutation_baseline.py / script 04."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def gated_peak(x: pd.Series, y: pd.Series, max_lag: int,
               n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """
    Hypothesis-blind peak (global max-|r| lag) + search-corrected circular-shift
    noise gate. The peak counts as real only if |r| exceeds the NULL_PCTILE-th
    percentile of the null max-|r| distribution across the lag window (the null
    searches the lags the same way the observation did, correcting for the
    multiple-lag search). Same null model as 04 / 05b / 06. Returns a dict (with
    the xcorr Series under 'cc'), or None if x/y cannot be cross-correlated.
    """
    cc = normalised_xcorr(x, y, max_lag)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])

    rng = np.random.default_rng(seed)
    n   = len(x)
    null_max = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n, 2)))
        cc_n  = normalised_xcorr(shift_values_only(x, shift), y, max_lag)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
    null_max = np.asarray(null_max)

    if null_max.size:
        threshold   = float(np.percentile(null_max, NULL_PCTILE))
        p_value     = float(np.mean(null_max >= abs(peak_corr)))
        significant = bool(abs(peak_corr) > threshold)
    else:
        threshold, p_value, significant = np.nan, np.nan, False

    return {
        'cc':              cc,
        'peak_lag':        peak_lag,
        'peak_corr':       round(peak_corr, 4),
        'noise_threshold': round(threshold, 4) if not np.isnan(threshold) else np.nan,
        'p_value':         round(p_value, 4) if not np.isnan(p_value) else np.nan,
        'significant':     significant,
    }


#%%
# ── Test 1: primary xcorr, daily and weekly ───────────────────────────────────
# Peaks are noise-gated (global max-|r| lag, kept only if it clears the
# circular-shift gate); r@0 is reported as a filter-stable fixed-lag anchor.
xcorr_curves = {}  # label -> (daily_res, weekly_res) gated_peak dicts
primary_records = []

for label, (yt, rd) in datasets.items():
    c_agg = aggregate_creator_daily(yt, ['overall__score'])
    r_agg = aggregate_reddit_daily(rd, ['overall__score'])

    s = max(c_agg.index.min(), r_agg.index.min())
    e = min(c_agg.index.max(), r_agg.index.max())
    d = pd.date_range(s, e, freq='D')
    c_agg = c_agg.reindex(d)
    r_agg = r_agg.reindex(d)

    c_d = deseasonalize_dow(c_agg['overall__score_unweighted'])
    r_d = deseasonalize_dow(r_agg['overall__score_unweighted'])

    # Daily (noise-gated peak)
    res_d = gated_peak(c_d, r_d, MAX_LAG_DAYS)
    cc_d  = res_d['cc'] if res_d else pd.Series(dtype=float)
    daily_r_at_0 = float(cc_d.get(0, np.nan)) if not cc_d.empty else np.nan

    # Weekly (noise-gated peak)
    c_w = c_agg['overall__score_unweighted'].resample('W-MON').mean()
    r_w = r_agg['overall__score_unweighted'].resample('W-MON').mean()
    res_w = gated_peak(c_w, r_w, MAX_LAG_WEEKS)
    cc_w  = res_w['cc'] if res_w else pd.Series(dtype=float)
    weekly_r_at_0 = float(cc_w.get(0, np.nan)) if not cc_w.empty else np.nan

    xcorr_curves[label] = (res_d, res_w)
    primary_records.append({
        'condition':           label,
        'test':                'primary_xcorr',
        'daily_peak_lag':      res_d['peak_lag']        if res_d else np.nan,
        'daily_peak_r':        res_d['peak_corr']       if res_d else np.nan,
        'daily_r_at_0':        round(daily_r_at_0, 4),
        'daily_noise_thresh':  res_d['noise_threshold'] if res_d else np.nan,
        'daily_p_value':       res_d['p_value']         if res_d else np.nan,
        'daily_significant':   res_d['significant']     if res_d else False,
        'weekly_peak_lag':     res_w['peak_lag']        if res_w else np.nan,
        'weekly_peak_r':       res_w['peak_corr']       if res_w else np.nan,
        'weekly_r_at_0':       round(weekly_r_at_0, 4),
        'weekly_noise_thresh': res_w['noise_threshold'] if res_w else np.nan,
        'weekly_p_value':      res_w['p_value']         if res_w else np.nan,
        'weekly_significant':  res_w['significant']     if res_w else False,
    })
    if res_d and res_w:
        print(f'\n[{label}] primary xcorr:')
        print(f'  daily   peak_lag={res_d["peak_lag"]:+3d}d  r={res_d["peak_corr"]:+.4f}  '
              f'r@0={daily_r_at_0:+.4f}  {"sig" if res_d["significant"] else "ns"} '
              f'(gate={res_d["noise_threshold"]:.3f})')
        print(f'  weekly  peak_lag={res_w["peak_lag"]:+3d}w  r={res_w["peak_corr"]:+.4f}  '
              f'r@0={weekly_r_at_0:+.4f}  {"sig" if res_w["significant"] else "ns"} '
              f'(gate={res_w["noise_threshold"]:.3f})')

#%%
# ── Test 2: creator engagement fields, weekly ─────────────────────────────────
# Player counts are filter-invariant (the confidence filter touches LLM content,
# not Steam counts), so load them once here; only the creator series changes.
engagement_records = []

pc = _loaders.load_player_counts(source='total').set_index('date').sort_index()['players']
pc_w = pc.resample('W-MON').mean()

for label, (yt, rd) in datasets.items():
    c_agg_eng = aggregate_creator_daily(yt, CREATOR_ENGAGEMENT_COLS)
    s = c_agg_eng.index.min()
    e = c_agg_eng.index.max()
    d = pd.date_range(s, e, freq='D')
    c_agg_eng = c_agg_eng.reindex(d)

    c_w = c_agg_eng.resample('W-MON').mean()

    for field in CREATOR_ENGAGEMENT_COLS:
        col = f'{field}_unweighted'
        if col not in c_w.columns:
            continue
        res = gated_peak(c_w[col].reindex(pc_w.index), pc_w, MAX_LAG_WEEKS)
        if res is None:
            continue
        r_at_1 = float(res['cc'].get(1, np.nan))
        engagement_records.append({
            'condition':       label,
            'field':           field,
            'peak_lag':        res['peak_lag'],
            'peak_corr':       res['peak_corr'],
            'r_at_lag1':       round(r_at_1, 4),
            'noise_threshold': res['noise_threshold'],
            'p_value':         res['p_value'],
            'significant':     res['significant'],
        })
        print(f'[{label}] {field.replace("creator__", ""):40s}  '
              f'peak_lag={res["peak_lag"]:+2d}w  r@1={r_at_1:+.4f}  '
              f'{"sig" if res["significant"] else "ns"} (gate={res["noise_threshold"]:.3f})')

#%%
# ── Test 3: late-season M3 regression ─────────────────────────────────────────
# Inline reproduction of script 03's M3 hierarchy.
# Reads revival metrics from 01_release_metrics.csv (must exist).

from analysis.shared.loaders import TABLES_DIR as TBLS

revival_path = TBLS / '01_release_metrics.csv'
if not revival_path.exists():
    print(f'\nSkipping M3 regression: {revival_path} not found. Run script 01 first.')
    m3_records = []
else:
    # Chronological season order + each season's own revival, from script 01's output.
    # season_name lives in the releases/metrics file; the content master carries it
    # as the 'current_season' column (see 5b_build_masters), so we pair the two on
    # matching name values.
    metrics = pd.read_csv(revival_path, parse_dates=['start_date'])
    season_order   = metrics.sort_values('start_date')['season_name'].tolist()
    revival_lookup = metrics.dropna(subset=['revival']).set_index('season_name')['revival']

    m3_records = []
    for label, (yt, rd) in datasets.items():
        if 'release_phase' not in yt.columns or 'current_season' not in yt.columns:
            print(f'[{label}] release_phase/current_season columns absent — skipping M3.')
            continue

        late = (yt[yt['release_phase'] == 'late_season']
                .groupby('current_season')['overall__score']
                .mean())

        # Pair each season's late-season sentiment with the *next* chronological
        # season's revival (revival_{n+1}); 'revival' is the source season's own
        # revival (revival_n), the autoregressive control. Mirrors script 03.
        rows = []
        for i, season in enumerate(season_order[:-1]):
            next_season = season_order[i + 1]
            rows.append({
                'season':           season,
                'late_season_sent': late.get(season, np.nan),
                'revival':          revival_lookup.get(season, np.nan),
                'next_revival':     revival_lookup.get(next_season, np.nan),
            })
        paired = pd.DataFrame(rows).dropna(
            subset=['late_season_sent', 'revival', 'next_revival'])

        if len(paired) < 8:
            print(f'[{label}] Insufficient seasons for M3 ({len(paired)}), skipping.')
            continue

        m1 = smf.ols('next_revival ~ revival',                    data=paired).fit()
        m2 = smf.ols('next_revival ~ late_season_sent',           data=paired).fit()
        m3 = smf.ols('next_revival ~ revival + late_season_sent', data=paired).fit()

        coef3 = m3.params.get('late_season_sent', np.nan)
        p3    = m3.pvalues.get('late_season_sent', np.nan)

        m3_records.append({
            'condition':     label,
            'n_seasons':     len(paired),
            'M1_adj_r2':     round(m1.rsquared_adj, 3),
            'M2_adj_r2':     round(m2.rsquared_adj, 3),
            'M3_adj_r2':     round(m3.rsquared_adj, 3),
            'M3_sent_beta':  round(coef3, 3),
            'M3_sent_p':     round(p3, 4),
        })
        print(f'\n[{label}] M3 regression:  '
              f'adj_R2={m3.rsquared_adj:.3f}  beta={coef3:.3f}  p={p3:.4f}')

#%%
# ── Figure: xcorr curves before/after filter ─────────────────────────────────
colors = {'all': '#95a5a6', 'filtered': '#e74c3c'}
linestyles = {'all': '-', 'filtered': '--'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# filled peak dot = clears the noise gate, hollow = within noise
for label, (res_d, res_w) in xcorr_curves.items():
    for ax, res in ((axes[0], res_d), (axes[1], res_w)):
        if res is None or res['cc'].empty:
            continue
        cc = res['cc']
        ax.plot(cc.index, cc.values,
                color=colors[label], lw=2, linestyle=linestyles[label],
                label=label)
        ax.scatter([res['peak_lag']], [res['peak_corr']], s=70, zorder=5,
                   facecolors=(colors[label] if res['significant'] else 'none'),
                   edgecolors=colors[label], linewidths=1.6)

for ax, gran, unit in zip(axes, ['Daily ±28', 'Weekly ±4'], ['days', 'weeks']):
    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel(f'Lag ({unit})  [+ = creator leads Reddit]', fontsize=10)
    ax.set_ylabel('Cross-correlation', fontsize=10)
    ax.set_title(f'{gran} {unit}: overall sentiment xcorr', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

fig.suptitle(
    f'Confidence-score self-consistency check\n'
    f'(dashed = low-confidence rows excluded; filled dot = peak clears noise gate; '
    f'YT: {pct_yt_excl:.1f}% excl., Reddit: {pct_rd_excl:.1f}% excl.)',
    fontsize=12,
)
plt.tight_layout()
out_fig = FIGURES_DIR / '13_confidence_filter_xcorr.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')
plt.show()

#%%
# ── Save summary table ────────────────────────────────────────────────────────
# Exclusion rates are persisted here (not just printed) — a reader of the table
# needs them to weigh how much a "survived the filter" result is worth.
exclusion_records = [
    {'test': 'exclusion_rate', 'source': 'youtube', 'n_total': n_yt_all,
     'n_kept': n_yt_filt, 'pct_excluded': round(pct_yt_excl, 2)},
    {'test': 'exclusion_rate', 'source': 'reddit', 'n_total': n_rd_all,
     'n_kept': n_rd_filt, 'pct_excluded': round(pct_rd_excl, 2)},
]

all_records = (
    exclusion_records
    + primary_records
    + [dict(test='engagement_xcorr', **r) for r in engagement_records]
    + [dict(test='M3_regression', **r) for r in m3_records]
)
summary_df = pd.DataFrame(all_records)
out_tbl = TABLES_DIR / '13_confidence_filter_summary.csv'
summary_df.to_csv(out_tbl, index=False)
print(f'Saved: {out_tbl}')
print(summary_df.to_string(index=False))
