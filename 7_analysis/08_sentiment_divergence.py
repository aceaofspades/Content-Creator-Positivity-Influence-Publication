#%%
"""
08_sentiment_divergence.py
---------------------------
Plots creator sentiment minus Reddit sentiment over time, overlaid with player
count, and tests whether the divergence carries any signal.

Disposition of this script (post-review): the ONLY finding is descriptive —
creators score a persistent ~1 point above Reddit (the "persistent positivity"
level). Everything else is a confirmatory null:
  - The divergence→player-count xcorr shows no peak above noise (gated).
  - Episode structure is NOT tested. Detecting extreme weeks requires centering
    the series, which removes the persistent level that IS the finding, and the
    circular-shift episode gate this script once used had no power (see the
    "No divergence episode test" cell). The earlier standardized/centered
    "ebb-and-flow" extremes table was removed for the same reason.
  - A calibration check (question E) shows the +1 gap is real in DIRECTION
    (human raters also score creators above submissions) but LLM-inflated in
    MAGNITUDE (creator-side over-scoring vs. humans), so the ~1-point size is an
    upper bound, not a calibrated effect.

Outputs:
  figures/08a_divergence_timeseries.png  — divergence + player count time series
  figures/08b_divergence_xcorr.png       — daily xcorr (gated; no peak above noise)
  figures/08c_divergence_weekly_xcorr.png — weekly xcorr (gated; no peak above noise)
  tables/08_divergence_xcorr.csv         — gated peak (lag/r/noise_threshold/p_value/significant)
  tables/08_calibration_check.csv        — per-source LLM vs human sentiment means (question E)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    load_player_counts, load_releases, ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow, rolling_7day,
)

SENTIMENT_FIELD      = 'overall__score'
MAX_LAG_DAYS         = 28
MAX_LAG_WEEKS        = 12

# Peak noise gate (mirrors scripts 04 / 05b / 06 / 06c): a bare abs().idxmax()
# peak is only reported as real if its |r| beats a circular-shift null of the
# max-|r| across the whole lag window (corrects for the multiple-lag search).
N_PERM      = 1000
NULL_PCTILE = 95
RANDOM_SEED = 42

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

    x is shuffled in the null; y is the fixed target. The null draws max-|r|
    ACROSS the lag window on each shuffle, correcting for the multiple-lag search
    behind abs().idxmax(). Returns (peak_lag, peak_corr, noise_threshold, p_value,
    significant) or None if the xcorr is empty.
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


xcorr_records = []   # gate results for both granularities -> 08_divergence_xcorr.csv

#%%
ensure_output_dirs()

yt       = load_youtube_creators()
rd       = load_reddit_submissions()
cm       = load_comments_master()
pc       = load_player_counts(source='total').set_index('date').sort_index()['players']
releases = load_releases()

creator_agg = aggregate_creator_daily(yt, [SENTIMENT_FIELD])
reddit_agg  = aggregate_reddit_daily(rd, [SENTIMENT_FIELD])
comment_agg = aggregate_comments_daily(cm, ['sentiment_score'])

# Align to common range (include comments in bounds)
start = max(creator_agg.index.min(), reddit_agg.index.min(),
            comment_agg.index.min(), pc.index.min())
end   = min(creator_agg.index.max(), reddit_agg.index.max(),
            comment_agg.index.max(), pc.index.max())
dates = pd.date_range(start, end, freq='D')

c_series  = creator_agg[f'{SENTIMENT_FIELD}_unweighted'].reindex(dates)
r_series  = reddit_agg[f'{SENTIMENT_FIELD}_unweighted'].reindex(dates)
cm_series = comment_agg['sentiment_score_unweighted'].reindex(dates)
pc_daily  = pc.reindex(dates)

# Primary divergence: creator - Reddit submissions (positive = creators more positive)
# The ONLY finding this series carries is the persistent positive LEVEL (creators
# score higher). It lives entirely in the raw, uncentered series. Do NOT standardize
# or center it to hunt for "episodes": z-scoring each source (or subtracting a
# rolling baseline) zeroes out that level by construction, so the centered series is
# a different construct that can no longer express the finding — and its residual
# week-to-week wiggle is noise (see the stability cell: 0/N weeks beat a shuffle
# null, and Var(creator-Reddit)=Var(c)+Var(r)-2Cov makes it the noisiest series here).
divergence = c_series - r_series
# Secondary divergence: creator - Reddit comments
divergence_comments = c_series - cm_series

#%%
# ── Figure 1: Time series ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 1, hspace=0.08, height_ratios=[1, 1.2, 1])

ax_pc  = fig.add_subplot(gs[0])
ax_div = fig.add_subplot(gs[1], sharex=ax_pc)
ax_raw = fig.add_subplot(gs[2], sharex=ax_pc)

# Player count
pc_smooth = rolling_7day(pc_daily)
ax_pc.plot(pc_smooth.index, pc_smooth.values, color='#2c3e50', lw=1.5)
ax_pc.set_ylabel('Players (7d MA)', fontsize=9)
ax_pc.tick_params(labelbottom=False)
ax_pc.grid(True, alpha=0.2)

# Divergence — show both creator-submissions and creator-comments divergences
div_smooth    = rolling_7day(divergence, min_periods=5)
div_cm_smooth = rolling_7day(divergence_comments, min_periods=5)
ax_div.plot(div_smooth.index, div_smooth.values, color='#8e44ad', lw=1.5,
            label='Creator − submissions')
ax_div.plot(div_cm_smooth.index, div_cm_smooth.values, color='#27ae60', lw=1.2,
            alpha=0.8, linestyle='--', label='Creator − comments')
ax_div.axhline(0, color='gray', lw=0.8, linestyle='--')
# Persistent baseline gap: the divergence oscillates around this, not around 0.
grand_gap_daily = divergence.mean()
ax_div.axhline(grand_gap_daily, color='#8e44ad', lw=0.9, linestyle=':', alpha=0.8,
               label=f'Mean gap = {grand_gap_daily:+.2f}')
ax_div.set_ylabel('Divergence\n(creator − Reddit)', fontsize=9)
ax_div.set_ylim(-2.2, 2.2)
ax_div.legend(fontsize=8, loc='upper right')
ax_div.tick_params(labelbottom=False)
ax_div.grid(True, alpha=0.2)

# Raw sentiment — all three sources
c_smooth  = rolling_7day(c_series)
r_smooth  = rolling_7day(r_series)
cm_smooth = rolling_7day(cm_series)
ax_raw.plot(c_smooth.index,  c_smooth.values,  color='#3498db', lw=1.3, alpha=0.85,
            label='Creator sentiment')
ax_raw.plot(r_smooth.index,  r_smooth.values,  color='#e67e22', lw=1.3, alpha=0.85,
            label='Reddit submissions')
ax_raw.plot(cm_smooth.index, cm_smooth.values, color='#27ae60', lw=1.0, alpha=0.75,
            label='Reddit comments')
ax_raw.axhline(0, color='gray', lw=0.6, linestyle=':')
ax_raw.set_ylabel('Sentiment score (−2 to +2)', fontsize=9)
ax_raw.set_ylim(-2.2, 2.2)
ax_raw.legend(fontsize=9, loc='upper right')
ax_raw.set_xlabel('Date')
ax_raw.grid(True, alpha=0.2)

# Season boundary lines on all panels
for _, rel in releases.iterrows():
    if pd.isna(rel['start_date']):
        continue
    for ax in [ax_pc, ax_div, ax_raw]:
        ax.axvline(rel['start_date'], color='black', lw=0.4, alpha=0.3)

# Figure titles live in the paper captions, not in the image.
plt.tight_layout()
out_a = FIGURES_DIR / '08a_divergence_timeseries.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

#%%
# ── Figure 2: Cross-correlation of divergence vs. player counts ───────────────
# Caveat: this is the noisiest xcorr in the project. The divergence is a DIFFERENCE
# of two noisy series, so Var(divergence) = Var(creator) + Var(Reddit) - 2Cov —
# variances add. Any peak is therefore harder to distinguish from noise than a
# single-source xcorr; the gate below is essential, not decorative.
pc_deseas  = deseasonalize_dow(pc_daily)
div_deseas = deseasonalize_dow(divergence)

cc = normalised_xcorr(div_deseas, pc_deseas, MAX_LAG_DAYS)
gp = gated_peak(div_deseas, pc_deseas, MAX_LAG_DAYS)
if not cc.empty and gp is not None:
    peak_lag, peak_corr, thr, p_val, sig = gp

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(cc.index, cc.values, color='#8e44ad', alpha=0.7, width=1)
    ax2.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
    peak_label = (f'Peak: lag={peak_lag}d  r={peak_corr:.3f}  p={p_val:.3f}' if sig
                  else f'Max |r|: lag={peak_lag}d  r={peak_corr:.3f}  (ns, p={p_val:.3f})')
    ax2.axvline(peak_lag, color='red' if sig else 'gray', lw=1.5, linestyle=':',
                label=peak_label)
    ax2.axhline(thr,  color='red', lw=0.8, linestyle='--', alpha=0.5)
    ax2.axhline(-thr, color='red', lw=0.8, linestyle='--', alpha=0.5,
                label=f'Noise gate (95th pct |r|={thr:.3f})')
    ax2.axhline(0, color='gray', lw=0.6)
    ax2.set_xlabel('Lag (days)  [+ = divergence leads player counts]', fontsize=11)
    ax2.set_ylabel('Cross-correlation coefficient', fontsize=11)
    ax2.set_title('Sentiment divergence → player counts  (deseasonalised)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    out_b = FIGURES_DIR / '08b_divergence_xcorr.png'
    plt.savefig(out_b, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_b}')
    plt.show()
    verdict = 'SIGNIFICANT' if sig else 'no peak above noise'
    print(f'Peak divergence->player-count lag: {peak_lag} days  r={peak_corr:.3f}  '
          f'p={p_val:.3f}  [{verdict}]')
    xcorr_records.append({'granularity': 'daily', 'peak_lag': peak_lag,
                          'peak_corr': round(peak_corr, 4),
                          'noise_threshold': round(thr, 4), 'p_value': round(p_val, 4),
                          'significant': sig})

#%%
# ── Figure 3: Weekly cross-correlation ───────────────────────────────────────
# DOW effects average out in weekly means; no deseasonalization applied.
div_weekly_xcorr = divergence.resample('W-MON').mean()
pc_weekly_xcorr  = pc_daily.resample('W-MON').mean()

cc_w = normalised_xcorr(div_weekly_xcorr, pc_weekly_xcorr, MAX_LAG_WEEKS)
gp_w = gated_peak(div_weekly_xcorr, pc_weekly_xcorr, MAX_LAG_WEEKS)
if not cc_w.empty and gp_w is not None:
    peak_lag_w, peak_corr_w, thr_w, p_val_w, sig_w = gp_w

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(cc_w.index, cc_w.values, color='#8e44ad', alpha=0.7, width=0.6)
    ax3.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
    peak_label_w = (f'Peak: lag={peak_lag_w}w  r={peak_corr_w:.3f}  p={p_val_w:.3f}' if sig_w
                    else f'Max |r|: lag={peak_lag_w}w  r={peak_corr_w:.3f}  (ns, p={p_val_w:.3f})')
    ax3.axvline(peak_lag_w, color='red' if sig_w else 'gray', lw=1.5, linestyle=':',
                label=peak_label_w)
    ax3.axhline(thr_w,  color='red', lw=0.8, linestyle='--', alpha=0.5)
    ax3.axhline(-thr_w, color='red', lw=0.8, linestyle='--', alpha=0.5,
                label=f'Noise gate (95th pct |r|={thr_w:.3f})')
    ax3.axhline(0, color='gray', lw=0.6)
    ax3.set_xlabel('Lag (weeks)  [+ = divergence leads player counts]', fontsize=11)
    ax3.set_ylabel('Cross-correlation coefficient', fontsize=11)
    ax3.set_title('Sentiment divergence → player counts  (weekly, raw)', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.25)
    plt.tight_layout()
    out_c = FIGURES_DIR / '08c_divergence_weekly_xcorr.png'
    plt.savefig(out_c, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_c}')
    plt.show()
    verdict_w = 'SIGNIFICANT' if sig_w else 'no peak above noise'
    print(f'Weekly peak divergence->player-count lag: {peak_lag_w} weeks  '
          f'r={peak_corr_w:.3f}  p={p_val_w:.3f}  [{verdict_w}]')
    xcorr_records.append({'granularity': 'weekly', 'peak_lag': peak_lag_w,
                          'peak_corr': round(peak_corr_w, 4),
                          'noise_threshold': round(thr_w, 4), 'p_value': round(p_val_w, 4),
                          'significant': sig_w})

# Persist gate results for both granularities
if xcorr_records:
    pd.DataFrame(xcorr_records).to_csv(TABLES_DIR / '08_divergence_xcorr.csv', index=False)
    print(f"Saved: {TABLES_DIR / '08_divergence_xcorr.csv'}")

#%%
# ── Finding: persistent positive level (the only thing this series carries) ──
# Creators score the same -2..+2 instrument ~1 point above Reddit submissions in
# EVERY week of the series. This persistent LEVEL is the sole finding of the
# divergence analysis, reported from the RAW, uncentered series — centering or
# standardizing would remove exactly this level (that is why there is no
# "episodes" table; see the "No divergence episode test" cell).
div_weekly = divergence.resample('W-MON').mean()
grand_gap  = div_weekly.mean()
print('\n-- Persistent creator-Reddit positive gap (the finding) --')
print(f'  mean weekly divergence   = {grand_gap:+.3f}  (creators more positive)')
print(f'  sd                       = {div_weekly.std():.3f}')
print(f'  min / max                = {div_weekly.min():+.3f} / {div_weekly.max():+.3f}')
print(f'  % weeks creator-positive = {100 * (div_weekly.dropna() > 0).mean():.1f}%  '
      f'(of {div_weekly.notna().sum()} weeks with creator/Reddit overlap)')
print('  Status: creators score persistently higher AS MEASURED. Direction is')
print('  human-confirmed (calibration cell below); cross-genre LLM calibration')
print('  inflates the magnitude, so treat the ~1-point size as an upper bound.')

# Persist the gap so the paper cites a table, not a printout. The share of
# creator-positive weeks is over weeks that HAVE an overlapping day -- the old
# printout used (div_weekly > 0).mean(), which counted no-overlap (NaN) weeks as
# "not creator-positive" and understated the share for sparse-creator games.
# pct_positive_all_weeks keeps that old denominator for traceability.
gap_records = []
for target, div in [('submissions', divergence), ('comments', divergence_comments)]:
    wk = div.resample('W-MON').mean()
    ok = wk.dropna()
    gap_records.append({
        'target':                 target,
        'n_weeks_total':          len(wk),
        'n_weeks_overlap':        len(ok),
        'mean_gap':               round(ok.mean(), 4),
        'sd_gap':                 round(ok.std(), 4),
        'min_gap':                round(ok.min(), 4),
        'max_gap':                round(ok.max(), 4),
        'pct_positive':           round(100 * (ok > 0).mean(), 2),
        'pct_positive_all_weeks': round(100 * (wk > 0).mean(), 2),
        'first_week':             ok.index.min().date(),
        'last_week':              ok.index.max().date(),
    })
gap_summary = pd.DataFrame(gap_records)

#%%
# ── No "divergence episode" test ─────────────────────────────────────────────
# An earlier cell gated each week's deviation from a 26-week rolling median
# against circular shifts of the same series. A circular shift only relocates
# the series' own extreme deviations, so the null maximum equals the observed
# maximum and no week can ever beat the gate: on synthetic data it reports 0
# weeks for white noise AND for a planted 7.5-sd six-week episode. The check had
# no power, so it was removed. The persistent level above is the finding; episode
# structure is not tested.
gap_summary.to_csv(TABLES_DIR / '08_gap_summary.csv', index=False)
print(f'  Saved {TABLES_DIR / "08_gap_summary.csv"}')

#%%
# ── Calibration check (question E): is the +1 gap real or an LLM artifact? ────
# Uses the existing human-validated test set (no new labeling). Compares the
# creator-minus-submission sentiment gap as scored by the LLM vs. by humans: if
# humans see the same gap it is real; if the LLM's gap is larger, cross-genre
# calibration inflates the magnitude. Skips gracefully if the test set is absent.
ts_path = TABLES_DIR.parents[1] / 'validation' / 'test_set.csv'
if ts_path.exists():
    ts  = pd.read_csv(ts_path)
    # Only human-graded rows: the test set can carry ungraded extension batches
    # (6_data_validation.py --extend), and averaging llm__ over all rows but
    # human__ over graded rows only would compare different samples.
    graded = ts['human_validated'].astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})
    print(f'\n(Calibration uses {int(graded.sum())} graded of {len(ts)} test-set rows)')
    ts  = ts[graded]
    grp = ts.groupby('content_type')
    cal = pd.DataFrame({'n': grp.size(),
                        'llm_mean':   grp['llm__sentiment_score'].mean(),
                        'human_mean': grp['human__sentiment_score'].mean()})
    cal['llm_minus_human'] = cal['llm_mean'] - cal['human_mean']
    print('\n-- Calibration check (per-source sentiment, LLM vs human) --')
    print(cal.round(3).to_string())
    if {'creator', 'submission'}.issubset(cal.index):
        llm_gap   = cal.loc['creator', 'llm_mean']   - cal.loc['submission', 'llm_mean']
        human_gap = cal.loc['creator', 'human_mean'] - cal.loc['submission', 'human_mean']
        infl = llm_gap - human_gap
        print(f'  LLM   creator-submission gap = {llm_gap:+.3f}')
        print(f'  human creator-submission gap = {human_gap:+.3f}  (direction confirmed real)')
        print(f'  LLM inflation over human      = {infl:+.3f}  '
              f'({100 * infl / llm_gap:.0f}% of the measured gap is calibration)')
    cal.to_csv(TABLES_DIR / '08_calibration_check.csv')

    # Bootstrap CIs (n = 25 per genre): resample items within each content type.
    rng_cal = np.random.default_rng(RANDOM_SEED)
    by_type = {k: g[['llm__sentiment_score', 'human__sentiment_score']].dropna().to_numpy()
               for k, g in ts.groupby('content_type')}
    if {'creator', 'submission'}.issubset(by_type):
        draws = []
        for _ in range(10_000):
            cr = by_type['creator'][rng_cal.integers(0, len(by_type['creator']), len(by_type['creator']))]
            sb = by_type['submission'][rng_cal.integers(0, len(by_type['submission']), len(by_type['submission']))]
            draws.append((cr[:, 0].mean() - cr[:, 1].mean(),                     # creator bias
                          sb[:, 0].mean() - sb[:, 1].mean(),                     # submission bias
                          cr[:, 1].mean() - sb[:, 1].mean(),                     # human gap
                          cr[:, 0].mean() - sb[:, 0].mean()))                    # LLM gap
        draws = np.asarray(draws)
        obs = [cal.loc['creator', 'llm_minus_human'], cal.loc['submission', 'llm_minus_human'],
               cal.loc['creator', 'human_mean'] - cal.loc['submission', 'human_mean'],
               cal.loc['creator', 'llm_mean'] - cal.loc['submission', 'llm_mean']]
        boot = pd.DataFrame({'quantity': ['creator_llm_minus_human', 'submission_llm_minus_human',
                                          'human_creator_minus_submission', 'llm_creator_minus_submission'],
                             'estimate': np.round(obs, 4),
                             'ci_lower': np.round(np.percentile(draws, 2.5, axis=0), 4),
                             'ci_upper': np.round(np.percentile(draws, 97.5, axis=0), 4)})
        print(boot.to_string(index=False))
        boot.to_csv(TABLES_DIR / '08_calibration_bootstrap.csv', index=False)
    print(f"Saved: {TABLES_DIR / '08_calibration_check.csv'}")
else:
    print('\n(Calibration check skipped: validation/test_set.csv not found.)')
