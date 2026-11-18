#%%
"""
04_creator_engagement_indicators.py
-------------------------------------
Tracks creator engagement fields over time and tests whether they lead
player-count changes.

Creator engagement fields (numeric in master, -2 to +2):
  creator__are_they_still_playing    (no=-2 … yes=+2)
  creator__are_they_burned_out       (yes=-2 … no=+2, INVERTED)
  creator__would_they_recommend_the_game

Robustness checks implemented:
  1. Baseline:      daily unweighted mean, deseasonalized player counts (DOW only)
  2. View-weighted: daily view-weighted mean (reach-adjusted signal)
  3. Weekly:        weekly aggregation at ±4 week lags (unweighted + view-weighted)
  4. First-diff:    xcorr vs. day-over-day player count change (tests inflection points)
  5. State %:       proportion of creator videos in concerning state each day

Produces:
  figures/04a_engagement_vs_players.png  — time series of all three fields + player count
  figures/04b_engagement_lead_lag.png    — daily unweighted xcorr vs. deseasonalized players
  figures/04c_view_weighted_xcorr.png    — daily view-weighted xcorr
  figures/04d_weekly_xcorr.png           — weekly xcorr ±4 weeks (unweighted + view-weighted)
  figures/04e_firstdiff_xcorr.png        — xcorr vs. first-differenced player counts
  figures/04f_creator_state_proportions.png — % of videos in concerning state over time
  figures/04g_state_proportion_xcorr.png — xcorr of state proportions vs. player counts
  tables/04_engagement_xcorr.csv         — baseline results (backward-compatible)
  tables/04_engagement_xcorr_extended.csv — all robustness checks combined

Deseasonalization note: DOW dummies only. Release-window effects are NOT removed,
so the dependent variable is player counts with release spikes intact — the thing
engagement fields are theoretically supposed to predict.

Peak reporting: the reported peak is the global max-|r| lag (hypothesis-blind),
but it is only labelled a "peak" when it clears a circular-shift noise gate
(N_PERM permutations; a bar must exceed NULL_PCTILE of the null max-|r|
distribution, which searches the lag window the same way the observation did).
Sub-threshold results are reported as "no peak above noise". Cleared peaks are
tagged by their relation to the directional hypothesis (consistent / contrary)
using predicted_sign and the creator-leads convention (lag >= 0). Direction is
used only to label a peak, never to select it. The null model matches
05b_permutation_baseline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.lagcorr import normalised_xcorr as _shared_normalised_xcorr
from analysis.shared.loaders import (
    load_youtube_creators, load_player_counts, load_releases,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
    CREATOR_ENGAGEMENT_COLS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, deseasonalize_dow, rolling_7day, reindex_daily,
)

MAX_LAG_DAYS  = 28
MAX_LAG_WEEKS = 4

N_PERM      = 1000   # circular-shift permutations for the noise gate
NULL_PCTILE = 95     # a peak must exceed this percentile of the null max-|r|
RANDOM_SEED = 42

# Release-window-controlled sweep (04h): exclude weeks within +/-W days of any
# release start and re-run the gated weekly xcorr, for each W below.
SWEEP_WIDTHS = [0, 7, 14, 21, 28, 42]
MIN_PAIRS    = 20    # minimum aligned week-pairs to compute a lagged correlation
RELIABLE_MIN_WEEKS = 150  # below this the masked sample is too thin to trust the r-trajectory

FIELD_LABELS = {
    'creator__are_they_still_playing':        'Still playing',
    'creator__are_they_burned_out':           'Not burned out (inverted)',
    'creator__would_they_recommend_the_game': 'Would recommend',
}

CONCERN_LABELS = {
    'creator__are_they_still_playing_pct_concerning':        '% not playing (score < 0)',
    'creator__are_they_burned_out_pct_concerning':           '% burned out (inverted score < 0)',
    'creator__would_they_recommend_the_game_pct_concerning': '% would not recommend (score < 0)',
}

colors = ['#3498db', '#e74c3c', '#2ecc71']

#%%
ensure_output_dirs()

yt = load_youtube_creators()
pc = load_player_counts(source='total').set_index('date').sort_index()['players']

#%%
# ── Aggregation ───────────────────────────────────────────────────────────────
creator_agg = aggregate_creator_daily(yt, CREATOR_ENGAGEMENT_COLS)

eng_cols  = [f'{c}_unweighted'    for c in CREATOR_ENGAGEMENT_COLS if c in yt.columns]
view_cols = [f'{c}_view_weighted' for c in CREATOR_ENGAGEMENT_COLS if c in yt.columns]

# Common daily date range
all_dates = pd.date_range(
    max(creator_agg.index.min(), pc.index.min()),
    min(creator_agg.index.max(), pc.index.max()),
    freq='D',
)
creator_daily = creator_agg[eng_cols].reindex(all_dates)
creator_view  = creator_agg[view_cols].reindex(all_dates)
pc_daily      = pc.reindex(all_dates)

#%%
# ── Deseasonalize player counts (DOW dummies only; release effects intact) ───
pc_deseas = deseasonalize_dow(pc_daily)

#%%
# ── Cross-correlation helper ──────────────────────────────────────────────────
def xcorr(x: pd.Series, y: pd.Series, max_lag: int) -> pd.Series:
    """
    Normalised cross-correlation of x (creator) vs. y (player counts).
    Positive lag = x leads y by that many calendar periods.

    Delegates to analysis/shared/lagcorr.normalised_xcorr. The old local version
    correlated dropna()'d rows by position, so on sparse daily creator series a
    "lag" was a count of observed rows, not days (gap-collapsing).
    """
    return _shared_normalised_xcorr(x, y, max_lag, min_pairs=max_lag * 2)


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed.
    Matches the null model in 05b_permutation_baseline.py."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def gated_peak(x: pd.Series, y: pd.Series, max_lag: int, predicted_sign: int,
               n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """
    Hypothesis-blind peak selection + search-corrected noise gate.

    Selection: the global max-|r| lag over the window. This does NOT look at
    the hypothesis — a contrary bar wins the peak if it is the largest.

    Gate: circular-shift the creator series `n_perm` times, rebuild the xcorr,
    and collect the null distribution of max-|r| ACROSS the window (so the null
    searches the lags the same way the observation did — this corrects for the
    multiple-lag search). The peak counts only if |r| exceeds NULL_PCTILE of
    that distribution; otherwise the panel reports 'no peak above noise'.

    Direction labels the cleared peak, it never selects it:
      predicted_sign = +1 (healthy-high fields) or -1 (concern proportion)
      creator-leads  = lag >= 0
    Returns a dict (incl. the xcorr Series under 'cc'), or None if x/y cannot
    be cross-correlated.
    """
    cc = xcorr(x, y, max_lag)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])

    rng = np.random.default_rng(seed)
    n   = len(x)
    null_max = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n, 2)))
        cc_n  = xcorr(shift_values_only(x, shift), y, max_lag)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
    null_max = np.asarray(null_max)

    if null_max.size:
        threshold   = float(np.percentile(null_max, NULL_PCTILE))
        p_value     = float(np.mean(null_max >= abs(peak_corr)))
        significant = bool(abs(peak_corr) > threshold)
    else:
        threshold, p_value, significant = np.nan, np.nan, False

    if not significant:
        relation = 'no peak above noise'
    else:
        sign_ok = np.sign(peak_corr) == predicted_sign
        lead_ok = peak_lag >= 0
        if sign_ok and lead_ok:
            relation = 'consistent with hypothesis'
        elif sign_ok and not lead_ok:
            relation = 'contrary (players lead)'
        elif not sign_ok and lead_ok:
            relation = 'contrary (opposite sign)'
        else:
            relation = 'contrary (opposite sign + players lead)'

    return {
        'cc':                  cc,
        'peak_lag':            peak_lag,
        'peak_corr':           round(peak_corr, 4),
        'noise_threshold':     round(threshold, 4) if not np.isnan(threshold) else np.nan,
        'p_value':             round(p_value, 4) if not np.isnan(p_value) else np.nan,
        'significant':         significant,
        'hypothesis_relation': relation,
    }


_REC_KEYS = ('peak_lag', 'peak_corr', 'noise_threshold', 'p_value',
             'significant', 'hypothesis_relation')


def _annotate_peak(ax, res, unit):
    """Draw the noise band + earned-peak / no-peak annotation on an axis."""
    thr = res['noise_threshold']
    ax.axhspan(-thr, thr, color='gray', alpha=0.12,
               label=f'noise band (95%): +/-{thr:.3f}')
    if res['significant']:
        ax.axvline(res['peak_lag'], color='red', lw=1.2, linestyle=':', alpha=0.85,
                   label=(f"Peak: lag={res['peak_lag']}{unit}  r={res['peak_corr']:.3f}\n"
                          f"{res['hypothesis_relation']} (p={res['p_value']:.3f})"))
    else:
        ax.axvline(res['peak_lag'], color='gray', lw=1.0, linestyle=':', alpha=0.5,
                   label=(f"No peak above noise\n"
                          f"max|r|={abs(res['peak_corr']):.3f} < {thr:.3f}"))


def _xcorr_panel(axes, col_list, creator_data, y_series, max_lag,
                 xlabel_suffix='days', title_suffix='', predicted_sign=1):
    """Plot noise-gated xcorr bars for each field, return list of result dicts.

    predicted_sign tags the cleared peak's relation to the hypothesis; it does
    not change which lag is selected (always the global max-|r|).
    """
    records = []
    for ax, col, color in zip(axes, col_list, colors):
        field_name = col.replace('_unweighted', '').replace('_view_weighted', '')
        label = FIELD_LABELS.get(field_name, field_name)

        res = gated_peak(creator_data[col], y_series, max_lag, predicted_sign)
        if res is None:
            ax.set_title(f'{label}\n(insufficient data)')
            continue
        cc = res['cc']

        ax.bar(cc.index, cc.values, color=color, alpha=0.7, width=0.8)
        ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.6)
        _annotate_peak(ax, res, xlabel_suffix[0])
        ax.legend(fontsize=7)
        ax.set_title(label + (f'\n{title_suffix}' if title_suffix else ''), fontsize=10)
        ax.set_xlabel(f'Lag ({xlabel_suffix})\n+ = creator leads player counts')
        ax.grid(True, alpha=0.25)
        rec = {k: res[k] for k in _REC_KEYS}
        rec['field'] = field_name
        records.append(rec)
    return records


def masked_lag_corr(x: pd.Series, y: pd.Series, max_lag: int, keep: pd.Series) -> pd.Series:
    """
    Date-aware lagged Pearson correlation on a regular weekly index, using only
    weeks where `keep` is True (e.g. non-release weeks). Positive lag = x leads y.

    Shifts happen on the regular index BEFORE masking, so 'lag h' always means a
    true h-week offset — no gap-collapsing when weeks are dropped. Per-lag Pearson
    matches the method script 14 uses for its CIs.
    """
    xk = x.where(keep)
    yk = y.where(keep)
    out = {}
    for h in range(-max_lag, max_lag + 1):
        pair = pd.concat([xk, yk.shift(-h)], axis=1).dropna()
        if len(pair) >= MIN_PAIRS:
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if not np.isnan(r):
                out[h] = float(r)
    return pd.Series(out, dtype=float)


def masked_gated_peak(x: pd.Series, y: pd.Series, max_lag: int, keep: pd.Series,
                      predicted_sign: int, n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """gated_peak for the release-controlled sweep: same hypothesis-blind
    selection + circular-shift noise gate, but on the masked (non-release) sample
    using the date-aware masked_lag_corr. Returns a dict (incl. 'n_weeks') or None."""
    cc = masked_lag_corr(x, y, max_lag, keep)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])

    xk  = x.where(keep)
    n_valid = int(xk.notna().sum())
    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n_valid, 2)))
        cc_n  = masked_lag_corr(shift_values_only(xk, shift), y, max_lag, keep)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
    null_max = np.asarray(null_max)

    if null_max.size:
        threshold   = float(np.percentile(null_max, NULL_PCTILE))
        p_value     = float(np.mean(null_max >= abs(peak_corr)))
        significant = bool(abs(peak_corr) > threshold)
    else:
        threshold, p_value, significant = np.nan, np.nan, False

    if not significant:
        relation = 'no peak above noise'
    else:
        sign_ok = np.sign(peak_corr) == predicted_sign
        lead_ok = peak_lag >= 0
        if sign_ok and lead_ok:
            relation = 'consistent with hypothesis'
        elif sign_ok and not lead_ok:
            relation = 'contrary (players lead)'
        elif not sign_ok and lead_ok:
            relation = 'contrary (opposite sign)'
        else:
            relation = 'contrary (opposite sign + players lead)'

    return {
        'peak_lag':            peak_lag,
        'peak_corr':           round(peak_corr, 4),
        'noise_threshold':     round(threshold, 4) if not np.isnan(threshold) else np.nan,
        'p_value':             round(p_value, 4) if not np.isnan(p_value) else np.nan,
        'significant':         significant,
        'hypothesis_relation': relation,
        'n_weeks':             int((keep & x.notna() & y.notna()).sum()),
    }


#%%
# ── Figure 04a: Time series ───────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(4, 1, hspace=0.4)

ax_pc  = fig.add_subplot(gs[0])
ax_eng = [fig.add_subplot(gs[i + 1], sharex=ax_pc) for i in range(3)]

pc_smooth = rolling_7day(pc_daily, min_periods=4)
ax_pc.plot(pc_smooth.index, pc_smooth.values, color='#2c3e50', lw=1.5)
ax_pc.set_ylabel('Players', fontsize=9)
ax_pc.set_title('Creator engagement indicators vs. player counts', fontsize=13)
ax_pc.grid(True, alpha=0.25)
ax_pc.tick_params(labelbottom=False)

for ax, col, color in zip(ax_eng, eng_cols, colors):
    field_name = col.replace('_unweighted', '')
    label = FIELD_LABELS.get(field_name, field_name)
    smoothed = rolling_7day(creator_daily[col], min_periods=4)
    ax.plot(smoothed.index, smoothed.values, color=color, lw=1.4, alpha=0.85)
    ax.axhline(0, color='gray', lw=0.7, linestyle=':')
    ax.set_ylabel(label, fontsize=8)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.25)
    if ax != ax_eng[-1]:
        ax.tick_params(labelbottom=False)

ax_eng[-1].set_xlabel('Date')
out_a = FIGURES_DIR / '04a_engagement_vs_players.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

#%%
# ── Figure 04b: Baseline — daily unweighted xcorr vs. deseasonalized players ─
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
xcorr_records = _xcorr_panel(axes2, eng_cols, creator_daily, pc_deseas, MAX_LAG_DAYS)
axes2[0].set_ylabel('Cross-correlation coefficient')
fig2.suptitle(f'Unweighted daily creator engagement → deseasonalized player counts'
              f'  (lags ±{MAX_LAG_DAYS} days)', fontsize=12)
plt.tight_layout()
out_b = FIGURES_DIR / '04b_engagement_lead_lag.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')
plt.show()

# Backward-compatible single-check table
xcorr_df = pd.DataFrame(xcorr_records)
print(xcorr_df.to_string(index=False))
xcorr_df.to_csv(TABLES_DIR / '04_engagement_xcorr.csv', index=False)

#%%
# ── Figure 04c: Robustness 1 — view-weighted aggregation ─────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
xcorr_view_records = _xcorr_panel(axes3, view_cols, creator_view, pc_deseas, MAX_LAG_DAYS)
axes3[0].set_ylabel('Cross-correlation coefficient')
fig3.suptitle(f'View-weighted daily creator engagement → deseasonalized player counts'
              f'  (lags ±{MAX_LAG_DAYS} days)', fontsize=12)
plt.tight_layout()
out_c = FIGURES_DIR / '04c_view_weighted_xcorr.png'
plt.savefig(out_c, dpi=150, bbox_inches='tight')
print(f'Saved: {out_c}')
plt.show()

#%%
# ── Figure 04d: Robustness 2 — weekly aggregation at ±4 week lags ────────────
# DOW effects average out in weekly means; no deseasonalization applied.
creator_weekly      = creator_daily.resample('W-MON').mean()
creator_view_weekly = creator_view.resample('W-MON').mean()
pc_weekly           = pc_daily.resample('W-MON').mean()

start_w = max(creator_weekly.index.min(), pc_weekly.index.min())
end_w   = min(creator_weekly.index.max(), pc_weekly.index.max())
all_weeks = pd.date_range(start_w, end_w, freq='W-MON')
creator_weekly      = creator_weekly.reindex(all_weeks)
creator_view_weekly = creator_view_weekly.reindex(all_weeks)
pc_weekly           = pc_weekly.reindex(all_weeks)

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9), sharey='row')
xcorr_weekly_records = []

for col_idx, (uw_col, vw_col, color) in enumerate(zip(eng_cols, view_cols, colors)):
    field_name = uw_col.replace('_unweighted', '')
    label = FIELD_LABELS.get(field_name, field_name)

    for row_idx, (col, weight_label, c_weekly) in enumerate([
        (uw_col, 'unweighted',    creator_weekly),
        (vw_col, 'view_weighted', creator_view_weekly),
    ]):
        ax = axes4[row_idx, col_idx]
        res = gated_peak(c_weekly[col], pc_weekly, MAX_LAG_WEEKS, predicted_sign=1)
        if res is None:
            ax.set_title(f'{label}\n(insufficient data)')
            continue
        cc = res['cc']

        ax.bar(cc.index, cc.values, color=color, alpha=0.7, width=0.6)
        ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.6)
        _annotate_peak(ax, res, 'w')
        ax.legend(fontsize=7)
        ax.set_xlabel('Lag (weeks)\n+ = creator leads player counts')
        ax.grid(True, alpha=0.25)
        rec = {k: res[k] for k in _REC_KEYS}
        rec.update({'field': field_name, 'weighting': weight_label, 'granularity': 'weekly'})
        xcorr_weekly_records.append(rec)

        if col_idx == 0:
            ax.set_ylabel(f'{weight_label}\nCross-corr.', fontsize=9)
        if row_idx == 0:
            ax.set_title(label, fontsize=10)

fig4.suptitle(f'Weekly creator engagement → raw weekly player counts  (lags ±{MAX_LAG_WEEKS} weeks)',
              fontsize=12)
plt.tight_layout()
out_d = FIGURES_DIR / '04d_weekly_xcorr.png'
plt.savefig(out_d, dpi=150, bbox_inches='tight')
print(f'Saved: {out_d}')
plt.show()

#%%
# ── Figure 04e: Robustness 3 — first-differenced player counts ───────────────
# First-differencing tests whether engagement predicts inflection points
# (day-over-day change) rather than levels.
pc_diff = pc_daily.diff()

fig5, axes5 = plt.subplots(2, 3, figsize=(15, 9), sharey='row')
xcorr_diff_records = []

for col_idx, (uw_col, vw_col, color) in enumerate(zip(eng_cols, view_cols, colors)):
    field_name = uw_col.replace('_unweighted', '')
    label = FIELD_LABELS.get(field_name, field_name)

    for row_idx, (col, weight_label, c_data) in enumerate([
        (uw_col, 'unweighted',    creator_daily),
        (vw_col, 'view_weighted', creator_view),
    ]):
        ax = axes5[row_idx, col_idx]
        res = gated_peak(c_data[col], pc_diff, MAX_LAG_DAYS, predicted_sign=1)
        if res is None:
            ax.set_title(f'{label}\n(insufficient data)')
            continue
        cc = res['cc']

        ax.bar(cc.index, cc.values, color=color, alpha=0.7, width=1)
        ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.6)
        _annotate_peak(ax, res, 'd')
        ax.legend(fontsize=7)
        ax.set_xlabel('Lag (days)\n+ = creator leads')
        ax.grid(True, alpha=0.25)
        rec = {k: res[k] for k in _REC_KEYS}
        rec.update({'field': field_name, 'weighting': weight_label, 'granularity': 'daily_diff'})
        xcorr_diff_records.append(rec)

        if col_idx == 0:
            ax.set_ylabel(f'{weight_label}\nCross-corr.', fontsize=9)
        if row_idx == 0:
            ax.set_title(label, fontsize=10)

fig5.suptitle(f'Creator engagement → first-differenced player counts  (lags ±{MAX_LAG_DAYS} days)',
              fontsize=12)
plt.tight_layout()
out_e = FIGURES_DIR / '04e_firstdiff_xcorr.png'
plt.savefig(out_e, dpi=150, bbox_inches='tight')
print(f'Saved: {out_e}')
plt.show()

#%%
# ── Figure 04f: Robustness 4 — % of videos in concerning state ───────────────
# Proportion-based metric captures clustering: "many creators burned out at once"
# is a different signal than "mean burnout score drifted slightly negative."
# Threshold < 0 = concerning (not actively playing / burned out / not recommending).
CONCERN_THRESHOLD = 0

concern_cols = [f'{c}_pct_concerning' for c in CREATOR_ENGAGEMENT_COLS if c in yt.columns]

yt_concern = yt.copy()
for c in CREATOR_ENGAGEMENT_COLS:
    if c not in yt.columns:
        continue
    yt_concern[f'{c}_pct_concerning'] = (
        (yt_concern[c] < CONCERN_THRESHOLD)
        .where(yt_concern[c].notna())
        .astype(float)
    )

concern_agg = yt_concern.groupby('date')[concern_cols].mean()
concern_agg.index = pd.to_datetime(concern_agg.index)
concern_daily = concern_agg.reindex(all_dates)

fig6, axes6 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
for ax, col, color in zip(axes6, concern_cols, colors):
    smoothed = rolling_7day(concern_daily[col], min_periods=4)
    ax.plot(smoothed.index, smoothed.values * 100, color=color, lw=1.4)
    ax.set_ylabel(CONCERN_LABELS.get(col, col), fontsize=8)
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', lw=0.7, linestyle=':')
    ax.grid(True, alpha=0.25)
    if ax != axes6[-1]:
        ax.tick_params(labelbottom=False)

axes6[0].set_title('% of creator videos in concerning engagement state (7-day smoothed)',
                   fontsize=13)
axes6[-1].set_xlabel('Date')
plt.tight_layout()
out_f = FIGURES_DIR / '04f_creator_state_proportions.png'
plt.savefig(out_f, dpi=150, bbox_inches='tight')
print(f'Saved: {out_f}')
plt.show()

#%%
# ── Figure 04g: State-proportion xcorr vs. deseasonalized player counts ──────
fig7, axes7 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Concern proportion is oriented opposite to the healthy-high fields:
# more creators in a concerning state should predict FEWER players (predicted_sign=-1).
xcorr_prop_records = _xcorr_panel(
    axes7, concern_cols, concern_daily, pc_deseas, MAX_LAG_DAYS, predicted_sign=-1
)
# Fix field names in prop records (they include _pct_concerning suffix)
for r in xcorr_prop_records:
    r['field'] = r['field'].replace('_pct_concerning', '')
axes7[0].set_ylabel('Cross-correlation coefficient')
fig7.suptitle(f'% creators in concerning state → deseasonalized player counts'
              f'  (lags ±{MAX_LAG_DAYS} days)', fontsize=12)
plt.tight_layout()
out_g = FIGURES_DIR / '04g_state_proportion_xcorr.png'
plt.savefig(out_g, dpi=150, bbox_inches='tight')
print(f'Saved: {out_g}')
plt.show()

#%%
# ── Figure 04h: Release-window-controlled sweep ──────────────────────────────
# Confound (common cause): releases drive BOTH creator output and player counts
# at the same time, manufacturing correlation that has nothing to do with creator
# influence. The noise gate does NOT address this — circular-shift preserves each
# series' own release-time spikes, and their alignment inflates r above the null.
# Here we progressively exclude weeks within +/-W days of any release start and
# re-run the gated weekly xcorr. If the +1-week lead survives as W widens, it is
# independent of the release calendar; if it decays to zero, it was the calendar.
# W=0 = full series. Wide windows are data-limited by the ~13-week season cadence
# (n_weeks panel shows where the sample runs out).
releases  = load_releases()
rel_ord   = pd.to_datetime(releases['start_date']).dropna().values.astype('datetime64[D]').astype(int)
week_ord  = pc_weekly.index.values.astype('datetime64[D]').astype(int)
nearest_release_dist = np.array([int(np.min(np.abs(w - rel_ord))) for w in week_ord])

sweep_records = []
for col in eng_cols:
    field_name = col.replace('_unweighted', '')
    for W in SWEEP_WIDTHS:
        keep = pd.Series(nearest_release_dist > W, index=pc_weekly.index)
        res  = masked_gated_peak(creator_weekly[col], pc_weekly, MAX_LAG_WEEKS,
                                 keep, predicted_sign=1)
        rec = {'field': field_name, 'exclude_days': W}
        if res is None:
            rec.update({'n_weeks': int((keep & creator_weekly[col].notna() & pc_weekly.notna()).sum()),
                        'peak_lag': np.nan, 'peak_corr': np.nan, 'noise_threshold': np.nan,
                        'p_value': np.nan, 'significant': False,
                        'hypothesis_relation': 'insufficient data'})
        else:
            rec.update(res)
        sweep_records.append(rec)

sweep_df = pd.DataFrame(sweep_records)[
    ['field', 'exclude_days', 'n_weeks', 'peak_lag', 'peak_corr',
     'noise_threshold', 'p_value', 'significant', 'hypothesis_relation']
]
print('\n── Release-controlled sweep (weekly, unweighted) ──')
print(sweep_df.to_string(index=False))
sweep_df.to_csv(TABLES_DIR / '04h_release_controlled_sweep.csv', index=False)

fig8, axes8 = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 1.4]})
for col, color in zip(eng_cols, colors):
    field_name = col.replace('_unweighted', '')
    label = FIELD_LABELS.get(field_name, field_name)
    sub = sweep_df[sweep_df['field'] == field_name].sort_values('exclude_days')
    xv  = sub['exclude_days'].to_numpy()
    rv  = sub['peak_corr'].to_numpy(dtype=float)
    tv  = sub['noise_threshold'].to_numpy(dtype=float)
    lv  = sub['peak_lag'].to_numpy(dtype=float)
    sig = sub['significant'].to_numpy(dtype=bool)

    axes8[0].plot(xv, rv, color=color, lw=1.6, alpha=0.85, label=label)
    # noise floor rises as weeks drop — plotting it shows when sig is lost to
    # power (floor rises to meet a stable r) vs effect decay (r itself falls).
    axes8[0].plot(xv, tv, color=color, lw=1.0, alpha=0.45, ls='--')
    axes8[0].scatter(xv[sig],  rv[sig],  color=color, s=55, zorder=5)
    axes8[0].scatter(xv[~sig], rv[~sig], facecolors='none', edgecolors=color, s=55, zorder=5)
    axes8[1].plot(xv, lv, color=color, lw=1.5, alpha=0.8, marker='o')

axes8[0].axhline(0, color='gray', lw=0.8, ls='--')
axes8[0].set_ylabel('Peak cross-correlation (signed)')
axes8[0].set_title('04h: Creator engagement -> player counts, release-window-controlled sweep\n'
                   '(weekly, +/-4wk lags; solid = peak r, dashed = 95% noise floor, '
                   'filled = clears gate)', fontsize=11)
axes8[0].legend(fontsize=9)
axes8[0].grid(True, alpha=0.25)

axes8[1].axhline(0, color='gray', lw=0.8, ls='--')
axes8[1].axhline(1, color='green', lw=0.8, ls=':', alpha=0.6, label='+1 week')
axes8[1].set_ylabel('Peak lag (weeks)\n+ = creator leads')
axes8[1].legend(fontsize=8)
axes8[1].grid(True, alpha=0.25)

nrec = (sweep_df[sweep_df['field'] == eng_cols[0].replace('_unweighted', '')]
        .sort_values('exclude_days'))
axes8[2].bar(nrec['exclude_days'].to_numpy(), nrec['n_weeks'].to_numpy(),
             width=3.0, color='#7f8c8d', alpha=0.75)
axes8[2].set_ylabel('Weeks\nretained')
axes8[2].set_xlabel('Excluded window around each release (+/- days)')
axes8[2].set_xticks(SWEEP_WIDTHS)
axes8[2].grid(True, alpha=0.25, axis='y')

plt.tight_layout()
out_h = FIGURES_DIR / '04h_release_controlled_sweep.png'
plt.savefig(out_h, dpi=150, bbox_inches='tight')
print(f'Saved: {out_h}')
plt.show()

print('\n── Release control: effect-size trajectory + significance ──')
print('  (the gate loses power as weeks drop, so the r-trajectory is the primary read;')
print('   a STABLE r with significance lost at wide W = power loss, not a release artifact)')
for col in eng_cols:
    field_name = col.replace('_unweighted', '')
    label = FIELD_LABELS.get(field_name, field_name)
    sub   = sweep_df[sweep_df['field'] == field_name].sort_values('exclude_days')
    valid = sub.dropna(subset=['peak_corr'])
    if valid.empty:
        print(f'  {label:<28s}: no valid widths')
        continue
    r0       = float(valid.iloc[0]['peak_corr'])
    reliable = valid[valid['n_weeks'] >= RELIABLE_MIN_WEEKS]
    wide     = reliable.iloc[-1] if not reliable.empty else valid.iloc[-1]
    r_wide   = float(wide['peak_corr'])
    w_wide   = int(wide['exclude_days'])
    ratio    = (r_wide / r0) if r0 != 0 else np.nan
    sig_w    = valid.loc[valid['significant'], 'exclude_days'].tolist()

    if np.isnan(ratio) or r0 == 0:
        traj = 'no baseline effect'
    elif ratio >= 0.8:
        traj = f'effect STABLE (r {r0:+.2f} -> {r_wide:+.2f} at W={w_wide}d): robust to the release calendar'
    elif ratio >= 0.5:
        traj = f'effect PARTLY attenuated (r {r0:+.2f} -> {r_wide:+.2f} at W={w_wide}d): releases carry some of it'
    else:
        traj = f'effect DECAYS (r {r0:+.2f} -> {r_wide:+.2f} at W={w_wide}d): largely release-driven'
    sig_note = f'significant at W={sig_w}d' if sig_w else 'never clears the gate'
    print(f'  {label:<28s}: {traj}; {sig_note}')

#%%
# ── 04i: Trend-treatment arm (levels / linear-detrended / first-differenced) ─
# Everything above runs on weekly LEVELS. Where creator engagement and player
# counts trend in opposite (or the same) directions, a levels correlation is
# dominated by the shared secular trend, which the release sweep cannot remove
# because a trend is not release-locked. This arm re-runs the gated weekly
# xcorr under three treatments so the paper cites a persisted table rather than
# an ad hoc run. Detrending is a separate OLS on time for each series (fitted on
# its non-NaN weeks); first differences are week-over-week on the regular index.
def _linear_detrend(s: pd.Series) -> pd.Series:
    t  = np.arange(len(s), dtype=float)
    ok = s.notna().values
    if ok.sum() < 3:
        return s * np.nan
    b, a = np.polyfit(t[ok], s.values[ok], 1)
    return s - (a + b * t)


def _time_corr(s: pd.Series) -> float:
    ok = s.notna()
    return float(np.corrcoef(np.arange(len(s))[ok.values], s[ok].values)[0, 1])


TREATMENTS = {
    'levels':     lambda s: s,
    'detrended':  _linear_detrend,
    'first_diff': lambda s: s.diff(),
}

trend_records = []
for uw_col in eng_cols:
    field_name = uw_col.replace('_unweighted', '')
    for treatment, fn in TREATMENTS.items():
        cx  = fn(creator_weekly[uw_col])
        py  = fn(pc_weekly)
        res = gated_peak(cx, py, MAX_LAG_WEEKS, predicted_sign=1)
        if res is None:
            continue
        rec = {k: res[k] for k in _REC_KEYS}
        rec.update({'field': field_name, 'treatment': treatment})
        for k in range(-MAX_LAG_WEEKS, MAX_LAG_WEEKS + 1):
            rec[f'r_at_{k:+d}'] = round(float(res['cc'].get(k, np.nan)), 4)
        rec['creator_time_corr'] = round(_time_corr(creator_weekly[uw_col]), 4)
        rec['players_time_corr'] = round(_time_corr(pc_weekly), 4)
        trend_records.append(rec)

trend_df = pd.DataFrame(trend_records)
lead_cols = ['field', 'treatment'] + list(_REC_KEYS)
trend_df  = trend_df[lead_cols + [c for c in trend_df.columns if c not in lead_cols]]
print('\n── 04i: Trend treatments (weekly, unweighted) ──')
print(trend_df[['field', 'treatment', 'peak_lag', 'peak_corr', 'p_value',
                'significant', 'r_at_+0', 'r_at_+1', 'r_at_+2']].to_string(index=False))
trend_df.to_csv(TABLES_DIR / '04i_trend_treatments.csv', index=False)
print(f'Saved: {TABLES_DIR / "04i_trend_treatments.csv"}')

# Both controls at once: detrend each full series first, then exclude release-
# adjacent weeks (the order 05's H1 battery uses). A field whose levels lead is
# carried by the release calendar can survive detrending alone, and one carried
# by a trend can survive release exclusion alone; only this cell tests both.
detrend_sweep = []
for col in eng_cols:
    field_name = col.replace('_unweighted', '')
    cx, py = _linear_detrend(creator_weekly[col]), _linear_detrend(pc_weekly)
    for W in SWEEP_WIDTHS:
        keep = pd.Series(nearest_release_dist > W, index=pc_weekly.index)
        res  = masked_gated_peak(cx, py, MAX_LAG_WEEKS, keep, predicted_sign=1)
        rec  = {'field': field_name, 'exclude_days': W}
        if res is None:
            rec.update({'hypothesis_relation': 'insufficient data', 'significant': False})
        else:
            rec.update(res)
            rec['r_at_+1'] = round(float(masked_lag_corr(cx, py, MAX_LAG_WEEKS, keep).get(1, np.nan)), 4)
        detrend_sweep.append(rec)

detrend_sweep_df = pd.DataFrame(detrend_sweep)
print('\n── 04i: Detrended release-controlled sweep (weekly, unweighted) ──')
print(detrend_sweep_df.to_string(index=False))
detrend_sweep_df.to_csv(TABLES_DIR / '04i_detrended_release_sweep.csv', index=False)
print(f'Saved: {TABLES_DIR / "04i_detrended_release_sweep.csv"}')

# Window edge and season phase. The detrended, release-excluded peaks above sit
# at +4 weeks, the edge of the window, so the true peak may lie beyond it. And
# two series locked to a ~10-12-week season cycle correlate at a fixed lag from
# cycle phase alone, which dropping release weeks does not remove. So: widen the
# window to +/-12 weeks, and residualize both series on weeks-since-last-release
# (dummies 0..SEASON_PHASE_CAP-1, later weeks pooled) instead of dropping weeks.
WIDE_LAG_WEEKS   = 12
SEASON_PHASE_CAP = 16
release_starts = rel_ord


def _season_phase_residualize(s: pd.Series) -> pd.Series:
    wk = s.index.values.astype('datetime64[D]').astype(int)
    since = np.full(len(wk), SEASON_PHASE_CAP, dtype=int)
    for i, w in enumerate(wk):
        prior = release_starts[release_starts <= w]
        if prior.size:
            since[i] = min((w - prior.max()) // 7, SEASON_PHASE_CAP)
    return s - s.groupby(pd.Series(since, index=s.index)).transform('mean')


wide_records = []
for col in eng_cols:
    field_name = col.replace('_unweighted', '')
    for treatment, cx, py in [
        ('detrended',                 _linear_detrend(creator_weekly[col]), _linear_detrend(pc_weekly)),
        ('season_phase',              _season_phase_residualize(creator_weekly[col]), _season_phase_residualize(pc_weekly)),
        ('detrended+season_phase',    _season_phase_residualize(_linear_detrend(creator_weekly[col])),
                                      _season_phase_residualize(_linear_detrend(pc_weekly))),
    ]:
        for L in (MAX_LAG_WEEKS, WIDE_LAG_WEEKS):
            res = gated_peak(cx, py, L, predicted_sign=1)
            if res is None:
                continue
            rec = {k: res[k] for k in _REC_KEYS}
            rec.update({'field': field_name, 'treatment': treatment, 'max_lag_weeks': L,
                        'r_at_+1': round(float(res['cc'].get(1, np.nan)), 4)})
            wide_records.append(rec)

wide_df = pd.DataFrame(wide_records)
wide_df = wide_df[['field', 'treatment', 'max_lag_weeks'] + list(_REC_KEYS) + ['r_at_+1']]
print('\n── 04i: Wide window and season-phase control (weekly, unweighted) ──')
print(wide_df.to_string(index=False))
wide_df.to_csv(TABLES_DIR / '04i_season_phase_wide.csv', index=False)
print(f'Saved: {TABLES_DIR / "04i_season_phase_wide.csv"}')

# The window-edge peaks are in the detrended, release-EXCLUDED series, so widen
# those too: same sweep as above at +/-12 weeks.
edge_records = []
for col in eng_cols:
    field_name = col.replace('_unweighted', '')
    cx, py = _linear_detrend(creator_weekly[col]), _linear_detrend(pc_weekly)
    for W in SWEEP_WIDTHS:
        keep = pd.Series(nearest_release_dist > W, index=pc_weekly.index)
        res  = masked_gated_peak(cx, py, WIDE_LAG_WEEKS, keep, predicted_sign=1)
        rec  = {'field': field_name, 'exclude_days': W, 'max_lag_weeks': WIDE_LAG_WEEKS}
        if res is None:
            rec.update({'hypothesis_relation': 'insufficient data', 'significant': False})
        else:
            rec.update(res)
        edge_records.append(rec)
edge_df = pd.DataFrame(edge_records)
print('\n── 04i: Detrended release-controlled sweep at +/-12 weeks ──')
print(edge_df.to_string(index=False))
edge_df.to_csv(TABLES_DIR / '04i_detrended_release_sweep_wide.csv', index=False)
print(f'Saved: {TABLES_DIR / "04i_detrended_release_sweep_wide.csv"}')

#%%
# ── Extended summary table ────────────────────────────────────────────────────
all_records = (
    [dict(r, weighting='unweighted',    granularity='daily')       for r in xcorr_records]
    + [dict(r, weighting='view_weighted', granularity='daily')     for r in xcorr_view_records]
    + xcorr_weekly_records
    + xcorr_diff_records
    + [dict(r, weighting='pct_concerning', granularity='daily')    for r in xcorr_prop_records]
)

summary_df = pd.DataFrame(all_records)[[
    'field', 'weighting', 'granularity', 'peak_lag', 'peak_corr',
    'noise_threshold', 'p_value', 'significant', 'hypothesis_relation',
]]
summary_df['peak_corr'] = summary_df['peak_corr'].round(4)
print(summary_df.to_string(index=False))
out_table = TABLES_DIR / '04_engagement_xcorr_extended.csv'
summary_df.to_csv(out_table, index=False)
print(f'\nSaved: {out_table}')
