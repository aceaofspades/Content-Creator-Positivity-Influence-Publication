#%%
"""
12_coverage_volume.py
----------------------
Tests whether creator activity VOLUME (independent of tone) co-moves with, leads,
or lags player counts -- the "any press is good press" hypothesis. Distinct claim
from the sentiment scripts (04-07): quantity of attention, not valence.

PRIMARY TREATMENT = FIRST DIFFERENCES (why this matters, read before editing):
  The volume series carry strong SECULAR TRENDS that player counts do NOT share.
  `video_count` trends UP over the series (the D2 creator ecosystem grew:
  corr(video_count, time) ~ +0.56), while player counts trend DOWN (~ -0.49).
  Those OPPOSING trends *mask* real co-movement in the raw levels: raw lag-0
  corr(video_count, players) ~ +0.09, but once the trend is removed it is ~ +0.48
  (first-differenced) / ~ +0.51 (linearly detrended). Conversely
  `view_weighted_volume` barely trends (~ +0.09 with time), so its raw lag-0
  r ~ +0.44 already reflects the de-trended co-movement.

  => Raw levels are MISLEADING here, and a level-based release sweep is confounded
     by trend (a trend is not release-locked, so excising release weeks leaves it
     intact). We therefore FIRST-DIFFERENCE both sides (Delta week-over-week) as the
     primary lens, and show the raw levels ONLY to illustrate the trend-masking.
     (An earlier version of this script reported a levels-based "the two volume
     twins disagree, so view_weighted is a view-count-bias artifact" verdict. That
     was itself a trend artifact -- video_count's upward trend masked its
     co-movement. RETRACTED: once de-trended, count and reach AGREE.)

DOW deseasonalization is a structural no-op at weekly scale (a W-MON mean already
contains one of each weekday, so the DOW component is a constant that drops out
under mean-centring). The preprocessing that matters here is TREND removal, above.

Volume series (weekly, from YouTube creators):
  video_count          -- weekly video count (pure count; IMMUNE to view-count bias)
  view_weighted_volume -- weekly sum of view_count (reach; EXPOSED to accumulation bias)
  positive_volume / negative_volume -- weekly counts of videos with overall__score >0 / <0

Valence (tone) constructs -- volume-normalised so they are not just scaled copies of
the release-driven total (positive+negative ~ total):
  positive_share    -- positive_volume / video_count
  net_valence_share -- (positive_volume - negative_volume) / video_count

Noise gate: every peak is gated with a max-across-lags circular-shift null (N_PERM
shuffles; peak must clear the 95th percentile of the null max-|r|; mirrors 04/05b).
Sub-threshold peaks are drawn grey and labelled (ns).

CCF-shape / asymmetry statistic (co-movement vs. a weak lead): reports r at lags
{-2,-1,0,+1,+2} and a near-zero asymmetry index A = (r[+1]+r[+2]) - (r[-1]+r[-2]),
gated two-sided. A>0 = coverage-lead shape; A<0 = players-lead; A~0/ns = symmetric
co-movement. On the first-differenced series the peak sits firmly at lag 0 and A is
symmetric -> pure contemporaneous coupling, no lead.

Release-controlled sweep (reuses 04h's machinery): progressively exclude weeks within
+/-W days of any release start (W = 0..42) and re-run the gated xcorr on the
first-differenced surviving weeks via date-aware per-lag Pearson (masked_lag_corr --
shifts on the regular index before masking, so 'lag h' is a true h-week offset). If
the lag-0 coupling survives as W widens, it is INDEPENDENT of the release calendar.

KEY FINDING: week-over-week changes in coverage volume co-move with changes in player
counts at r ~ 0.4, at LAG 0, and this SURVIVES excluding +/-21-28 days around every
release (204-260 of 425 weeks remain). `video_count` (bias-immune) and
`view_weighted_volume` (bias-exposed) AGREE. So the coupling is real, release-
independent, trend-robust, and measurement-robust -- but it is strictly CONCURRENT
(lag 0), so it is co-movement / common-cause, NOT a creator lead. "Any press is good
press" holds as a correlation (tone-independent), not as a causal driver. The
elite-influence null (no lead) is intact and cleaner.

view_count is accumulated views at collection time, not contemporary reach at upload;
contemporary reach is not recoverable. It barely trends here, so accumulation is not
the driver; robustness is confirmed by video_count (immune) agreeing with
view_weighted_volume (exposed) once de-trended.

Outputs:
  figures/12a_coverage_volume_timeseries.png   -- raw series + player counts (context)
  figures/12b_coverage_volume_xcorr.png        -- first-differenced gated xcorr + levels-vs-diff contrast
  figures/12c_valence_test.png                 -- differenced volume-normalised tone vs players
  figures/12d_release_controlled_sweep.png     -- differenced sweep: count vs view-weighted
  figures/12e_ccf_shape.png                    -- differenced r at +/-2 lags + asymmetry
  tables/12_coverage_volume_xcorr.csv          -- peak/gate/CCF-shape/asymmetry (treatment col)
  tables/12_release_controlled_sweep.csv       -- differenced sweep trajectory per series x width
  tables/12_valence_test.csv                   -- differenced tone results
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive: save figures to file, never open a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_player_counts, load_releases,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import rolling_7day

MAX_LAG_WEEKS  = 8
N_PERM         = 1_000
NULL_PCTILE    = 95
MIN_PERM_SHIFT = 4    # weeks; avoid near-identity circular shifts
RANDOM_SEED    = 42

# Release-controlled sweep (mirrors 04h): exclude weeks within +/-W days of any
# release start and re-run the gated (first-differenced) weekly xcorr for each W.
SWEEP_WIDTHS       = [0, 7, 14, 21, 28, 42]
MIN_PAIRS          = 20    # min aligned week-pairs to compute a lagged correlation
RELIABLE_MIN_WEEKS = 150   # below this the masked sample is too thin to trust r-trajectory


def save_csv(df: pd.DataFrame, path) -> None:
    """Write a CSV, but don't crash the whole run if the file is locked (open in
    Excel / OneDrive sync). Warn and continue so already-computed outputs survive."""
    try:
        df.to_csv(path, index=False)
        print(f'Saved: {path}')
    except PermissionError:
        print(f'WARNING: could not write {path} (file is locked / open elsewhere). '
              f'Close it and re-run to persist this table.')

#%%
ensure_output_dirs()

yt       = load_youtube_creators()
pc       = load_player_counts(source='total').set_index('date').sort_index()['players']
releases = load_releases()

pc.index = pd.to_datetime(pc.index)

#%%
# ── Build weekly coverage series (raw levels) ────────────────────────────────
yt = yt.copy()
yt['date'] = pd.to_datetime(yt['date'])
yt_indexed = yt.set_index('date').sort_index()

video_count          = yt_indexed.resample('W-MON').size().rename('video_count')
view_weighted_volume = yt_indexed['view_count'].resample('W-MON').sum().rename('view_weighted_volume')
positive_volume      = (yt_indexed['overall__score'].resample('W-MON')
                        .apply(lambda x: (x > 0).sum()).rename('positive_volume'))
negative_volume      = (yt_indexed['overall__score'].resample('W-MON')
                        .apply(lambda x: (x < 0).sum()).rename('negative_volume'))

view_any_observed = yt_indexed['view_count'].resample('W-MON').count() > 0
view_weighted_volume[~view_any_observed] = np.nan

zero_weeks = video_count == 0
positive_volume[zero_weeks] = np.nan
negative_volume[zero_weeks] = np.nan

positive_share    = (positive_volume / video_count).rename('positive_share')
net_valence_share = ((positive_volume - negative_volume) / video_count).rename('net_valence_share')

weekly = pd.DataFrame({
    'video_count':          video_count,
    'view_weighted_volume': view_weighted_volume,
    'positive_volume':      positive_volume,
    'negative_volume':      negative_volume,
    'positive_share':       positive_share,
    'net_valence_share':    net_valence_share,
})

pc_weekly = pc.resample('W-MON').mean()

start_w   = max(weekly.index.min(), pc_weekly.index.min())
end_w     = min(weekly.index.max(), pc_weekly.index.max())
all_weeks = pd.date_range(start_w, end_w, freq='W-MON')
weekly    = weekly.reindex(all_weeks)
pc_weekly = pc_weekly.reindex(all_weeks)

# ── First differences (PRIMARY treatment) ────────────────────────────────────
weekly_d = weekly.diff()
pc_weekly_d = pc_weekly.diff().rename('players')

# ── Trend diagnostics (why levels mislead) ───────────────────────────────────
_t = np.arange(len(all_weeks), dtype=float)
def _trend_r(s: pd.Series) -> float:
    v = s.to_numpy(dtype=float)
    m = ~np.isnan(v)
    return float(np.corrcoef(_t[m], v[m])[0, 1]) if m.sum() > 10 else np.nan

trend_video   = _trend_r(weekly['video_count'])
trend_view    = _trend_r(weekly['view_weighted_volume'])
trend_players = _trend_r(pc_weekly)

print(f'Weekly range: {start_w.date()} -> {end_w.date()}  ({len(all_weeks)} weeks)')
print('\nSecular trend (corr with time) -- why raw levels mislead:')
print(f'  video_count   vs time: {trend_video:+.3f}   (ecosystem grew)')
print(f'  view_weighted vs time: {trend_view:+.3f}   (roughly flat)')
print(f'  players       vs time: {trend_players:+.3f}   (declined)')
print('  -> video_count and players trend in OPPOSITE directions, so raw levels')
print('     mask the real co-movement. First differences remove the trend.')
print('\nMean weekly tone-share (volume-normalised):')
print(f'  positive_share    : {weekly["positive_share"].mean():.3f}')
print(f'  net_valence_share : {weekly["net_valence_share"].mean():.3f}')
print('\nNOTE: view_count is accumulated at collection time, not contemporary reach at '
      'upload. It barely trends here, so accumulation is not the driver; robustness is '
      'judged by whether video_count (immune) and view_weighted_volume (exposed) agree.')


#%%
# ── Cross-correlation + noise-gate machinery (mirrors script 04) ─────────────
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed. Preserves the
    series' marginal distribution and (with sparse NaNs) its own autocorrelation;
    destroys only the cross-timing with the target. Matches the null in 04/05b."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def _ccf_shape(cc: pd.Series):
    """r at lags -2..+2 and the near-zero asymmetry index A."""
    def g(h):
        v = cc.get(h, np.nan)
        return float(v) if v is not None else np.nan
    r = {'r_m2': g(-2), 'r_m1': g(-1), 'r_0': g(0), 'r_p1': g(1), 'r_p2': g(2)}
    asym = (g(1) + g(2)) - (g(-1) + g(-2))
    return r, asym


def analyze_series(s: pd.Series, y: pd.Series, max_lag: int,
                   n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """Hypothesis-blind peak + max-|r| noise gate + CCF-shape + asymmetry test.
    One circular-shift permutation loop builds BOTH the peak null (max|r| across
    lags) and the asymmetry null. Returns a result dict or None."""
    cc = normalised_xcorr(s, y, max_lag)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])
    rshape, asym = _ccf_shape(cc)

    rng = np.random.default_rng(seed)
    n   = len(s)
    null_max, null_asym = [], []
    for _ in range(n_perm):
        shift = int(rng.integers(MIN_PERM_SHIFT, max(n, MIN_PERM_SHIFT + 1)))
        cc_n  = normalised_xcorr(shift_values_only(s, shift), y, max_lag)
        if not cc_n.empty:
            null_max.append(float(cc_n.abs().max()))
            _, a_n = _ccf_shape(cc_n)
            if not np.isnan(a_n):
                null_asym.append(a_n)
    null_max  = np.asarray(null_max)
    null_asym = np.asarray(null_asym)

    if null_max.size:
        threshold   = float(np.percentile(null_max, NULL_PCTILE))
        p_value     = float(np.mean(null_max >= abs(peak_corr)))
        significant = bool(abs(peak_corr) > threshold)
    else:
        threshold, p_value, significant = np.nan, np.nan, False

    asym_p = float(np.mean(np.abs(null_asym) >= abs(asym))) if null_asym.size else np.nan
    if np.isnan(asym_p) or asym_p >= 0.05:
        asym_verdict = 'symmetric (co-movement, no lead)'
    elif asym > 0:
        asym_verdict = 'right-heavy (coverage leads)'
    else:
        asym_verdict = 'left-heavy (players lead)'

    return {
        'cc':              cc,
        'peak_lag':        peak_lag,
        'peak_corr':       round(peak_corr, 4),
        'corr_at_0':       round(rshape['r_0'], 4),
        'noise_threshold': round(threshold, 4) if not np.isnan(threshold) else np.nan,
        'p_value':         round(p_value, 4) if not np.isnan(p_value) else np.nan,
        'significant':     significant,
        'r_m2':            round(rshape['r_m2'], 4),
        'r_m1':            round(rshape['r_m1'], 4),
        'r_p1':            round(rshape['r_p1'], 4),
        'r_p2':            round(rshape['r_p2'], 4),
        'asymmetry':       round(asym, 4),
        'asym_p':          round(asym_p, 4) if not np.isnan(asym_p) else np.nan,
        'asym_verdict':    asym_verdict,
        'n_perm':          len(null_max),
    }


def _lagged_corr_np(xv: np.ndarray, yv: np.ndarray, max_lag: int) -> dict:
    """Numpy core of the date-aware lagged Pearson (both arrays may hold NaN).
    Pairs xv[t] with yv[t+h] (i.e. y.shift(-h)); positive lag = x leads y."""
    n = len(xv)
    out = {}
    for h in range(-max_lag, max_lag + 1):
        if h >= 0:
            a, b = xv[0:n - h], yv[h:n]
        else:
            a, b = xv[-h:n], yv[0:n + h]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < MIN_PAIRS:
            continue
        aa = a[m] - a[m].mean()
        bb = b[m] - b[m].mean()
        denom = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
        if denom > 0:
            out[h] = float((aa * bb).sum() / denom)
    return out


def masked_lag_corr(x: pd.Series, y: pd.Series, max_lag: int, keep: pd.Series) -> pd.Series:
    """Date-aware lagged Pearson on a regular weekly index, using only weeks where
    `keep` is True. Shifts happen on the regular index BEFORE masking, so 'lag h'
    always means a true h-week offset (no gap-collapsing). Mirrors 04h. NOTE: the
    input is already first-differenced on the full index, so a diff at a kept week
    may span one release boundary -- a minor edge effect at wide W."""
    xv = x.where(keep).to_numpy(dtype=float)
    yv = y.where(keep).to_numpy(dtype=float)
    return pd.Series(_lagged_corr_np(xv, yv, max_lag), dtype=float)


def masked_gated_peak(x: pd.Series, y: pd.Series, max_lag: int, keep: pd.Series,
                      n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """Release-sweep gated peak: hypothesis-blind selection + circular-shift gate
    on the masked sample via date-aware masked_lag_corr."""
    cc = masked_lag_corr(x, y, max_lag, keep)
    if cc.empty:
        return None
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])

    # Numpy null loop: circular-shift the kept, non-NaN x values in place, keeping
    # NaN positions fixed (identical semantics to shift_values_only).
    xk_v      = x.where(keep).to_numpy(dtype=float)
    yv        = y.where(keep).to_numpy(dtype=float)
    valid_idx = np.where(~np.isnan(xk_v))[0]
    base_vals = xk_v[valid_idx]
    n_valid   = len(base_vals)
    rng       = np.random.default_rng(seed)
    null_max  = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, max(n_valid, 2)))
        perm_xv = xk_v.copy()
        perm_xv[valid_idx] = np.roll(base_vals, shift % max(n_valid, 1))
        cc_n = _lagged_corr_np(perm_xv, yv, max_lag)
        if cc_n:
            null_max.append(max(abs(v) for v in cc_n.values()))
    null_max = np.asarray(null_max)

    if null_max.size:
        threshold   = float(np.percentile(null_max, NULL_PCTILE))
        p_value     = float(np.mean(null_max >= abs(peak_corr)))
        significant = bool(abs(peak_corr) > threshold)
    else:
        threshold, p_value, significant = np.nan, np.nan, False

    return {
        'peak_lag':        peak_lag,
        'peak_corr':       round(peak_corr, 4),
        'corr_at_0':       round(float(cc.get(0, np.nan)), 4),
        'noise_threshold': round(threshold, 4) if not np.isnan(threshold) else np.nan,
        'p_value':         round(p_value, 4) if not np.isnan(p_value) else np.nan,
        'significant':     significant,
        'n_weeks':         int((keep & x.notna() & y.notna()).sum()),
    }


def _draw_gated_bars(ax, res, color, unit='w'):
    """Bars + noise band + gated (red) / sub-threshold (grey, ns) peak line."""
    cc, thr = res['cc'], res['noise_threshold']
    ax.bar(cc.index, cc.values, color=color, alpha=0.7, width=0.6)
    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.6)
    if not np.isnan(thr):
        ax.axhspan(-thr, thr, color='gray', alpha=0.12)
    if res['significant']:
        ax.axvline(res['peak_lag'], color='red', lw=1.5, linestyle=':',
                   label=f'Peak: lag={res["peak_lag"]:+d}{unit}  r={res["peak_corr"]:+.3f}\n'
                         f'p={res["p_value"]:.3f} (clears gate)')
    else:
        ax.axvline(res['peak_lag'], color='gray', lw=1.2, linestyle=':',
                   label=f'Max|r|: lag={res["peak_lag"]:+d}{unit}  r={res["peak_corr"]:+.3f}\n'
                         f'(ns, p={res["p_value"]:.3f})')
    ax.legend(fontsize=7, loc='upper left')


#%%
# ── Figure 12a: Timeseries overview (raw context) ────────────────────────────
daily_range = pd.date_range(start_w, end_w, freq='D')
pc_daily    = pc.reindex(daily_range)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 1, hspace=0.08)
ax_pc  = fig.add_subplot(gs[0])
ax_cnt = fig.add_subplot(gs[1], sharex=ax_pc)
ax_vol = fig.add_subplot(gs[2], sharex=ax_pc)

pc_smooth = rolling_7day(pc_daily)
ax_pc.plot(pc_smooth.index, pc_smooth.values, color='#2c3e50', lw=1.5)
ax_pc.set_ylabel('Players (7d MA)', fontsize=9)
ax_pc.tick_params(labelbottom=False)
ax_pc.grid(True, alpha=0.2)

for _, rel in releases.iterrows():
    if pd.isna(rel['start_date']):
        continue
    for ax in [ax_pc, ax_cnt, ax_vol]:
        ax.axvline(rel['start_date'], color='black', lw=0.4, alpha=0.3)

ax_cnt.plot(weekly.index, weekly['video_count'],
            color='#3498db', lw=1.4, alpha=0.9, label='All videos')
ax_cnt.fill_between(weekly.index, weekly['positive_volume'].fillna(0),
                    alpha=0.4, color='#2ecc71', label='Positive (score > 0)')
ax_cnt.fill_between(weekly.index, -weekly['negative_volume'].fillna(0),
                    alpha=0.4, color='#e74c3c', label='Negative (score < 0)')
ax_cnt.axhline(0, color='gray', lw=0.6)
ax_cnt.set_ylabel('Videos per week', fontsize=9)
ax_cnt.legend(fontsize=8, loc='upper right')
ax_cnt.tick_params(labelbottom=False)
ax_cnt.grid(True, alpha=0.2)

ax_vol.plot(weekly.index, weekly['view_weighted_volume'] / 1e6, color='#8e44ad', lw=1.3)
ax_vol.set_ylabel('View-weighted volume\n(millions)', fontsize=9)
ax_vol.set_xlabel('Date')
ax_vol.grid(True, alpha=0.2)

fig.suptitle('Creator coverage volume and player counts over time (raw levels)', fontsize=14)
out_a = FIGURES_DIR / '12a_coverage_volume_timeseries.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.close(fig)


#%%
# ── Primary: first-differenced gated xcorr for each volume series ─────────────
coverage_specs = [
    ('video_count',          '#3498db', 'Video count (raw)'),
    ('view_weighted_volume', '#8e44ad', 'View-weighted volume'),
    ('positive_volume',      '#2ecc71', 'Positive-sentiment videos'),
    ('negative_volume',      '#e74c3c', 'Negative-sentiment videos'),
]

diff_res, xcorr_records = {}, []
for name, color, label in coverage_specs:
    res = analyze_series(weekly_d[name], pc_weekly_d, MAX_LAG_WEEKS)
    diff_res[name] = res
    if res is None:
        print(f'{label}: insufficient data.')
        continue
    rec = {k: res[k] for k in res if k != 'cc'}
    xcorr_records.append({'series': name, 'treatment': 'first_diff', **rec})
    flag = 'SIG' if res['significant'] else 'ns '
    print(f'[diff] {label:<28s} peak={res["peak_lag"]:+d}w r={res["peak_corr"]:+.3f} '
          f'p={res["p_value"]:.3f} [{flag}]  r0={res["corr_at_0"]:+.3f}  '
          f'asym={res["asymmetry"]:+.3f} (p={res["asym_p"]:.3f}) -> {res["asym_verdict"]}')

# Levels results for the two bias-robustness twins (to illustrate trend-masking)
lvl_res = {}
for name in ['video_count', 'view_weighted_volume']:
    res = analyze_series(weekly[name], pc_weekly, MAX_LAG_WEEKS)
    lvl_res[name] = res
    if res is not None:
        rec = {k: res[k] for k in res if k != 'cc'}
        xcorr_records.append({'series': name, 'treatment': 'levels', **rec})
        print(f'[levels] {name:<26s} r0={res["corr_at_0"]:+.3f}  '
              f'peak={res["peak_lag"]:+d}w r={res["peak_corr"]:+.3f} '
              f'(trend-masked; compare to [diff] above)')


#%%
# ── Figure 12b: differenced gated xcorr (2x2) + levels-vs-diff contrast ───────
fig2 = plt.figure(figsize=(14, 13))
gs2  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)
ax_grid = [fig2.add_subplot(gs2[0, 0]), fig2.add_subplot(gs2[0, 1]),
           fig2.add_subplot(gs2[1, 0]), fig2.add_subplot(gs2[1, 1])]
ax_ct = fig2.add_subplot(gs2[2, :])

for ax, (name, color, label) in zip(ax_grid, coverage_specs):
    res = diff_res.get(name)
    if res is None:
        ax.set_title(f'{label}\n(insufficient data)', fontsize=9)
        continue
    _draw_gated_bars(ax, res, color)
    ax.set_title(f'{label} (Delta week-over-week)', fontsize=10)
    ax.set_xlabel('Lag (weeks)  [+ = coverage leads players]', fontsize=8)
    ax.set_ylabel('Cross-correlation', fontsize=8)
    ax.grid(True, alpha=0.25)

# Levels-vs-differences contrast (the trend-masking illustration)
contrast = [('video_count', '#3498db'), ('view_weighted_volume', '#8e44ad')]
for name, color in contrast:
    dl = diff_res.get(name)
    lv = lvl_res.get(name)
    lab = 'video count' if name == 'video_count' else 'view-weighted'
    if dl is not None:
        ax_ct.plot(dl['cc'].index, dl['cc'].values, color=color, lw=2.0, alpha=0.9,
                   label=f'{lab} DIFFERENCED (r@0={dl["corr_at_0"]:+.2f})')
    if lv is not None:
        ax_ct.plot(lv['cc'].index, lv['cc'].values, color=color, lw=1.4, alpha=0.6,
                   ls='--', label=f'{lab} LEVELS (r@0={lv["corr_at_0"]:+.2f}, trend-masked)')
ax_ct.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax_ct.axhline(0, color='gray', lw=0.6)
ax_ct.set_title('Why levels mislead: removing the trend unmasks the co-movement '
                '(esp. video_count, +0.09 -> +0.48 at lag 0)', fontsize=10)
ax_ct.set_xlabel('Lag (weeks)  [+ = coverage leads players]', fontsize=9)
ax_ct.set_ylabel('Cross-correlation', fontsize=9)
ax_ct.legend(fontsize=8, loc='upper left')
ax_ct.grid(True, alpha=0.25)

fig2.suptitle('Creator coverage volume -> player counts  (first differences, gated +/-8w)',
              fontsize=13)
out_b = FIGURES_DIR / '12b_coverage_volume_xcorr.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')
plt.close(fig2)


#%%
# ── Figure 12e: CCF shape (+/-2 lags) + asymmetry per series (differenced) ────
fig5, axes5 = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, color, label) in zip(axes5.ravel(), coverage_specs):
    res = diff_res.get(name)
    if res is None:
        ax.set_title(f'{label}\n(insufficient data)', fontsize=9)
        continue
    lags = [-2, -1, 0, 1, 2]
    vals = [res['r_m2'], res['r_m1'], res['corr_at_0'], res['r_p1'], res['r_p2']]
    bar_colors = ['#95a5a6', '#95a5a6', color, color, color]
    ax.bar(lags, vals, color=bar_colors, alpha=0.8, width=0.6)
    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(f'{label}\nA={res["asymmetry"]:+.3f} (p={res["asym_p"]:.3f}) '
                 f'-> {res["asym_verdict"]}', fontsize=9)
    ax.set_xlabel('Lag (weeks)  [+ = coverage leads]', fontsize=8)
    ax.set_ylabel('r', fontsize=8)
    ax.set_xticks(lags)
    ax.grid(True, alpha=0.25, axis='y')
fig5.suptitle('CCF shape near lag 0 (first differences): a lag-0 peak with symmetric\n'
              'shoulders = pure contemporaneous coupling, no lead in either direction',
              fontsize=12)
plt.tight_layout()
out_e = FIGURES_DIR / '12e_ccf_shape.png'
plt.savefig(out_e, dpi=150, bbox_inches='tight')
print(f'Saved: {out_e}')
plt.close(fig5)


#%%
# ── Release-controlled sweep on FIRST-DIFFERENCED series (mirrors 04h) ────────
rel_ord  = pd.to_datetime(releases['start_date']).dropna().values.astype('datetime64[D]').astype(int)
week_ord = pc_weekly.index.values.astype('datetime64[D]').astype(int)
nearest_release_dist = np.array([int(np.min(np.abs(w - rel_ord))) for w in week_ord])

sweep_series = ['video_count', 'view_weighted_volume', 'positive_volume',
                'negative_volume', 'net_valence_share']
sweep_records = []
for name in sweep_series:
    for W in SWEEP_WIDTHS:
        keep = pd.Series(nearest_release_dist > W, index=pc_weekly.index)
        res  = masked_gated_peak(weekly_d[name], pc_weekly_d, MAX_LAG_WEEKS, keep)
        rec  = {'series': name, 'exclude_days': W}
        if res is None:
            rec.update({'n_weeks': int((keep & weekly_d[name].notna() & pc_weekly_d.notna()).sum()),
                        'peak_lag': np.nan, 'peak_corr': np.nan, 'corr_at_0': np.nan,
                        'noise_threshold': np.nan, 'p_value': np.nan, 'significant': False})
        else:
            rec.update(res)
        sweep_records.append(rec)

sweep_df = pd.DataFrame(sweep_records)[
    ['series', 'exclude_days', 'n_weeks', 'peak_lag', 'peak_corr', 'corr_at_0',
     'noise_threshold', 'p_value', 'significant']
]
print('\n-- Release-controlled sweep (first-differenced, weekly) --')
print(sweep_df.to_string(index=False))
save_csv(sweep_df, TABLES_DIR / '12_release_controlled_sweep.csv')

print('\n-- Release control: lag-0 coupling trajectory (differenced) --')
print('  (peak sits at lag 0; report corr_at_0. STABLE r losing sig only as n falls')
print('   = power loss, not decay. Agreement of video_count vs view_weighted = bias-robust.)')
for name in sweep_series:
    sub   = sweep_df[sweep_df['series'] == name].sort_values('exclude_days')
    valid = sub.dropna(subset=['corr_at_0'])
    if valid.empty:
        print(f'  {name:<22s}: no valid widths')
        continue
    r0       = float(valid.iloc[0]['corr_at_0'])
    reliable = valid[valid['n_weeks'] >= RELIABLE_MIN_WEEKS]
    wide     = reliable.iloc[-1] if not reliable.empty else valid.iloc[-1]
    r_wide, w_wide = float(wide['corr_at_0']), int(wide['exclude_days'])
    ratio    = (r_wide / r0) if r0 != 0 else np.nan
    sig_w    = valid.loc[valid['significant'], 'exclude_days'].tolist()
    if np.isnan(ratio) or r0 == 0:
        traj = 'no baseline coupling'
    elif ratio >= 0.7:
        traj = f'STABLE (r@0 {r0:+.2f} -> {r_wide:+.2f} @W={w_wide}d): release-independent coupling'
    elif ratio >= 0.4:
        traj = f'PARTLY attenuated (r@0 {r0:+.2f} -> {r_wide:+.2f} @W={w_wide}d)'
    else:
        traj = f'DECAYS (r@0 {r0:+.2f} -> {r_wide:+.2f} @W={w_wide}d): largely release-driven'
    sig_note = f'sig at W={sig_w}d' if sig_w else 'never clears the gate'
    print(f'  {name:<22s}: {traj}; {sig_note}')

#%%
# ── Figure 12d: differenced sweep for the bias-robustness pair ────────────────
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 1.4]})
pair_specs = [('video_count', '#3498db', 'Video count (immune to view accumulation)'),
              ('view_weighted_volume', '#8e44ad', 'View-weighted volume (exposed to accumulation)')]
for name, color, label in pair_specs:
    sub = sweep_df[sweep_df['series'] == name].sort_values('exclude_days')
    xv  = sub['exclude_days'].to_numpy()
    rv  = sub['corr_at_0'].to_numpy(dtype=float)   # lag-0 coupling (peak is at 0)
    tv  = sub['noise_threshold'].to_numpy(dtype=float)
    lv  = sub['peak_lag'].to_numpy(dtype=float)
    sig = sub['significant'].to_numpy(dtype=bool)
    axes4[0].plot(xv, rv, color=color, lw=1.6, alpha=0.85, label=label)
    axes4[0].plot(xv, tv, color=color, lw=1.0, alpha=0.45, ls='--')
    axes4[0].scatter(xv[sig],  rv[sig],  color=color, s=55, zorder=5)
    axes4[0].scatter(xv[~sig], rv[~sig], facecolors='none', edgecolors=color, s=55, zorder=5)
    axes4[1].plot(xv, lv, color=color, lw=1.5, alpha=0.8, marker='o', label=label)

axes4[0].axhline(0, color='gray', lw=0.8, ls='--')
axes4[0].set_ylabel('Lag-0 coupling (Delta vs Delta)')
axes4[0].set_title('12d: Coverage volume <-> player counts, release-window-controlled sweep\n'
                   '(first-differenced; solid = lag-0 r, dashed = 95% noise floor, filled = clears gate)',
                   fontsize=11)
axes4[0].legend(fontsize=8)
axes4[0].grid(True, alpha=0.25)
axes4[1].axhline(0, color='gray', lw=0.8, ls='--')
axes4[1].set_ylabel('Peak lag (weeks)\n(+ = coverage leads)')
axes4[1].legend(fontsize=7)
axes4[1].grid(True, alpha=0.25)

nrec = sweep_df[sweep_df['series'] == 'video_count'].sort_values('exclude_days')
axes4[2].bar(nrec['exclude_days'].to_numpy(), nrec['n_weeks'].to_numpy(),
             width=3.0, color='#7f8c8d', alpha=0.75)
axes4[2].set_ylabel('Weeks\nretained')
axes4[2].set_xlabel('Excluded window around each release (+/- days)')
axes4[2].set_xticks(SWEEP_WIDTHS)
axes4[2].grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out_d = FIGURES_DIR / '12d_release_controlled_sweep.png'
plt.savefig(out_d, dpi=150, bbox_inches='tight')
print(f'Saved: {out_d}')
plt.close(fig4)


#%%
# ── Fixed valence test: differenced volume-normalised tone vs players ─────────
# positive_share / net_valence_share divide out the volume envelope, so any
# coupling they carry is about TONE, not quantity. Differenced for consistency.
valence_specs = [
    ('positive_share',    '#27ae60', 'Positive share (positive / total)'),
    ('net_valence_share', '#2980b9', 'Net valence share ((pos - neg) / total)'),
]
valence_records = []
fig3, ax3 = plt.subplots(figsize=(11, 6))
ref = diff_res.get('video_count')
if ref is not None:
    ax3.plot(ref['cc'].index, ref['cc'].values, color='#95a5a6', lw=1.4, ls=':',
             alpha=0.8, label='Video count Delta (volume reference)')
for name, color, label in valence_specs:
    res = analyze_series(weekly_d[name], pc_weekly_d, MAX_LAG_WEEKS)
    if res is None:
        print(f'{label}: insufficient data.')
        continue
    rec = {k: res[k] for k in res if k != 'cc'}
    valence_records.append({'series': name, 'treatment': 'first_diff', **rec})
    cc = res['cc']
    ax3.plot(cc.index, cc.values, color=color, lw=2.0, alpha=0.9, label=label)
    ax3.scatter([res['peak_lag']], [res['peak_corr']], color=color, s=70,
                marker=('o' if res['significant'] else 'x'), zorder=5)
    flag = 'SIG' if res['significant'] else 'ns'
    print(f'[diff] {label:<40s} peak={res["peak_lag"]:+d}w r={res["peak_corr"]:+.3f} '
          f'p={res["p_value"]:.3f} [{flag}]')

ax3.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax3.axhline(0, color='gray', lw=0.6)
ax3.set_xlabel('Lag (weeks)  [+ = coverage leads players]', fontsize=11)
ax3.set_ylabel('Cross-correlation', fontsize=11)
ax3.set_title('Fixed valence test (first differences): does TONE track players beyond volume?\n'
              'Volume-normalised shares (o = clears gate, x = ns). Flat/ns = tone adds nothing.',
              fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.25)
plt.tight_layout()
out_c = FIGURES_DIR / '12c_valence_test.png'
plt.savefig(out_c, dpi=150, bbox_inches='tight')
print(f'Saved: {out_c}')
plt.close(fig3)

if valence_records:
    save_csv(pd.DataFrame(valence_records), TABLES_DIR / '12_valence_test.csv')


#%%
# ── Save main summary table ──────────────────────────────────────────────────
xcorr_df = pd.DataFrame(xcorr_records)
print('\nFull summary:')
print(xcorr_df.to_string(index=False))
save_csv(xcorr_df, TABLES_DIR / '12_coverage_volume_xcorr.csv')
