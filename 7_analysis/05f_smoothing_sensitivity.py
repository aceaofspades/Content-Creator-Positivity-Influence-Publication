#%%
"""
05f_smoothing_sensitivity.py
-----------------------------
Sensitivity of the primary creator→Reddit cross-correlation to the amount
of pre-computation smoothing.

The main scripts (05, 05b–05e) compute xcorr on raw deseasonalised daily
series. If the null result were a noise-floor artifact, heavier smoothing
should reveal a systematic positive creator-lead lag. If the null is
genuine, peak lag should stay near zero regardless of window width.

Four centered rolling-mean windows are tested: 1 (raw), 3, 7, 14 days.
Smoothing is applied to both the creator and Reddit series before xcorr
computation. Results are compared for overall__score and the full set of
topic fields.

Outputs:
  figures/05f_smoothing_overall.png   — xcorr curves overlaid by window
  figures/05f_smoothing_fields.png    — field peak-lag boxplots by window
  tables/05f_smoothing_sensitivity.csv — peak lag + peak r per window
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, deseasonalize_dow,
)

MAX_LAG_DAYS = 28
MIN_OVERLAP  = 90
WINDOWS      = [1, 3, 7, 14]   # days; 1 = no smoothing (baseline)
WINDOW_COLORS = {1: '#2c3e50', 3: '#3498db', 7: '#e67e22', 14: '#e74c3c'}

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
topic_cols = get_topic_cols(yt)
all_fields = ['overall__score'] + topic_cols

creator_agg = aggregate_creator_daily(yt, all_fields)
reddit_agg  = aggregate_reddit_daily(rd, all_fields)

start = max(creator_agg.index.min(), reddit_agg.index.min())
end   = min(creator_agg.index.max(), reddit_agg.index.max())
dates = pd.date_range(start, end, freq='D')
creator_agg = creator_agg.reindex(dates)
reddit_agg  = reddit_agg.reindex(dates)

# Deseasonalise once; smoothing is applied on top of these residuals
c_base_deseas = deseasonalize_dow(creator_agg['overall__score_unweighted'].reindex(dates))
r_base_deseas = deseasonalize_dow(reddit_agg['overall__score_unweighted'].reindex(dates))

creator_deseas_all = {}
reddit_deseas_all  = {}
for field in all_fields:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col in creator_agg.columns:
        creator_deseas_all[field] = deseasonalize_dow(creator_agg[c_col].reindex(dates))
    if r_col in reddit_agg.columns:
        reddit_deseas_all[field]  = deseasonalize_dow(reddit_agg[r_col].reindex(dates))

#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


def apply_smooth(series: pd.Series, window: int) -> pd.Series:
    if window == 1:
        return series
    return series.rolling(window, center=True, min_periods=1).mean()


#%%
# ── Overall__score: xcorr curves by window ────────────────────────────────────
overall_records = []
xcorr_by_window = {}

for w in WINDOWS:
    c_smooth = apply_smooth(c_base_deseas, w)
    r_smooth = apply_smooth(r_base_deseas, w)
    cc = normalised_xcorr(c_smooth, r_smooth, MAX_LAG_DAYS)
    if cc.empty:
        continue
    xcorr_by_window[w] = cc
    peak_lag  = int(cc.abs().idxmax())
    peak_corr = float(cc.loc[peak_lag])
    corr_at_0 = float(cc.get(0, np.nan))
    overall_records.append({
        'window_days': w,
        'peak_lag':    peak_lag,
        'peak_corr':   round(peak_corr, 4),
        'corr_at_0':   round(corr_at_0, 4),
    })
    print(f'Window={w:2d}d  peak_lag={peak_lag:+3d}  peak_r={peak_corr:+.4f}')

#%%
fig1, ax1 = plt.subplots(figsize=(12, 5))
for w, cc in xcorr_by_window.items():
    label = f'{w}d window' if w > 1 else '1d (no smoothing, baseline)'
    ax1.plot(cc.index, cc.values, label=label,
             color=WINDOW_COLORS[w], lw=2.0, alpha=0.85)
ax1.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
ax1.axhline(0, color='gray', lw=0.6)
ax1.set_xlabel('Lag (days)  [+ = creator leads Reddit]', fontsize=11)
ax1.set_ylabel('Cross-correlation', fontsize=11)
ax1.set_title(
    'Creator→Reddit overall sentiment: xcorr sensitivity to pre-computation smoothing\n'
    '(unweighted, deseasonalised; centered rolling mean applied before xcorr)',
    fontsize=12,
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
plt.tight_layout()
out1 = FIGURES_DIR / '05f_smoothing_overall.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f'Saved: {out1}')
plt.show()

#%%
# ── Topic fields: peak-lag distribution by window ─────────────────────────────
field_records = []
for w in WINDOWS:
    peak_lags = []
    for field in topic_cols:
        if field not in creator_deseas_all or field not in reddit_deseas_all:
            continue
        c_s = apply_smooth(creator_deseas_all[field], w)
        r_s = apply_smooth(reddit_deseas_all[field], w)
        both = pd.concat([c_s, r_s], axis=1).dropna()
        if len(both) < MIN_OVERLAP:
            continue
        cc = normalised_xcorr(c_s, r_s, MAX_LAG_DAYS)
        if cc.empty:
            continue
        peak_lags.append(int(cc.abs().idxmax()))

    if not peak_lags:
        continue
    med  = float(np.median(peak_lags))
    frac = float(np.mean([v > 0 for v in peak_lags]))
    print(f'Window={w:2d}d  n_fields={len(peak_lags)}  '
          f'median_lag={med:+.1f}  frac_positive={frac:.3f}')
    for lag in peak_lags:
        field_records.append({'window_days': w, 'peak_lag': lag})

    overall_records_w = [r for r in overall_records if r['window_days'] == w]
    if overall_records_w:
        overall_records_w[0]['n_fields']          = len(peak_lags)
        overall_records_w[0]['median_field_lag']   = round(med, 2)
        overall_records_w[0]['frac_positive_lag']  = round(frac, 4)

field_lag_df = pd.DataFrame(field_records)

fig2, ax2 = plt.subplots(figsize=(10, 5))
data_by_window = [field_lag_df[field_lag_df['window_days'] == w]['peak_lag'].values
                  for w in WINDOWS]
bp = ax2.boxplot(data_by_window, vert=True, patch_artist=True,
                 medianprops=dict(color='black', lw=2))
for patch, w in zip(bp['boxes'], WINDOWS):
    patch.set_facecolor(WINDOW_COLORS[w])
    patch.set_alpha(0.7)
ax2.set_xticks(range(1, len(WINDOWS) + 1))
ax2.set_xticklabels([f'{w}d' if w > 1 else '1d\n(baseline)' for w in WINDOWS], fontsize=10)
ax2.axhline(0, color='black', lw=1.5, linestyle='--', alpha=0.7)
ax2.set_ylabel('Peak lag (days)  [+ = creator leads Reddit]', fontsize=11)
ax2.set_xlabel('Smoothing window', fontsize=11)
ax2.set_title('Topic field peak-lag distribution vs. smoothing window', fontsize=12)
ax2.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out2 = FIGURES_DIR / '05f_smoothing_fields.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
plt.show()

#%%
summary_df = pd.DataFrame(overall_records)
out_tbl = TABLES_DIR / '05f_smoothing_sensitivity.csv'
summary_df.to_csv(out_tbl, index=False)
print(f'\nSaved: {out_tbl}')
print(summary_df.to_string(index=False))
