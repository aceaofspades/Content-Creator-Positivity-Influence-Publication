#%%
"""
05e_reddit_comments_lead_lag.py
--------------------------------
Lead/lag analysis using Reddit *comments* as the community sentiment target.

Comments are structurally different from submissions: faster reaction, higher
volume, more emotional, less filtered. If creator influence exists but operates
on comments rather than submissions, script 05 misses it entirely.

Two tests:
  1. Creator overall__score → comment sentiment_score
     Mirrors script 05's primary test at daily ±28 days and weekly ±12 weeks.
     All creator weighting schemes × both comment weighting schemes.

  2. Creator topic fields → comment sentiment_score  (one-sided)
     Tests whether creator discussion of specific topics predicts overall comment
     mood. Not symmetric — comments lack topic fields — but captures whether
     topic salience in creator content correlates with community-level response.
     Annotated by the same TOPIC_CATEGORIES as script 05.

Null: strengthens the overall null across two distinct community measures.
Creator lead on comments but not submissions would be a major finding requiring
paper revision. Required before submission — reviewers will ask.

Outputs:
  figures/05e_comments_xcorr_daily.png     — creator overall → comments, daily ±28d
  figures/05e_comments_xcorr_weekly.png    — creator overall → comments, weekly ±12w
  figures/05e_comments_topic_heatmap.png   — creator topics → comments, heatmap
  tables/05e_comments_xcorr_daily.csv      — daily overall xcorr summary
  tables/05e_comments_xcorr_weekly.csv     — weekly overall xcorr summary
  tables/05e_comments_topic_xcorr.csv      — per-topic peak lag + corr vs comments
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_comments_daily,
    deseasonalize_dow,
)
from analysis.shared.parsers import TOPIC_CATEGORIES

MAX_LAG_DAYS      = 28
MAX_LAG_WEEKS     = 12
MIN_OVERLAP       = 90    # minimum days of non-NaN overlap to include a field
MIN_OVERLAP_WEEKS = 20

TOPIC_CATEGORY = {f'topic__{k}': v for k, v in TOPIC_CATEGORIES.items()}
TOPIC_CATEGORY['overall__score'] = 'Overall'

creator_schemes = ['unweighted', 'view_weighted', 'creator_weighted']
comment_schemes = ['unweighted', 'engagement_weighted']
scheme_colors   = {
    'unweighted':          '#95a5a6',
    'view_weighted':       '#3498db',
    'creator_weighted':    '#8e44ad',
    'engagement_weighted': '#e67e22',
}

#%%
ensure_output_dirs()

yt = load_youtube_creators()
cm = load_comments_master()

topic_cols = get_topic_cols(yt)
all_fields = ['overall__score'] + topic_cols

#%%
# Aggregate under all schemes
creator_agg = aggregate_creator_daily(yt, all_fields)
comment_agg = aggregate_comments_daily(cm, ['sentiment_score'])

# Common date range
start = max(creator_agg.index.min(), comment_agg.index.min())
end   = min(creator_agg.index.max(), comment_agg.index.max())
dates = pd.date_range(start, end, freq='D')
creator_agg = creator_agg.reindex(dates)
comment_agg = comment_agg.reindex(dates)

print(f'Analysis window: {start.date()} → {end.date()}  ({len(dates)} days)')
print(f'Comment rows: {len(cm):,}')


#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


#%%
# ── Figure 1: Creator overall__score → comments, daily ±28 days ──────────────
fig, axes = plt.subplots(len(comment_schemes), 1,
                          figsize=(12, 4 * len(comment_schemes)),
                          sharex=True, sharey=True)

daily_records = []

for ax, cs in zip(axes, comment_schemes):
    cm_col = f'sentiment_score_{cs}'
    cm_s   = deseasonalize_dow(comment_agg[cm_col].reindex(dates))

    for crs in creator_schemes:
        c_col = f'overall__score_{crs}'
        c_s   = deseasonalize_dow(creator_agg[c_col].reindex(dates))
        cc    = normalised_xcorr(c_s, cm_s, MAX_LAG_DAYS)
        if cc.empty:
            continue
        ax.plot(cc.index, cc.values, label=f'Creator: {crs}',
                color=scheme_colors[crs], lw=1.8, alpha=0.85)
        peak_lag  = cc.abs().idxmax()
        peak_corr = cc.loc[peak_lag]
        daily_records.append({
            'comment_scheme': cs,
            'creator_scheme': crs,
            'peak_lag':       peak_lag,
            'peak_corr':      round(peak_corr, 4),
            'corr_at_0':      round(cc.get(0, np.nan), 4),
        })

    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(f'Comments: {cs}', fontsize=10)
    ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel('Lag (days)  [+ = creator leads comments]')
fig.suptitle('Creator overall sentiment → Reddit comments sentiment\n'
             '(daily ±28 days, deseasonalised)', fontsize=13)
plt.tight_layout()
out_a = FIGURES_DIR / '05e_comments_xcorr_daily.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

daily_df = pd.DataFrame(daily_records)
daily_df.to_csv(TABLES_DIR / '05e_comments_xcorr_daily.csv', index=False)
print(f'Saved: {TABLES_DIR / "05e_comments_xcorr_daily.csv"}')
print('\nDaily peak lags (creator → comments):')
print(daily_df.to_string(index=False))


#%%
# ── Figure 2: Creator overall__score → comments, weekly ±12 weeks ────────────
creator_weekly = creator_agg.resample('W-MON').mean()
comment_weekly = comment_agg.resample('W-MON').mean()

start_w   = max(creator_weekly.index.min(), comment_weekly.index.min())
end_w     = min(creator_weekly.index.max(), comment_weekly.index.max())
all_weeks = pd.date_range(start_w, end_w, freq='W-MON')
creator_weekly = creator_weekly.reindex(all_weeks)
comment_weekly = comment_weekly.reindex(all_weeks)

fig2, axes2 = plt.subplots(len(comment_schemes), 1,
                            figsize=(12, 4 * len(comment_schemes)),
                            sharex=True, sharey=True)

weekly_records = []

for ax, cs in zip(axes2, comment_schemes):
    cm_col = f'sentiment_score_{cs}'
    cm_s   = comment_weekly[cm_col]

    for crs in creator_schemes:
        c_col = f'overall__score_{crs}'
        c_s   = creator_weekly[c_col]
        both  = pd.concat([c_s, cm_s], axis=1).dropna()
        if len(both) < MIN_OVERLAP_WEEKS:
            continue
        cc = normalised_xcorr(c_s, cm_s, MAX_LAG_WEEKS)
        if cc.empty:
            continue
        ax.plot(cc.index, cc.values, label=f'Creator: {crs}',
                color=scheme_colors[crs], lw=1.8, alpha=0.85)
        peak_lag  = cc.abs().idxmax()
        peak_corr = cc.loc[peak_lag]
        weekly_records.append({
            'comment_scheme': cs,
            'creator_scheme': crs,
            'peak_lag':       peak_lag,
            'peak_corr':      round(peak_corr, 4),
            'corr_at_0':      round(cc.get(0, np.nan), 4),
            'n_weeks':        len(both),
        })

    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(f'Comments: {cs}', fontsize=10)
    ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

axes2[-1].set_xlabel('Lag (weeks)  [+ = creator leads comments]')
fig2.suptitle('Creator overall sentiment → Reddit comments sentiment\n'
              '(weekly ±12 weeks)', fontsize=13)
plt.tight_layout()
out_b = FIGURES_DIR / '05e_comments_xcorr_weekly.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')
plt.show()

weekly_df = pd.DataFrame(weekly_records)
weekly_df.to_csv(TABLES_DIR / '05e_comments_xcorr_weekly.csv', index=False)
print(f'Saved: {TABLES_DIR / "05e_comments_xcorr_weekly.csv"}')
print('\nWeekly peak lags (creator → comments):')
print(weekly_df.to_string(index=False))


#%%
# ── Figure 3: Creator topic fields → comment sentiment_score ─────────────────
# One-sided test: does creator discussion of specific topics predict overall
# comment mood? Not symmetric — comments lack topic fields — but captures
# whether topic salience in creator content correlates with community response.
# Annotated by natural TOPIC_CATEGORIES, same as script 05.

import matplotlib.cm as cm_cmap

cm_base = deseasonalize_dow(comment_agg['sentiment_score_unweighted'].reindex(dates))

topic_records = []
lag_matrix    = {}

for field in topic_cols:
    c_col = f'{field}_unweighted'
    if c_col not in creator_agg.columns:
        continue
    c_s  = deseasonalize_dow(creator_agg[c_col].reindex(dates))
    both = pd.concat([c_s, cm_base], axis=1).dropna()
    if len(both) < MIN_OVERLAP:
        continue
    cc = normalised_xcorr(c_s, cm_base, MAX_LAG_DAYS)
    if cc.empty:
        continue
    lag_matrix[field] = cc
    peak_lag  = cc.abs().idxmax()
    peak_corr = cc.loc[peak_lag]
    topic_records.append({
        'field':     field,
        'category':  TOPIC_CATEGORY.get(field, 'Other'),
        'peak_lag':  peak_lag,
        'peak_corr': round(peak_corr, 4),
        'corr_at_0': round(cc.get(0, np.nan), 4),
        'n_overlap': len(both),
    })

topic_df = pd.DataFrame(topic_records).sort_values('peak_lag')
topic_df.to_csv(TABLES_DIR / '05e_comments_topic_xcorr.csv', index=False)
print(f'\n{len(topic_df)} topic fields with sufficient data (≥{MIN_OVERLAP} days overlap).')
print('Peak lag distribution (creator topic → comment sentiment):')
print(topic_df['peak_lag'].describe().round(2))
print('\nBy category:')
print(topic_df.groupby('category')['peak_lag'].describe().round(1))

# ── Comparison with script 05 submission results (if available) ───────────────
sub_path = TABLES_DIR / '05_field_xcorr_summary.csv'
if sub_path.exists():
    sub_df = pd.read_csv(sub_path)
    merged = topic_df[['field', 'peak_lag', 'peak_corr']].merge(
        sub_df[['field', 'peak_lag', 'peak_corr']].rename(
            columns={'peak_lag': 'sub_peak_lag', 'peak_corr': 'sub_peak_corr'}),
        on='field', how='inner')
    print(f'\nComparison with submissions (n={len(merged)} common fields):')
    print(f'  Comments median peak lag: {topic_df["peak_lag"].median():.1f} days')
    print(f'  Submissions median peak lag: {sub_df["peak_lag"].median():.1f} days')
    lag_corr = merged['peak_lag'].corr(merged['sub_peak_lag'])
    print(f'  Lag rank correlation (comments vs submissions): r={lag_corr:.3f}')

#%%
if lag_matrix:
    categories_present = sorted(topic_df['category'].unique())
    cat_palette = {c: cm_cmap.tab10(i / 10) for i, c in enumerate(categories_present)}

    # Sort rows: first by category, then by corr at lag=0 within category
    heatmap_df = pd.DataFrame(lag_matrix).T
    sort_order = topic_df.sort_values(['category', 'corr_at_0'],
                                       ascending=[True, False])['field']
    heatmap_df = heatmap_df.reindex(sort_order)

    fig3, axes3 = plt.subplots(1, 2, figsize=(18, max(6, len(heatmap_df) * 0.28)),
                                gridspec_kw={'width_ratios': [3, 1]})

    # Left: heatmap by lag
    im = axes3[0].imshow(heatmap_df.values, aspect='auto', cmap='RdBu_r',
                         vmin=-0.35, vmax=0.35,
                         extent=[-MAX_LAG_DAYS - 0.5, MAX_LAG_DAYS + 0.5,
                                 len(heatmap_df) - 0.5, -0.5])
    plt.colorbar(im, ax=axes3[0], label='Cross-correlation coefficient')
    axes3[0].axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
    axes3[0].set_yticks(range(len(heatmap_df)))
    axes3[0].set_yticklabels([f.replace('topic__', '') for f in heatmap_df.index], fontsize=7)
    axes3[0].set_xlabel('Lag (days)  [+ = creator topic leads comment sentiment]', fontsize=10)
    axes3[0].set_title('Cross-correlation by creator topic and lag', fontsize=11)

    # Right: peak lag bar chart coloured by category
    n = len(topic_df)
    bar_y     = np.arange(n)
    bar_order = topic_df.sort_values(['category', 'corr_at_0'],
                                      ascending=[True, False])
    bar_colors = [cat_palette.get(row['category'], 'gray') for _, row in bar_order.iterrows()]
    axes3[1].barh(bar_y, bar_order['peak_lag'], color=bar_colors, alpha=0.8, height=0.8)
    axes3[1].axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
    axes3[1].axvline(topic_df['peak_lag'].median(), color='red', lw=1.2, linestyle=':',
                     label=f'Median: {topic_df["peak_lag"].median():.0f}d')
    axes3[1].set_yticks([])
    axes3[1].set_xlabel('Peak lag (days)', fontsize=9)
    axes3[1].set_title('Peak lag', fontsize=11)
    axes3[1].legend(fontsize=8)
    axes3[1].grid(True, alpha=0.25, axis='x')

    # Category legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_palette[c], alpha=0.8)
               for c in categories_present]
    axes3[0].legend(handles, categories_present, fontsize=7, loc='lower left',
                    title='Category', title_fontsize=8)

    fig3.suptitle('Creator topic discussion → Reddit comment sentiment\n'
                  '(daily ±28 days, deseasonalised, sorted by category)', fontsize=13)
    plt.tight_layout()
    out_c = FIGURES_DIR / '05e_comments_topic_heatmap.png'
    plt.savefig(out_c, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_c}')
    plt.show()
