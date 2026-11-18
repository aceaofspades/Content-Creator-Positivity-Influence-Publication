#%%
"""
05_creator_reddit_lead_lag.py
------------------------------
PRIMARY elite-influence test: do YouTube creators' sentiment scores lead
Reddit community sentiment, and if so, by how many days?

Approach:
  1. Aggregate creator and Reddit sentiment to daily under all weighting schemes
  2. Deseasonalize both series (day-of-week dummies)
  3. Cross-correlation at lags 0–MAX_LAG_DAYS (overall__score + all topic__ fields)
  4. Granger causality test on residuals after removing release-window effects
  5. Field-level cross-correlations: run on every topic__ field individually,
     plot distribution of peak lags — no pre-specified buckets

Outputs:
  figures/05a_overall_xcorr.png           — cross-corr for overall__score, all weighting combos
  figures/05b_field_lag_distribution.png  — histogram of peak lags across all topic fields
  figures/05c_field_xcorr_heatmap.png     — heatmap of corr by field × lag
  figures/05d_weekly_overall_xcorr.png    — weekly overall xcorr (no deseasonalization needed)
  figures/05e_weekly_field_lag_dist.png   — weekly field peak-lag histogram
  figures/05f_permutation_test.png        — null distribution for expertise-cluster test
  tables/05_field_xcorr_summary.csv       — daily peak lag + corr per field, with hypothesis label
  tables/05_field_xcorr_summary_weekly.csv — same, at weekly granularity
  tables/05_granger_results.csv           — Granger causality test results
  tables/05_permutation_results.csv       — permutation test summary for expertise cluster
  tables/05_weighting_robustness.csv      — central test summary per Reddit weighting scheme
  figures/05_weighting_robustness.png     — field-level peak-lag distribution, both Reddit schemes
  tables/05_field_xcorr_summary_engagement_weighted.csv        — daily field summary, EW Reddit
  tables/05_field_xcorr_summary_weekly_engagement_weighted.csv — weekly field summary, EW Reddit
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import correlate
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow, rolling_7day,
)
from analysis.shared.parsers import TOPIC_CATEGORIES

MAX_LAG_DAYS    = 28
GRANGER_MAX_LAG = 14    # max lag for Granger test (in days; kept shorter for power)
MIN_OVERLAP     = 90    # minimum days of overlapping non-NaN data for a field to be included

# Natural category per field — derived from TOPIC_CATEGORIES in parsers.py,
# which mirrors the STANDARD_TOPICS groupings without any subjective reclassification.
TOPIC_CATEGORY = {f'topic__{k}': v for k, v in TOPIC_CATEGORIES.items()}
TOPIC_CATEGORY['overall__score'] = 'Overall'

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
cm = load_comments_master()
topic_cols = get_topic_cols(yt)
all_fields = ['overall__score'] + topic_cols

#%%
# Aggregate under all schemes
creator_agg = aggregate_creator_daily(yt, all_fields)
reddit_agg  = aggregate_reddit_daily(rd, all_fields)
comment_agg = aggregate_comments_daily(cm, ['sentiment_score'])

# Common date range (include comments in bounds)
start = max(creator_agg.index.min(), reddit_agg.index.min(), comment_agg.index.min())
end   = min(creator_agg.index.max(), reddit_agg.index.max(), comment_agg.index.max())
dates = pd.date_range(start, end, freq='D')
creator_agg = creator_agg.reindex(dates)
reddit_agg  = reddit_agg.reindex(dates)
comment_agg = comment_agg.reindex(dates)

#%%
# normalised_xcorr now lives in analysis/shared/lagcorr.py: calendar-correct lags
# (the old local copy correlated dropna()'d rows by position -- gap-collapsing).
from analysis.shared.lagcorr import normalised_xcorr  # noqa: E402


#%%
# ── Figure 1: Overall__score cross-correlation, all weighting combos ──────────
creator_schemes = ['unweighted', 'view_weighted', 'creator_weighted']
reddit_schemes  = ['unweighted', 'engagement_weighted']
comment_schemes = ['unweighted', 'engagement_weighted']
scheme_colors   = {'unweighted': '#95a5a6', 'view_weighted': '#3498db',
                   'creator_weighted': '#8e44ad', 'engagement_weighted': '#e67e22'}

# Build target list: (agg_df, field_col, panel_title, target_label)
xcorr_targets = (
    [(reddit_agg,  f'overall__score_{rs}',    f'Reddit submissions ({rs})',  'submission')
     for rs in reddit_schemes] +
    [(comment_agg, f'sentiment_score_{cs}',   f'Reddit comments ({cs})',     'comment')
     for cs in comment_schemes]
)

fig, axes = plt.subplots(len(xcorr_targets), 1, figsize=(12, 4 * len(xcorr_targets)),
                          sharex=True, sharey=True)

for ax, (target_agg, r_col, panel_title, _ttype) in zip(axes, xcorr_targets):
    r_series = deseasonalize_dow(target_agg[r_col].dropna().reindex(dates))

    for cs in creator_schemes:
        c_col = f'overall__score_{cs}'
        c_series = deseasonalize_dow(creator_agg[c_col].dropna().reindex(dates))
        cc = normalised_xcorr(c_series, r_series, MAX_LAG_DAYS)
        if cc.empty:
            continue
        ax.plot(cc.index, cc.values, label=f'Creator: {cs}',
                color=scheme_colors[cs], lw=1.8, alpha=0.85)

    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(panel_title, fontsize=10)
    ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel('Lag (days)  [+ = creator leads Reddit]')
# Figure title lives in the paper caption, not in the image.
plt.tight_layout()
out_a = FIGURES_DIR / '05a_overall_xcorr.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

#%%
# ── Field-level cross-correlations (unweighted baseline) ─────────────────────
r_base = deseasonalize_dow(reddit_agg['overall__score_unweighted'].reindex(dates))
c_base = deseasonalize_dow(creator_agg['overall__score_unweighted'].reindex(dates))

field_records = []
lag_matrix    = {}   # field → corr series (for heatmap)

for field in all_fields:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col not in creator_agg.columns or r_col not in reddit_agg.columns:
        continue

    c_s = deseasonalize_dow(creator_agg[c_col].reindex(dates))
    r_s = deseasonalize_dow(reddit_agg[r_col].reindex(dates))
    both = pd.concat([c_s, r_s], axis=1).dropna()
    if len(both) < MIN_OVERLAP:
        continue

    cc = normalised_xcorr(c_s, r_s, MAX_LAG_DAYS)
    if cc.empty:
        continue

    lag_matrix[field] = cc
    peak_lag  = cc.abs().idxmax()
    peak_corr = cc.loc[peak_lag]
    field_records.append({
        'field':        field,
        'peak_lag':     peak_lag,
        'peak_corr':    peak_corr,
        'corr_at_0':    cc.get(0, np.nan),
        'n_overlap':    len(both),
        'category': TOPIC_CATEGORY.get(field, 'Other'),
    })

field_df = pd.DataFrame(field_records).sort_values('peak_lag')
field_df.to_csv(TABLES_DIR / '05_field_xcorr_summary.csv', index=False)
print(f'\n{len(field_df)} fields with sufficient data. Peak lag distribution:')
print(field_df['peak_lag'].describe())
print('\nBy category:')
print(field_df.groupby('category')['peak_lag'].describe().round(1))

#%%
# ── Figure 2: Distribution of peak lags by topic category ────────────────────
# One box per natural category — shows spread and median without subjective re-labeling.
import matplotlib.cm as cm

categories_present = field_df[field_df['category'] != 'Overall']['category'].unique()
categories_sorted  = sorted(categories_present)
cat_palette = {c: cm.tab10(i / 10) for i, c in enumerate(categories_sorted)}

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6),
                            gridspec_kw={'width_ratios': [2, 1]})

# Left: box plot per category
cat_data  = [field_df[field_df['category'] == c]['peak_lag'].values
             for c in categories_sorted]
bp = axes2[0].boxplot(cat_data, vert=False, patch_artist=True,
                      medianprops=dict(color='black', lw=2))
for patch, cat in zip(bp['boxes'], categories_sorted):
    patch.set_facecolor(cat_palette[cat])
    patch.set_alpha(0.7)
axes2[0].set_yticks(range(1, len(categories_sorted) + 1))
axes2[0].set_yticklabels(categories_sorted, fontsize=9)
axes2[0].axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.6)
axes2[0].set_xlabel('Peak lag (days)  [+ = creator leads Reddit]', fontsize=10)
axes2[0].set_title('Peak lag distribution by topic category', fontsize=12)
axes2[0].grid(True, alpha=0.25, axis='x')

# Right: histogram of all fields regardless of category
axes2[1].hist(field_df[field_df['category'] != 'Overall']['peak_lag'],
              bins=range(-MAX_LAG_DAYS - 1, MAX_LAG_DAYS + 2),
              color='#7f8c8d', alpha=0.75, edgecolor='white')
axes2[1].axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.7)
axes2[1].axvline(field_df['peak_lag'].median(), color='#e74c3c', lw=1.5,
                 linestyle=':', label=f'Median: {field_df["peak_lag"].median():.0f}d')
axes2[1].set_xlabel('Peak lag (days)', fontsize=10)
axes2[1].set_ylabel('Number of fields', fontsize=10)
axes2[1].set_title('All topic fields', fontsize=12)
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.25)

fig2.suptitle('Creator→Reddit peak lag across all topic fields', fontsize=13)
plt.tight_layout()
out_b = FIGURES_DIR / '05b_field_lag_distribution.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')
plt.show()

#%%
# ── Figure 3: Heatmap of corr by field × lag ─────────────────────────────────
if lag_matrix:
    heatmap_df = pd.DataFrame(lag_matrix).T   # fields as rows, lags as columns
    heatmap_df = heatmap_df.sort_values(
        by=heatmap_df.columns[heatmap_df.columns.get_loc(0)], ascending=False
    )

    fig3, ax3 = plt.subplots(figsize=(16, max(6, len(heatmap_df) * 0.28)))
    im = ax3.imshow(heatmap_df.values, aspect='auto', cmap='RdBu_r',
                    vmin=-0.4, vmax=0.4,
                    extent=[-MAX_LAG_DAYS - 0.5, MAX_LAG_DAYS + 0.5,
                            len(heatmap_df) - 0.5, -0.5])
    plt.colorbar(im, ax=ax3, label='Cross-correlation coefficient')
    ax3.axvline(0, color='black', lw=1.2, linestyle='--', alpha=0.7)
    ax3.set_yticks(range(len(heatmap_df)))
    ax3.set_yticklabels(
        [f.replace('topic__', '').replace('overall__', 'OVERALL: ') for f in heatmap_df.index],
        fontsize=7,
    )
    ax3.set_xlabel('Lag (days)  [+ = creator leads Reddit]', fontsize=11)
    ax3.set_title('Creator→Reddit cross-correlation by field and lag', fontsize=13)
    plt.tight_layout()
    out_c = FIGURES_DIR / '05c_field_xcorr_heatmap.png'
    plt.savefig(out_c, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_c}')
    plt.show()

#%%
# ── Granger causality on overall__score (deseasonalised, unweighted) ─────────
# Tests both creator → submissions and creator → comments.
granger_records = []


def _granger_date_correct(target: pd.Series, creator: pd.Series, lag: int):
    """
    Granger F-test at one lag order, with lags measured in calendar days.

    Both series must sit on the same regular daily index (NaN for days with no
    data). Lags are built by shifting on that index and rows are dropped only
    AFTER shifting, so y_lag{k} / x_lag{k} are always exactly k days back.

    This replaces statsmodels' grangercausalitytests, which treats its input
    rows as consecutive. The old code passed it the dropna()'d series, so in
    games with sparse daily creator coverage a "lag 1" could span several days
    (the gap-collapsing defect already fixed in 05g). D2's dense series was
    unaffected; Warframe's 14/14 daily pattern was partly this artifact.

    Restricted (own lags) and unrestricted (+ creator lags) models are fitted
    on the SAME rows; the F statistic matches statsmodels' ssr_ftest.

    Returns (f_stat, p_value, creator_coefs, n_obs) or None if too few rows.
    The coefficients carry the sign: the F-test is unsigned, and a creator
    block predicting *lower* subsequent Reddit sentiment rejects the same null
    as one predicting higher while meaning the opposite.
    """
    df = pd.DataFrame({'y': target})
    for k in range(1, lag + 1):
        df[f'y_lag{k}'] = target.shift(k)
    for k in range(1, lag + 1):
        df[f'x_lag{k}'] = creator.shift(k)
    df = df.dropna()
    y_cols = [f'y_lag{k}' for k in range(1, lag + 1)]
    x_cols = [f'x_lag{k}' for k in range(1, lag + 1)]
    if len(df) < 3 * lag + 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_r = sm.OLS(df['y'], sm.add_constant(df[y_cols], has_constant='add')).fit()
        m_u = sm.OLS(df['y'], sm.add_constant(df[y_cols + x_cols], has_constant='add')).fit()
    f_stat = ((m_r.ssr - m_u.ssr) / lag) / (m_u.ssr / m_u.df_resid)
    p_val = float(stats.f.sf(f_stat, lag, m_u.df_resid))
    return float(f_stat), p_val, np.asarray(m_u.params[x_cols]), len(df)


def _run_granger(target_series: pd.Series, target_label: str,
                 creator_series: pd.Series) -> None:
    # Both inputs are on the regular `dates` index (NaN-padded) -- do NOT
    # dropna() them before calling, or the lags stop being calendar days.
    for lag in range(1, GRANGER_MAX_LAG + 1):
        res = _granger_date_correct(target_series, creator_series, lag)
        if res is None:
            print(f'Granger ({target_label}) lag={lag:2d}d  insufficient complete rows')
            continue
        f_stat, p_val, coefs, n_obs = res

        if coefs.size:
            coef_sum    = float(coefs.sum())
            i_max       = int(np.argmax(np.abs(coefs)))
            coef_max    = float(coefs[i_max])
            lag_at_max  = i_max + 1
            frac_pos    = float((coefs > 0).mean())
        else:
            coef_sum = coef_max = frac_pos = np.nan
            lag_at_max = np.nan

        sign = ('positive' if coef_sum > 0 else
                'negative' if coef_sum < 0 else 'zero') if coefs.size else 'unknown'
        # Only the positive direction is consistent with elite influence:
        # higher creator sentiment preceding higher community sentiment.
        relation = ('consistent with hypothesis' if sign == 'positive' else
                    'contrary (inverse relation)' if sign == 'negative' else
                    'undetermined')

        granger_records.append({
            'target': target_label, 'lag': lag,
            'f_stat': f_stat, 'p_value': p_val,
            'n_obs': n_obs,   # complete rows at THIS lag order (shrinks with lag)
            # Individual creator-lag coefficients (x_lag1 first). The sum can sit
            # near zero while F is significant when adjacent lags cancel -- the
            # sign of the SUM is then not a direction, and these show why.
            'creator_coefs': ';'.join(f'{c:+.4f}' for c in coefs),
            'creator_coef_sum': coef_sum,
            'creator_coef_max_abs': coef_max,
            'lag_at_max_abs': lag_at_max,
            'frac_creator_lags_positive': frac_pos,
            'sign': sign,
            'hypothesis_relation': relation,
        })
        flag = ' *' if p_val < 0.05 else '  '
        print(f'Granger ({target_label}) lag={lag:2d}d  F={f_stat:.3f}  '
              f'p={p_val:.4f}{flag} sum(creator coefs)={coef_sum:+.4f} [{sign}]  n={n_obs}')

# Regular daily index, NaN-padded (no dropna -- see _granger_date_correct)
c_g = c_base
r_g = r_base
cm_g = deseasonalize_dow(comment_agg['sentiment_score_unweighted'].reindex(dates))

print('\n── Granger: creator → Reddit submissions ──')
_run_granger(r_g, 'submissions', c_g)
print('\n── Granger: creator → Reddit comments ──')
_run_granger(cm_g, 'comments', c_g)
print('\n-- Granger: creator -> Reddit submissions (engagement-weighted) --')
r_g_ew = deseasonalize_dow(
    reddit_agg['overall__score_engagement_weighted'].reindex(dates))
_run_granger(r_g_ew, 'submissions_engagement_weighted', c_g)

if granger_records:
    granger_df = pd.DataFrame(granger_records)
    granger_df.to_csv(TABLES_DIR / '05_granger_results.csv', index=False)
    print(f'\nSaved Granger results to {TABLES_DIR / "05_granger_results.csv"}')

    # Per-target summary. A scatter of one or two isolated significant lags is
    # what noise looks like across 14 nested tests; a target significant at
    # (nearly) every lag with a consistent sign is a pattern. Report both, and
    # always with the sign — an inverse relation rejects the same null as a
    # positive one but contradicts the hypothesis.
    print('\n-- Granger summary (sign is the direction of the creator block) --')
    for tgt, grp in granger_df.groupby('target'):
        sig = grp[grp['p_value'] < 0.05]
        if sig.empty:
            print(f'  {tgt:35s} 0/{len(grp)} lags p<0.05  (min p={grp["p_value"].min():.4f})')
            continue
        signs = sig['sign'].value_counts().to_dict()
        print(f'  {tgt:35s} {len(sig)}/{len(grp)} lags p<0.05  '
              f'min p={sig["p_value"].min():.4f}  '
              f'lags={sorted(sig["lag"].tolist())}  signs={signs}')

#%%
# ── Weekly aggregation: overall__score ───────────────────────────────────────
# Motivated by 04 finding that engagement signals are weekly-scale phenomena.
# DOW effects average out at weekly granularity; no deseasonalization applied.
MAX_LAG_WEEKS    = 4
MIN_OVERLAP_WEEKS = 20

creator_weekly_agg = creator_agg.resample('W-MON').mean()
reddit_weekly_agg  = reddit_agg.resample('W-MON').mean()
comment_weekly_agg = comment_agg.resample('W-MON').mean()

start_w   = max(creator_weekly_agg.index.min(), reddit_weekly_agg.index.min(),
                comment_weekly_agg.index.min())
end_w     = min(creator_weekly_agg.index.max(), reddit_weekly_agg.index.max(),
                comment_weekly_agg.index.max())
all_weeks = pd.date_range(start_w, end_w, freq='W-MON')
creator_weekly_agg = creator_weekly_agg.reindex(all_weeks)
reddit_weekly_agg  = reddit_weekly_agg.reindex(all_weeks)
comment_weekly_agg = comment_weekly_agg.reindex(all_weeks)

weekly_xcorr_targets = (
    [(reddit_weekly_agg,  f'overall__score_{rs}',  f'Reddit submissions ({rs})')
     for rs in reddit_schemes] +
    [(comment_weekly_agg, f'sentiment_score_{cs}', f'Reddit comments ({cs})')
     for cs in comment_schemes]
)

fig4, axes4 = plt.subplots(len(weekly_xcorr_targets), 1,
                            figsize=(12, 4 * len(weekly_xcorr_targets)),
                            sharex=True, sharey=True)
for ax, (target_wagg, r_col, panel_title) in zip(axes4, weekly_xcorr_targets):
    r_series = target_wagg[r_col]
    for cs in creator_schemes:
        c_col = f'overall__score_{cs}'
        c_series = creator_weekly_agg[c_col]
        cc = normalised_xcorr(c_series, r_series, MAX_LAG_WEEKS)
        if cc.empty:
            continue
        ax.plot(cc.index, cc.values, label=f'Creator: {cs}',
                color=scheme_colors[cs], lw=1.8, alpha=0.85)
    ax.axvline(0, color='black', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_title(panel_title, fontsize=10)
    ax.set_ylabel('Cross-correlation')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)

axes4[-1].set_xlabel('Lag (weeks)  [+ = creator leads Reddit]')
fig4.suptitle('Overall sentiment: creator → Reddit (weekly aggregation)\n'
              '(top panels: submissions; bottom panels: comments)', fontsize=13)
plt.tight_layout()
out_d = FIGURES_DIR / '05d_weekly_overall_xcorr.png'
plt.savefig(out_d, dpi=150, bbox_inches='tight')
print(f'Saved: {out_d}')
plt.show()

#%%
# ── Weekly field-level cross-correlations ────────────────────────────────────
weekly_field_records = []

for field in all_fields:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col not in creator_weekly_agg.columns or r_col not in reddit_weekly_agg.columns:
        continue
    c_s = creator_weekly_agg[c_col]
    r_s = reddit_weekly_agg[r_col]
    both = pd.concat([c_s, r_s], axis=1).dropna()
    if len(both) < MIN_OVERLAP_WEEKS:
        continue
    cc = normalised_xcorr(c_s, r_s, MAX_LAG_WEEKS)
    if cc.empty:
        continue
    peak_lag  = cc.abs().idxmax()
    peak_corr = cc.loc[peak_lag]
    weekly_field_records.append({
        'field':      field,
        'peak_lag':   peak_lag,
        'peak_corr':  round(peak_corr, 4),
        'corr_at_0':  round(cc.get(0, np.nan), 4),
        'n_overlap':  len(both),
        'category': TOPIC_CATEGORY.get(field, 'Other'),
    })

weekly_field_df = pd.DataFrame(weekly_field_records).sort_values('peak_lag')
weekly_field_df.to_csv(TABLES_DIR / '05_field_xcorr_summary_weekly.csv', index=False)
print(f'\n{len(weekly_field_df)} fields with sufficient weekly data.')
print('Weekly peak lag distribution:')
print(weekly_field_df['peak_lag'].describe())
print('\nBy category:')
print(weekly_field_df.groupby('category')['peak_lag'].describe().round(2))

weekly_cats = [c for c in categories_sorted if c in weekly_field_df['category'].values]
fig5, ax5 = plt.subplots(figsize=(10, 5))
cat_data_w = [weekly_field_df[weekly_field_df['category'] == c]['peak_lag'].values
              for c in weekly_cats]
bp5 = ax5.boxplot(cat_data_w, vert=True, patch_artist=True,
                  medianprops=dict(color='black', lw=2))
for patch, cat in zip(bp5['boxes'], weekly_cats):
    patch.set_facecolor(cat_palette.get(cat, 'gray'))
    patch.set_alpha(0.7)
ax5.set_xticks(range(1, len(weekly_cats) + 1))
ax5.set_xticklabels(weekly_cats, rotation=30, ha='right', fontsize=9)
ax5.axhline(0, color='black', lw=1.5, linestyle='--', alpha=0.7)
ax5.set_ylabel('Peak lag (weeks)\n[+ = creator leads Reddit]', fontsize=10)
ax5.set_title('Weekly: peak lag by topic category', fontsize=13)
ax5.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
out_e = FIGURES_DIR / '05e_weekly_field_lag_dist.png'
plt.savefig(out_e, dpi=150, bbox_inches='tight')
print(f'Saved: {out_e}')
plt.show()

#%%
# ── Inspect negative-lag cluster by natural category ─────────────────────────
# The -22 to -26 day cluster previously labeled "expertise" maps to:
# Sandbox & Balance (super_balance, pvp_balance, weapon_balance) and
# Progression & Economy (infusion_system, currency_economy, mod_system, power_level_system).
CLUSTER_THRESHOLD = -20

cluster_df = field_df[field_df['peak_lag'] <= CLUSTER_THRESHOLD].sort_values('peak_lag')
print(f'--- Fields with peak_lag <= {CLUSTER_THRESHOLD} days ---')
print(cluster_df[['field', 'category', 'peak_lag', 'peak_corr', 'n_overlap']].to_string(index=False))

print(f'\n--- Mean peak lag by category (daily) ---')
cat_summary = (field_df[field_df['category'] != 'Overall']
               .groupby('category')['peak_lag']
               .agg(['mean', 'median', 'count'])
               .sort_values('mean'))
print(cat_summary.round(1).to_string())

#%%
# ── Permutation test: per-category mean peak lag vs. null ────────────────────
# Tests whether any category's mean peak lag is significantly more negative (or positive)
# than expected under independence. Null: circular-shift non-NaN creator values.
# p-value: one-tailed left (more negative than null) per category.
N_PERM = 1000

topic_fields_for_perm = [f for f in field_df['field'].tolist() if f != 'overall__score']
obs_cat_means = (field_df[field_df['field'].isin(topic_fields_for_perm)]
                 .groupby('category')['peak_lag'].mean())
print(f'Observed category mean peak lags:\n{obs_cat_means.sort_values().round(2)}')

# Pre-compute deseasonalised series for all topic fields
perm_creator_deseas = {}
perm_reddit_deseas  = {}
for field in topic_fields_for_perm:
    c_col = f'{field}_unweighted'
    r_col = f'{field}_unweighted'
    if c_col in creator_agg.columns:
        perm_creator_deseas[field] = deseasonalize_dow(creator_agg[c_col].reindex(dates))
    if r_col in reddit_agg.columns:
        perm_reddit_deseas[field]  = deseasonalize_dow(reddit_agg[r_col].reindex(dates))


def _shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values in place, keeping NaN positions fixed."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


np.random.seed(42)
# perm_cat_means[i] = dict of {category: mean_peak_lag} for permutation i
perm_cat_means = {cat: [] for cat in obs_cat_means.index}

for i in range(N_PERM):
    shift = np.random.randint(1, len(dates))
    perm_by_field = {}   # field → peak lag under this permutation
    for field in topic_fields_for_perm:
        if field not in perm_creator_deseas or field not in perm_reddit_deseas:
            continue
        c_s = _shift_values_only(perm_creator_deseas[field], shift)
        r_s = perm_reddit_deseas[field]
        both = pd.concat([c_s, r_s], axis=1).dropna()
        if len(both) < MIN_OVERLAP:
            continue
        cc = normalised_xcorr(c_s, r_s, MAX_LAG_DAYS)
        if cc.empty:
            continue
        perm_by_field[field] = int(cc.abs().idxmax())

    # Aggregate by category
    perm_field_df = pd.DataFrame([
        {'field': f, 'peak_lag': l, 'category': TOPIC_CATEGORY.get(f, 'Other')}
        for f, l in perm_by_field.items()
    ])
    if not perm_field_df.empty:
        for cat, grp in perm_field_df.groupby('category'):
            if cat in perm_cat_means:
                perm_cat_means[cat].append(grp['peak_lag'].mean())

    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{N_PERM} permutations done...')

# Compute p-values per category (one-tailed left: is observed mean more negative than null?)
perm_results = []
for cat in sorted(obs_cat_means.index):
    null = np.array(perm_cat_means.get(cat, []))
    if len(null) == 0:
        continue
    obs  = obs_cat_means[cat]
    p    = (null <= obs).mean()
    perm_results.append({
        'category':       cat,
        'obs_mean_lag':   round(obs, 2),
        'null_mean':      round(null.mean(), 2),
        'null_std':       round(null.std(), 2),
        'p_left_tail':    round(p, 4),
    })
    print(f'  {cat:<28}  obs={obs:+.1f}  null={null.mean():+.1f}±{null.std():.1f}  p={p:.4f}')

perm_df = pd.DataFrame(perm_results)
perm_df.to_csv(TABLES_DIR / '05_permutation_results.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "05_permutation_results.csv"}')

# Figure 05f: per-category observed vs null
n_cats = len(perm_results)
fig6, axes6 = plt.subplots(2, (n_cats + 1) // 2, figsize=(16, 8))
axes6 = axes6.flatten()
for ax_i, row in enumerate(perm_results):
    cat  = row['category']
    null = np.array(perm_cat_means[cat])
    obs  = row['obs_mean_lag']
    axes6[ax_i].hist(null, bins=25, color=cat_palette.get(cat, 'gray'),
                     alpha=0.7, edgecolor='white')
    axes6[ax_i].axvline(obs, color='black', lw=2,
                        label=f'obs={obs:.1f}d  p={row["p_left_tail"]:.3f}')
    axes6[ax_i].axvline(row['null_mean'], color='gray', lw=1.2, linestyle='--')
    axes6[ax_i].set_title(cat, fontsize=8)
    axes6[ax_i].legend(fontsize=7)
    axes6[ax_i].grid(True, alpha=0.2)

for ax in axes6[len(perm_results):]:
    ax.set_visible(False)

fig6.suptitle('Permutation test: per-category mean peak lag vs. null\n'
              '(circular shift of non-NaN creator values, n=1000)',
              fontsize=12)
plt.tight_layout()
out_f = FIGURES_DIR / '05f_permutation_test.png'
plt.savefig(out_f, dpi=150, bbox_inches='tight')
print(f'Saved: {out_f}')
plt.show()

#%%
# ── Weighting robustness: central lead/lag test under BOTH Reddit schemes ─────
# The field-level tables, Granger, and permutation test above use unweighted
# Reddit only. Here we re-run the central creator->Reddit peak-lag test under
# engagement-weighted Reddit as well, UNCONDITIONALLY, and report both schemes
# regardless of outcome. This is the direct demonstration of "null robust to
# Reddit weighting" that script 11 could only infer from a level-offset argument.
# Creator side is held at unweighted (matches the field-level design); only the
# Reddit aggregation scheme varies.

def _weighting_field_summary(creator_daily, reddit_daily, reddit_scheme,
                             dates_idx, max_lag, min_overlap, deseasonalize):
    """Peak-lag / peak-corr per field for creator(unweighted) x reddit(scheme)."""
    recs = []
    for field in all_fields:
        c_col = f'{field}_unweighted'
        r_col = f'{field}_{reddit_scheme}'
        if c_col not in creator_daily.columns or r_col not in reddit_daily.columns:
            continue
        c_s = creator_daily[c_col].reindex(dates_idx)
        r_s = reddit_daily[r_col].reindex(dates_idx)
        if deseasonalize:
            c_s = deseasonalize_dow(c_s)
            r_s = deseasonalize_dow(r_s)
        both = pd.concat([c_s, r_s], axis=1).dropna()
        if len(both) < min_overlap:
            continue
        cc = normalised_xcorr(c_s, r_s, max_lag)
        if cc.empty:
            continue
        peak_lag = int(cc.abs().idxmax())
        recs.append({
            'field':     field,
            'peak_lag':  peak_lag,
            'peak_corr': round(float(cc.loc[peak_lag]), 4),
            'corr_at_0': round(float(cc.get(0, np.nan)), 4),
            'n_overlap': len(both),
            'category':  TOPIC_CATEGORY.get(field, 'Other'),
        })
    return pd.DataFrame(recs)


_wt_specs = {
    'daily':  dict(cre=creator_agg,        red=reddit_agg,
                   didx=dates,     max_lag=MAX_LAG_DAYS,  min_overlap=MIN_OVERLAP,       des=True),
    'weekly': dict(cre=creator_weekly_agg, red=reddit_weekly_agg,
                   didx=all_weeks, max_lag=MAX_LAG_WEEKS, min_overlap=MIN_OVERLAP_WEEKS, des=False),
}

wt_field_dfs   = {}
robustness_rows = []
for gran, sp in _wt_specs.items():
    for rs in reddit_schemes:
        fdf = _weighting_field_summary(sp['cre'], sp['red'], rs, sp['didx'],
                                       sp['max_lag'], sp['min_overlap'], sp['des'])
        wt_field_dfs[(gran, rs)] = fdf
        topic = fdf[fdf['field'] != 'overall__score']
        ov    = fdf[fdf['field'] == 'overall__score']
        robustness_rows.append({
            'granularity':       gran,
            'reddit_scheme':     rs,
            'n_topic_fields':    len(topic),
            'median_peak_lag':   topic['peak_lag'].median(),
            'mean_peak_lag':     round(topic['peak_lag'].mean(), 2),
            # Two conventions, named so neither is mistaken for the other: the
            # paper's "creator-leading" fraction is strictly lag > 0.
            'pct_peak_lag_gt0':  round((topic['peak_lag'] > 0).mean() * 100, 1),
            'pct_peak_lag_ge0':  round((topic['peak_lag'] >= 0).mean() * 100, 1),
            'overall_peak_lag':  int(ov['peak_lag'].iloc[0]) if len(ov) else np.nan,
            'overall_peak_corr': round(float(ov['peak_corr'].iloc[0]), 4) if len(ov) else np.nan,
        })

# Persist engagement-weighted field summaries (parity with the unweighted CSVs)
wt_field_dfs[('daily', 'engagement_weighted')].sort_values('peak_lag').to_csv(
    TABLES_DIR / '05_field_xcorr_summary_engagement_weighted.csv', index=False)
wt_field_dfs[('weekly', 'engagement_weighted')].sort_values('peak_lag').to_csv(
    TABLES_DIR / '05_field_xcorr_summary_weekly_engagement_weighted.csv', index=False)

robustness_df = pd.DataFrame(robustness_rows)
robustness_df.to_csv(TABLES_DIR / '05_weighting_robustness.csv', index=False)
print('\n-- Central lead/lag test: robustness to Reddit weighting --')
print(robustness_df.to_string(index=False))

# Per-field stability: are the INDIVIDUAL field peak lags stable across schemes?
# Instability under reweighting is itself evidence the field-level peaks are noise.
for gran in ('daily', 'weekly'):
    uw = wt_field_dfs[(gran, 'unweighted')].set_index('field')['peak_lag']
    ew = wt_field_dfs[(gran, 'engagement_weighted')].set_index('field')['peak_lag']
    common = uw.index.intersection(ew.index)
    shift  = ew.reindex(common) - uw.reindex(common)
    unit   = 'd' if gran == 'daily' else 'wk'
    big    = 7 if gran == 'daily' else 1
    lag_r  = np.corrcoef(uw.reindex(common), ew.reindex(common))[0, 1]
    print(f'  {gran}: per-field lag corr across schemes r={lag_r:+.3f}; '
          f'|shift|>{big}{unit} in {(shift.abs() > big).sum()}/{len(common)} fields '
          f'(mean shift {shift.mean():+.2f}{unit})')

#%%
# ── Figure: peak-lag distribution overlay, both Reddit schemes ───────────────
fig7, axes7 = plt.subplots(1, 2, figsize=(15, 5))
scheme_hist_colors = {'unweighted': '#2980b9', 'engagement_weighted': '#e67e22'}
for ax, (gran, mlag) in zip(axes7, [('daily', MAX_LAG_DAYS), ('weekly', MAX_LAG_WEEKS)]):
    for rs in reddit_schemes:
        t = wt_field_dfs[(gran, rs)]
        t = t[t['field'] != 'overall__score']['peak_lag']
        ax.hist(t, bins=range(-mlag - 1, mlag + 2), alpha=0.5,
                color=scheme_hist_colors[rs], edgecolor='white',
                label=f'Reddit {rs} (med {t.median():+.0f})')
        ax.axvline(t.median(), color=scheme_hist_colors[rs], lw=1.5, linestyle=':')
    ax.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.7)
    unit = 'days' if gran == 'daily' else 'weeks'
    ax.set_xlabel(f'Peak lag ({unit})  [+ = creator leads Reddit]', fontsize=10)
    ax.set_ylabel('Number of topic fields', fontsize=10)
    ax.set_title(f'{gran.capitalize()} field-level peak lags', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
fig7.suptitle('Central lead/lag test under both Reddit weighting schemes\n'
              '(field-level peak-lag distribution: unweighted vs. engagement-weighted Reddit)',
              fontsize=13)
plt.tight_layout()
out_g = FIGURES_DIR / '05_weighting_robustness.png'
plt.savefig(out_g, dpi=150, bbox_inches='tight')
print(f'Saved: {out_g}')
plt.show()

#%%
# ── H1 diagnostic battery: trend, stationarity, release calendar ─────────────
# Applied UNIFORMLY to every game's weekly overall-score H1 test, against both
# community targets. (Previously these checks existed only for H3 / volume, and
# were first run on H1 ad hoc for Warframe after its lead appeared -- running
# them on every game removes that asymmetry.) Appended at the end of the script
# with its own RNG so no earlier output in 05 can change.
#
#   A. ADF unit-root test on each weekly series (levels).
#   B. Gated weekly xcorr (+-4 wk) under three treatments: levels, linear
#      detrending (each series separately), first differences. Estimator is the
#      date-aware per-lag Pearson used by 04h/12 (shift on the regular weekly
#      index, then drop NaN pairs), so r can differ slightly from the
#      normalised-xcorr headline above. Gate = circular shift of the non-NaN
#      creator values (same null model as 04/05b/06), 95th pct of max|r|.
#   C. Release-week exclusion (weeks within +-W days of a release start dropped
#      AFTER treatment, so lags stay true offsets) for levels and detrended.
#   D. Date-correct weekly Granger (_granger_date_correct), lags 1-4, both
#      directions (creator->community and community->creator), all treatments.
#
# Reading: a creator lead that survives detrending and release exclusion is not
# a trend or release-calendar artifact; one that vanishes under first
# differencing is a slow (multi-week) component rather than week-to-week
# transmission. A lead that APPEARS under detrending would mean a trend was
# masking it -- the check cuts both ways.
from statsmodels.tsa.stattools import adfuller
from analysis.shared.loaders import load_releases

DIAG_MAX_LAG   = 4
DIAG_N_PERM    = 1000
DIAG_MIN_PAIRS = 30
DIAG_WIDTHS    = [0, 14, 28]
_diag_rng = np.random.default_rng(20261018)


def _diag_lagcorr(xv: np.ndarray, yv: np.ndarray, max_lag: int) -> dict:
    """corr(x_t, y_{t+h}) for h in -max_lag..max_lag on aligned arrays with NaN."""
    n = len(xv)
    out = {}
    for h in range(-max_lag, max_lag + 1):
        a, b = (xv[:n - h], yv[h:]) if h >= 0 else (xv[-h:], yv[:n + h])
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < DIAG_MIN_PAIRS:
            continue
        aa, bb = a[m] - a[m].mean(), b[m] - b[m].mean()
        d = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
        if d > 0:
            out[h] = float((aa * bb).sum() / d)
    return out


def _diag_gated(x: pd.Series, y: pd.Series, keep: pd.Series):
    xv = x.where(keep).to_numpy(dtype=float)
    yv = y.where(keep).to_numpy(dtype=float)
    obs = _diag_lagcorr(xv, yv, DIAG_MAX_LAG)
    if not obs:
        return None
    pk = max(obs, key=lambda k: abs(obs[k]))
    valid = ~np.isnan(xv)
    vals = xv[valid]
    null = []
    for _ in range(DIAG_N_PERM):
        xs = xv.copy()
        xs[valid] = np.roll(vals, int(_diag_rng.integers(1, max(len(vals), 2))))
        v = _diag_lagcorr(xs, yv, DIAG_MAX_LAG)
        null.append(max(abs(t) for t in v.values()) if v else 0.0)
    thr = float(np.percentile(null, 95))
    p = float(np.mean(np.asarray(null) >= abs(obs[pk])))
    sig = abs(obs[pk]) > thr
    if not sig:
        rel = 'no peak above noise'
    elif pk > 0 and obs[pk] > 0:
        rel = 'creator leads (consistent with hypothesis)'
    elif pk > 0:
        rel = 'creator-leading lag, negative r (contrary)'
    elif pk == 0:
        rel = 'concurrent (lag 0)'
    else:
        rel = 'community leads'
    row = {'peak_lag': int(pk), 'peak_corr': round(obs[pk], 4),
           'noise_threshold': round(thr, 4), 'p_value': round(p, 4),
           'significant': bool(sig), 'reading': rel,
           'n_weeks_kept': int(keep.sum())}
    row.update({f'r_at_{h:+d}': (round(obs[h], 4) if h in obs else np.nan)
                for h in range(-DIAG_MAX_LAG, DIAG_MAX_LAG + 1)})
    return row


def _detrend(s: pd.Series) -> pd.Series:
    t = np.arange(len(s), dtype=float)
    m = s.notna().to_numpy()
    if m.sum() < 3:
        return s
    b = np.polyfit(t[m], s.to_numpy(dtype=float)[m], 1)
    return s - (b[0] * t + b[1])


diag_c = creator_weekly_agg['overall__score_unweighted']
diag_targets = {
    'submissions': reddit_weekly_agg['overall__score_unweighted'],
    'comments':    comment_weekly_agg['sentiment_score_unweighted'],
}
diag_treat = {
    'levels':     lambda s: s,
    'detrended':  _detrend,
    'first_diff': lambda s: s.diff(),
}

# Distance (days) from each week to the nearest release start
_rel = pd.to_datetime(load_releases()['start_date']).dropna()
_wk_ord = all_weeks.values.astype('datetime64[D]').astype(np.int64)
_rel_ord = _rel.values.astype('datetime64[D]').astype(np.int64)
if len(_rel_ord):
    _rel_dist = np.array([int(np.min(np.abs(w - _rel_ord))) for w in _wk_ord])
else:
    _rel_dist = np.full(len(_wk_ord), 10 ** 6)

# A. ADF
adf_rows = []
_t = np.arange(len(all_weeks))
for nm, s in [('creator', diag_c)] + list(diag_targets.items()):
    si = s.interpolate(limit_direction='both')
    if si.notna().sum() > 20:
        a = adfuller(si.dropna(), autolag='AIC')
        m = s.notna().to_numpy()
        adf_rows.append({'series': nm, 'adf_stat': round(a[0], 3), 'p_value': round(a[1], 4),
                         'n_weeks': int(m.sum()),
                         'corr_with_time': round(float(np.corrcoef(_t[m], s.to_numpy()[m])[0, 1]), 3)})
adf_df = pd.DataFrame(adf_rows)
adf_df.to_csv(TABLES_DIR / '05_h1_adf.csv', index=False)

# B + C. Gated xcorr by treatment and release-exclusion width
diag_rows = []
for tgt, y_raw in diag_targets.items():
    for tr, f in diag_treat.items():
        x, y = f(diag_c), f(y_raw)
        for W in DIAG_WIDTHS:
            if W and tr == 'first_diff':
                continue
            keep_arr = (_rel_dist > W) if W else np.ones(len(all_weeks), bool)
            keep = pd.Series(keep_arr, index=all_weeks)
            res = _diag_gated(x, y, keep)
            if res:
                diag_rows.append({'target': tgt, 'treatment': tr, 'exclude_days': W, **res})
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(TABLES_DIR / '05_h1_diagnostics.csv', index=False)

# D. Weekly date-correct Granger, both directions
gw_rows = []
for tgt, y_raw in diag_targets.items():
    for tr, f in diag_treat.items():
        x, y = f(diag_c), f(y_raw)
        for direction, (yy, xx) in {'creator->community': (y, x),
                                    'community->creator': (x, y)}.items():
            for L in range(1, DIAG_MAX_LAG + 1):
                res = _granger_date_correct(yy, xx, L)
                if res is None:
                    continue
                F, p, coefs, n = res
                gw_rows.append({'target': tgt, 'treatment': tr, 'direction': direction,
                                'lag_weeks': L, 'f_stat': round(F, 3), 'p_value': round(p, 4),
                                'coef_sum': round(float(coefs.sum()), 4),
                                'lagged_coefs': ';'.join(f'{c:+.4f}' for c in coefs),
                                'n_obs': n})
gw_df = pd.DataFrame(gw_rows)
gw_df.to_csv(TABLES_DIR / '05_h1_granger_weekly.csv', index=False)

print('\n-- H1 diagnostic battery (weekly, unweighted overall score) --')
print(adf_df.to_string(index=False))
print(diag_df[['target', 'treatment', 'exclude_days', 'n_weeks_kept', 'peak_lag',
               'peak_corr', 'p_value', 'reading']].to_string(index=False))
if not gw_df.empty:
    _s = (gw_df.assign(sig=gw_df.p_value < 0.05)
          .groupby(['target', 'treatment', 'direction'])
          .agg(sig_lags=('sig', 'sum'), min_p=('p_value', 'min'), n_lags=('lag_weeks', 'size')))
    print(_s.to_string())
print(f"Saved: {TABLES_DIR / '05_h1_adf.csv'}, 05_h1_diagnostics.csv, 05_h1_granger_weekly.csv")
