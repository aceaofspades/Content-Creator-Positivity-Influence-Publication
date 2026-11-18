#%%
"""
03_sentiment_decay_curves.py
-----------------------------
Within-season sentiment decay curves using release_phase and days_since_release.

Produces:
  1. Mean sentiment arc by release_phase bucket (launch_week / early_season /
     mid_season / late_season) — creators vs. Reddit side by side
  2. Late-season creator sentiment vs. next-season revival metric scatter —
     tests whether end-of-season negativity predicts next launch performance

Outputs:
  figures/03a_sentiment_by_phase.png
  figures/03b_late_season_vs_revival.png
  tables/03_late_season_revival_correlation.csv
  tables/03_hierarchical_regression.csv  — M1/M2/M3 adj-R², sentiment coef, partial r
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import aggregate_creator_daily, aggregate_reddit_daily

SENTIMENT_FIELD = 'overall__score'
PHASE_ORDER     = ['launch_week', 'early_season', 'mid_season', 'late_season']
PHASE_LABELS    = {
    'launch_week':   'Launch week',
    'early_season':  'Early season',
    'mid_season':    'Mid season',
    'late_season':   'Late season',
}

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
cm = load_comments_master()

#%%
# ── Figure 1: Sentiment by release phase ─────────────────────────────────────
def phase_summary(df: pd.DataFrame, field: str = SENTIMENT_FIELD) -> pd.DataFrame:
    """Mean ± SE of sentiment by release_phase, across all seasons."""
    grp = df.groupby('release_phase')[field]
    summary = pd.DataFrame({
        'mean': grp.mean(),
        'se':   grp.sem(),
        'n':    grp.count(),
    }).loc[[p for p in PHASE_ORDER if p in grp.groups]]
    return summary


yt_phase = phase_summary(yt)
rd_phase = phase_summary(rd)

# Include comments if release_phase is present in the master
if 'release_phase' in cm.columns:
    cm_phase = phase_summary(cm, field='sentiment_score')
    print(f'Comments phase data available: {dict(cm_phase["n"])}')
else:
    cm_phase = None
    print('Comments master lacks release_phase — omitting from phase chart.')

sources = [
    (yt_phase, 'YouTube creators',    '#3498db', SENTIMENT_FIELD),
    (rd_phase, 'Reddit submissions',   '#e67e22', SENTIMENT_FIELD),
]
if cm_phase is not None:
    sources.append((cm_phase, 'Reddit comments', '#27ae60', 'sentiment_score'))

fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources) + 1, 5), sharey=True)
if len(sources) == 1:
    axes = [axes]

for ax, (phase_df, title, color, _field) in zip(axes, sources):
    x = np.arange(len(phase_df))
    labels = [PHASE_LABELS.get(p, p) for p in phase_df.index]
    ax.bar(x, phase_df['mean'], yerr=phase_df['se'] * 1.96,
           color=color, alpha=0.75, capsize=5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Mean sentiment score (−2 to +2)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_ylim(-2.2, 2.2)
    ax.grid(axis='y', alpha=0.3)
    for xi, (_, row) in zip(x, phase_df.iterrows()):
        ax.text(xi, row['mean'] + row['se'] * 2 + 0.05, f"n={row['n']:,}",
                ha='center', va='bottom', fontsize=8, color='#555555')

fig.suptitle('Within-season sentiment by release phase', fontsize=14)
plt.tight_layout()
out_a = FIGURES_DIR / '03a_sentiment_by_phase.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'Saved: {out_a}')
plt.show()

#%%
# ── Figure 2: Late-season creator sentiment vs. next-season revival ───────────
# Load release metrics produced by 01_release_categorization.py
metrics_path = TABLES_DIR / '01_release_metrics.csv'
if not metrics_path.exists():
    metrics_path = TABLES_DIR / '02_release_metrics.csv'  # legacy fallback
if not metrics_path.exists():
    print('Run 01_release_categorization.py first to generate release metrics.')
else:
    metrics = pd.read_csv(metrics_path, parse_dates=['start_date'])

    # Late-season creator sentiment per season
    late_sent = (
        yt[yt['release_phase'] == 'late_season']
        .groupby('current_season')[SENTIMENT_FIELD]
        .mean()
        .rename('late_season_creator_sentiment')
    )

    # Merge: each season's late sentiment → *next* season's revival
    season_order = metrics.sort_values('start_date')['season_name'].tolist()
    pairs = []
    for i, season in enumerate(season_order[:-1]):
        next_season = season_order[i + 1]
        late = late_sent.get(season, np.nan)
        revival = metrics.loc[metrics['season_name'] == next_season, 'revival']
        if not revival.empty:
            pairs.append({
                'season':            season,
                'next_season':       next_season,
                'late_season_sent':  late,
                'next_revival':      revival.values[0],
            })

    pairs_df = pd.DataFrame(pairs).dropna()

    if len(pairs_df) >= 3:
        r, p = stats.pearsonr(pairs_df['late_season_sent'], pairs_df['next_revival'])
        print(f'\nLate-season sentiment -> next-season revival:  r={r:.3f}  p={p:.3f}  n={len(pairs_df)}')

        pairs_df.to_csv(TABLES_DIR / '03_late_season_revival_correlation.csv', index=False)

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        ax2.scatter(pairs_df['late_season_sent'], pairs_df['next_revival'],
                    s=60, alpha=0.8, color='#8e44ad', edgecolors='white')
        def short_name(s: str) -> str:
            for prefix in ('Season of the ', 'Season of '):
                if s.startswith(prefix):
                    return s[len(prefix):]
            return s

        for _, row in pairs_df.iterrows():
            ax2.annotate(short_name(row['next_season']),
                         (row['late_season_sent'], row['next_revival']),
                         fontsize=7, xytext=(4, 4), textcoords='offset points', color='#555555')

        # Regression line
        m, b = np.polyfit(pairs_df['late_season_sent'], pairs_df['next_revival'], 1)
        xs = np.linspace(pairs_df['late_season_sent'].min(), pairs_df['late_season_sent'].max(), 100)
        ax2.plot(xs, m * xs + b, 'r--', lw=1.5, alpha=0.7, label=f'r = {r:.3f}, p = {p:.3f}, slope = {m:.3f}')

        ax2.set_xlabel('Creator sentiment in the season before this launch\n(mean overall score, late-season videos only)', fontsize=11)
        ax2.set_ylabel('Revival at launch (peak / prior-year mean daily players)', fontsize=11)
        ax2.set_title('Does pre-launch creator mood predict revival?\n'
                      'Each dot = one season launch; X = how creators felt the season before', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out_b = FIGURES_DIR / '03b_late_season_vs_revival.png'
        plt.savefig(out_b, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_b}')
        plt.show()
    else:
        print('Not enough season pairs with complete data for the scatter plot.')

#%%
# ── Hierarchical regression: does sentiment add beyond the revival lag? ────────
# Tests whether late-season creator sentiment carries information about the next
# launch beyond what the prior season's own revival already predicts.
#
# M1 (autoregressive): revival_n+1 ~ revival_n
# M2 (sentiment only): revival_n+1 ~ sentiment_n          [= what scatter shows]
# M3 (full):           revival_n+1 ~ revival_n + sentiment_n
#
# Key result: coefficient + p-value on sentiment in M3.
# If it survives after controlling for revival_n, the r=0.42 is not just
# two sticky series tracking the same slow trend.

if 'pairs_df' in dir() and len(pairs_df) >= 8:
    # Add revival_n: the source season's own revival
    revival_lookup = metrics.set_index('season_name')['revival']
    pairs_df = pairs_df.copy()
    pairs_df['revival_n'] = pairs_df['season'].map(revival_lookup)
    reg_df = pairs_df.dropna(subset=['revival_n', 'late_season_sent', 'next_revival'])

    y  = reg_df['next_revival'].values
    r_n  = sm.add_constant(reg_df[['revival_n']])
    s_n  = sm.add_constant(reg_df[['late_season_sent']])
    rs_n = sm.add_constant(reg_df[['revival_n', 'late_season_sent']])

    m1 = sm.OLS(y, r_n).fit()
    m2 = sm.OLS(y, s_n).fit()
    m3 = sm.OLS(y, rs_n).fit()

    # Lag-1 autocorrelations as confound-risk diagnostic
    autocorr_revival  = reg_df['revival_n'].corr(reg_df['next_revival'])
    autocorr_sent     = (
        pairs_df['late_season_sent']
        .corr(pairs_df['late_season_sent'].shift(-1))
    )

    # Partial correlation: correlate residuals of (revival_n+1 ~ revival_n)
    # with residuals of (sentiment_n ~ revival_n)
    resid_y   = sm.OLS(y, r_n).fit().resid
    resid_s   = sm.OLS(reg_df['late_season_sent'].values, r_n).fit().resid
    partial_r, partial_p = stats.pearsonr(resid_s, resid_y)

    sent_coef = m3.params['late_season_sent']
    sent_t    = m3.tvalues['late_season_sent']
    sent_p    = m3.pvalues['late_season_sent']

    print('\n── Hierarchical regression ──────────────────────────────────')
    print(f'  n = {len(reg_df)} season pairs')
    print(f'  Lag-1 autocorr  revival:   {autocorr_revival:.3f}')
    print(f'  Lag-1 autocorr  sentiment: {autocorr_sent:.3f}')
    print(f'  M1  revival_n only:         adj-R2 = {m1.rsquared_adj:.3f}')
    print(f'  M2  sentiment only:         adj-R2 = {m2.rsquared_adj:.3f}')
    print(f'  M3  revival_n + sentiment:  adj-R2 = {m3.rsquared_adj:.3f}')
    print(f'  Sentiment coef in M3:  beta={sent_coef:.3f}  t={sent_t:.2f}  p={sent_p:.3f}')
    print(f'  Partial r (sentiment | revival_n): {partial_r:.3f}  p={partial_p:.3f}')
    if sent_p < 0.05:
        print('  -> Sentiment survives after controlling for revival lag.')
    else:
        print('  -> Sentiment does NOT survive after controlling for revival lag.')

    reg_records = [
        {'model': 'M1_revival_only',    'adj_r2': round(m1.rsquared_adj, 3),
         'sentiment_beta': np.nan,       'sentiment_t': np.nan, 'sentiment_p': np.nan},
        {'model': 'M2_sentiment_only',  'adj_r2': round(m2.rsquared_adj, 3),
         'sentiment_beta': round(m2.params['late_season_sent'], 3),
         'sentiment_t':    round(m2.tvalues['late_season_sent'], 3),
         'sentiment_p':    round(m2.pvalues['late_season_sent'], 3)},
        {'model': 'M3_full',            'adj_r2': round(m3.rsquared_adj, 3),
         'sentiment_beta': round(sent_coef, 3),
         'sentiment_t':    round(sent_t, 3),
         'sentiment_p':    round(sent_p, 3)},
        {'model': 'diagnostics',
         'adj_r2': np.nan,
         'sentiment_beta': round(partial_r, 3),
         'sentiment_t':    round(partial_p, 3),
         'sentiment_p':    np.nan},
    ]
    reg_df_out = pd.DataFrame(reg_records)
    reg_df_out.to_csv(TABLES_DIR / '03_hierarchical_regression.csv', index=False)
    print(f'\n  Saved: {TABLES_DIR / "03_hierarchical_regression.csv"}')
