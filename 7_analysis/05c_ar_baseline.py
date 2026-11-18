#%%
"""
05c_ar_baseline.py
-------------------
Autoregressive baseline comparison for the creator→Reddit lead/lag analysis.

Tests whether creator sentiment predicts Reddit sentiment beyond what Reddit's
own recent history already predicts. Without this, apparent lead relationships
could reflect shared autocorrelation rather than genuine information flow.

Approach:
  M1: AR(p) on the community target alone (p selected by AIC, 1..14)
  M2: AR(p) + creator sentiment at lag h (one model per test lag h)

For each h, the comparison asks: does creator_{t-h} improve predictions of
target_t after controlling for target_{t-1..p}? The creator term's t-stat and
p-value in M2 are a single-lag Granger-type test. The R² and BIC delta show
practical effect size.

Two community targets are tested — Reddit **submissions** (overall__score) and
Reddit **comments** (sentiment_score) — matching the targets in script 05's
Granger test. These are two parameterisations of the same question, not a test
and its validator: script 05 runs a joint F-test over creator lags 1..L, this
runs a t-test on a single creator lag h with an AIC-selected AR order. Where
they disagree that is a specification difference, and both should be reported.

Sign is reported alongside significance in both. A negative creator
coefficient rejects the same null as a positive one but means the opposite:
higher creator sentiment preceding *lower* community sentiment is evidence
against elite influence, not for it.

Test lags (daily): h = 1, 2, 7, 14 days
  Positive h means creator at t-h predicts reddit at t
  (creator leads reddit by h days — the elite-influence direction)

Test lags (weekly): h = 1, 2, 3, 4 weeks (motivated by script 04 finding that
  weekly granularity doubles engagement-player correlations)

All series: deseasonalized daily overall__score (DOW dummies removed), or
weekly means for the weekly analysis.

Outputs:
  figures/05c_ar_baseline.png             — delta-R² bars at each lag, both granularities
  tables/05c_ar_baseline_daily.csv        — full model comparison table, daily
  tables/05c_ar_baseline_weekly.csv       — full model comparison table, weekly
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    deseasonalize_dow,
)

MAX_AR_ORDER     = 14
TEST_LAGS_DAILY  = [1, 2, 7, 14]
TEST_LAGS_WEEKLY = [1, 2, 3, 4]
FIELD            = 'overall__score'
COMMENT_FIELD    = 'sentiment_score'

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
cm = load_comments_master()

creator_agg = aggregate_creator_daily(yt, [FIELD])
reddit_agg  = aggregate_reddit_daily(rd,  [FIELD])
comment_agg = aggregate_comments_daily(cm, [COMMENT_FIELD])

start = max(creator_agg.index.min(), reddit_agg.index.min(), comment_agg.index.min())
end   = min(creator_agg.index.max(), reddit_agg.index.max(), comment_agg.index.max())
dates = pd.date_range(start, end, freq='D')
creator_agg = creator_agg.reindex(dates)
reddit_agg  = reddit_agg.reindex(dates)
comment_agg = comment_agg.reindex(dates)

c_daily = deseasonalize_dow(creator_agg[f'{FIELD}_unweighted'].reindex(dates))
r_daily = deseasonalize_dow(reddit_agg[f'{FIELD}_unweighted'].reindex(dates))
# Comments are tested as a second community target. Script 05's Granger test
# already runs on both submissions and comments; running the AR baseline on
# submissions only left the strongest Granger results in several games with no
# complementary single-lag test at all.
m_daily = deseasonalize_dow(comment_agg[f'{COMMENT_FIELD}_unweighted'].reindex(dates))

print(f'Daily series: {len(dates)} days, '
      f'creator non-NaN={c_daily.notna().sum()}, '
      f'submissions non-NaN={r_daily.notna().sum()}, '
      f'comments non-NaN={m_daily.notna().sum()}')

DAILY_TARGETS  = [('submissions', r_daily), ('comments', m_daily)]

#%%
# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_ar_df(reddit: pd.Series, p: int) -> pd.DataFrame:
    """Build lag matrix for AR(p): y and y_lag1..p, drop NaN rows."""
    df = pd.DataFrame({'y': reddit})
    for i in range(1, p + 1):
        df[f'y_lag{i}'] = reddit.shift(i)
    return df.dropna()


def select_ar_order(reddit: pd.Series, max_order: int) -> int:
    """
    Select AR lag order by AIC (1..max_order).

    Lags are built on the regular date index, so y_lag{i} is always i calendar
    periods back. (This used to dropna() first, which made a "lag" count
    observed rows in sparse series.) Every candidate order is scored on the
    same rows -- those complete at max_order -- because AIC is only comparable
    across models fitted to the same data.
    """
    common = _build_ar_df(reddit, max_order).index
    best_p, best_aic = 1, np.inf
    for p in range(1, max_order + 1):
        df = _build_ar_df(reddit, p).loc[common]
        if len(df) < p + 10:
            continue
        X = sm.add_constant(df.drop('y', axis=1), has_constant='add')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.OLS(df['y'], X).fit(disp=0)
        if res.aic < best_aic:
            best_aic, best_p = res.aic, p
    return best_p


def fit_ar_pair(target: pd.Series, creator: pd.Series, p: int, h: int):
    """
    Fit AR(p) and AR(p)+creator_{t-h} on the SAME rows. Returns (m1, m2, n).

    The restricted model has to be estimated on the unrestricted model's
    sample. Fitting AR(p) on every non-NaN target day while fitting
    AR(p)+creator on the smaller creator-overlap subset makes R² and BIC
    describe different datasets, so delta_r2 / delta_bic stop being model
    comparisons at all. Where creator coverage is sparse this is not a subtle
    bias: Warframe reported a BIC drop of ~3 100 from adding a single
    regressor, which is impossible at equal n. Only the creator t-stat (a
    within-model quantity) was unaffected.
    """
    combined = pd.DataFrame({'y': target, 'creator': creator})
    df = pd.DataFrame({'y': combined['y']})
    for i in range(1, p + 1):
        df[f'y_lag{i}'] = combined['y'].shift(i)
    df['creator_lag'] = combined['creator'].shift(h)
    df = df.dropna()
    if len(df) < p + 10:
        return None, None, 0

    ar_cols = [f'y_lag{i}' for i in range(1, p + 1)]
    X_restricted   = sm.add_constant(df[ar_cols], has_constant='add')
    X_unrestricted = sm.add_constant(df[ar_cols + ['creator_lag']], has_constant='add')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m1 = sm.OLS(df['y'], X_restricted).fit(disp=0)
        m2 = sm.OLS(df['y'], X_unrestricted).fit(disp=0)
    return m1, m2, len(df)


def run_ar_baseline(target: pd.Series, creator: pd.Series, target_label: str,
                    test_lags: list, lag_col: str, unit: str,
                    max_order: int) -> tuple:
    """Run the AR-baseline comparison for one target series. Returns (df, p)."""
    best_p = select_ar_order(target, max_order)
    print(f'\n[{target_label}] AR order selected by AIC ({unit}): p={best_p}')

    records = []
    for h in test_lags:
        m1, m2, n = fit_ar_pair(target, creator, best_p, h)
        if m2 is None:
            print(f'  lag={h}{unit[0]} - insufficient data')
            continue

        creator_idx = [i for i, name in enumerate(m2.model.exog_names)
                       if 'creator' in name]
        if not creator_idx:
            continue
        ci = creator_idx[0]

        record = {
            'target':        target_label,
            lag_col:         h,
            'ar_order':      best_p,
            'n_obs':         n,
            'ar_only_r2':    round(m1.rsquared, 5),
            'ar_only_bic':   round(m1.bic, 2),
            'ar_plus_r2':    round(m2.rsquared, 5),
            'ar_plus_bic':   round(m2.bic, 2),
            'delta_r2':      round(m2.rsquared - m1.rsquared, 5),
            'delta_bic':     round(m2.bic - m1.bic, 2),
            'creator_coef':  round(m2.params.iloc[ci], 5),
            'creator_tstat': round(m2.tvalues.iloc[ci], 3),
            'creator_pval':  round(m2.pvalues.iloc[ci], 4),
        }
        # Sign matters as much as significance: a negative creator coefficient
        # is evidence against elite influence, not for it.
        record['sign'] = 'positive' if record['creator_coef'] > 0 else 'negative'
        record['hypothesis_relation'] = (
            'consistent with hypothesis' if record['creator_coef'] > 0
            else 'contrary (inverse relation)'
        )
        records.append(record)

        sig = '**' if record['creator_pval'] < 0.01 else ('*' if record['creator_pval'] < 0.05 else '')
        print(f'  lag={h}{unit[0]} - dR2={record["delta_r2"]:+.5f}  '
              f'dBIC={record["delta_bic"]:+.1f}  '
              f'creator coef={record["creator_coef"]:+.4f}  '
              f'p={record["creator_pval"]:.4f}{sig}  [{record["sign"]}]')

    return pd.DataFrame(records), best_p


#%%
# ── Daily analysis (both community targets) ──────────────────────────────────
daily_frames, daily_orders = [], {}
for label, series in DAILY_TARGETS:
    df_t, p_t = run_ar_baseline(series, c_daily, label, TEST_LAGS_DAILY,
                                'lag_days', 'days', MAX_AR_ORDER)
    daily_orders[label] = p_t
    if not df_t.empty:
        daily_frames.append(df_t)

daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
daily_df.to_csv(TABLES_DIR / '05c_ar_baseline_daily.csv', index=False)
print(f'\nSaved: {TABLES_DIR / "05c_ar_baseline_daily.csv"}')

#%%
# ── Weekly analysis (both community targets) ─────────────────────────────────
creator_weekly = creator_agg.resample('W-MON').mean()
reddit_weekly  = reddit_agg.resample('W-MON').mean()
comment_weekly = comment_agg.resample('W-MON').mean()

start_w = max(creator_weekly.index.min(), reddit_weekly.index.min(),
              comment_weekly.index.min())
end_w   = min(creator_weekly.index.max(), reddit_weekly.index.max(),
              comment_weekly.index.max())
weeks   = pd.date_range(start_w, end_w, freq='W-MON')
creator_weekly = creator_weekly.reindex(weeks)
reddit_weekly  = reddit_weekly.reindex(weeks)
comment_weekly = comment_weekly.reindex(weeks)

c_col = f'{FIELD}_unweighted'
r_col = f'{FIELD}_unweighted'
m_col = f'{COMMENT_FIELD}_unweighted'
c_weekly = creator_weekly[c_col] if c_col in creator_weekly.columns else pd.Series(dtype=float)
r_weekly = reddit_weekly[r_col]  if r_col in reddit_weekly.columns  else pd.Series(dtype=float)
m_weekly = comment_weekly[m_col] if m_col in comment_weekly.columns else pd.Series(dtype=float)

WEEKLY_TARGETS = [('submissions', r_weekly), ('comments', m_weekly)]

weekly_frames, weekly_orders = [], {}
for label, series in WEEKLY_TARGETS:
    df_t, p_t = run_ar_baseline(series, c_weekly, label, TEST_LAGS_WEEKLY,
                                'lag_weeks', 'weeks', min(MAX_AR_ORDER, 4))
    weekly_orders[label] = p_t
    if not df_t.empty:
        weekly_frames.append(df_t)

weekly_df = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
weekly_df.to_csv(TABLES_DIR / '05c_ar_baseline_weekly.csv', index=False)
print(f'Saved: {TABLES_DIR / "05c_ar_baseline_weekly.csv"}')

#%%
# ── Summary: where the creator term is significant, and in which direction ───
print('\n-- AR-baseline summary (creator term) --')
for gran, frame in [('daily', daily_df), ('weekly', weekly_df)]:
    if frame.empty:
        continue
    for tgt, grp in frame.groupby('target'):
        sig = grp[grp['creator_pval'] < 0.05]
        if sig.empty:
            print(f'  {gran:6s} {tgt:12s} 0/{len(grp)} lags p<0.05  '
                  f'(min p={grp["creator_pval"].min():.4f})')
        else:
            print(f'  {gran:6s} {tgt:12s} {len(sig)}/{len(grp)} lags p<0.05  '
                  f'signs={sig["sign"].value_counts().to_dict()}  '
                  f'coefs={[f"{v:+.3f}" for v in sig["creator_coef"]]}')

#%%
# ── Figure: ΔR² at each lag (2 targets × 2 granularities) ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def _bar_panel(ax, df, lag_col, unit_label, title):
    if df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Red = significant and positive (the only cell consistent with elite
    # influence); blue = significant but negative (contrary); grey = ns.
    colors = ['#e74c3c' if (p < 0.05 and c > 0) else
              '#2980b9' if p < 0.05 else '#95a5a6'
              for p, c in zip(df['creator_pval'], df['creator_coef'])]
    bars = ax.bar(df[lag_col].astype(str), df['delta_r2'], color=colors,
                  edgecolor='white', linewidth=0.5)

    # Annotate p-values
    for bar, pval in zip(bars, df['creator_pval']):
        label = '**' if pval < 0.01 else ('*' if pval < 0.05 else f'p={pval:.2f}')
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00005,
                label, ha='center', va='bottom', fontsize=9)

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel(f'Creator lead lag ({unit_label})\n[creator_{"{t-h}"}  predicts  reddit_t]',
                  fontsize=10)
    ax.set_ylabel('ΔR²  (AR+creator − AR alone)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='p<0.05, positive'),
                       Patch(facecolor='#2980b9', label='p<0.05, contrary'),
                       Patch(facecolor='#95a5a6', label='p ≥ 0.05')]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

for col, (tgt_label, tgt_title) in enumerate([('submissions', 'Reddit submissions'),
                                              ('comments', 'Reddit comments')]):
    d_sub = daily_df[daily_df['target'] == tgt_label] if not daily_df.empty else pd.DataFrame()
    w_sub = weekly_df[weekly_df['target'] == tgt_label] if not weekly_df.empty else pd.DataFrame()
    _bar_panel(axes[0, col], d_sub, 'lag_days', 'days',
               f'Daily -> {tgt_title}: AR({daily_orders.get(tgt_label, "?")}) '
               f'vs AR+creator')
    _bar_panel(axes[1, col], w_sub, 'lag_weeks', 'weeks',
               f'Weekly -> {tgt_title}: AR({weekly_orders.get(tgt_label, "?")}) '
               f'vs AR+creator')

fig.suptitle(
    'Does creator sentiment improve community-sentiment predictions beyond an AR baseline?\n'
    'ΔR² = AR+creator − AR alone  (creator overall__score, unweighted; '
    'restricted and unrestricted models fitted on the same rows)',
    fontsize=13,
)
plt.tight_layout()
out_fig = FIGURES_DIR / '05c_ar_baseline.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_fig}')
plt.show()
