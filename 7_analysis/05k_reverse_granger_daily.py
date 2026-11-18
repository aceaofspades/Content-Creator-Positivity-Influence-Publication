#%%
"""
05k_reverse_granger_daily.py
============================
Daily Granger tests in BOTH directions, so the creator -> community block in
script 05 can be read against its mirror. Uses 05's date-correct construction
(lags built on the regular daily index, incomplete rows dropped afterwards) on
the same deseasonalized unweighted series, lag orders 1..14.

Output: tables/05k_granger_daily_both.csv
  target, direction, lag, f_stat, p_value, n_obs, coef_sum
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    ensure_output_dirs, TABLES_DIR, GAME_OF_FOCUS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily, deseasonalize_dow,
)

MAX_LAG = 14
FIELD, COMMENT_FIELD = 'overall__score', 'sentiment_score'


def granger_date_correct(target: pd.Series, source: pd.Series, lag: int):
    """F-test of source lags 1..lag added to target's own lags 1..lag (same rows)."""
    df = pd.DataFrame({'y': target})
    for k in range(1, lag + 1):
        df[f'y_lag{k}'] = target.shift(k)
        df[f'x_lag{k}'] = source.shift(k)
    df = df.dropna()
    y_cols = [f'y_lag{k}' for k in range(1, lag + 1)]
    x_cols = [f'x_lag{k}' for k in range(1, lag + 1)]
    if len(df) < 3 * lag + 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_r = sm.OLS(df['y'], sm.add_constant(df[y_cols], has_constant='add')).fit()
        m_u = sm.OLS(df['y'], sm.add_constant(df[y_cols + x_cols], has_constant='add')).fit()
    f = ((m_r.ssr - m_u.ssr) / lag) / (m_u.ssr / m_u.df_resid)
    return float(f), float(stats.f.sf(f, lag, m_u.df_resid)), len(df), float(m_u.params[x_cols].sum())


#%%
ensure_output_dirs()
c_agg = aggregate_creator_daily(load_youtube_creators(), [FIELD])[f'{FIELD}_unweighted']
targets = {
    'submissions': aggregate_reddit_daily(load_reddit_submissions(), [FIELD])[f'{FIELD}_unweighted'],
    'comments':    aggregate_comments_daily(load_comments_master(), [COMMENT_FIELD])[f'{COMMENT_FIELD}_unweighted'],
}
# Same three-way overlap as script 05's Granger section.
start = max(c_agg.index.min(), *(t.index.min() for t in targets.values()))
end   = min(c_agg.index.max(), *(t.index.max() for t in targets.values()))
days  = pd.date_range(start, end, freq='D')
c = deseasonalize_dow(c_agg.reindex(days))

rows = []
for target, t_agg in targets.items():
    t = deseasonalize_dow(t_agg.reindex(days))
    for direction, y, x in [('creator->community', t, c), ('community->creator', c, t)]:
        for lag in range(1, MAX_LAG + 1):
            res = granger_date_correct(y, x, lag)
            if res is None:
                continue
            f, p, n, s = res
            rows.append({'game': GAME_OF_FOCUS, 'target': target, 'direction': direction, 'lag': lag,
                         'f_stat': round(f, 3), 'p_value': p, 'n_obs': n, 'coef_sum': round(s, 4)})

out = pd.DataFrame(rows)
out.to_csv(TABLES_DIR / '05k_granger_daily_both.csv', index=False)
for (tg, d), g in out.groupby(['target', 'direction']):
    sig = g[g.p_value < 0.05].lag.tolist()
    print(f'{tg:11s} {d:19s} {len(sig)}/{len(g)} sig lags={sig} min p={g.p_value.min():.4g}')
print(f'Saved {TABLES_DIR / "05k_granger_daily_both.csv"}')
