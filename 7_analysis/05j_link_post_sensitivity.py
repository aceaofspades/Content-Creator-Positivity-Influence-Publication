#%%
"""
05j_link_post_sensitivity.py
============================
Does the H1 result depend on Reddit submissions that merely repost creator videos?

A submission linking to a panel creator's video is scored partly on that video's
content, which would mechanically couple the two series. This re-runs the gated
creator -> submissions cross-correlation (overall__score, unweighted; daily
deseasonalized +/-28 d and weekly +/-4 wk; same circular-shift gate as 05b) on
three samples:
  all          -- every eligible submission (the headline sample)
  no_panel     -- drops submissions whose URL or body contains a panel video ID
  no_youtube   -- drops every submission with a YouTube link in its URL or body

Output: tables/05j_link_post_sensitivity.csv
"""

import sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.lagcorr import normalised_xcorr
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, ensure_output_dirs, TABLES_DIR, GAME_OF_FOCUS,
)
from analysis.shared.aggregators import aggregate_creator_daily, aggregate_reddit_daily, deseasonalize_dow

N_PERM        = 1000
NULL_PCTILE   = 95
RANDOM_SEED   = 42
MAX_LAG_DAYS  = 28
MAX_LAG_WEEKS = 4
FIELD         = 'overall__score'


def shift_values_only(series: pd.Series, shift: int) -> pd.Series:
    """Circular-shift non-NaN values, keeping NaN positions fixed (05b null)."""
    result = series.copy()
    valid  = series.notna()
    vals   = series[valid].values
    result[valid] = np.roll(vals, shift % max(len(vals), 1))
    return result


def gated_peak(x: pd.Series, y: pd.Series, max_lag: int, rng) -> dict:
    cc = normalised_xcorr(x, y, max_lag)
    peak_lag = int(cc.abs().idxmax())
    obs = abs(float(cc.loc[peak_lag]))
    n = int(x.notna().sum())
    null = []
    for _ in range(N_PERM):
        cc_n = normalised_xcorr(shift_values_only(x, int(rng.integers(1, max(n, 2)))), y, max_lag)
        if not cc_n.empty:
            null.append(float(cc_n.abs().max()))
    null = np.asarray(null)
    thr = float(np.percentile(null, NULL_PCTILE))
    return {'peak_lag': peak_lag, 'peak_r': round(float(cc.loc[peak_lag]), 4),
            'noise_threshold': round(thr, 4),
            'p_value': round((int((null >= obs).sum()) + 1) / (len(null) + 1), 4),
            'significant': obs > thr, 'r_at_0': round(float(cc.get(0, np.nan)), 4)}


#%%
ensure_output_dirs()
rng = np.random.default_rng(RANDOM_SEED)
yt, rd = load_youtube_creators(), load_reddit_submissions()

text = (rd['url'].fillna('').astype(str) + ' ' + rd['url_overridden_by_dest'].fillna('').astype(str)
        + ' ' + rd['selftext'].fillna('').astype(str))
# URL-decode (attribution links carry an encoded v=...) and drop Reddit's
# markdown escapes (youtu.be/abc\_def) so IDs are matched in their plain form.
text = text.map(unquote).str.replace('\\', '', regex=False)
panel_ids = set(yt['video_id'].dropna().astype(str))
# A link, not the bare word "YouTube".
has_youtube = text.str.contains(r'youtu(?:\.be/|be\.com/)', case=False, regex=True)
# Pull every YouTube video ID out of the linking texts and match against the panel.
VIDEO_ID = r'(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})'
linked_ids = text[has_youtube].str.findall(VIDEO_ID)
links_panel = pd.Series(False, index=rd.index)
links_panel[linked_ids.index] = linked_ids.apply(lambda ids: any(v in panel_ids for v in ids))
print(f'{GAME_OF_FOCUS}: {int(has_youtube.sum())} submissions link to YouTube; '
      f'{int(links_panel.sum())} link to a panel video')

samples = {'all': rd, 'no_panel': rd[~links_panel], 'no_youtube': rd[~has_youtube]}
c_agg = aggregate_creator_daily(yt, [FIELD])[f'{FIELD}_unweighted']

rows = []
for name, sub in samples.items():
    r_agg = aggregate_reddit_daily(sub, [FIELD])[f'{FIELD}_unweighted']
    start, end = max(c_agg.index.min(), r_agg.index.min()), min(c_agg.index.max(), r_agg.index.max())
    days = pd.date_range(start, end, freq='D')
    c_d, r_d = deseasonalize_dow(c_agg.reindex(days)), deseasonalize_dow(r_agg.reindex(days))
    c_w = c_agg.reindex(days).resample('W-MON').mean()
    r_w = r_agg.reindex(days).resample('W-MON').mean()
    for gran, x, y, L in [('daily', c_d, r_d, MAX_LAG_DAYS), ('weekly', c_w, r_w, MAX_LAG_WEEKS)]:
        res = gated_peak(x, y, L, rng)
        rows.append({'game': GAME_OF_FOCUS, 'sample': name, 'granularity': gran,
                     'n_submissions': len(sub), 'n_dropped': len(rd) - len(sub), **res})
        print(f'{name:10s} {gran:6s} dropped={len(rd) - len(sub):5d} peak {res["peak_lag"]:+d} '
              f'r={res["peak_r"]:+.3f} p={res["p_value"]:.3f} sig={res["significant"]}')

pd.DataFrame(rows).to_csv(TABLES_DIR / '05j_link_post_sensitivity.csv', index=False)
print(f'Saved {TABLES_DIR / "05j_link_post_sensitivity.csv"}')
