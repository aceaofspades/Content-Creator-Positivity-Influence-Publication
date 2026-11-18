#%%
"""
00b_player_count_audit.py
=========================
Data-quality audit of the player-count series every analysis uses
(load_player_counts(source='total'), which is Steam-only except for Destiny 2).

Reports, per game: source label, date range, days present vs expected, missing
days and the longest gap, duplicate dates, zero or near-zero days, days dropped
by the game's analysis_start (launch) cut, and abrupt day-over-day jumps
(ratio > JUMP_RATIO or < 1/JUMP_RATIO) that could mark a level shift or a
collection break. Destiny 2 also reports the Steam series' first date, which is
its Steam launch (2019-10-01, together with free-to-play and cross-save).

Output: tables/00b_player_count_audit.csv, tables/00b_player_count_jumps.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared import loaders as L

JUMP_RATIO     = 3.0
NEAR_ZERO_FRAC = 0.01   # a day below 1% of the series median counts as near-zero

#%%
L.ensure_output_dirs()
raw = pd.read_parquet(L.DATA_DIR / f'{L._GAME_SLUG}_player_counts.parquet')
source = 'total' if 'total' in set(raw['source']) else sorted(set(raw['source']))[0]
full = raw[raw['source'] == source].copy()
full['date'] = pd.to_datetime(full['date'])
used = L.load_player_counts(source='total')
used['date'] = pd.to_datetime(used['date'])

s = used.set_index('date')['players'].sort_index()
expected = pd.date_range(s.index.min(), s.index.max(), freq='D')
missing = expected.difference(s.index)
gaps = pd.Series(missing).diff().ne(pd.Timedelta(days=1)).cumsum() if len(missing) else pd.Series(dtype=int)
longest_gap = int(gaps.value_counts().max()) if len(missing) else 0
med = float(s.median())
ratio = s / s.shift(1)
jumps = s[(ratio > JUMP_RATIO) | (ratio < 1 / JUMP_RATIO)]

row = {
    'game':               L.GAME_OF_FOCUS,
    'source_used':        source,
    'first_date':         s.index.min().date(),
    'last_date':          s.index.max().date(),
    'days_present':       len(s),
    'days_expected':      len(expected),
    'missing_days':       len(missing),
    'longest_gap_days':   longest_gap,
    'duplicate_dates':    int(used['date'].duplicated().sum()),
    'near_zero_days':     int((s < NEAR_ZERO_FRAC * med).sum()),
    'median_players':     int(med),
    'days_cut_by_launch': int((full['date'] < L.ANALYSIS_START).sum()) if L.ANALYSIS_START is not None else 0,
    'abrupt_jumps':       len(jumps),
}
if 'steam' in set(raw['source']) and source != 'steam':
    row['steam_series_first_date'] = pd.to_datetime(raw.loc[raw['source'] == 'steam', 'date']).min().date()

print(pd.Series(row).to_string())
pd.DataFrame([row]).to_csv(L.TABLES_DIR / '00b_player_count_audit.csv', index=False)
pd.DataFrame({'date': jumps.index.date, 'players': jumps.values,
              'ratio_to_previous_day': ratio[jumps.index].round(3).values}).to_csv(
    L.TABLES_DIR / '00b_player_count_jumps.csv', index=False)
