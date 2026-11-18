#%%
"""
5b_build_masters.py
===================
Assembles master parquets from LLM-processed sources for the active game.

Active game is controlled by a single variable in analysis/shared/game_config.py.
Run after step 5 completes for all sources.

Inputs (paths from game_config.py):
  channels_parquet/{game}/processed/{youtube_filename}
  reddit_posts/{game}/processed/*_llm.parquet
  other_data/{game}/daily_player_counts*.csv
  other_data/{game}/releases.csv

Outputs -> analysis/{data_slug}/data/:
  {data_slug}_content_master.parquet   (reddit submissions + youtube creators)
  {data_slug}_comments_master.parquet
  {data_slug}_player_counts.parquet    (one row per source per day)
  {data_slug}_releases.parquet

Usage:
  python 5b_build_masters.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shared.parsers import parse_submissions_batch, parse_comments_batch, parse_youtube_batch
from analysis.shared.game_config import get_config as _get_config


# =============================================================================
# PATHS  (derived from analysis/shared/game_config.py — edit GAME there)
# =============================================================================

BASE = Path(__file__).parent
_cfg = _get_config()

_DATA_SLUG            = _cfg['data_slug']
OUTPUT_DIR            = BASE / 'analysis' / _DATA_SLUG / 'data'
YOUTUBE_PROCESSED_DIR = BASE / _cfg['youtube_dir'] / 'processed'
_YOUTUBE_FILENAME     = _cfg['youtube_filename']
_YOUTUBE_FALLBACK_DIR = (
    BASE / _cfg['youtube_fallback_punctuated_dir']
    if 'youtube_fallback_punctuated_dir' in _cfg else None
)

REDDIT_SUBMISSIONS = [
    (BASE / rel, sub) for rel, sub in _cfg['reddit_submissions']
]
REDDIT_COMMENTS = [
    (BASE / rel, sub) for rel, sub in _cfg['reddit_comments']
]
PLAYER_COUNT_SOURCES = [
    (BASE / rel, src, dcol, pcol)
    for rel, src, dcol, pcol in _cfg['player_count_sources']
]
RELEASES_PATH = BASE / _cfg['releases_path']


# =============================================================================
# HELPERS
# =============================================================================

def to_date(series: pd.Series, unit=None) -> pd.Series:
    dt = pd.to_datetime(series, unit=unit, errors='coerce')
    if hasattr(dt.dt, 'tz') and dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()


def add_release_context(df: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    if releases.empty or 'date' not in df.columns:
        return df

    df = df.copy()
    df['current_season'] = None
    df['days_since_release'] = np.nan
    df['days_until_next_release'] = np.nan
    df['release_phase'] = None

    releases_sorted = releases.sort_values('start_date').reset_index(drop=True)
    dates = df['date']
    valid = dates.notna()
    assigned = pd.Series(False, index=df.index)

    for _, rel in releases_sorted.iterrows():
        start = rel['start_date']
        end = rel.get('end_date', pd.NaT)
        mask = valid & ~assigned & (dates >= start)
        if pd.notna(end):
            mask = mask & (dates <= end)
        if mask.any():
            df.loc[mask, 'current_season'] = rel.get('season_name')
            df.loc[mask, 'days_since_release'] = (dates[mask] - start).dt.days
            assigned |= mask

    starts = releases_sorted['start_date'].values
    valid_dates = dates[valid]
    next_idx = np.searchsorted(starts, valid_dates.values, side='right')
    has_next = next_idx < len(starts)
    if has_next.any():
        next_dates = pd.to_datetime(starts[next_idx[has_next]])
        df.loc[valid_dates.index[has_next], 'days_until_next_release'] = (
            next_dates - valid_dates[has_next]
        ).dt.days.values

    d = df['days_since_release']
    df.loc[d < 0,                'release_phase'] = 'pre_release'
    df.loc[(d >= 0) & (d <= 7),  'release_phase'] = 'launch_week'
    df.loc[(d > 7)  & (d <= 30), 'release_phase'] = 'early_season'
    df.loc[(d > 30) & (d <= 60), 'release_phase'] = 'mid_season'
    df.loc[d > 60,               'release_phase'] = 'late_season'

    return df


def clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != 'object':
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        col_types = non_null.apply(type)
        if (col_types == bool).any():
            df[col] = df[col].apply(
                lambda x: str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None
            )
        elif (col_types == dict).any() or (col_types == list).any():
            # Dicts and lists must be serialized — Arrow can't handle mixed object types.
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
        elif (col_types == bytes).any():
            # bytes + NaN float mix — replace non-bytes with None so Arrow sees a
            # nullable bytes column rather than a mixed-type object column.
            df[col] = df[col].apply(lambda x: x if isinstance(x, bytes) else None)
        elif (col_types.isin([int, float])).any() and (col_types == str).any():
            # Mixed str/numeric. Two sub-cases:
            # (a) All non-null values coerce to numeric -> the strings are numeric strings;
            #     collapse to a float column.
            # (b) Only some coerce -> the floats are placeholder artifacts (e.g. 0.0 where
            #     a source wrote 0 instead of null); keep strings, drop the floats.
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() == non_null.shape[0]:
                df[col] = converted
            else:
                # Keep only genuine string values; replace float artifacts with None.
                df[col] = df[col].apply(lambda x: x if isinstance(x, str) else None)
    return df


# =============================================================================
# LOADERS
# =============================================================================

def load_youtube() -> pd.DataFrame:
    combined_path = YOUTUBE_PROCESSED_DIR / _YOUTUBE_FILENAME

    if combined_path.exists():
        print(f'Loading YouTube: processed/{combined_path.name}')
        df = pd.read_parquet(combined_path)
        print(f'  {len(df):,} videos')
    elif _YOUTUBE_FALLBACK_DIR is not None:
        # D2-only fallback: per-creator files from before the combine step existed.
        # Run _rectify_d2_youtube.py once to build the combined file and retire this path.
        paths = sorted(_YOUTUBE_FALLBACK_DIR.glob('*_punctuated_llm.parquet'))
        if not paths:
            raise FileNotFoundError(
                f'YouTube file not found: {combined_path}\n'
                f'Fallback dir also empty: {_YOUTUBE_FALLBACK_DIR}\n'
                f'Run _rectify_d2_youtube.py to build the combined file.'
            )
        print(f'Loading YouTube: {len(paths)} individual creator files (D2 fallback — run _rectify_d2_youtube.py)')
        dfs = []
        for p in paths:
            part = pd.read_parquet(p)
            part['creator'] = p.stem.replace('_punctuated_llm', '')
            dfs.append(part)
        df = pd.concat(dfs, ignore_index=True)
        print(f'  {len(df):,} videos combined')
    else:
        raise FileNotFoundError(f'YouTube file not found: {combined_path}')

    if 'creator' not in df.columns:
        df['creator'] = df['channel_title']

    df['source_type'] = 'youtube_creator'
    df['date'] = to_date(df['published_at'])
    df['analysis_eligible'] = (
        df['llm_error'].isna() if 'llm_error' in df.columns else True
    )
    print(f'  {df["date"].min().date()} -> {df["date"].max().date()}  |  eligible: {df["analysis_eligible"].sum():,}')

    return parse_youtube_batch(df, llm_column='llm_response', legacy_format=False)


def load_submissions() -> pd.DataFrame:
    print('Loading Reddit submissions')
    dfs = []
    for path, subreddit in REDDIT_SUBMISSIONS:
        df = pd.read_parquet(path)
        df['subreddit'] = subreddit
        print(f'  {subreddit}: {len(df):,}')
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).copy()
    combined['source_type'] = 'reddit_submission'
    combined['date'] = to_date(combined['created_utc'], unit='s')
    combined['analysis_eligible'] = (
        combined['llm_error'].isna() if 'llm_error' in combined.columns else True
    )
    print(f'  Total: {len(combined):,}  |  {combined["date"].min().date()} -> {combined["date"].max().date()}  |  eligible: {combined["analysis_eligible"].sum():,}')

    return parse_submissions_batch(combined)


def load_comments() -> pd.DataFrame:
    print('Loading Reddit comments')
    dfs = []
    for path, subreddit in REDDIT_COMMENTS:
        df = pd.read_parquet(path)
        df['subreddit'] = subreddit
        print(f'  {subreddit}: {len(df):,}')
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).copy()
    combined['source_type'] = 'reddit_comment'
    combined['date'] = to_date(combined['created_utc'], unit='s')
    combined['analysis_eligible'] = (
        combined['llm_error'].isna() if 'llm_error' in combined.columns else True
    )
    print(f'  Total: {len(combined):,}  |  {combined["date"].min().date()} -> {combined["date"].max().date()}  |  eligible: {combined["analysis_eligible"].sum():,}')

    return parse_comments_batch(combined)


def load_player_counts() -> pd.DataFrame:
    print('Loading player counts')
    dfs = []
    for path, source, date_col, player_col in PLAYER_COUNT_SOURCES:
        df = pd.read_csv(path)
        df['date'] = to_date(df[date_col])
        df['players'] = pd.to_numeric(df[player_col], errors='coerce')
        df['source'] = source
        df = df[['date', 'players', 'source']].dropna(subset=['date'])
        # Aggregate sub-daily rows to one row per day (peak concurrent players)
        df = (
            df.groupby(['date', 'source'], as_index=False)['players']
            .max()
            .dropna(subset=['players'])
        )
        print(f'  {source}: {len(df):,} days  |  {df["date"].min().date()} -> {df["date"].max().date()}')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values(['source', 'date']).reset_index(drop=True)


def load_releases() -> pd.DataFrame:
    print('Loading releases')
    df = pd.read_csv(RELEASES_PATH)
    df['start_date'] = to_date(df['start_date'])
    if 'end_date' in df.columns:
        df['end_date'] = to_date(df['end_date'])
    df = df.sort_values('start_date').reset_index(drop=True)
    print(f'  {len(df):,} releases  |  {df["start_date"].min().date()} -> {df["start_date"].max().date()}')
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    releases      = load_releases()
    player_counts = load_player_counts()

    youtube     = load_youtube()
    submissions = load_submissions()
    content     = add_release_context(
        pd.concat([submissions, youtube], ignore_index=True),
        releases
    )

    comments = add_release_context(load_comments(), releases)

    masters = {
        f'{_DATA_SLUG}_content_master':  content,
        f'{_DATA_SLUG}_comments_master': comments,
        f'{_DATA_SLUG}_player_counts':   player_counts,
        f'{_DATA_SLUG}_releases':        releases,
    }

    print(f'\nSaving to {OUTPUT_DIR}')
    for name, df in masters.items():
        path = OUTPUT_DIR / f'{name}.parquet'
        clean_for_parquet(df).to_parquet(path, index=False)
        print(f'  {name}: {len(df):,} rows -> {path.name}')

    print('\nDone.')


if __name__ == '__main__':
    main()
