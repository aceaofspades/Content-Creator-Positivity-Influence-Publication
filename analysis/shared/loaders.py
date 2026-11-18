#%%
import os
import warnings
from pathlib import Path
import pandas as pd

from analysis.shared.game_config import get_config as _get_config
from analysis.shared.game_config import get_game_labels as _get_game_labels

# ── Game / filter configuration ───────────────────────────────────────────────
# Change GAME in analysis/shared/game_config.py — everything below is derived.
_cfg = _get_config()
GAME_OF_FOCUS = _cfg['game_label']
# Every meta__game_of_focus spelling accepted as the active game. For most games
# this is just [GAME_OF_FOCUS]; CS needs both the CS:GO- and CS2-era labels since
# the name changes mid-series at the Sept-2023 relaunch.
GAME_OF_FOCUS_LABELS = _get_game_labels(_cfg)
_GAME_SLUG    = _cfg['data_slug']
_PC_SOURCES   = {src for _, src, _, _ in _cfg['player_count_sources']}
# Launch date for games whose data predates release (see game_config.py); None = no cut.
ANALYSIS_START = pd.Timestamp(_cfg['analysis_start']) if _cfg.get('analysis_start') else None

ANALYSIS_ELIGIBLE_ONLY = True
# Sensitivity parameter: when True, drops rows where meta__confidence_score <= CONFIDENCE_THRESHOLD
EXCLUDE_LOW_CONFIDENCE = False
CONFIDENCE_THRESHOLD = 0

# Figures are always written to disk; this only controls whether they are also
# popped up on screen. Default False so batch runs stay quiet. Set the env var
# SHOW_FIGURES=1 (or flip this) when working interactively in VSCode cells.
SHOW_FIGURES = os.environ.get('SHOW_FIGURES', '').strip() in ('1', 'true', 'True', 'yes')


def show_figure() -> None:
    """
    plt.show() replacement honouring SHOW_FIGURES.

    Under a non-interactive backend plt.show() is already a no-op, but it emits
    a UserWarning per call that buries real output in batch logs. Calling this
    instead keeps the logs readable and makes the intent explicit at the call
    site: the savefig above it is what matters, the display is optional.
    """
    if not SHOW_FIGURES:
        return
    import matplotlib.pyplot as plt
    plt.show()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = PROJECT_ROOT / 'analysis' / _GAME_SLUG / 'data'
OUTPUT_DIR  = PROJECT_ROOT / 'analysis' / _GAME_SLUG / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR  = OUTPUT_DIR / 'tables'

# ── Malformed duplicate columns produced by a parsing quirk in 5b_build_masters
# The correctly-named versions already exist; these are safe to drop.
_MALFORMED_COLS = [
    'sentiment_dim__is_the_game_respecting_player time',
    'sentiment_dim__is_the_game_heading_in_a good_direction',
    'sentiment_dim__are_the_developers_making_good_good_decisions',
    'sentiment_dim__is_pve_in_a-good_state',
]

# ── Canonical column lists ────────────────────────────────────────────────────
SENTIMENT_DIM_COLS = [
    'sentiment_dim__is_the_game_fun_right_now',
    'sentiment_dim__is_there_enough_content_to_do',
    'sentiment_dim__is_the_game_respecting_player_time',
    'sentiment_dim__are_the_developers_making_good_decisions',
    'sentiment_dim__is_the_monetization_fair',
    'sentiment_dim__is_the_game_heading_in_a_good_direction',
    'sentiment_dim__is_the_game_worth_playing',
    'sentiment_dim__is_pvp_in_a_good_state',
    'sentiment_dim__is_pve_in_a_good_state',
]

CREATOR_ENGAGEMENT_COLS = [
    'creator__are_they_still_playing',
    'creator__are_they_burned_out',
    'creator__would_they_recommend_the_game',
]

# All topic__ columns — populated dynamically from the parquet at load time
# (use get_topic_cols() rather than hard-coding)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _apply_content_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if ANALYSIS_ELIGIBLE_ONLY:
        mask &= df['analysis_eligible'] == True
    mask &= df['meta__game_of_focus'].isin(GAME_OF_FOCUS_LABELS)
    if EXCLUDE_LOW_CONFIDENCE:
        mask &= df['meta__confidence_score'] > CONFIDENCE_THRESHOLD
    if ANALYSIS_START is not None:
        mask &= pd.to_datetime(df['date']) >= ANALYSIS_START
    drop = [c for c in _MALFORMED_COLS if c in df.columns]
    return df.loc[mask].drop(columns=drop)


# ── Public loaders ────────────────────────────────────────────────────────────
def load_content_master() -> pd.DataFrame:
    """Full content master (Reddit submissions + YouTube creators), filtered."""
    df = pd.read_parquet(DATA_DIR / f'{_GAME_SLUG}_content_master.parquet')
    return _apply_content_filters(df)


def load_youtube_creators() -> pd.DataFrame:
    """YouTube creator rows only, filtered."""
    df = load_content_master()
    return df[df['source_type'] == 'youtube_creator'].copy()


def load_reddit_submissions() -> pd.DataFrame:
    """Reddit submission rows only, filtered."""
    df = load_content_master()
    return df[df['source_type'] == 'reddit_submission'].copy()


def load_comments_master() -> pd.DataFrame:
    """
    Comments master. No game filter applied — game relevance is inherited from
    the submission level (all comments here belong to D2 submissions).
    """
    df = pd.read_parquet(DATA_DIR / f'{_GAME_SLUG}_comments_master.parquet')
    if ANALYSIS_ELIGIBLE_ONLY:
        df = df[df['analysis_eligible'] == True]
    if EXCLUDE_LOW_CONFIDENCE and 'meta__confidence_score' in df.columns:
        df = df[df['meta__confidence_score'] > CONFIDENCE_THRESHOLD]
    if ANALYSIS_START is not None:
        df = df[pd.to_datetime(df['date']) >= ANALYSIS_START]
    return df.copy()


def load_player_counts(source: str = 'total') -> pd.DataFrame:
    """
    Player counts filtered by source.
    source='total'  → all-platform counts (preferred when available)
    source='steam'  → Steam-only counts
    If the requested source is not present for the active game, falls back to
    the first available source with a warning.
    See steam_proxy_results/ for when each is appropriate.
    """
    if source not in _PC_SOURCES:
        fallback = next(iter(sorted(_PC_SOURCES)))
        warnings.warn(
            f"Player count source '{source}' not available for {GAME_OF_FOCUS}; "
            f"using '{fallback}' instead.",
            stacklevel=2,
        )
        source = fallback
    df = pd.read_parquet(DATA_DIR / f'{_GAME_SLUG}_player_counts.parquet')
    df = df[df['source'] == source]
    if ANALYSIS_START is not None:
        df = df[pd.to_datetime(df['date']) >= ANALYSIS_START]
    return df[['date', 'players']].copy()


def available_player_count_sources() -> set:
    """
    Player-count source labels configured for the active game.
    Only Destiny 2 has all-platform ('total') counts; every other game is
    Steam-only, so figure labels should not assert 'all platforms' blindly.
    """
    return set(_PC_SOURCES)


def load_releases() -> pd.DataFrame:
    """Season/expansion release calendar with parsed date columns."""
    df = pd.read_parquet(DATA_DIR / f'{_GAME_SLUG}_releases.parquet').copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date']   = pd.to_datetime(df['end_date'])
    return df


# ── Utility ───────────────────────────────────────────────────────────────────
def get_topic_cols(df: pd.DataFrame) -> list:
    """Return all topic__ column names present in df, excluding other_mean/other_count."""
    return [c for c in df.columns
            if c.startswith('topic__')
            and not c.endswith(('_mean', '_count'))]


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist. Call at the top of each analysis script."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
