#%%
# ── Single change point ────────────────────────────────────────────────────────
# Change GAME here; 5b_build_masters.py and every 7_analysis/ script follow.
# Can also be overridden without editing this file by setting the environment
# variable ANALYSIS_GAME (used by the multi-game batch runner).
import os

GAME = os.environ.get('ANALYSIS_GAME', 'destiny2')
# 'destiny2' | 'warframe' | 'csgo' | 'dota2' | 'deadbydaylight'

# ── Per-game configurations ────────────────────────────────────────────────────
# All paths are relative to the project root.
#
# 'game_label'  — canonical display name; also the default meta__game_of_focus filter.
# 'game_labels' — OPTIONAL list of every meta__game_of_focus spelling that denotes
#                 this game. The LLM was not constrained to a fixed vocabulary, so
#                 it emits naming variants. This matters most for CS: the label
#                 flips from 'CS:GO' to 'CS2' at the Sept-2023 relaunch, so an
#                 exact-match filter would sever the time series in half. When
#                 absent, the filter falls back to [game_label] exactly.
# 'analysis_start' — OPTIONAL launch date (YYYY-MM-DD). Content, comments and player
#                 counts dated before it are dropped by loaders.py. Set where the
#                 subreddit or player series predates the game's launch: r/DestinyTheGame
#                 carries Destiny 1-era posts before Destiny 2's 2017-09-06 launch, the
#                 D2 player series has four near-empty pre-launch days, and Dead by
#                 Daylight's Reddit coverage begins before its 2016-06-14 release.
_CONFIGS = {
    'destiny2': {
        # Label matched against meta__game_of_focus in the content master
        'game_label': 'Destiny 2',
        'analysis_start': '2017-09-06',
        # Slug used for analysis/{slug}/data/ and master parquet filenames
        'data_slug': 'destiny2',
        # YouTube: dir containing processed/ and (optionally) punctuated/ subdirs
        'youtube_dir': 'channels_parquet/destiny2',
        # Filename inside processed/ for the combined LLM-processed parquet
        'youtube_filename': 'combined_punctuated_llm.parquet',
        # D2-only: fall back to per-creator files if the combined file is absent
        'youtube_fallback_punctuated_dir': 'channels_parquet/destiny2/punctuated',
        # Reddit processed parquets: list of (relative_path, subreddit_label)
        'reddit_submissions': [
            ('reddit_posts/destiny2/processed/Destiny_2_2017-2024_submissions_filtered_20_llm.parquet',       'Destiny_2'),
            ('reddit_posts/destiny2/processed/destiny2_2017-2024_submissions_filtered_20_llm.parquet',        'destiny2'),
            ('reddit_posts/destiny2/processed/DestinyTheGame_2017-2024_submissions_filtered_20_llm.parquet',  'DestinyTheGame'),
            ('reddit_posts/destiny2/processed/Destiny_2_2025_submissions_filtered_llm.parquet',               'Destiny_2'),
            ('reddit_posts/destiny2/processed/destiny2_2025_submissions_filtered_llm.parquet',                'destiny2'),
            ('reddit_posts/destiny2/processed/DestinyTheGame_2025_submissions_filtered_llm.parquet',          'DestinyTheGame'),
        ],
        'reddit_comments': [
            ('reddit_posts/destiny2/processed/Destiny_2_2017-2024_comments_filtered_20_llm.parquet',       'Destiny_2'),
            ('reddit_posts/destiny2/processed/destiny2_2017-2024_comments_filtered_20_llm.parquet',        'destiny2'),
            ('reddit_posts/destiny2/processed/DestinyTheGame_2017-2024_comments_filtered_20_llm.parquet',  'DestinyTheGame'),
            ('reddit_posts/destiny2/processed/Destiny_2_2025_comments_filtered_llm.parquet',               'Destiny_2'),
            ('reddit_posts/destiny2/processed/destiny2_2025_comments_filtered_llm.parquet',                'destiny2'),
            ('reddit_posts/destiny2/processed/DestinyTheGame_2025_comments_filtered_llm.parquet',          'DestinyTheGame'),
        ],
        # Player count CSVs: (relative_path, source_label, date_col, players_col)
        'player_count_sources': [
            ('other_data/destiny2/daily_player_counts_steam_only.csv', 'steam', 'DateTime', 'Players'),
            ('other_data/destiny2/daily_player_counts.csv',            'total', 'date',     'players'),
        ],
        'releases_path': 'other_data/destiny2/releases.csv',
        # 05d: analyse expansion eras year_number 1..7. Year 8 (Year of Prophecy)
        # was excluded as too recent/incomplete when that analysis was specified.
        # Kept as an explicit cap so the published D2 result is reproducible —
        # the releases file has since gained end dates that make Year 8 *look*
        # complete, so this can no longer be re-derived from the data.
        'era_max_year_number': 7,
        # 16: non-community accounts. NOTE: this list intentionally omits
        # 'DestinyTheGame-ModTeam' (~1.3k posts) because the published D2
        # concentration figures were computed without it. Add it and re-run 16
        # when you next refresh the D2 numbers.
        'excluded_authors': ['[deleted]', '[removed]', 'AutoModerator', 'destiny2-ModTeam'],
    },

    # NOTE: the raw-data folders for Warframe were renamed from the collection-era
    # short slug 'wf' to 'warframe'; only the YouTube *filename* still carries the
    # old prefix. Do not reintroduce 'wf/' directory paths here.
    'warframe': {
        'game_label': 'Warframe',
        'game_labels': ['Warframe', 'WARFRAME', 'warframe'],
        'data_slug': 'warframe',
        'youtube_dir': 'channels_parquet/warframe',
        'youtube_filename': 'wf_combined_punctuated_llm.parquet',
        'reddit_submissions': [
            ('reddit_posts/warframe/processed/Warframe_2015-2025_submissions_filtered_llm.parquet', 'Warframe'),
        ],
        'reddit_comments': [
            ('reddit_posts/warframe/processed/Warframe_2015-2025_comments_filtered_llm.parquet', 'Warframe'),
        ],
        'player_count_sources': [
            ('other_data/warframe/daily_player_counts_steam_only.csv', 'steam', 'DateTime', 'Players'),
        ],
        'releases_path': 'other_data/warframe/releases.csv',
        'excluded_authors': ['[deleted]', '[removed]', 'AutoModerator', 'Warframe-ModTeam'],
    },

    'csgo': {
        'game_label': 'CS2',
        # The Sept-2023 CS:GO -> CS2 relaunch changes the label mid-series; both
        # eras (and the LLM's spelling variants) must be accepted or the panel
        # splits at the relaunch. 'Counter-Strike 1.6' and 'Counter-Strike: Source'
        # are deliberately excluded — those are different games.
        'game_labels': [
            'CS2', 'CS:GO', 'CSGO', 'CS', 'CS:GO/CS2',
            'Counter-Strike', 'Counter-Strike 2',
            'Counter-Strike: Global Offensive', 'Counter-Strike Global Offensive',
        ],
        'data_slug': 'csgo',
        'youtube_dir': 'channels_parquet/csgo',
        'youtube_filename': 'combined_punctuated_llm.parquet',
        'reddit_submissions': [
            ('reddit_posts/csgo/processed/GlobalOffensive_2015-2025_submissions_filtered_llm.parquet', 'GlobalOffensive'),
        ],
        'reddit_comments': [
            ('reddit_posts/csgo/processed/GlobalOffensive_2015-2025_comments_filtered_llm.parquet', 'GlobalOffensive'),
        ],
        'player_count_sources': [
            ('other_data/csgo/daily_player_counts_steam_only.csv', 'steam', 'DateTime', 'Players'),
        ],
        'releases_path': 'other_data/csgo/releases.csv',
        'excluded_authors': ['[deleted]', '[removed]', 'AutoModerator', 'GlobalOffensive-ModTeam'],
    },

    'dota2': {
        'game_label': 'Dota 2',
        'game_labels': ['Dota 2', 'DOTA 2', 'Dota2', 'DotA 2', 'dota 2'],
        'data_slug': 'dota2',
        'youtube_dir': 'channels_parquet/dota2',
        'youtube_filename': 'combined_punctuated_llm.parquet',
        'reddit_submissions': [
            ('reddit_posts/dota2/processed/DotA2_2015-2025_submissions_filtered_llm.parquet', 'DotA2'),
        ],
        'reddit_comments': [
            ('reddit_posts/dota2/processed/DotA2_2015-2025_comments_filtered_llm.parquet', 'DotA2'),
        ],
        'player_count_sources': [
            ('other_data/dota2/daily_player_counts_steam_only.csv', 'steam', 'DateTime', 'Players'),
        ],
        'releases_path': 'other_data/dota2/releases.csv',
        'excluded_authors': ['[deleted]', '[removed]', 'AutoModerator', 'DotA2-ModTeam'],
    },

    'deadbydaylight': {
        'game_label': 'Dead by Daylight',
        'analysis_start': '2016-06-14',
        # 'Dead by Daylight Mobile' is excluded — separate product, separate audience.
        'game_labels': ['Dead by Daylight', 'Dead By Daylight', 'dead by daylight', 'DBD', 'DbD'],
        'data_slug': 'deadbydaylight',
        'youtube_dir': 'channels_parquet/deadbydaylight',
        'youtube_filename': 'combined_punctuated_llm.parquet',
        'reddit_submissions': [
            ('reddit_posts/deadbydaylight/processed/deadbydaylight_2015-2025_submissions_filtered_llm.parquet', 'deadbydaylight'),
        ],
        'reddit_comments': [
            ('reddit_posts/deadbydaylight/processed/deadbydaylight_2015-2025_comments_filtered_llm.parquet', 'deadbydaylight'),
        ],
        'player_count_sources': [
            ('other_data/deadbydaylight/daily_player_counts_steam_only.csv', 'steam', 'DateTime', 'Players'),
        ],
        'releases_path': 'other_data/deadbydaylight/releases.csv',
        'excluded_authors': ['[deleted]', '[removed]', 'AutoModerator', 'deadbydaylight-ModTeam'],
    },
}


def get_config() -> dict:
    """Return the configuration dict for the active GAME."""
    if GAME not in _CONFIGS:
        raise ValueError(f'Unknown game {GAME!r}. Add it to _CONFIGS in game_config.py. '
                         f'Available: {list(_CONFIGS)}')
    return _CONFIGS[GAME]


def get_game_labels(cfg: dict = None) -> list:
    """
    Every meta__game_of_focus spelling that counts as the active game.

    Falls back to [game_label] when a config declares no aliases, so games
    configured before alias support behave exactly as they did before.
    """
    cfg = cfg or get_config()
    return list(cfg.get('game_labels') or [cfg['game_label']])
