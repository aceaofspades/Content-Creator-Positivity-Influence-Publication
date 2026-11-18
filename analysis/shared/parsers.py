#%%
"""
LLM Response Parsers
====================
Functions to parse and flatten LLM response JSON from Reddit and YouTube sources.

Handles:
- Nested JSON flattening with prefixes
- Missing keys (graceful handling)
- "unclear" / "NA" → None conversion
- Legacy YouTube format translation
- Score validation and mapping
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Optional

SCORE_RANGE = (-2, 2)

LEGACY_SCORE_MAP = {
    -5: -2, -4: -2,
    -3: -1, -2: -1, -1: -1,
    0: 0,
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2,
}

CREATOR_ENGAGEMENT_MAP = {
    'yes': 2, 'mostly': 1, 'sometimes': 0, 'rarely': -1, 'no': -2,
    'conditional': 0,
    'unclear': None, 'NA': None, 'na': None, None: None,
}

NULL_VALUES = {'unclear', 'NA', 'na', 'N/A', 'n/a', None, ''}

# ── Topic vocabulary ─────────────────────────────────────────────────────────
# Single source of truth: category -> topics. TOPIC_CATEGORIES and
# STANDARD_TOPICS are both derived from it, so they can no longer drift apart.
# They previously did: the Narrative category and three Social & Onboarding
# topics were defined in the Destiny 2 prompt but present in neither constant,
# so every narrative score the model produced was discarded.
#
# The vocabulary is the UNION of all five games' prompt-defined topics, not a
# per-game switch. Each game's `modular_prompt_categories.txt` emits its own
# names — Counter-Strike asks about `map_pool` and `case_pricing`, Dota about
# `hero_balance` and `patch_meta`, Dead by Daylight about `killer_balance` and
# `camping_tunneling_slugging` — and anything outside this set is collapsed
# into topic__other_mean/topic__other_count with no column of its own. With a
# Destiny-2-only vocabulary that silently deleted most of the topic layer for
# the other games: standard-topic hits per creator video ran 22.8 for Destiny 2
# against 4.0 (Dota), 0.54 (CS) and 0.28 (DBD). Their topic-level results were
# not nulls, they were an absence of measurement.
#
# Union rather than per-game because parsers.py stays self-contained and
# game-agnostic, and because a mixed-vocabulary corpus still parses: Warframe
# was processed with the Destiny 2 prompt, so its rows carry D2 topic names,
# and a per-game switch would discard them wholesale. A game only ever
# populates its own topics; the rest stay NULL, which is exactly how topic
# fields already behave (conditionally populated — missing != neutral).
#
# No topic name maps to two different categories across the five prompts, so
# the union is unambiguous.
_TOPICS_BY_CATEGORY: dict = {
    'Content & Activities': (
        # Destiny 2
        'raid_content', 'dungeon_content', 'strike_content', 'nightfall_content',
        'seasonal_activity_content', 'story_campaign_content', 'exotic_missions',
        'open_world_patrol', 'content_quantity', 'content_quality', 'endgame_content',
        'solo_content', 'casual_content',
        # Warframe
        'mission_design', 'endgame_missions', 'nightwave_content', 'open_world_content',
        'railjack_content', 'duviri_content', 'boss_content', 'new_game_modes',
        'new_warframe_releases',
        # Counter-Strike
        'map_pool', 'map_design', 'game_modes', 'premier_system', 'workshop_content',
        # Dota 2
        'hero_pool', 'new_hero_releases', 'seasonal_events', 'custom_games',
        # Dead by Daylight
        'chapter_releases', 'licensed_chapters', 'killer_roster',
        # Model-emitted, not prompt-defined; long-standing columns in the D2 masters
        'replayability',
    ),
    'PvP': (
        # Destiny 2
        'pvp_maps', 'pvp_modes', 'pvp_matchmaking', 'pvp_balance', 'pvp_rewards',
        'trials_of_osiris', 'iron_banner', 'competitive_playlist', 'anti_cheat',
        # Counter-Strike
        'competitive_matchmaking', 'premier_matchmaking', 'matchmaking_quality',
        'team_communication',
        # Dota 2
        'ranked_matchmaking', 'mmr_system', 'anti_cheat_smurfing', 'behavior_score',
        'turbo_mode', 'captains_mode',
        # Dead by Daylight
        'grade_system', 'swf_vs_solo_balance', 'dc_penalty_system', 'reporting_system',
    ),
    'Sandbox & Balance': (
        # Destiny 2
        'weapon_balance', 'ability_balance', 'class_balance', 'subclass_design',
        'super_balance', 'exotic_balance', 'pve_sandbox', 'pvp_sandbox',
        'buildcrafting_depth', 'gunplay_feel',
        # Warframe
        'warframe_balance', 'warframe_kit_design', 'riven_system', 'helminth_system',
        'operator_drifter_system',
        # Counter-Strike
        'economy_balance', 'utility_balance', 'movement_mechanics', 'recoil_system',
        'hitbox_registration',
        # Dota 2
        'hero_balance', 'item_balance', 'patch_meta', 'hero_reworks',
        'ability_interactions', 'map_changes', 'gameplay_feel',
        # Dead by Daylight
        'killer_balance', 'survivor_balance', 'perk_balance', 'map_balance',
        'chase_mechanics', 'gen_speed_balance', 'camping_tunneling_slugging',
        'add_on_balance', 'hook_sacrifice_system',
    ),
    'Progression & Economy': (
        # Destiny 2
        'loot_drops', 'rng_drop_rates', 'crafting_system', 'power_level_system',
        'artifact_system', 'mod_system', 'masterwork_system', 'currency_economy',
        'infusion_system', 'grind_length', 'time_investment_respect', 'catch_up_mechanics',
        # Warframe
        'mastery_rank_system', 'forma_system', 'void_relic_system', 'syndicate_system',
        'focus_system', 'trading_system',
        # Counter-Strike
        'rank_system', 'drop_system', 'case_system', 'steam_marketplace', 'cs2_migration',
        # Dota 2
        'mmr_rank_progression', 'seasonal_pass_system',
        # Dead by Daylight
        'bloodpoint_economy', 'prestige_system', 'shrine_of_secrets', 'grade_pip_system',
        'iridescent_shard_economy',
    ),
    'Rewards & Loot': (
        # Destiny 2
        'loot_quality', 'loot_variety', 'cosmetic_rewards', 'playlist_rewards',
        'pinnacle_rewards', 'adept_weapons', 'title_seal_rewards', 'triumph_rewards',
        # Warframe
        'mission_rotation_rewards', 'prime_incarnon_weapons', 'honoria_rewards',
        # Counter-Strike
        'case_drops', 'operation_rewards', 'souvenir_packages', 'skin_quality',
        'skin_variety',
        # Dota 2
        'event_rewards', 'arcana_immortal', 'seasonal_pass_rewards',
        # Dead by Daylight
        'rift_rewards', 'shrine_perk_availability', 'cosmetic_quality', 'cosmetic_variety',
    ),
    'Inventory & UI': (
        # Destiny 2
        'vault_space', 'inventory_management', 'ui_design', 'menu_navigation',
        'loadout_system', 'collections_system',
        # Warframe
        'inventory_capacity',
        # Dota 2
        'ui_hud_design', 'hero_loadout_system',
        # Model-emitted, not prompt-defined; long-standing columns in the D2 masters
        'postmaster',
    ),
    'Monetization': (
        # Destiny 2
        'eververse_pricing', 'season_pass_value', 'expansion_pricing',
        'dungeon_key_pricing', 'cosmetic_availability', 'fomo_limited_time',
        'free_to_play_experience', 'silver_economy',
        # Warframe
        'market_pricing', 'prime_access_pricing', 'platinum_economy',
        # Counter-Strike
        'case_pricing', 'skin_market_economy', 'operation_pricing', 'prime_status',
        # Dota 2
        'seasonal_pass_pricing', 'arcana_pricing', 'steam_marketplace_economy',
        'ti_prize_pool',
        # Dead by Daylight
        'chapter_pricing', 'licensed_cosmetic_pricing', 'rift_pricing',
        'auric_cell_pricing', 'dlc_model',
    ),
    'Technical': (
        # Destiny 2
        'server_stability', 'tick_rate_netcode', 'bug_fixes', 'performance_optimization',
        'load_times', 'error_codes', 'matchmaking_times',
        # Counter-Strike
        'subtick_system',
        # Dota 2
        'client_quality', 'replay_spectator_tools',
        # Dead by Daylight
        'hit_validation', 'dedicated_server_quality',
    ),
    'Developer Communication': (
        # Destiny 2
        'communication_transparency', 'patch_frequency', 'community_feedback_response',
        'content_roadmap', 'dev_updates', 'sandbox_update_frequency', 'live_service_model',
        # Dota 2
        'patch_design_philosophy',
        # Dead by Daylight
        'bug_fix_responsiveness',
    ),
    'Social & Onboarding': (
        # Destiny 2
        'new_player_experience', 'returning_player_experience', 'in_game_lfg',
        'clan_features', 'social_spaces', 'guided_games', 'tutorials_guidance',
        'friend_systems',
        # Counter-Strike
        'community_servers', 'team_finding', 'coaching_system',
        # Dota 2
        'party_matchmaking', 'toxicity_behavior_enforcement',
        # Dead by Daylight
        'toxicity_behavior',
    ),
    'Narrative': (
        # Destiny 2
        'story_writing', 'character_development', 'lore_accessibility', 'cutscene_quality',
        'seasonal_storytelling', 'expansion_storytelling', 'villain_writing',
        # Warframe
        'quest_storytelling',
        # Counter-Strike
        'operation_lore',
        # Dota 2
        'hero_lore', 'world_lore', 'event_storylines', 'dragons_blood',
        # Dead by Daylight
        'character_lore', 'licensed_lore_representation', 'entity_lore', 'tome_storylines',
    ),
}

# topic name -> category, flattened from the definition above.
TOPIC_CATEGORIES: dict = {
    topic: category
    for category, topics in _TOPICS_BY_CATEGORY.items()
    for topic in topics
}

# Any topic outside this set is an "other" topic (LLM-generated, aggregated
# into topic__other_mean / topic__other_count rather than given a column).
STANDARD_TOPICS = frozenset(TOPIC_CATEGORIES)


#%%
# =============================================================================
# CORE PARSING UTILITIES
# =============================================================================

def safe_json_loads(value: Any) -> Optional[dict]:
    """Safely parse JSON string to dict. Returns None if result is not a dict."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def normalize_value(value: Any) -> Any:
    """Normalize a value - convert null-like values to None."""
    if isinstance(value, (list, dict)):
        return value
    if value in NULL_VALUES:
        return None
    if isinstance(value, str) and value.lower() in {'unclear', 'na', 'n/a'}:
        return None
    return value


def validate_score(value: Any) -> Optional[int]:
    """Validate and convert a score to integer within valid range."""
    normalized = normalize_value(value)
    if normalized is None:
        return None
    
    try:
        score = int(normalized)
        min_score, max_score = SCORE_RANGE
        if min_score <= score <= max_score:
            return score
        # Clip to range if out of bounds
        return max(min_score, min(max_score, score))
    except (ValueError, TypeError):
        return None


def flatten_dict(d: dict, parent_key: str = '', sep: str = '__') -> dict:
    """
    Recursively flatten a nested dictionary.
    
    Example:
        {'a': {'b': 1, 'c': 2}} -> {'a__b': 1, 'a__c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, normalize_value(v)))
    
    return dict(items)


#%%
# =============================================================================
# REDDIT SUBMISSION PARSER
# =============================================================================

def parse_submission_llm_response(llm_response: Any) -> dict:
    """
    Parse LLM response from a Reddit submission.
    
    Returns a flat dictionary with prefixed keys:
    - overall__score, overall__summary
    - sentiment_dim__is_the_game_fun_right_now, etc.
    - topic__new_player_experience, etc.
    - creator__are_they_still_playing, etc.
    - outlook__optimism_about_future
    - meta__confidence_score, meta__game_relevance, etc.
    - emotional_tone (no prefix, top-level)
    """
    parsed = safe_json_loads(llm_response)
    if parsed is None:
        return {}
    
    result = {}
    
    # Overall state of game
    if 'overall_state_of_game' in parsed:
        overall = parsed['overall_state_of_game']
        if isinstance(overall, dict):
            result['overall__score'] = validate_score(overall.get('score'))
            result['overall__summary'] = normalize_value(overall.get('summary'))
    
    # Sentiment dimensions
    if 'sentiment_dimensions' in parsed:
        dims = parsed['sentiment_dimensions']
        if isinstance(dims, dict):
            for key, value in dims.items():
                result[f'sentiment_dim__{key}'] = validate_score(value)
    
    # Topic sentiments - flatten known topics to individual columns,
    # group "other" (LLM-generated) topics together
    if 'topic_sentiments' in parsed:
        topics = parsed['topic_sentiments']
        if isinstance(topics, dict):
            # Flatten nested structure (legacy format nests under categories)
            flattened = flatten_dict(topics)
            other_scores = []
            for key, value in flattened.items():
                score = validate_score(value)
                if key in STANDARD_TOPICS:
                    result[f'topic__{key}'] = score
                elif key not in ('unclear', 'NA', 'na'):
                    # "Other" topic - collect for aggregation
                    if score is not None:
                        other_scores.append(score)
            # Aggregate other topics
            result['topic__other_mean'] = sum(other_scores) / len(other_scores) if other_scores else None
            result['topic__other_count'] = len(other_scores)
    
    # Creator engagement (categorical strings → numeric via mapping)
    if 'creator_engagement' in parsed:
        creator = parsed['creator_engagement']
        if isinstance(creator, dict):
            for key, value in creator.items():
                result[f'creator__{key}'] = map_creator_engagement_string(value, key)
    
    # Outlook
    if 'outlook' in parsed:
        outlook = parsed['outlook']
        if isinstance(outlook, dict):
            result['outlook__optimism_about_future'] = validate_score(outlook.get('optimism_about_future'))
    
    # Emotional tone (top-level)
    if 'emotional_tone' in parsed:
        result['emotional_tone'] = normalize_value(parsed['emotional_tone'])
    
    # Meta information
    if 'meta' in parsed:
        meta = parsed['meta']
        if isinstance(meta, dict):
            for key, value in meta.items():
                if key == 'confidence_score':
                    result[f'meta__{key}'] = validate_score(value)
                elif key in ('video_types_applicable', 'doubts_and_caveats'):
                    # Keep lists/long text as-is
                    result[f'meta__{key}'] = value
                else:
                    result[f'meta__{key}'] = normalize_value(value)
    
    return result


#%%
# =============================================================================
# REDDIT COMMENT PARSER
# =============================================================================

def parse_comment_llm_response(llm_response: Any) -> dict:
    """
    Parse LLM response from a Reddit comment.
    
    Expected format:
    {
        "sentiment_score": 0,
        "topics_mentioned": [],
        "emotional_tone": "neutral",
        "player_status": "NA"
    }
    
    Returns flat dictionary (no prefixes needed, already flat).
    """
    parsed = safe_json_loads(llm_response)
    if parsed is None:
        return {}
    
    return {
        'sentiment_score': validate_score(parsed.get('sentiment_score')),
        'topics_mentioned': parsed.get('topics_mentioned', []),  # Keep as list
        'emotional_tone': normalize_value(parsed.get('emotional_tone')),
        'player_status': normalize_value(parsed.get('player_status')),
    }


#%%
# =============================================================================
# LEGACY YOUTUBE FORMAT TRANSLATION
# =============================================================================

def map_legacy_score(score: Any) -> Optional[int]:
    """Map legacy -5 to 5 score to -2 to 2 scale."""
    if score is None or score in NULL_VALUES:
        return None
    
    try:
        score_int = int(score)
        return LEGACY_SCORE_MAP.get(score_int, validate_score(score_int))
    except (ValueError, TypeError):
        return None


def map_creator_engagement_string(value: Any, field_name: str = '') -> Optional[int]:
    """Map creator engagement string values to numeric."""
    if value is None or value in NULL_VALUES:
        return None
    
    value_str = str(value).lower().strip()
    
    # Special handling for burnout (inverted scale)
    if field_name == 'are_they_burned_out':
        burnout_map = {
            'no': 2,
            'slightly': 1,
            'partially': 0,
            'mostly': -1,
            'yes': -2,
        }
        return burnout_map.get(value_str)
    
    return CREATOR_ENGAGEMENT_MAP.get(value_str)


def translate_legacy_youtube_format(model_response: Any) -> dict:
    """
    Translate legacy YouTube model_response format to match Reddit submission format.
    
    Key differences handled:
    - video_type_* fields moved into meta
    - 'is_bungie_making_good_decisions' → 'are_the_developers_making_good_decisions'
    - Deeply nested topic_sentiments flattened
    - Creator engagement strings → numeric
    - Scores mapped from -5/5 to -2/2
    - Missing meta fields added as None
    
    DELETE THIS FUNCTION after rerunning YouTube data with updated prompt.
    """
    parsed = safe_json_loads(model_response)
    if parsed is None:
        return {}
    
    result = {}
    
    # -------------------------------------------------------------------------
    # Overall state of game
    # -------------------------------------------------------------------------
    if 'overall_state_of_game' in parsed:
        overall = parsed['overall_state_of_game']
        if isinstance(overall, dict):
            result['overall__score'] = map_legacy_score(overall.get('score'))
            result['overall__summary'] = normalize_value(overall.get('summary'))
    
    # -------------------------------------------------------------------------
    # Sentiment dimensions (with key renaming)
    # -------------------------------------------------------------------------
    if 'sentiment_dimensions' in parsed:
        dims = parsed['sentiment_dimensions']
        if isinstance(dims, dict):
            for key, value in dims.items():
                # Rename bungie → developers
                new_key = key.replace('is_bungie_making_good_decisions', 
                                       'are_the_developers_making_good_decisions')
                result[f'sentiment_dim__{new_key}'] = map_legacy_score(value)
    
    # -------------------------------------------------------------------------
    # Topic sentiments - flatten known topics, group others
    # -------------------------------------------------------------------------
    if 'topic_sentiments' in parsed:
        topics = parsed['topic_sentiments']
        if isinstance(topics, dict):
            flattened = flatten_dict(topics)
            other_scores = []
            for key, value in flattened.items():
                score = map_legacy_score(value)
                if key in STANDARD_TOPICS:
                    result[f'topic__{key}'] = score
                elif key not in ('unclear', 'NA', 'na'):
                    if score is not None:
                        other_scores.append(score)
            result['topic__other_mean'] = sum(other_scores) / len(other_scores) if other_scores else None
            result['topic__other_count'] = len(other_scores)
    
    # -------------------------------------------------------------------------
    # Creator engagement (strings → numeric)
    # -------------------------------------------------------------------------
    if 'creator_engagement' in parsed:
        creator = parsed['creator_engagement']
        if isinstance(creator, dict):
            for key, value in creator.items():
                result[f'creator__{key}'] = map_creator_engagement_string(value, key)
    
    # -------------------------------------------------------------------------
    # Outlook
    # -------------------------------------------------------------------------
    if 'outlook' in parsed:
        outlook = parsed['outlook']
        if isinstance(outlook, dict):
            result['outlook__optimism_about_future'] = map_legacy_score(
                outlook.get('optimism_about_future')
            )
    
    # -------------------------------------------------------------------------
    # Emotional tone
    # -------------------------------------------------------------------------
    if 'emotional_tone' in parsed:
        result['emotional_tone'] = normalize_value(parsed['emotional_tone'])
    
    # -------------------------------------------------------------------------
    # Meta (constructed from top-level fields in legacy format)
    # -------------------------------------------------------------------------
    # Video type fields (top-level in legacy → nested in meta)
    if 'video_type_primary' in parsed:
        result['meta__video_type_primary'] = normalize_value(parsed['video_type_primary'])
    if 'video_types_applicable' in parsed:
        result['meta__video_types_applicable'] = parsed['video_types_applicable']
    
    # Legacy format doesn't have these meta fields - set to None
    result['meta__confidence_score'] = None  # Not in legacy format
    result['meta__game_relevance'] = None
    result['meta__transcript_quality'] = None
    result['meta__sarcasm_detected'] = None
    result['meta__is_sponsored_by_game'] = None
    
    return result


#%%
# =============================================================================
# BATCH PARSING FUNCTIONS
# =============================================================================

def parse_submissions_batch(df: pd.DataFrame, llm_column: str = 'llm_response') -> pd.DataFrame:
    parsed_rows = df[llm_column].apply(parse_submission_llm_response)
    parsed_df = pd.DataFrame(parsed_rows.tolist(), index=df.index)
    return pd.concat([df, parsed_df], axis=1).copy()


def parse_comments_batch(df: pd.DataFrame, llm_column: str = 'llm_response') -> pd.DataFrame:
    parsed_rows = df[llm_column].apply(parse_comment_llm_response)
    parsed_df = pd.DataFrame(parsed_rows.tolist(), index=df.index)
    return pd.concat([df, parsed_df], axis=1).copy()


def parse_youtube_batch(
    df: pd.DataFrame,
    llm_column: str = 'model_response',
    legacy_format: bool = True
) -> pd.DataFrame:
    if legacy_format:
        parsed_rows = df[llm_column].apply(translate_legacy_youtube_format)
    else:
        parsed_rows = df[llm_column].apply(parse_submission_llm_response)
    parsed_df = pd.DataFrame(parsed_rows.tolist(), index=df.index)
    return pd.concat([df, parsed_df], axis=1).copy()


#%%
# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Test submission parser
    test_submission = '''
    {
        "overall_state_of_game": {"score": 1, "summary": "Game is decent"},
        "sentiment_dimensions": {
            "is_the_game_fun_right_now": 1,
            "is_there_enough_content_to_do": -1
        },
        "topic_sentiments": {
            "new_player_experience": "unclear",
            "in_game_lfg": 1
        },
        "creator_engagement": {
            "are_they_still_playing": "unclear",
            "are_they_burned_out": "unclear"
        },
        "outlook": {"optimism_about_future": 0},
        "emotional_tone": "neutral",
        "meta": {
            "confidence_score": 1,
            "game_relevance": "mostly_relevant"
        }
    }
    '''
    
    print("=== Submission Parser Test ===")
    result = parse_submission_llm_response(test_submission)
    for k, v in sorted(result.items()):
        print(f"  {k}: {v}")
    
    # Test legacy YouTube parser
    test_legacy_youtube = '''
    {
        "video_type_primary": "discussion",
        "overall_state_of_game": {"score": -3, "summary": "Game is struggling"},
        "sentiment_dimensions": {
            "is_the_game_fun_right_now": -2,
            "is_bungie_making_good_decisions": 0
        },
        "topic_sentiments": {
            "content_activities": {"raid_content": "NA", "dungeon_content": 1},
            "pvp": {"pvp_balance": -2}
        },
        "creator_engagement": {
            "are_they_still_playing": "yes",
            "are_they_burned_out": "partially",
            "would_they_recommend_the_game": "conditional"
        },
        "outlook": {"optimism_about_future": 3},
        "emotional_tone": "frustrated"
    }
    '''
    
    print("\n=== Legacy YouTube Parser Test ===")
    result = translate_legacy_youtube_format(test_legacy_youtube)
    for k, v in sorted(result.items()):
        print(f"  {k}: {v}")
    
    # Test comment parser
    test_comment = '''
    {
        "sentiment_score": -1,
        "topics_mentioned": ["pvp", "matchmaking"],
        "emotional_tone": "frustrated",
        "player_status": "active"
    }
    '''
    
    print("\n=== Comment Parser Test ===")
    result = parse_comment_llm_response(test_comment)
    for k, v in sorted(result.items()):
        print(f"  {k}: {v}")
