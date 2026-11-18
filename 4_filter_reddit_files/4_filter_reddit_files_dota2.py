#%%
#!/usr/bin/env python3
"""
Reddit JSONL Filter Script (Streaming Version)
Processes large files in chunks without loading everything into memory.
"""

import json
import random
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dictionary terms - submission must contain at least ONE of these to be included
REQUIRED_TERMS = [
    # === GAME & COMPANY ===
    "dota 2",
    "dota2",
    "dota",
    "valve",
    "icefrog",
    "defense of the ancients",

    # === ROLES ===
    "carry",
    "support",
    "hard support",
    "soft support",
    "offlane",
    "midlane",
    "mid lane",
    "safe lane",
    "safelane",
    "pos 1",
    "pos 2",
    "pos 3",
    "pos 4",
    "pos 5",
    "position 1",
    "position 5",
    "core",
    "roamer",
    "roaming",
    "jungler",
    "jungling",

    # === HEROES (All) ===
    "abaddon",
    "alchemist",
    "ancient apparition",
    "anti-mage",
    "anti mage",
    "arc warden",
    "axe",
    "bane",
    "batrider",
    "beastmaster",
    "bloodseeker",
    "bounty hunter",
    "brewmaster",
    "bristleback",
    "broodmother",
    "centaur warrunner",
    "chaos knight",
    "chen",
    "clinkz",
    "clockwerk",
    "crystal maiden",
    "dark seer",
    "dark willow",
    "dawnbreaker",
    "dazzle",
    "death prophet",
    "disruptor",
    "doom",
    "dragon knight",
    "drow ranger",
    "earthshaker",
    "earth spirit",
    "elder titan",
    "ember spirit",
    "enchantress",
    "enigma",
    "faceless void",
    "grimstroke",
    "gyrocopter",
    "hoodwink",
    "huskar",
    "invoker",
    "io",
    "jakiro",
    "juggernaut",
    "keeper of the light",
    "kunkka",
    "legion commander",
    "leshrac",
    "lich",
    "lifestealer",
    "lina",
    "lion",
    "lone druid",
    "luna",
    "lycan",
    "magnus",
    "marci",
    "mars",
    "medusa",
    "meepo",
    "mirana",
    "monkey king",
    "morphling",
    "muerta",
    "naga siren",
    "nature's prophet",
    "necrophos",
    "night stalker",
    "nyx assassin",
    "ogre magi",
    "omniknight",
    "oracle",
    "outworld destroyer",
    "outworld devourer",
    "pangolier",
    "phantom assassin",
    "phantom lancer",
    "phoenix",
    "primal beast",
    "puck",
    "pudge",
    "pugna",
    "queen of pain",
    "razor",
    "riki",
    "ringmaster",
    "rubick",
    "sand king",
    "shadow demon",
    "shadow fiend",
    "shadow shaman",
    "silencer",
    "skywrath mage",
    "slardar",
    "slark",
    "snapfire",
    "sniper",
    "spectre",
    "spirit breaker",
    "storm spirit",
    "sven",
    "techies",
    "templar assassin",
    "terrorblade",
    "tidehunter",
    "timbersaw",
    "tinker",
    "tiny",
    "treant protector",
    "troll warlord",
    "tusk",
    "underlord",
    "undying",
    "ursa",
    "vengeful spirit",
    "venomancer",
    "viper",
    "visage",
    "void spirit",
    "warlock",
    "weaver",
    "windranger",
    "winter wyvern",
    "witch doctor",
    "wraith king",
    "zeus",
    "kez",

    # === KEY ITEMS ===
    "bkb",
    "black king bar",
    "aegis",
    "buyback",
    "buybacks",
    "divine rapier",
    "rapier",
    "hex",
    "eul",
    "aghanim",
    "shard",
    "bloodthorn",
    "daedalus",
    "butterfly",
    "linkens sphere",
    "linken",
    "mjollnir",
    "satanic",
    "heart of tarrasque",
    "manta",
    "diffusal blade",
    "skull basher",
    "abyssal blade",
    "assault cuirass",
    "shiva's guard",
    "refresher",
    "refresher orb",
    "arcane blink",
    "nullifier",
    "gleipnir",
    "kaya",
    "witch blade",
    "parasma",
    "boots of travel",
    "bot",
    "phase boots",
    "treads",
    "power treads",
    "arcane boots",
    "tranquil boots",
    "magic wand",
    "magic stick",
    "bottle",
    "observer ward",
    "sentry ward",
    "smoke",
    "smoke of deceit",
    "town portal scroll",
    "tp",
    "blink dagger",
    "shadow blade",
    "silver edge",
    "pipe",
    "crimson guard",
    "lotus orb",
    "force staff",
    "ghost scepter",
    "glimmer cape",
    "solar crest",
    "vladmir",
    "mekansm",
    "guardian greaves",
    "urn",
    "spirit vessel",
    "veil of discord",
    "octarine core",
    "eternal shroud",
    "hood of defiance",
    "cloak",
    "poor man's shield",
    "ring of basilius",
    "aquila",
    "bracer",
    "null talisman",
    "wraith band",
    "quarterstaff",
    "javelin",
    "broadsword",
    "claymore",
    "crystalys",
    "demon edge",
    "sacred relic",
    "eagle",
    "eaglesong",
    "yasha",
    "sange",
    "kaya and sange",
    "sange and yasha",
    "yasha and kaya",
    "talisman of evasion",
    "ultimate orb",
    "mystic staff",
    "platemail",
    "chainmail",
    "ring of health",
    "void stone",
    "perseverance",
    "point booster",
    "soul booster",

    # === MECHANICS ===
    "roshan",
    "aegis",
    "cheese",
    "mega creeps",
    "barracks",
    "lane creep",
    "neutral creep",
    "camp",
    "stacking",
    "pulling",
    "deny",
    "creep block",
    "creep cut",
    "fog of war",
    "high ground",
    "gank",
    "gank",
    "gank",
    "teamfight",
    "teamfight",
    "push",
    "split push",
    "rat dota",
    "deathball",
    "snowball",
    "farm",
    "farming",
    "cs",
    "last hit",
    "last hits",
    "gold per minute",
    "gpm",
    "experience per minute",
    "xpm",
    "hero damage",
    "net worth",
    "comeback",
    "comeback mechanic",
    "buyback",
    "scan",
    "ward",
    "wards",
    "observer",
    "sentry",
    "deward",
    "dewarding",
    "vision",
    "detection",
    "courier",
    "flying courier",
    "bottle crowing",
    "rune",
    "bounty rune",
    "power rune",
    "water rune",
    "outpost",
    "shrine",

    # === COMPETITIVE ===
    "the international",
    "ti",
    "ti10",
    "ti11",
    "ti12",
    "ti13",
    "dpc",
    "dota pro circuit",
    "major",
    "minor",
    "bali major",
    "lima major",
    "berlin major",
    "riyadh masters",
    "dreamleague",
    "esl one",
    "eppg",
    "open qualifier",
    "regional qualifier",
    "group stage",
    "main event",
    "upper bracket",
    "lower bracket",
    "grand finals",
    "team liquid",
    "og",
    "team secret",
    "evil geniuses",
    "eg",
    "navi",
    "natus vincere",
    "alliance",
    "virtus.pro",
    "vp",
    "psa",
    "team spirit",
    "tundra",
    "gaimin gladiators",
    "entity",
    "betboom",
    "talon",
    "1win",
    "nouns",
    "ana",
    "miracle",
    "topson",
    "puppey",
    "ceb",
    "sumail",
    "n0tail",

    # === GAME MODES ===
    "ranked",
    "unranked",
    "turbo",
    "ability draft",
    "all random",
    "single draft",
    "random draft",
    "all pick",
    "captains mode",
    "captains draft",
    "mmr",
    "matchmaking",
    "medal",
    "herald",
    "guardian",
    "crusader",
    "archon",
    "legend",
    "ancient",
    "divine",
    "immortal",
    "leaderboard",
    "recalibration",
    "behavior score",
    "low priority",
    "report",

    # === PATCHES & UPDATES ===
    "patch",
    "7.",
    "nerf",
    "buff",
    "rework",
    "meta",
    "meta shift",
    "patch notes",
    "update",
    "balance",
    "rebalance",

    # === COMMUNITY TERMS ===
    "dotabuff",
    "opendota",
    "stratz",
    "pick rate",
    "win rate",
    "winrate",
    "pickrate",
    "ban rate",
    "combo",
    "trilane",
    "dual lane",
    "solo lane",
    "lane",
    "laning",
    "laning phase",
    "mid game",
    "late game",
    "early game",
    "comeback",
    "stomp",
    "feed",
    "feeder",
    "tilted",
    "tilt",
    "flame",
    "muted",
    "mute",
    "gg wp",
    "gg",
]


# Random sampling rate (0.0 to 1.0) - 1.0 = keep all, 0.2 = keep 20%
SAMPLE_RATE = 0.2

# Output directory for filtered files
OUTPUT_DIR = "../reddit_posts/dota/filtered"

# File pairs to process — update subreddit names and year ranges as needed
FILE_PAIRS = [
    {
        "submissions": "../reddit_posts/dota/raw/DotA2_2015-2025_submissions.jsonl",
        "comments": "../reddit_posts/dota/raw/DotA2_2015-2025_comments.jsonl",
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def get_submission_text(sub: dict) -> str:
    title = sub.get("title", "") or ""
    body = sub.get("selftext", "") or ""
    return f"{title} {body}".strip()


def get_comment_text(com: dict) -> str:
    return com.get("body", "") or ""


def passes_word_count_filter(word_count: int, score: int, num_comments: int) -> bool:
    """
    Word count rules:
    - 20+ words → always include
    - 10-19 words → include if score ≥ 5 or comments ≥ 3
    - 4-10 words → include only if score ≥ 20 or comments ≥ 10
    - <3 words → never include
    """
    if word_count >= 20:
        return True
    elif 10 <= word_count <= 19:
        return score >= 5 or num_comments >= 3
    elif 4 <= word_count <= 10:
        return score >= 20 or num_comments >= 10
    else:
        return False


def contains_required_term(text: str, terms: list) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def is_top_level_comment(com: dict) -> bool:
    parent_id = com.get("parent_id", "")
    return parent_id.startswith("t3_")


def get_output_path(input_path: str) -> str:
    p = Path(input_path)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{p.stem}_filtered{p.suffix}")


def pct(part, whole):
    return (part / whole * 100) if whole else 0


def stream_jsonl(filepath: str):
    """Generator that yields parsed JSON objects from a JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_lines(filepath: str) -> int:
    """Count lines in file without loading into memory."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# =============================================================================
# MAIN PROCESSING (STREAMING)
# =============================================================================

def process_file_pair(submissions_path: str, comments_path: str):

    print(f"\n{'='*70}")
    print(f"Processing: {Path(submissions_path).name}")
    print(f"{'='*70}")

    print("\nCounting records...")
    total_submissions = count_lines(submissions_path)
    total_comments = count_lines(comments_path)
    print(f"  Total submissions: {total_submissions:,}")
    print(f"  Total comments: {total_comments:,}")

    step1_sub_removed = 0
    step1_com_removed = 0
    step2_sub_removed = 0
    step2_com_removed = 0
    step3_sub_removed = 0
    step3_com_removed = 0

    print("\nPass 1: Filtering submissions...")

    surviving_ids_step1 = set()
    surviving_ids_step2 = set()
    surviving_ids_final = set()

    sub_output_path = get_output_path(submissions_path)

    for sub in stream_jsonl(submissions_path):
        sub_id = sub.get("id", "")
        text = get_submission_text(sub)
        word_count = count_words(text)
        score = sub.get("score", 0) or 0
        num_comments = sub.get("num_comments", 0) or 0

        if not passes_word_count_filter(word_count, score, num_comments):
            step1_sub_removed += 1
            continue

        surviving_ids_step1.add(sub_id)

        if not contains_required_term(text, REQUIRED_TERMS):
            step2_sub_removed += 1
            continue

        surviving_ids_step2.add(sub_id)

    if SAMPLE_RATE < 1.0:
        sample_size = int(len(surviving_ids_step2) * SAMPLE_RATE)
        surviving_ids_final = set(random.sample(list(surviving_ids_step2), sample_size))
        step3_sub_removed = len(surviving_ids_step2) - len(surviving_ids_final)
    else:
        surviving_ids_final = surviving_ids_step2

    print("  Writing filtered submissions...")
    final_sub_count = 0
    with open(sub_output_path, 'w', encoding='utf-8') as out:
        for sub in stream_jsonl(submissions_path):
            if sub.get("id", "") in surviving_ids_final:
                out.write(json.dumps(sub) + '\n')
                final_sub_count += 1

    print("\nPass 2: Filtering comments...")

    valid_parents_step1 = {f"t3_{sid}" for sid in surviving_ids_step1}
    valid_parents_step2 = {f"t3_{sid}" for sid in surviving_ids_step2}
    valid_parents_final = {f"t3_{sid}" for sid in surviving_ids_final}

    com_output_path = get_output_path(comments_path)
    final_com_count = 0

    with open(com_output_path, 'w', encoding='utf-8') as out:
        for com in stream_jsonl(comments_path):
            parent_id = com.get("parent_id", "")

            if not is_top_level_comment(com):
                step1_com_removed += 1
                continue

            if parent_id not in valid_parents_step1:
                step1_com_removed += 1
                continue

            text = get_comment_text(com)
            word_count = count_words(text)
            score = com.get("score", 0) or 0

            if not passes_word_count_filter(word_count, score, 0):
                step1_com_removed += 1
                continue

            if parent_id not in valid_parents_step2:
                step2_com_removed += 1
                continue

            if parent_id not in valid_parents_final:
                step3_com_removed += 1
                continue

            out.write(json.dumps(com) + '\n')
            final_com_count += 1

    after_step1_sub = total_submissions - step1_sub_removed
    after_step2_sub = after_step1_sub - step2_sub_removed
    after_step3_sub = after_step2_sub - step3_sub_removed

    after_step1_com = total_comments - step1_com_removed
    after_step2_com = after_step1_com - step2_com_removed
    after_step3_com = after_step2_com - step3_com_removed

    print(f"\n  [Step 1: Word Count & Top-Level Filter]")
    print(f"    Submissions: {step1_sub_removed:,} removed")
    print(f"      → {pct(step1_sub_removed, total_submissions):.1f}% of total")
    print(f"      → {after_step1_sub:,} remaining")
    print(f"    Comments: {step1_com_removed:,} removed")
    print(f"      → {pct(step1_com_removed, total_comments):.1f}% of total")
    print(f"      → {after_step1_com:,} remaining")

    print(f"\n  [Step 2: Dictionary Filter]")
    print(f"    Submissions: {step2_sub_removed:,} removed")
    print(f"      → {pct(step2_sub_removed, total_submissions):.1f}% of total | {pct(step2_sub_removed, after_step1_sub):.1f}% of step 1 output")
    print(f"      → {after_step2_sub:,} remaining")
    print(f"    Comments: {step2_com_removed:,} removed")
    print(f"      → {pct(step2_com_removed, total_comments):.1f}% of total | {pct(step2_com_removed, after_step1_com):.1f}% of step 1 output")
    print(f"      → {after_step2_com:,} remaining")

    print(f"\n  [Step 3: Random Sampling ({SAMPLE_RATE*100:.0f}%)]")
    print(f"    Submissions: {step3_sub_removed:,} removed")
    print(f"      → {pct(step3_sub_removed, total_submissions):.1f}% of total | {pct(step3_sub_removed, after_step2_sub):.1f}% of step 2 output")
    print(f"      → {after_step3_sub:,} remaining")
    print(f"    Comments: {step3_com_removed:,} removed")
    print(f"      → {pct(step3_com_removed, total_comments):.1f}% of total | {pct(step3_com_removed, after_step2_com):.1f}% of step 2 output")
    print(f"      → {after_step3_com:,} remaining")

    total_sub_removed = step1_sub_removed + step2_sub_removed + step3_sub_removed
    total_com_removed = step1_com_removed + step2_com_removed + step3_com_removed

    print(f"\n  [FINAL SUMMARY]")
    print(f"    Submissions: {total_submissions:,} → {final_sub_count:,}")
    print(f"      → {total_sub_removed:,} removed ({pct(total_sub_removed, total_submissions):.1f}%)")
    print(f"      → {pct(final_sub_count, total_submissions):.1f}% retained")
    print(f"    Comments: {total_comments:,} → {final_com_count:,}")
    print(f"      → {total_com_removed:,} removed ({pct(total_com_removed, total_comments):.1f}%)")
    print(f"      → {pct(final_com_count, total_comments):.1f}% retained")

    print(f"\n  Output files:")
    print(f"    → {sub_output_path}")
    print(f"    → {com_output_path}")


def main():
    print("Reddit JSONL Filter (Streaming)")
    print(f"Sample rate: {SAMPLE_RATE * 100:.0f}%")
    print(f"Required terms: {REQUIRED_TERMS}")

    for pair in FILE_PAIRS:
        process_file_pair(pair["submissions"], pair["comments"])

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
