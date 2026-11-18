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
    "dead by daylight",
    "deadbydaylight",
    "dbd",
    "behaviour interactive",
    "behavior interactive",
    "bhvr",

    # === ROLES ===
    "killer",
    "survivor",
    "survivors",
    "the entity",
    "entity",

    # === KILLERS (Licensed & Original) ===
    "the trapper",
    "trapper",
    "the wraith",
    "wraith",
    "the hillbilly",
    "hillbilly",
    "the nurse",
    "nurse",
    "the shape",
    "michael myers",
    "myers",
    "the hag",
    "hag",
    "the doctor",
    "the huntress",
    "huntress",
    "the cannibal",
    "leatherface",
    "the nightmare",
    "freddy",
    "freddy krueger",
    "the pig",
    "pig",
    "amanda",
    "the clown",
    "clown",
    "the spirit",
    "spirit",
    "the legion",
    "legion",
    "the plague",
    "plague",
    "the ghost face",
    "ghostface",
    "ghost face",
    "the demogorgon",
    "demogorgon",
    "the oni",
    "oni",
    "the deathslinger",
    "deathslinger",
    "the executioner",
    "executioner",
    "pyramid head",
    "the blight",
    "blight",
    "the twins",
    "twins",
    "victor",
    "the trickster",
    "trickster",
    "the nemesis",
    "nemesis",
    "the cenobite",
    "cenobite",
    "pinhead",
    "the artist",
    "artist",
    "the onryō",
    "onryo",
    "sadako",
    "the dredge",
    "dredge",
    "the mastermind",
    "mastermind",
    "wesker",
    "the knight",
    "knight",
    "the skull merchant",
    "skull merchant",
    "the singularity",
    "singularity",
    "the xenomorph",
    "xenomorph",
    "the good guy",
    "chucky",
    "the unknown",
    "unknown",
    "the lich",
    "the vecna",
    "vecna",
    "the dark lord",
    "the ghostface",
    "the minotaur",

    # === SURVIVORS (Licensed & Original) ===
    "dwight",
    "meg",
    "claudette",
    "jake",
    "nea",
    "laurie strode",
    "laurie",
    "ace",
    "bill",
    "feng",
    "feng min",
    "david",
    "quentin",
    "tapp",
    "kate",
    "adam",
    "jeff",
    "jane",
    "ash",
    "ash williams",
    "nancy",
    "steve",
    "yui",
    "zarina",
    "cheryl",
    "cheryl mason",
    "felix",
    "elodie",
    "yun-jin",
    "jill",
    "jill valentine",
    "leon",
    "leon kennedy",
    "mikaela",
    "jonah",
    "haddie",
    "ada",
    "ada wong",
    "rebecca",
    "vittorio",
    "thalita",
    "renato",
    "nicolas",
    "ellen ripley",
    "ripley",
    "alan wake",
    "gabriel",
    "sable",
    "aestri",
    "lara croft",
    "lara",

    # === CORE MECHANICS ===
    "generator",
    "gen",
    "gens",
    "gen rush",
    "gen regression",
    "gen speed",
    "gen defense",
    "hook",
    "hooks",
    "unhook",
    "sacrifice",
    "struggle phase",
    "second hook",
    "death hook",
    "basement",
    "basement hook",
    "hatch",
    "escape",
    "escaped",
    "killed",
    "4k",
    "3k",
    "2k",
    "0k",
    "blood points",
    "bp",
    "iridescent",
    "gold pip",
    "silver pip",
    "black pip",
    "pip",
    "depip",
    "safety pip",
    "emblem",
    "survivor emblem",
    "killer emblem",
    "chest",
    "totem",
    "dull totem",
    "hex totem",
    "cleanse",
    "bless",
    "chase",
    "vault",
    "window",
    "pallet",
    "drop pallet",
    "stun",
    "flashlight",
    "sabo",
    "sabotage",
    "body block",
    "bodyblock",
    "dead hard",
    "speed boost",
    "bloodlust",
    "terror radius",
    "heartbeat",
    "lunge",
    "basic attack",
    "m1",
    "m2",
    "killer power",
    "power",
    "obsession",

    # === PERKS (Killer) ===
    "pop goes the weasel",
    "pop",
    "corrupt intervention",
    "corrupt",
    "pain resonance",
    "ruin",
    "hex: ruin",
    "undying",
    "hex: undying",
    "tinkerer",
    "thanatophobia",
    "surveillance",
    "overcharge",
    "call of brine",
    "eruption",
    "deadlock",
    "deadman's switch",
    "nowhere to hide",
    "grim embrace",
    "dragon's grip",
    "sloppy butcher",
    "franklin's demise",
    "franklin",
    "bamboozle",
    "superior anatomy",
    "lightborn",
    "nurse's calling",
    "bbq and chili",
    "bbq",
    "barbecue and chili",
    "iron grasp",
    "agitation",
    "brutal strength",
    "enduring",
    "fire up",
    "whispers",

    # === PERKS (Survivor) ===
    "dead hard",
    "adrenaline",
    "self care",
    "sprint burst",
    "borrowed time",
    "bt",
    "decisive strike",
    "ds",
    "we'll make it",
    "prove thyself",
    "prove",
    "lithe",
    "quick and quiet",
    "urban evasion",
    "hope",
    "off the record",
    "otr",
    "exhaustion perk",
    "exhaustion",
    "circle of healing",
    "head on",
    "flashbang",
    "made for this",
    "mft",
    "iron will",
    "windows of opportunity",
    "windows",
    "kindred",
    "wake up",
    "botany knowledge",
    "unbreakable",
    "mettle of man",
    "object of obsession",
    "ooo",
    "dark sense",
    "bond",
    "spine chill",
    "alert",

    # === ADD-ONS & ITEMS ===
    "toolbox",
    "medkit",
    "map",
    "key",
    "offering",
    "map offering",
    "ebony mori",
    "ivory mori",
    "green mori",
    "mori",
    "rainbow map",
    "add-on",
    "addon",
    "yellow add-on",
    "brown add-on",
    "purple add-on",
    "iridescent add-on",
    "ultra rare",

    # === GAME PROGRESSION ===
    "bloodweb",
    "prestige",
    "p3",
    "prestige 3",
    "p100",
    "grade",
    "ash grade",
    "bronze grade",
    "silver grade",
    "gold grade",
    "iridescent grade",
    "rank reset",
    "rank",
    "mmr",
    "matchmaking",
    "sbmm",
    "skill based matchmaking",
    "red rank",
    "rank 1",
    "survivor rank",
    "killer rank",

    # === MAPS & REALMS ===
    "macmillan estate",
    "coldwind farm",
    "crotus prenn",
    "haddonfield",
    "backwater swamp",
    "lery's memorial",
    "red forest",
    "springwood",
    "gideon meat plant",
    "autohaven wreckers",
    "dead dawg saloon",
    "midwich elementary",
    "racoon city",
    "raccoon city",
    "hawkins",
    "hawkins lab",
    "mount ormond",
    "eyrie of crows",
    "garden of joy",
    "shattered square",
    "toba landing",
    "dvarka deepwood",
    "sactum of wrath",
    "junkyard",
    "swamp",
    "indoor map",
    "outdoor map",
    "map rework",
    "map update",

    # === CHAPTERS & DLC ===
    "chapter",
    "chapter update",
    "dlc",
    "licensed chapter",
    "original chapter",
    "paragraph",
    "tome",
    "rift",
    "battle pass",
    "cosmetic",
    "outfit",
    "skin",
    "charm",
    "cells",
    "auric cells",
    "shards",
    "iridescent shards",
    "store",
    "shrine of secrets",
    "shrine",

    # === COMMUNITY TERMS ===
    "tunneling",
    "tunnel",
    "camping",
    "camp",
    "face camp",
    "face camping",
    "slugging",
    "slug",
    "genrush",
    "gen rush",
    "swf",
    "swf group",
    "solo queue",
    "solo survivor",
    "soloq",
    "bully squad",
    "bm",
    "bad manners",
    "gg",
    "hatch escape",
    "dc",
    "disconnect",
    "afk",
    "toxic",
    "altruism",
    "looping",
    "loop",
    "mindgame",
    "mind game",
    "killer main",
    "survivor main",
    "survivor sided",
    "killer sided",
    "balanced",
    "fun killer",
    "strong killer",
    "weak killer",
    "strong survivor",
    "m1 killer",
    "endgame collapse",
    "egc",
    "dbd wiki",
    "the fog",
    "entity displeased",
    "entity pleased",
    "archive",
    "lore",
    "ptb",
    "public test build",
    "test server",
    "patch notes",
    "patch",
    "hotfix",
    "update",
    "nerf",
    "buff",
    "rework",
    "meta",
    "overpower",
]


# Random sampling rate (0.0 to 1.0) - 1.0 = keep all, 0.2 = keep 20%
SAMPLE_RATE = 0.2

# Output directory for filtered files
OUTPUT_DIR = "../reddit_posts/dbd/filtered"

# File pairs to process — update subreddit names and year ranges as needed
FILE_PAIRS = [
    {
        "submissions": "../reddit_posts/dbd/raw/deadbydaylight_2015-2025_submissions.jsonl",
        "comments": "../reddit_posts/dbd/raw/deadbydaylight_2015-2025_comments.jsonl",
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
