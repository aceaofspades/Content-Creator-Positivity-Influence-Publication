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
# Covers both CS:GO era and CS2 (same subreddit community)
REQUIRED_TERMS = [
    # === GAME & COMPANY ===
    "cs2",
    "counter-strike 2",
    "counter strike 2",
    "csgo",
    "cs:go",
    "cs go",
    "counter-strike",
    "counter strike",
    "valve",
    "cs",
    "global offensive",

    # === GAME MODES ===
    "competitive",
    "premier",
    "wingman",
    "deathmatch",
    "dm",
    "casual",
    "arms race",
    "demolition",
    "retake",
    "danger zone",
    "scrimmage",
    "faceit",
    "esea",
    "hltv",
    "matchmaking",
    "ranked",
    "unranked",

    # === RANKS ===
    "silver",
    "gold nova",
    "gn",
    "master guardian",
    "mg",
    "distinguished master guardian",
    "dmg",
    "legendary eagle",
    "le",
    "legendary eagle master",
    "lem",
    "supreme master first class",
    "smfc",
    "global elite",
    "ge",
    "rating",
    "elo",
    "premier rating",
    "faceit level",
    "faceit elo",
    "rank up",
    "derank",

    # === MAPS ===
    "dust2",
    "dust 2",
    "mirage",
    "inferno",
    "nuke",
    "overpass",
    "ancient",
    "anubis",
    "vertigo",
    "cache",
    "cobblestone",
    "train",
    "agency",
    "office",
    "italy",
    "tuscan",
    "thera",
    "edin",
    "mills",
    "map pool",
    "active duty",
    "reserve pool",
    "map rotation",
    "ct side",
    "t side",

    # === WEAPONS ===
    "ak-47",
    "ak47",
    "ak",
    "m4a4",
    "m4a1-s",
    "m4a1",
    "m4",
    "awp",
    "deagle",
    "desert eagle",
    "usp-s",
    "usp",
    "glock",
    "p250",
    "five-seven",
    "tec-9",
    "cz75",
    "r8 revolver",
    "r8",
    "famas",
    "galil",
    "galil ar",
    "sg 553",
    "aug",
    "scout",
    "ssg 08",
    "g3sg1",
    "scar-20",
    "mac-10",
    "mp9",
    "mp5",
    "mp7",
    "pp-bizon",
    "ump-45",
    "p90",
    "negev",
    "m249",
    "nova",
    "xm1014",
    "sawed-off",
    "mag-7",
    "knife",
    "bayonet",
    "karambit",
    "butterfly knife",
    "m9 bayonet",
    "huntsman",
    "flip knife",
    "gut knife",
    "stiletto",
    "navaja",
    "shadow daggers",
    "skeleton knife",
    "talon knife",
    "ursus knife",
    "falchion knife",
    "bowie knife",
    "classic knife",
    "paracord knife",
    "survival knife",
    "nomad knife",

    # === GRENADES & UTILITIES ===
    "flashbang",
    "flash",
    "smoke grenade",
    "smoke",
    "molotov",
    "incendiary",
    "he grenade",
    "fragmentation",
    "decoy",
    "lineup",
    "pop flash",
    "one-way smoke",

    # === GAME MECHANICS ===
    "spray pattern",
    "spray control",
    "recoil",
    "tap fire",
    "burst fire",
    "aim punch",
    "counter strafing",
    "counter strafe",
    "bunny hop",
    "bhopping",
    "bhop",
    "peek",
    "wide peek",
    "shoulder peek",
    "jiggle peek",
    "entry frag",
    "trade",
    "trade kill",
    "clutch",
    "eco",
    "force buy",
    "full buy",
    "pistol round",
    "pistol",
    "economy",
    "buy phase",
    "freeze time",
    "bomb plant",
    "defuse",
    "ct",
    "terrorist",
    "t",
    "site",
    "mid control",
    "retake",
    "rotate",
    "rotation",
    "boost",
    "crouch boost",
    "pixel boost",

    # === SKINS & ECONOMY ===
    "skin",
    "skins",
    "float",
    "wear",
    "factory new",
    "fn",
    "minimal wear",
    "mw",
    "field-tested",
    "ft",
    "well-worn",
    "ww",
    "battle-scarred",
    "bs",
    "pattern",
    "fade",
    "doppler",
    "crimson web",
    "marble fade",
    "tiger tooth",
    "case hardened",
    "blue gem",
    "sticker",
    "souvenir",
    "collection",
    "capsule",
    "case",
    "operation case",
    "clutch case",
    "snakebite case",
    "fracture case",
    "recoil case",
    "revolution case",
    "kilowatt case",
    "gallery case",
    "trade up",
    "trade up contract",
    "stattrack",
    "souvenir package",
    "steam market",
    "buff",
    "csmoney",
    "skinport",
    "cs.money",
    "trading",
    "gambling",
    "case opening",

    # === COMPETITIVE SCENE ===
    "major",
    "rio major",
    "paris major",
    "copenhagen major",
    "shanghai major",
    "blast",
    "esl pro league",
    "iem",
    "intel extreme masters",
    "pgl",
    "starladder",
    "pro league",
    "hltv ranking",
    "top 30",
    "s1mple",
    "niko",
    "zywoo",
    "device",
    "fallen",
    "coldzera",
    "kennyS",
    "get_right",
    "f0rest",
    "karrigan",
    "broky",
    "apeks",
    "navi",
    "team vitality",
    "faze clan",
    "faze",
    "team spirit",
    "g2",
    "astralis",
    "fnatic",
    "heroic",
    "nip",
    "ninjas in pyjamas",
    "virtus.pro",
    "ence",
    "cloud9",
    "complexity",
    "liquid",
    "team liquid",
    "mouz",
    "mousesports",
    "big",
    "big clan",
    "outsiders",
    "9 pandas",
    "perfect world",
    "pro100",

    # === ANTI-CHEAT & INTEGRITY ===
    "vac",
    "vac ban",
    "overwatch",
    "cheater",
    "cheating",
    "wallhack",
    "aimbot",
    "griefing",
    "griefer",
    "prime",
    "trust factor",
    "trust score",
    "anticheat",
    "anti-cheat",
    "vacnet",
    "prime matchmaking",

    # === CS2 SPECIFIC (post-launch) ===
    "sub-tick",
    "subtick",
    "tick rate",
    "tickrate",
    "128 tick",
    "64 tick",
    "cs2 update",
    "limited test",
    "beta",
    "source 2",
    "volumetric smoke",
    "responsive smokes",
    "cs2 beta",
    "premier mode",
    "steam workshop",
    "workshop map",
    "cs2 rating",

    # === COMMUNITY TERMS ===
    "hacker",
    "smurfs",
    "smurf",
    "smurfing",
    "boosting",
    "rank boost",
    "mm",
    "third party",
    "third-party",
    "pugs",
    "pug",
    "5v5",
    "clutch",
    "ace",
    "four-k",
    "4k",
    "knife kill",
    "knife round",
    "3k",
    "mvp",
    "flash assist",
    "pistol ace",
    "gg",
    "ez",
    "rekt",
    "shroud",
    "s1mple",
    "flusha",
    "olofmeister",
    "olof",
]


# Random sampling rate (0.0 to 1.0) - 1.0 = keep all, 0.2 = keep 20%
SAMPLE_RATE = 0.2

# Output directory for filtered files
OUTPUT_DIR = "../reddit_posts/csgo/filtered"

# File pairs to process — update subreddit names and year ranges as needed
# Primary subreddit is r/GlobalOffensive (covers both CS:GO and CS2 eras)
FILE_PAIRS = [
    {
        "submissions": "../reddit_posts/csgo/raw/GlobalOffensive_2015-2025_submissions.jsonl",
        "comments": "../reddit_posts/csgo/raw/GlobalOffensive_2015-2025_comments.jsonl",
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
