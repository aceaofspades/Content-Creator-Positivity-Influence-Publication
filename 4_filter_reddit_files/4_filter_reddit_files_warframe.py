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
    "warframe",
    "digital extremes",
    "de ",
    "de's",
    "tenno",

    # === PLAYER ARCHETYPES ===
    "operator",
    "drifter",
    "transference",
    "void child",
    "lotus",
    "the lotus",
    "natah",
    "margulis",
    "teshin",
    "ordis",
    "cephalon",

    # === WARFRAMES (Popular / Frequently Discussed) ===
    "excalibur",
    "mag",
    "volt",
    "rhino",
    "ash",
    "hydroid",
    "nova",
    "mesa",
    "saryn",
    "nezha",
    "mirage",
    "limbo",
    "loki",
    "nidus",
    "inaros",
    "trinity",
    "oberon",
    "equinox",
    "wisp",
    "gauss",
    "grendel",
    "protea",
    "xaku",
    "lavos",
    "sevagoth",
    "caliban",
    "yareli",
    "harrow",
    "octavia",
    "titania",
    "revenant",
    "khora",
    "garuda",
    "baruuk",
    "hildryn",
    "vlad",
    "gyre",
    "kullervo",
    "citrine",
    "dagath",
    "styanax",
    "voruna",
    "cyte-09",
    "jade",
    "atlas",
    "banshee",
    "chroma",
    "ember",
    "frost",
    "hydroid",
    "nekros",
    "vauban",
    "wukong",
    "zephyr",
    "nyx",
    "valkyr",
    "ivara",
    "vectis",
    "gauss",

    # === WEAPONS (Iconic) ===
    "braton",
    "paris",
    "skana",
    "lex",
    "glaive",
    "kogake",
    "tiberon",
    "soma",
    "hek",
    "boltor",
    "paris prime",
    "boar prime",
    "acrid",
    "tonkor",
    "ignis wraith",
    "catchmoon",
    "tombfinger",
    "kitgun",
    "zaw",
    "riven",
    "rivens",
    "riven mod",
    "kuva weapon",
    "tenet weapon",
    "incarnon",
    "incarnon adapter",
    "zariman weapons",

    # === GAME MODES & ACTIVITIES ===
    "sortie",
    "sorties",
    "arbitration",
    "arbitrations",
    "nightmare",
    "eidolon",
    "eidolons",
    "teralyst",
    "gantulyst",
    "hydrolyst",
    "profit-taker",
    "exploiter orb",
    "mother token",
    "isolation vault",
    "isolation vaults",
    "railjack",
    "rj",
    "void storm",
    "sanctuary onslaught",
    "steel path",
    "circuit",
    "duviri",
    "duviri paradox",
    "undercroft",
    "cascadia",
    "zariman",
    "angels of the zariman",
    "circuit",
    "void cascade",
    "netracell",
    "deep archimedean",
    "alchemy",
    "mirror defense",
    "excavation",
    "interception",
    "survival",
    "defense",
    "mobile defense",
    "deception",
    "extermination",
    "spy",
    "rescue",
    "assassination",
    "crossfire",
    "open world",
    "bounty",
    "bounties",

    # === OPEN WORLD HUBS ===
    "cetus",
    "fortuna",
    "cambion drift",
    "deimos",
    "orb vallis",
    "plains of eidolon",
    "poe",
    "vallis",
    "derelict",

    # === SYNDICATES ===
    "syndicate",
    "syndicates",
    "steel meridian",
    "arbiters of hexis",
    "cephalon suda",
    "the perrin sequence",
    "red veil",
    "new loka",
    "ostrons",
    "solaris united",
    "entrati",
    "holdfasts",
    "kahl",
    "narmer",
    "syndicate standing",

    # === PROGRESSION & SYSTEMS ===
    "mastery rank",
    "mr",
    "mastery",
    "forma",
    "catalyst",
    "reactor",
    "orokin",
    "primed mod",
    "prime",
    "primes",
    "relic",
    "relics",
    "void fissure",
    "void fissures",
    "baro",
    "baro ki'teer",
    "void trader",
    "arcane",
    "arcanes",
    "arcane helmet",
    "augment",
    "augment mod",
    "exilus",
    "umbral",
    "umbral forma",
    "galvanized",
    "galvanized mod",
    "modding",
    "mod capacity",
    "polarization",
    "polarity",
    "stance mod",
    "stance",
    "aura",
    "aura mod",
    "corrosive projection",
    "streamline",
    "continuity",
    "intensity",
    "overextended",
    "blind rage",
    "fleeting expertise",
    "transient fortitude",
    "primed continuity",
    "primed flow",
    "hunter munitions",
    "viral",
    "heat",
    "cold",
    "toxin",
    "electricity",
    "corrosive",
    "radiation",
    "magnetic",
    "gas",
    "blast",

    # === NIGHTWAVE ===
    "nightwave",
    "wolf of saturn six",
    "intermission",
    "nora",
    "nora night",
    "nora's choice",
    "wolf cred",

    # === EVENTS ===
    "plague star",
    "orphix venom",
    "operation scarlet spear",
    "operation railjack",
    "crimson spear",
    "operation ascendant challenge",

    # === CURRENCIES & RESOURCES ===
    "platinum",
    "plat",
    "ducats",
    "credits",
    "standing",
    "kuva",
    "endo",
    "forma",
    "orokin cell",
    "neural sensor",
    "control module",
    "neurodes",
    "argon crystal",
    "circuits",
    "alloy plate",
    "ferrite",
    "plastids",
    "polymer bundle",
    "rubedo",
    "salvage",

    # === ENEMIES & FACTIONS ===
    "grineer",
    "corpus",
    "infested",
    "orokin",
    "sentient",
    "murmur",
    "narmer",
    "worm queen",
    "kela de thaym",
    "ambulas",
    "hyena pack",
    "the ropalolyst",
    "jackal",
    "vay hek",
    "corrupted",
    "stalker",
    "zanuka",
    "acolyte",
    "sister of parvos",
    "kuva lich",
    "lich",
    "liches",
    "requiem",
    "parazon",

    # === CONTENT UPDATES (Major) ===
    "the war within",
    "the sacrifice",
    "the new war",
    "the duviri paradox",
    "jade shadows",
    "1999",
    "cross play",
    "cross-play",
    "cross save",
    "cross-save",
    "update",
    "hotfix",
    "prime access",
    "prime vault",

    # === COMMUNITY & META TERMS ===
    "frame",
    "frames",
    "warframes",
    "fashion frame",
    "fashionframe",
    "tennogen",
    "cosmetic",
    "skin",
    "syandana",
    "helmet",
    "ephemera",
    "glyph",
    "voidshell",
    "market",
    "trade chat",
    "warframe.market",
    "trade",
    "trading",
    "plat trade",
    "gg",
    "mr check",
    "endgame",
    "late game",
    "new player",
    "returning player",
    "daily tribute",
    "login reward",
    "founder",
    "excal umbra",
    "umbra",
    "paracesis",
    "chat mod",
    "dog days",
    "conclave",
    "liset",
    "landing craft",
    "orbiter",
    "dojo",
    "clan",
    "research",
    "drydock",

    # === THIRD-PARTY TOOLS ===
    "overframe",
    "warframe builder",
    "semlar",
    "nexus stats",
    "warframe wiki",
    "wf",
]


# Random sampling rate (0.0 to 1.0) - 1.0 = keep all, 0.2 = keep 20%
SAMPLE_RATE = 0.2

# Output directory for filtered files
OUTPUT_DIR = "../reddit_posts/wf/filtered"

# File pairs to process — update subreddit names and year ranges as needed
FILE_PAIRS = [
    {
        "submissions": "../reddit_posts/wf/raw/Warframe_2015-2025_submissions.jsonl",
        "comments": "../reddit_posts/wf/raw/Warframe_2015-2025_comments.jsonl",
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
