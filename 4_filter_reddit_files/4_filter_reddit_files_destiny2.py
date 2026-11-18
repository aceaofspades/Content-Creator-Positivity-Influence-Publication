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
# Modify this list as needed
REQUIRED_TERMS = [
    # === GAME & COMPANY ===
    "destiny",
    "destiny 2",
    "destiny2",
    "bungie",
    "d2",  # Common shorthand
    
    # === CLASSES ===
    "guardian",
    "guardians",
    "titan",
    "warlock",
    "hunter",
    "lightbearer",
    
    # === SUBCLASSES & ELEMENTS ===
    "stasis",
    "strand",
    "nightstalker",
    "gunslinger",
    "arcstrider",
    "revenant",
    "threadrunner",
    "sentinel",
    "striker",
    "sunbreaker",
    "behemoth",
    "berserker",
    "voidwalker",
    "stormcaller",
    "dawnblade",
    "shadebinder",
    "broodweaver",
    "prismatic",
    
    # === PVP MODES & TERMS ===
    "crucible",
    "trials of osiris",
    "iron banner",
    "iron banana",
    "shaxx",
    "lord shaxx",
    "flawless",
    "competitive",
    "comp",
    "glory rank",
    "valor rank",
    "pvp"
    
    # === PVE ACTIVITIES ===
    "nightfall",
    "grandmaster",
    "strike playlist",
    "vanguard ops",
    "lost sector",
    "legend lost sector",
    "gambit",
    "drifter",
    "reckoning",
    "pve"
    
    # === RAIDS (Current & Legacy) ===
    "vault of glass",
    "vog",
    "king's fall",
    "kf",
    "crota's end",
    "ce",
    "deep stone crypt",
    "dsc",
    "garden of salvation",
    "gos",
    "last wish",
    "lw",
    "vow of the disciple",
    "votd",
    "vow",
    "root of nightmares",
    "ron",
    "salvation's edge",
    "se",
    "leviathan",
    "crown of sorrow",
    "cos",
    "cs",
    "scourge of the past",
    "sotp",
    "scourge",
    
    # === DUNGEONS ===
    "shattered throne",
    "pit of heresy",
    "pit",
    "prophecy",
    "grasp of avarice",
    "grasp",
    "duality",
    "spire of the watcher",
    "spire",
    "ghosts of the deep",
    "ghosts",
    "warlord's ruin",
    "warlord",
    "vesper's host",
    "vesper",
    "dungeon key",
    
    # === EXPANSIONS ===
    "forsaken",
    "shadowkeep",
    "beyond light",
    "witch queen",
    "lightfall",
    "the final shape",
    "neomuna",
    "pale heart",
    "dreaming city",
    "europa",
    "throne world",
    
    # === KEY CHARACTERS ===
    "cayde",
    "cayde-6",
    "zavala",
    "ikora",
    "osiris",
    "saint-14",
    "crow",
    "mara sov",
    "eris morn",
    "savathun",
    "oryx",
    "xivu arath",
    "calus",
    "rhulk",
    "nezarec",
    "the witness",
    "eramis",
    
    # === ENEMY FACTIONS ===
    "fallen",
    "eliksni",
    "hive",
    "cabal",
    "taken",
    "scorn",
    "tormentor",
    "lucent hive",
    "pyramid ship",
    
    # === ICONIC EXOTIC WEAPONS ===
    "gjallarhorn",
    "ghorn",
    "gjally",
    "thorn",
    "last word",
    "ace of spades",
    "hawkmoon",
    "dead man's tale",
    "le monarque",
    "divinity",
    "izanagi",
    "whisper of the worm",
    "outbreak perfected",
    "touch of malice",
    "anarchy",
    "witherhoard",
    "xenophage",
    "lament",
    "vex mythoclast",
    "tarrabah",
    "eyes of tomorrow",
    "cloudstrike",
    "still hunt",
    "conditional finality",
    "heartshadow",
    "collective obligation",
    "quicksilver storm",
    "deterministic chaos",
    "winterbite",
    "final warning",
    "wish-ender",
    "malfeasance",
    "bad juju",
    "outbreak prime",
    "sleeper simulant",
    "thunderlord",
    "one thousand voices",
    "1k voices",
    
    # === ICONIC LEGENDARY WEAPONS ===
    "fatebringer",
    "midnight coup",
    "spare rations",
    "austringer",
    "luna's howl",
    "not forgotten",
    "recluse",
    "mountaintop",
    "beloved",
    "felwinter's lie",
    "palindrome",
    "eyasluna",
    "calus mini-tool",
    "ikelos",
    "falling guillotine",
    "funnelweb",
    "forbearance",
    "stormchaser",
    "cataclysmic",
    "doom of chelchis",
    "nation of beasts",
    "zaouli's bane",
    "smite of merain",
    "succession",
    "heritage",
    "commemoration",
    "posterity",
    "reconstruction",
    
    # === EXOTIC ARMOR (Popular) ===
    "one-eyed mask",
    "oem",
    "dunemarchers",
    "synthoceps",
    "heart of inmost light",
    "hoil",
    "cuirass of the falling star",
    "loreley splendor",
    "wormhusk",
    "stompees",
    "st0mp-ee5",
    "omnioculus",
    "gyrfalcon",
    "orpheus rig",
    "star-eater scales",
    "shards of galanor",
    "transversive steps",
    "ophidian aspect",
    "contraverse hold",
    "necrotic grip",
    "osmiomancy",
    "sunbracers",
    "phoenix protocol",
    "lunafaction",
    "starfire protocol",
    "celestial nighthawk",
    "falling star",
    "crown of tempests",
    "nezarec's sin",
    "eye of another world",
    "mask of bakris",
    "assassin's cowl",
    "renewal grasps",
    "caliban's hand",
    "nothing manacles",
    "verity's brow",
    "briarbinds",
    "abeyant leap",
    
    # === WEAPON TYPES (Destiny-specific phrasing) ===
    "hand cannon",
    "pulse",
    "scout",
    "auto",
    "trace",
    "linear",
    "fusion",
    "rocket launcher",
    "grenade launcher",
    "machine gun",
    "glaive",
    "special ammo",
    "heavy ammo",
    "primary ammo",
    "special",
    "heavy",
    "primary",
    "kinetic",
    "energy",
    "power",
    
    # === PERKS (Popular/Meta) ===
    "kill clip",
    "rampage",
    "desperado",
    "firefly",
    "explosive payload",
    "vorpal",
    "firing line",
    "triple tap",
    "fourth time's the charm",
    "fttc",
    "reconstruction",
    "overflow",
    "subsistence",
    "demolitionist",
    "incandescent",
    "voltshot",
    "repulsor brace",
    "destabilizing rounds",
    "hatchling",
    "one for all",
    "ofa",
    "frenzy",
    "bait and switch",
    "bns",
    "envious assassin",
    "cascade point",
    "precision instrument",
    
    # === GAME SYSTEMS & MECHANICS ===
    "light level",
    "power level",
    "power",
    "pinnacle",
    "pinnacles",
    "powerful gear",
    "artifact",
    "seasonal artifact",
    "champion",
    "barrier champion",
    "overload champion",
    "unstoppable champion",
    "anti barrier",
    "anti overload",
    "antibarrier",
    "antioverload",
    "masterwork",
    "adept weapon",
    "adept mod",
    "adepts",
    "crafting",
    "weapon crafting",
    "enhanced perk",
    "red border",
    "deepsight",
    "resonance",
    "infusion",
    "ascendant shard",
    "golf ball",
    "enhancement prism",
    "enhancement core",
    "exotic cipher",
    "umbral engram",
    "prime engram",
    "exotic engram",
    "focusing",
    "decoding",
    "postmaster",
    "loadout",
    "buildcrafting",
    "warmind cell",
    "charged with light",
    "elemental well",
    "armor charge",
    
    # === SUPERS & ABILITIES ===
    "golden gun",
    "blade barrage",
    "spectral blades",
    "shadowshot",
    "tether",
    "arc staff",
    "gathering storm",
    "silence and squall",
    "thundercrash",
    "fist of havoc",
    "sentinel shield",
    "ward of dawn",
    "bubble",
    "burning maul",
    "hammer of sol",
    "glacial quake",
    "bladefury",
    "nova bomb",
    "nova warp",
    "stormtrance",
    "chaos reach",
    "daybreak",
    "well of radiance",
    "winter's wrath",
    "needlestorm",
    "song of flame",
    "twilight arsenal",
    "storm's edge",
    
    # === CURRENCIES & MATERIALS ===
    "glimmer",
    "legendary shard",
    "bright dust",
    "silver",
    "spoils of conquest",
    "strange coin",
    "upgrade module",
    
    # === VENDORS & NPCS ===
    "xur",
    "xûr",
    "banshee-44",
    "banshee",
    "ada-1",
    "lord saladin",
    "rahool",
    "variks",
    "devrim",
    "failsafe",
    "nimbus",
    "rohan",
    "quinn",
    "micah-10",
    "micah",
    
    # === LOCATIONS ===
    "tower",
    "helm",
    "h.e.l.m.",
    "cosmodrome",
    "nessus",
    "edz",
    "european dead zone",
    "tangled shore",
    "throne world",
    "enclave",
    "pouka pond",
    "strands",
    "savathun's throne world",
    "ascendant plane",
    "ascendant realm",
    
    # === COMMUNITY & META TERMS ===
    "twab",
    "this week at bungie",
    "bungie plz",
    "pls",
    "dcv",
    "destiny content vault",
    "content vault",
    "sunsetting",
    "sunset",
    "god roll",
    "roll",
    "lfg",
    "fireteam",
    "fireteam finder",
    "solo player",
    "new light",
    "eververse",
    "transmog",
    "ornament",
    "shader",
    "finisher",
    "fomo",
    "bounty",
    "bounties",
    "triumph",
    "seal",
    "title",
    "gilded",
    "conqueror",
    "flawless",
    "rivensbane",
    "blacksmith",
    "shadow",
    "enlightened",
    "harbinger",
    "cursebreaker",
    "dredgen",
    "unbroken",
    
    # === THIRD-PARTY TOOLS ===
    "dim",
    "destiny item manager",
    "ishtar",
    "ishtar commander",
    "braytech",
    "light.gg",
    "d2armorpicker",
    "destinytracker",
    "raid.report",
    "raid report",
    "dungeon.report",
    "dungeon report",
    "trials.report",
    "trials report",
    
    # === SPECIFIC MISSIONS & QUESTS ===
    "whisper",
    "zero hour",
    "harbinger",
    "presage",
    "vox obscura",
    "operation seraph",
    "avalon",
    "starcrossed",
    "dual destiny",
    "excision",
    "exotic mission",
    "secret mission",
    
    # === SEASONS ===
    "season pass",
    "season of",
    "episode",
    "wish",
    "witch",
    "deep",
    "defiance",
    "seraph",
    "plunder",
    "haunted",
    "risen",
    "lost",
    "splicer",
    "chosen",
    "hunt",
    "arrivals",
    "worthy",
    "dawn",
    "undying",
    "echoes",
    "revenant",
    "heresy",
    
    # === ENDGAME TERMS ===
    "master",
    "legend",
    "contest mode",
    "day one",
    "world first",
    "raid race",
    "gm nightfall",
    "adept",
    "timelost",
    "harrowed",
    "deepsight",
    
    # === BALANCE & SANDBOX ===
    "sandbox",
    "sandbox update",
    "tuning",
    "nerf",
    "buff",
    "ability cooldown",
    "super regen",
    "ability spam",
    "ability meta",
    "special",
    "heavy",
    
    # === MISC DESTINY-SPECIFIC ===
    "ghost",
    "shell",
    "sparrow",
    "jumpship",
    "kiosk",
    "monument to lost lights",
    "collections",
    "vault space",
    "vault",
    "engram",
    "exotic quest",
    "catalyst",
    "exotic catalyst",
    "class item",
    "artifact mod",
    "champion mod",
    "origin trait",
    "intrinsic",
    "archetype",
    "frame",
    "aggressive",
    "adaptive",
    "precision",
    "rapid-fire",
    "lightweight",
    "high-impact",
]


# Random sampling rate (0.0 to 1.0) - 1.0 = keep all, 0.2 = keep 20%
SAMPLE_RATE = 0.2

# Output directory for filtered files — change per game, e.g. "../reddit_posts/wf/filtered"
OUTPUT_DIR = "../reddit_posts/destiny2/filtered"

# File pairs to process — change paths per game / year range
FILE_PAIRS = [
    {
        "submissions": "../reddit_posts/destiny2/raw/Destiny_2_2025_submissions.jsonl",
        "comments": "../reddit_posts/destiny2/raw/Destiny_2_2025_comments.jsonl",
    },
    {
        "submissions": "../reddit_posts/destiny2/raw/destiny2_2025_submissions.jsonl",
        "comments": "../reddit_posts/destiny2/raw/destiny2_2025_comments.jsonl",
    },
    {
        "submissions": "../reddit_posts/destiny2/raw/DestinyTheGame_2025_submissions.jsonl",
        "comments": "../reddit_posts/destiny2/raw/DestinyTheGame_2025_comments.jsonl",
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
    
    # --- Count totals first ---
    print("\nCounting records...")
    total_submissions = count_lines(submissions_path)
    total_comments = count_lines(comments_path)
    print(f"  Total submissions: {total_submissions:,}")
    print(f"  Total comments: {total_comments:,}")
    
    # Track stats
    step1_sub_removed = 0
    step1_com_removed = 0
    step2_sub_removed = 0
    step2_com_removed = 0
    step3_sub_removed = 0
    step3_com_removed = 0
    
    # =======================================================================
    # PASS 1: Process submissions (streaming)
    # =======================================================================
    print("\nPass 1: Filtering submissions...")
    
    # We need to track surviving submission IDs for comment filtering
    # This is unavoidable memory usage, but IDs are small strings
    surviving_ids_step1 = set()  # After word count filter
    surviving_ids_step2 = set()  # After dictionary filter
    surviving_ids_final = set()  # After sampling
    
    sub_output_path = get_output_path(submissions_path)
    
    # Step 1 & 2: Stream through submissions, apply filters
    for sub in stream_jsonl(submissions_path):
        sub_id = sub.get("id", "")
        text = get_submission_text(sub)
        word_count = count_words(text)
        score = sub.get("score", 0) or 0
        num_comments = sub.get("num_comments", 0) or 0
        
        # Step 1: Word count filter
        if not passes_word_count_filter(word_count, score, num_comments):
            step1_sub_removed += 1
            continue
        
        surviving_ids_step1.add(sub_id)
        
        # Step 2: Dictionary filter
        if not contains_required_term(text, REQUIRED_TERMS):
            step2_sub_removed += 1
            continue
        
        surviving_ids_step2.add(sub_id)
    
    # Step 3: Random sampling (if enabled)
    if SAMPLE_RATE < 1.0:
        sample_size = int(len(surviving_ids_step2) * SAMPLE_RATE)
        surviving_ids_final = set(random.sample(list(surviving_ids_step2), sample_size))
        step3_sub_removed = len(surviving_ids_step2) - len(surviving_ids_final)
    else:
        surviving_ids_final = surviving_ids_step2
    
    # Write surviving submissions
    print("  Writing filtered submissions...")
    final_sub_count = 0
    with open(sub_output_path, 'w', encoding='utf-8') as out:
        for sub in stream_jsonl(submissions_path):
            if sub.get("id", "") in surviving_ids_final:
                out.write(json.dumps(sub) + '\n')
                final_sub_count += 1
    
    # =======================================================================
    # PASS 2: Process comments (streaming)
    # =======================================================================
    print("\nPass 2: Filtering comments...")
    
    # Build lookup sets with t3_ prefix for parent matching
    valid_parents_step1 = {f"t3_{sid}" for sid in surviving_ids_step1}
    valid_parents_step2 = {f"t3_{sid}" for sid in surviving_ids_step2}
    valid_parents_final = {f"t3_{sid}" for sid in surviving_ids_final}
    
    com_output_path = get_output_path(comments_path)
    final_com_count = 0
    
    with open(com_output_path, 'w', encoding='utf-8') as out:
        for com in stream_jsonl(comments_path):
            parent_id = com.get("parent_id", "")
            
            # Must be top-level comment
            if not is_top_level_comment(com):
                step1_com_removed += 1
                continue
            
            # Parent must have survived step 1
            if parent_id not in valid_parents_step1:
                step1_com_removed += 1
                continue
            
            # Word count filter for comment
            text = get_comment_text(com)
            word_count = count_words(text)
            score = com.get("score", 0) or 0
            
            if not passes_word_count_filter(word_count, score, 0):
                step1_com_removed += 1
                continue
            
            # Step 2: Parent must have survived dictionary filter
            if parent_id not in valid_parents_step2:
                step2_com_removed += 1
                continue
            
            # Step 3: Parent must have survived sampling
            if parent_id not in valid_parents_final:
                step3_com_removed += 1
                continue
            
            out.write(json.dumps(com) + '\n')
            final_com_count += 1
    
    # =======================================================================
    # REPORTING
    # =======================================================================
    
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