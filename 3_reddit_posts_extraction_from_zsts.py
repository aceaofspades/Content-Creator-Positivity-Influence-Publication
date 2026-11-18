#%%
"""
Extract subreddit data from Pushshift dumps for a specific year.
Jupyter Notebook version - configure variables in the cell below.

Prerequisites:
    pip install zstandard
    
Download data from Academic Torrents:
    https://academictorrents.com/details/3e3f64dee22dc304cdd2546254ca1f8e8ae542b4
"""

import json
import zstandard as zstd
from datetime import datetime
from pathlib import Path
from typing import Generator
from calendar import timegm

#%%
# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# List of subreddits to process (without r/ prefix)
SUBREDDITS = [
    "GlobalOffensive",
]

# Year range to extract (inclusive)
START_YEAR = 2015
END_YEAR = 2025  # Set equal to START_YEAR for a single year

# Directory containing the downloaded .zst files
INPUT_DIR = "reddit_posts/csgo/zsts"  # Change per game, e.g. "reddit_posts/csgo/zsts"

# Output directory for extracted files
OUTPUT_DIR = "reddit_posts/csgo/raw"  # Change per game, e.g. "reddit_posts/csgo/raw"

# Set to True to overwrite existing output files
OVERWRITE = False

# Set to True to skip submissions/comments
SKIP_SUBMISSIONS = False
SKIP_COMMENTS = False

# Set to True to only keep top-level comments (direct replies to submissions, not replies to other comments)
TOP_LEVEL_COMMENTS_ONLY = True

#%%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_year_timestamps(year: int) -> tuple[int, int]:
    """Get start and end Unix timestamps for a given year."""
    start = timegm(datetime(year, 1, 1, 0, 0, 0).timetuple())
    end = timegm(datetime(year, 12, 31, 23, 59, 59).timetuple())
    return start, end


def get_year_range_timestamps(start_year: int, end_year: int) -> tuple[int, int]:
    """Get start and end Unix timestamps for a year range."""
    start = timegm(datetime(start_year, 1, 1, 0, 0, 0).timetuple())
    end = timegm(datetime(end_year, 12, 31, 23, 59, 59).timetuple())
    return start, end


def format_timestamp(unix_ts: int) -> str:
    """Convert Unix timestamp to human-readable format."""
    return datetime.utcfromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S UTC')


def read_zst_file(filepath: Path) -> Generator[dict, None, None]:
    """
    Stream read a zstandard compressed ndjson file, yielding one JSON object at a time.
    Memory efficient - doesn't load entire file into memory.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                chunk = reader.read(65536)  # Read 64KB at a time
                if not chunk:
                    break
                
                text_chunk = previous_line + chunk.decode('utf-8', errors='ignore')
                lines = text_chunk.split('\n')
                
                # Keep the last potentially incomplete line for next iteration
                previous_line = lines[-1]
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue


def is_in_year(record: dict, start_ts: int, end_ts: int) -> bool:
    """Check if a record's creation timestamp falls within the target year."""
    created = record.get('created_utc', 0)
    if isinstance(created, str):
        created = int(float(created))
    return start_ts <= created <= end_ts


def is_top_level_comment(record: dict) -> bool:
    """
    Check if a comment is a top-level comment (direct reply to a submission).
    Top-level comments have parent_id starting with 't3_' (submission).
    Replies to other comments have parent_id starting with 't1_' (comment).
    """
    parent_id = record.get('parent_id', '')
    # Handle cases where parent_id might not be a string
    if not isinstance(parent_id, str):
        return False
    return parent_id.startswith('t3_')

#%%
# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_data(
    input_file: Path, 
    output_file: Path, 
    start_ts: int, 
    end_ts: int,
    data_type: str,
    year_label: str,
    top_level_only: bool = False
) -> int:
    """
    Extract records for a specific year range from the zst file.
    Returns the count of extracted records.
    """
    print(f"\n  Processing {data_type} from: {input_file.name}")
    if top_level_only and data_type == "comments":
        print(f"  Filtering: TOP-LEVEL COMMENTS ONLY")
    print(f"  Output: {output_file}")
    
    count = 0
    total_scanned = 0
    filtered_out = 0
    progress_interval = 100000 if data_type == "submissions" else 500000
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for record in read_zst_file(input_file):
            total_scanned += 1
            
            if total_scanned % progress_interval == 0:
                msg = f"    Scanned {total_scanned:,} records, found {count:,} from {year_label}"
                if top_level_only and data_type == "comments":
                    msg += f" (skipped {filtered_out:,} reply comments)"
                print(msg + "...")
            
            if is_in_year(record, start_ts, end_ts):
                # For comments, optionally filter to top-level only
                if top_level_only and data_type == "comments":
                    if not is_top_level_comment(record):
                        filtered_out += 1
                        continue
                
                out.write(json.dumps(record) + '\n')
                count += 1
    
    msg = f"  ✓ Extracted {count:,} {data_type} (scanned {total_scanned:,} total)"
    if top_level_only and data_type == "comments":
        msg += f", filtered out {filtered_out:,} reply comments"
    print(msg)
    return count


def process_subreddit(
    subreddit: str,
    input_dir: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    start_ts: int,
    end_ts: int,
    overwrite: bool = False,
    process_submissions: bool = True,
    process_comments: bool = True,
    top_level_comments_only: bool = False
) -> dict:
    """
    Process a single subreddit's data.
    Returns a dict with counts of extracted submissions and comments.
    """
    # Create year label for filenames and display
    if start_year == end_year:
        year_label = str(start_year)
    else:
        year_label = f"{start_year}-{end_year}"
    
    print(f"\n{'='*60}")
    print(f"Processing r/{subreddit} for {year_label}")
    print(f"{'='*60}")
    
    results = {"subreddit": subreddit, "submissions": 0, "comments": 0, "skipped": []}
    
    # Define file paths
    submissions_input = input_dir / f"{subreddit}_submissions.zst"
    comments_input = input_dir / f"{subreddit}_comments.zst"
    
    submissions_output = output_dir / f"{subreddit}_{year_label}_submissions.jsonl"
    comments_output = output_dir / f"{subreddit}_{year_label}_comments.jsonl"
    
    # Process submissions
    if process_submissions:
        if submissions_output.exists() and not overwrite:
            print(f"\n  ⏭️  Skipping submissions - output already exists: {submissions_output.name}")
            results["skipped"].append("submissions")
        elif not submissions_input.exists():
            print(f"\n  ⚠️  Submissions file not found: {submissions_input}")
        else:
            results["submissions"] = extract_data(
                submissions_input, submissions_output, start_ts, end_ts, "submissions", year_label
            )
    
    # Process comments
    if process_comments:
        if comments_output.exists() and not overwrite:
            print(f"\n  ⏭️  Skipping comments - output already exists: {comments_output.name}")
            results["skipped"].append("comments")
        elif not comments_input.exists():
            print(f"\n  ⚠️  Comments file not found: {comments_input}")
        else:
            results["comments"] = extract_data(
                comments_input, comments_output, start_ts, end_ts, "comments", year_label,
                top_level_only=top_level_comments_only
            )
    
    return results


def print_summary(all_results: list[dict], year_label: str):
    """Print a summary of all processed subreddits."""
    print(f"\n{'='*60}")
    print(f"SUMMARY - {year_label} Data Extraction")
    print(f"{'='*60}")
    print(f"{'Subreddit':<25} {'Submissions':>12} {'Comments':>12} {'Status':<15}")
    print("-" * 60)
    
    total_subs = 0
    total_comments = 0
    
    for r in all_results:
        status = "complete"
        if r["skipped"]:
            status = f"skipped: {', '.join(r['skipped'])}"
        
        print(f"{r['subreddit']:<25} {r['submissions']:>12,} {r['comments']:>12,} {status:<15}")
        total_subs += r["submissions"]
        total_comments += r["comments"]
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_subs:>12,} {total_comments:>12,}")
    print()

#%%
# =============================================================================
# RUN EXTRACTION
# =============================================================================

# Setup paths
input_dir = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Create year label for display and filenames
if START_YEAR == END_YEAR:
    year_label = str(START_YEAR)
else:
    year_label = f"{START_YEAR}-{END_YEAR}"

# Get timestamps for the target year range
start_ts, end_ts = get_year_range_timestamps(START_YEAR, END_YEAR)

print("=" * 60)
print("Reddit Data Extractor")
print("=" * 60)
print(f"Years:       {year_label}")
print(f"Date range:  {format_timestamp(start_ts)} to {format_timestamp(end_ts)}")
print(f"Input dir:   {input_dir.absolute()}")
print(f"Output dir:  {output_dir.absolute()}")
print(f"Subreddits:  {', '.join(SUBREDDITS)}")
print(f"Overwrite:   {OVERWRITE}")
print(f"Top-level comments only: {TOP_LEVEL_COMMENTS_ONLY}")

# Process each subreddit
all_results = []
for subreddit in SUBREDDITS:
    # Clean up subreddit name (remove r/ prefix if present)
    subreddit = subreddit.replace("r/", "").strip()
    
    results = process_subreddit(
        subreddit=subreddit,
        input_dir=input_dir,
        output_dir=output_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
        start_ts=start_ts,
        end_ts=end_ts,
        overwrite=OVERWRITE,
        process_submissions=not SKIP_SUBMISSIONS,
        process_comments=not SKIP_COMMENTS,
        top_level_comments_only=TOP_LEVEL_COMMENTS_ONLY
    )
    all_results.append(results)

# Print summary
print_summary(all_results, year_label)

#%%
# =============================================================================
# LOAD RESULTS (OPTIONAL)
# =============================================================================

# Uncomment and modify to load extracted data into pandas:

# import pandas as pd
# 
# subreddit = "DestinyTheGame"
# year_range = "2017-2024"  # or just "2017" for single year
#
# submissions = pd.read_json(
#     f"reddit_posts/destiny2/raw/{subreddit}_{year_range}_submissions.jsonl",
#     lines=True
# )
#
# comments = pd.read_json(
#     f"reddit_posts/destiny2/raw/{subreddit}_{year_range}_comments.jsonl",
#     lines=True
# )
# 
# print(f"Loaded {len(submissions):,} submissions and {len(comments):,} comments")
# submissions.head()