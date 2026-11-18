#%%
"""
Combine per-creator punctuated transcript parquet files into a single sorted file.

Reads all *_punctuated.parquet files from INPUT_DIR (excluding already-combined
files and LLM output files), concatenates them, sorts ascending by tiktoken token
count of the transcript column, and writes a single combined parquet.

Sorting by token count groups items into the same small/medium/large context
bucket, reducing context window thrashing on the vLLM server.

To retarget for another game, update INPUT_DIR (and OUTPUT_FILE if desired).
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

#%%
# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

INPUT_DIR = Path(r"channels_parquet/dbd/punctuated")
OUTPUT_FILE = INPUT_DIR.parent / "combined" / "combined_punctuated.parquet"

# Column containing the transcript text to tokenize for sort ordering
TEXT_COLUMN = "transcript_text"

# tiktoken encoding — must match what 5_vllm_text_analysis.py uses internally
TIKTOKEN_ENCODING = "cl100k_base"

# Token range boundaries (mirrored from 5_vllm_text_analysis.py)
TOKEN_RANGES = [
    ("small",  0,      10_000),
    ("medium", 10_000, 45_000),
    ("large",  45_000, float("inf")),
]

#%%
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def count_tokens(text: str | None, enc: tiktoken.Encoding) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(enc.encode(text, disallowed_special=()))


def bucket_label(token_count: int) -> str:
    for name, lo, hi in TOKEN_RANGES:
        if lo <= token_count < hi:
            return name
    return "large"


#%%
# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # Collect input files: *_punctuated.parquet, excluding combined_* and *_llm.parquet
    input_files = sorted([
        p for p in INPUT_DIR.glob("*_punctuated.parquet")
        if not p.stem.startswith("combined_") and not p.stem.endswith("_llm")
    ])

    if not input_files:
        print(f"No *_punctuated.parquet files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} creator file(s) in {INPUT_DIR}\n")

    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    frames: list[pd.DataFrame] = []
    per_file_counts: list[tuple[str, int]] = []

    for path in tqdm(input_files, desc="Loading files"):
        df = pd.read_parquet(path)
        per_file_counts.append((path.name, len(df)))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    total_rows = len(combined)

    print(f"\nCombined {total_rows:,} rows from {len(input_files)} files.")
    print("\nPer-file row counts:")
    for name, count in per_file_counts:
        print(f"  {name}: {count:,}")

    # Compute token counts for sorting (not written to output)
    print(f"\nComputing tiktoken counts on '{TEXT_COLUMN}' ({TIKTOKEN_ENCODING})...")
    token_counts = [
        count_tokens(text, enc)
        for text in tqdm(combined[TEXT_COLUMN], desc="Tokenizing", unit="row")
    ]

    # Sort ascending by token count (nulls/empty = 0, sort to top)
    sort_indices = sorted(range(total_rows), key=lambda i: token_counts[i])
    combined = combined.iloc[sort_indices].reset_index(drop=True)
    sorted_counts = [token_counts[i] for i in sort_indices]

    # Token bucket distribution summary
    bucket_counts: dict[str, int] = {name: 0 for name, _, _ in TOKEN_RANGES}
    for tc in sorted_counts:
        bucket_counts[bucket_label(tc)] += 1

    print("\nToken bucket distribution (post-sort):")
    for name, count in bucket_counts.items():
        pct = 100 * count / total_rows if total_rows else 0
        print(f"  {name:8s}: {count:6,}  ({pct:.1f}%)")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {OUTPUT_FILE} ...")
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, OUTPUT_FILE, compression="zstd")

    print(f"Done. {total_rows:,} rows written to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
