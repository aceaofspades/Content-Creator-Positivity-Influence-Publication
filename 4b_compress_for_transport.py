#%%
"""
4b_compress_for_transport.py
============================
Packs filtered Reddit JSONL files into a single .tar.zst archive for
transport to the vLLM processing machine.

Run after step 4 (filtering) and before step 5 (vLLM analysis).
The output archive can be unpacked on the remote machine with:

    # Linux/Mac (zstd installed):
    tar --use-compress-program=zstd -xf <archive> -C <output_dir>

    # Python (any platform, zstandard installed):
    python -c "
    import tarfile, zstandard as zstd
    with open('<archive>', 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                tar.extractall('<output_dir>')
    "
"""

import tarfile
import zstandard as zstd
from pathlib import Path

#%%
# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Files to include — list of glob patterns or explicit paths (strings)
# Glob examples:  "reddit_posts/destiny2/filtered/*_2025_*_filtered.jsonl"
#                 "reddit_posts/destiny2/filtered/*_filtered_20.jsonl"
INPUT_GLOBS = [
    "reddit_posts/deadbydaylight/filtered/*_filtered.jsonl",
]

# Output archive path
OUTPUT_FILE = "reddit_posts/deadbydaylight/filtered/deadbydaylight_2015-2025_filtered.tar.zst"

# Compression level: 1 (fastest) – 22 (smallest). 19 is a good balance for text.
COMPRESSION_LEVEL = 19

#%%
# =============================================================================
# COMPRESS
# =============================================================================

root = Path(__file__).parent

# Resolve all input files
input_files: list[Path] = []
for pattern in INPUT_GLOBS:
    matches = sorted(root.glob(pattern))
    if not matches:
        print(f"  WARNING: no files matched pattern: {pattern}")
    input_files.extend(matches)

if not input_files:
    raise FileNotFoundError("No input files found. Check INPUT_GLOBS.")

output_path = root / OUTPUT_FILE
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Compressing {len(input_files)} file(s) → {output_path.name}  (zstd level {COMPRESSION_LEVEL})")

total_bytes_in = 0
cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)

with open(output_path, "wb") as f_out:
    with cctx.stream_writer(f_out, closefd=False) as compressor:
        with tarfile.open(fileobj=compressor, mode="w|") as tar:
            for f in input_files:
                size = f.stat().st_size
                total_bytes_in += size
                tar.add(f, arcname=f.name)
                print(f"  + {f.name}  ({size / 1_000_000:.1f} MB)")

total_bytes_out = output_path.stat().st_size
ratio = total_bytes_in / total_bytes_out if total_bytes_out else 0

print(f"\nDone.")
print(f"  Uncompressed:  {total_bytes_in / 1_000_000:.1f} MB")
print(f"  Compressed:    {total_bytes_out / 1_000_000:.1f} MB")
print(f"  Ratio:         {ratio:.1f}x")
print(f"  Output:        {output_path}")
