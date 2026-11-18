#%%
"""
This script restores punctuation for the 'transcript_text' column in Parquet files.
It is designed to be run as a Jupyter Notebook (interactive) or a script.

Features:
- Accepts a single file, multiple files, or a folder (top-level .parquet files only).
- Checkpoints progress every N rows to avoid data loss.
- Heuristic detection to skip already-punctuated text.
- Cleans common transcript artifacts.
- Outputs to a 'punctuated' subfolder by default, or a custom output folder.
"""

import sys
import shutil
from pathlib import Path
from typing import Union, List, Optional

# Try to import required libraries
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from punctfix import PunctFixer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: uv add pandas pyarrow punctfix tqdm rich")
    sys.exit(1)

#%%
# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Input: Can be a single file, list of files, or a folder path
# Examples:
#   INPUT_PATHS = Path(r"channels_parquet/Skarrow9.parquet")  # Single file
#   INPUT_PATHS = [Path(r"file1.parquet"), Path(r"file2.parquet")]  # Multiple files
#   INPUT_PATHS = Path(r"channels_parquet")  # Folder (processes all .parquet in top level)
INPUT_PATHS: Union[Path, List[Path]] = Path(r"channels_parquet/destiny2")

# Output folder (set to None to use default: input_parent/punctuated/)
# If set, all output files go to this folder
OUTPUT_DIR: Optional[Path] = None

# Directory for temporary checkpoints (relative to each input file's parent)
CHECKPOINT_BASE_DIR: Optional[Path] = None  # None = use input file's parent/checkpoints/

# How many rows to process before saving a checkpoint
CHECKPOINT_INTERVAL = 100

# Device for PunctFixer ('cuda' or 'cpu')
DEVICE = "cuda"


#%%
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def resolve_input_files(input_paths: Union[Path, List[Path]]) -> List[Path]:
    """
    Resolve input to a list of .parquet file paths.
    
    - If a single file: return [file]
    - If a list of files: return the list (validated)
    - If a folder: return all .parquet files in top level (no subdirectories)
    """
    if isinstance(input_paths, list):
        files = []
        for p in input_paths:
            p = Path(p)
            if p.is_file() and p.suffix.lower() == ".parquet":
                files.append(p)
            elif p.is_file():
                print(f"Warning: Skipping non-parquet file: {p}")
            else:
                print(f"Warning: File not found: {p}")
        return files
    
    path = Path(input_paths)
    
    if path.is_file():
        if path.suffix.lower() == ".parquet":
            return [path]
        else:
            raise ValueError(f"Input file is not a .parquet file: {path}")
    
    if path.is_dir():
        # Only top-level .parquet files, no subdirectories
        files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".parquet"]
        if not files:
            raise FileNotFoundError(f"No .parquet files found in folder: {path}")
        return sorted(files)
    
    raise FileNotFoundError(f"Input path not found: {path}")


def get_output_path(input_file: Path, output_dir: Optional[Path]) -> Path:
    """
    Determine the output path for a given input file.
    
    Default: input_parent/punctuated/filename_punctuated.parquet
    If output_dir is set: output_dir/filename_punctuated.parquet
    """
    output_filename = f"{input_file.stem}_punctuated.parquet"
    
    if output_dir is not None:
        return Path(output_dir) / output_filename
    else:
        return input_file.parent / "punctuated" / output_filename


def get_checkpoint_dir(input_file: Path, checkpoint_base: Optional[Path]) -> Path:
    """Determine checkpoint directory for a given input file."""
    if checkpoint_base is not None:
        return Path(checkpoint_base) / input_file.stem
    else:
        return input_file.parent / "checkpoints" / input_file.stem


def is_likely_punctuated(text: str) -> bool:
    """Check if text likely already has punctuation."""
    if not isinstance(text, str) or not text.strip():
        return False
    punct_chars = ".?!"
    count = sum(1 for c in text if c in punct_chars)
    words = len(text.split())
    if words < 2:
        return False
    return (count / words) > 0.02


def clean_transcript(text: str) -> str:
    """Remove common subtitle artifacts."""
    if not isinstance(text, str):
        return text
    return (
        text.replace("[Applause]", "")
        .replace("[Laughter]", "")
        .replace("[Music]", "")
        .replace("[music]", "")
        .strip()
    )


#%%
# -------------------------------------------------------------------
# Single File Processing
# -------------------------------------------------------------------

def process_single_file(
    input_file: Path,
    output_file: Path,
    checkpoint_dir: Path,
    fixer: PunctFixer,
    checkpoint_interval: int = 100
) -> bool:
    """
    Process a single parquet file and add punctuation.
    
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"Output:     {output_file}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load source dataframe
    df_source = pd.read_parquet(input_file)
    total_rows = len(df_source)
    print(f"Total rows: {total_rows}")
    
    if "transcript_text" not in df_source.columns:
        print(f"ERROR: DataFrame has no 'transcript_text' column")
        return False
    
    # Check for existing checkpoints to resume
    start_index = 0
    existing_checkpoints = list(checkpoint_dir.glob("chunk_*.parquet"))
    if existing_checkpoints:
        print(f"Found {len(existing_checkpoints)} existing checkpoints.")
        max_end_index = 0
        try:
            for ckpt in existing_checkpoints:
                end_val = int(ckpt.stem.split("_")[-1])
                if end_val > max_end_index:
                    max_end_index = end_val
            start_index = max_end_index
            print(f"Resuming from row {start_index}...")
        except ValueError:
            print("Could not parse checkpoint filenames. Starting over.")
    
    # Main processing loop
    pbar = tqdm(total=total_rows, initial=start_index, unit="rows", desc=input_file.name)
    
    current_batch_data = []
    current_batch_start = start_index
    
    for idx in range(start_index, total_rows):
        row = df_source.iloc[idx].copy()
        original_text = row["transcript_text"]
        
        final_text = original_text
        should_process = True
        
        if not isinstance(original_text, str) or not original_text.strip():
            should_process = False
        elif is_likely_punctuated(original_text):
            should_process = False
        
        if should_process:
            cleaned = clean_transcript(original_text)
            if cleaned:
                try:
                    final_text = fixer.punctuate(cleaned)
                except Exception:
                    final_text = original_text
            else:
                final_text = original_text
        
        row["transcript_text"] = final_text
        current_batch_data.append(row)
        pbar.update(1)
        
        # Checkpoint logic
        if len(current_batch_data) >= checkpoint_interval or idx == total_rows - 1:
            end_idx = idx + 1
            chunk_df = pd.DataFrame(current_batch_data)
            
            chunk_path = checkpoint_dir / f"chunk_{current_batch_start}_{end_idx}.parquet"
            chunk_df.to_parquet(chunk_path, index=False)
            
            current_batch_data = []
            current_batch_start = end_idx
    
    pbar.close()
    print("Processing complete!")
    
    # Merge checkpoints into final file
    print("Merging checkpoints into final file...")
    all_checkpoints = sorted(checkpoint_dir.glob("chunk_*.parquet"))
    
    if not all_checkpoints:
        print("ERROR: No checkpoints found!")
        return False
    
    dfs = [pd.read_parquet(ckpt) for ckpt in tqdm(all_checkpoints, desc="Loading chunks")]
    final_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Saving final output to {output_file}...")
    final_df.to_parquet(output_file, index=False)
    print("Done.")
    
    # Cleanup checkpoints (Optional - uncomment to enable)
    # shutil.rmtree(checkpoint_dir)
    # print("Checkpoint directory removed.")
    
    return True


#%%
# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

def main():
    # Resolve input files
    print("Resolving input files...")
    try:
        input_files = resolve_input_files(INPUT_PATHS)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} file(s) to process:")
    for f in input_files:
        print(f"  - {f}")
    
    # Initialize model once (shared across all files)
    print(f"\nLoading PunctFixer on {DEVICE}...")
    try:
        fixer = PunctFixer(language="en", device=DEVICE)
    except Exception as e:
        print(f"Failed to load PunctFixer on {DEVICE}, falling back to CPU. Error: {e}")
        fixer = PunctFixer(language="en", device="cpu")
    
    # Process each file
    results = []
    for input_file in input_files:
        output_file = get_output_path(input_file, OUTPUT_DIR)
        checkpoint_dir = get_checkpoint_dir(input_file, CHECKPOINT_BASE_DIR)
        
        success = process_single_file(
            input_file=input_file,
            output_file=output_file,
            checkpoint_dir=checkpoint_dir,
            fixer=fixer,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
        results.append((input_file, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful
    print(f"Processed: {len(results)} files")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for f, s in results:
            if not s:
                print(f"  - {f}")


#%%
if __name__ == "__main__":
    main()