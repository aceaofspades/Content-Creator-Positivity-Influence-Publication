#%%
"""
Channel video scraper (test: Aztecross, scrapetube + yt-dlp transcripts)

Requirements:
    pip install scrapetube google-api-python-client pandas pyarrow tqdm rich

Env:
    YOUTUBE_DATA_API_KEY=<your_api_key>

Notes:
- Per current constraints, this variant DOES NOT attempt to fetch transcripts via
  youtube-transcript-api or other transcript scrapers. The transcript columns are
  included as placeholders so the schema remains compatible.
"""

import os
import random
import subprocess
import time
from pathlib import Path
import shutil
import re  # NEW: for stripping VTT tags / metadata

import scrapetube

import pandas as pd
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
from rich.traceback import install
from tqdm.auto import tqdm

from googleapiclient.discovery import build

#%%

install()

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "channels_parquet" / "destiny2"

# Only Aztecross as a test; add more creators easily here later.
# KEY: creator display name (used for parquet name)
# VALUE: channel_username used by scrapetube (no leading @)
CHANNELS: dict[str, str] = { #Destiny 2
    "Aztecross": "AztecrossGaming",
    "MyNameIsByf": "mynameisbyf",
    "Ehroar": "Ehroar",
    "CammyCakes": "cammycakesgaming",
    "Sweatcicle": "sweatcicle",
    "Th3Jez": "Th3Jez",
    "FalloutPlays": "FalloutPlays",
    "KackisHD": "KackisHD",
    "xHOUNDISHx": "xHOUNDISHx",
    "IFrostBolt": "IFrostBolt",
    "Plunderdabooty": "Plunderdabooty",
    "SirDimetrious": "SirDimetrious",
    "MuffinBandit": "MuffinBandit",
    "DudditsII": "DudditsII",
    "Luckyy10P": "LUCKYY10P",
    "Datto": "DattoDoesDestiny",
    "Gladd": "GladdDestiny",
    "Zavalr": "Zavalr",
    "Skarrow9": "Skarrow9",
    "Mactics": "MacticsG1",
    "Evaze": "EvazeOfficial",
}

# CHANNELS: dict[str, str] = { # dbd
#     "Otzdarva": "notOtzdarva",
#     "ScottJund": "ScottJund",
#     "Yerv": "YervDBD",
#     "Ardetha": "Ardetha",
#     "TrU3Ta1ent": "TrU3Ta1ent",
#     "HybridPanda": "Panda",
#     "Dowsey": "Dowsey",
#     "OhTofu": "OhTofu",
# }

# CHANNELS: dict[str, str] = { # warframe
#     "Brozime": "Brozime",
#     "TacticalPotato": "TacticalPotato",
#     "StallordD": "StallordD",
#     "TheKengineer": "TheKengineer",
#     "iFlynn": "xiFlynn",
#     "kyaii": "kyaii",
#     "WarframeFlo": "WarframeFlo",
#     "GazTTV": "GazTTV",
#     "Pupsker": "Pupsker",
# }

# CHANNELS: dict[str, str] = { # dota
#     "Purge": "PurgeGamers",
#     "Gorgc": "Gorgc",
#     "Jenkins": "Jenkins69",
#     "BSJ": "BananaSlamJamma",
#     "WagaGaming": "WagaGaming",
#     "Baumi": "Baumi",
#     "SirActionSlacks": "SirActionSlacks",
# }

# CHANNELS: dict[str, str] = { # csgo
#     "3kliksphilip": "3kliksphilip",
#     "TheWarOwl": "TheWarOwl",
#     "TDM_Heyzeus": "TDM_Heyzeus",
#     "vooCSGO": "vooCSGO",
#     "NadeKing": "NadeKing",
#     "n0thing": "n0thing",
#     "fl0m": "fl0m",
# }

MAX_VIDEOS_PER_CHANNEL = 2000
COOLDOWN_MIN_S = 12
COOLDOWN_MAX_S = 20

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def build_youtube_client():
    api_key = os.environ.get("youtube_data_api_key")
    if not api_key:
        raise RuntimeError("youtube_data_api_key env var is not set.")
    return build("youtube", "v3", developerKey=api_key)



def get_recent_video_ids(channel_username: str, limit: int = MAX_VIDEOS_PER_CHANNEL) -> list[str]:
    """Use scrapetube to get up to `limit` most recent video IDs for a channel.

    `channel_username` is the YouTube handle without the leading @, e.g. "AztecrossGaming".
    """
    ids: list[str] = []
    videos = scrapetube.get_channel(
        channel_username=channel_username,
        limit=limit,
        sort_by="newest",
        content_type="videos",
    )
    for video in videos:
        vid = video.get("videoId")
        if not vid:
            continue
        ids.append(str(vid))
    return ids



def fetch_video_metadata(youtube, video_id: str) -> dict:
    """Fetch metadata for a single video via YouTube Data API."""
    resp = (
        youtube.videos()
        .list(
            part="snippet,liveStreamingDetails,recordingDetails,status,statistics",
            id=video_id,
        )
        .execute()
    )

    items = resp.get("items", [])
    if not items:
        # Gracefully handle missing; still return minimal info
        return {
            "video_id": video_id,
            "channel_id": None,
            "channel_title": None,
            "title": None,
            "description": None,
            "published_at": None,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "live_actualStartTime": None,
            "live_actualEndTime": None,
            "live_scheduledStartTime": None,
            "snippet_liveBroadcastContent": None,
            "recording_recordingDate": None,
            "status_uploadStatus": None,
            "view_count": None,
        }

    item = items[0]
    snippet = item.get("snippet", {}) or {}
    live = item.get("liveStreamingDetails", {}) or {}
    rec = item.get("recordingDetails", {}) or {}
    status = item.get("status", {}) or {}
    stats = item.get("statistics", {}) or {}

    # Normalize view_count to an integer (or None) to avoid mixed dtypes later
    vc_raw = stats.get("viewCount")
    try:
        view_count = int(vc_raw) if vc_raw is not None else None
    except (TypeError, ValueError):
        view_count = None

    return {
        "video_id": video_id,
        "channel_id": snippet.get("channelId"),
        "channel_title": snippet.get("channelTitle"),
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "published_at": snippet.get("publishedAt"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "live_actualStartTime": live.get("actualStartTime"),
        "live_actualEndTime": live.get("actualEndTime"),
        "live_scheduledStartTime": live.get("scheduledStartTime"),
        "snippet_liveBroadcastContent": snippet.get("liveBroadcastContent"),
        "recording_recordingDate": rec.get("recordingDate"),
        "status_uploadStatus": status.get("uploadStatus"),
        "view_count": view_count,
    }


# Cache the resolved yt-dlp executable path once
_YTDLP_PATH: str | None = None



def _resolve_ytdlp_path() -> str | None:
    """Return full path to yt-dlp executable if found, else None.

    Tries common variants for Windows and POSIX.
    """
    global _YTDLP_PATH
    if _YTDLP_PATH is not None:
        return _YTDLP_PATH

    candidates = [
        shutil.which("yt-dlp"),
        shutil.which("yt-dlp.exe"),
        shutil.which("yt_dlp"),
    ]
    for c in candidates:
        if c:
            _YTDLP_PATH = c
            return _YTDLP_PATH

    return None



def _clean_vtt_text_line(s: str) -> str:
    """Strip VTT metadata / tags from a single cue text line.

    - Removes things like <00:00:00.400>, <c>, </c>, <c.some_style>, etc.
    - Collapses extra whitespace.
    """
    # Remove any <...> tags (timestamps, cue styling, etc.)
    s = re.sub(r"<[^>]+>", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s



def fetch_transcript_via_ytdlp(video_id: str) -> tuple[str | None, str | None]:
    """Fetch transcript using yt-dlp auto-subs + basic VTT parsing.

    Returns:
        transcript_text (or None),
        transcript_failure_reason (or None on success)
    """
    ytdlp_exec = _resolve_ytdlp_path()
    if not ytdlp_exec:
        # Avoid spawning a nonexistent process; make failure explicit in the data
        return None, "yt_dlp_not_found"

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Temporary directory for VTT files per run
    tmp_dir = BASE_OUTPUT_DIR / "_tmp_vtt"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp output template; actual files will be like: <video_id>.<lang>.vtt
    output_template = str(tmp_dir / video_id)

    try:
        subprocess.run(
            [
                ytdlp_exec,
                "--write-auto-sub",
                "--skip-download",
                "-o",
                output_template,
                url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # Extra safety, in case the binary disappears between resolution and run
        return None, "yt_dlp_not_found_at_runtime"
    except subprocess.CalledProcessError as e:
        return None, f"yt_dlp_failed: {e}"

    # Collect any VTT files produced for this video_id
    vtt_files = list(tmp_dir.glob(f"{video_id}*.en*.vtt")) or list(tmp_dir.glob(f"{video_id}*.vtt"))
    vtt_files = vtt_files[:1]
    if not vtt_files:
        return None, "yt_dlp_no_vtt_subtitles"

    cleaned_lines: list[str] = []
    for vtt_path in vtt_files:
        try:
            raw_lines = vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for line in raw_lines:
            s = line.strip()
            if not s:
                continue
            # Skip WEBVTT header and timestamp/cue lines
            if s.startswith("WEBVTT"):
                continue
            if "-->" in s:
                continue
            # Skip metadata lines that yt-dlp leaves in the VTT
            lower = s.lower()
            if lower.startswith("kind:") or lower.startswith("language:"):
                continue

            # Clean out VTT tags / timestamps such as <00:00:00.400><c>, </c>, etc.
            cleaned = _clean_vtt_text_line(s)
            if cleaned:
                cleaned_lines.append(cleaned)

    if not cleaned_lines:
        # Cleanup VTT files for this video
        for vtt_path in vtt_files:
            try:
                vtt_path.unlink()
            except OSError:
                pass
        return None, "yt_dlp_empty_transcript"

    # Build transcript while reducing obvious duplication from overlapping cues
    transcript_parts: list[str] = []
    for line in cleaned_lines:
        if not transcript_parts:
            transcript_parts.append(line)
            continue

        prev = transcript_parts[-1]

        if line == prev:
            # Exact duplicate of previous cue text
            continue
        if prev in line:
            # New line extends previous (e.g., "Hello everyone" vs
            # "Hello everyone, it is your boy...") – keep only the longer one.
            transcript_parts[-1] = line
            continue
        if line in prev:
            # New line is a subset of previous – skip it.
            continue

        transcript_parts.append(line)

    transcript_text = " ".join(transcript_parts).strip() if transcript_parts else None

    # Cleanup VTT files for this video
    for vtt_path in vtt_files:
        try:
            vtt_path.unlink()
        except OSError:
            pass

    if not transcript_text:
        return None, "yt_dlp_empty_transcript"

    return transcript_text, None



def ensure_creator_paths(creator_name: str) -> Path:
    """Return parquet path directly in BASE_OUTPUT_DIR (no subfolder)."""
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = BASE_OUTPUT_DIR / f"{creator_name}.parquet"
    return parquet_path



def load_existing(parquet_path: Path) -> tuple[pd.DataFrame | None, set[str]]:
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "video_id" in df.columns:
            existing_ids = set(df["video_id"].astype(str))
        else:
            existing_ids = set()
        return df, existing_ids
    return None, set()



def save_parquet(parquet_path: Path, existing_df: pd.DataFrame | None, new_records: list[dict]) -> pd.DataFrame:
    """Merge and persist records, returning the up-to-date DataFrame.

    Includes defensive handling so that a type issue will not destroy all
    progress; we attempt a numeric cast for problematic columns, then fall
    back to string if needed.
    """
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    df_new = pd.DataFrame(new_records)

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["video_id"])
    else:
        combined = df_new

    # First attempt to write as-is
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_parquet(parquet_path, index=False)
    except ArrowInvalid as e:
        print(f"Parquet write failed with ArrowInvalid, attempting dtype fix: {e}")

        # Try to coerce known numeric columns to nullable integers
        for col in ["view_count"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")

        try:
            combined.to_parquet(parquet_path, index=False)
        except ArrowInvalid as e2:
            # As a last resort, store everything as string so data is not lost
            print(f"Second parquet write failed; coercing all columns to string: {e2}")
            combined = combined.astype("string")
            combined.to_parquet(parquet_path, index=False)

    print(f"Saved {len(combined)} rows to {parquet_path}")
    return combined


# -------------------------------------------------------------------
# Channel scraping
# -------------------------------------------------------------------


FLUSH_EVERY = 10  # how many new videos to buffer before writing to disk



def scrape_creator(creator_name: str, channel_username: str, youtube):
    parquet_path = ensure_creator_paths(creator_name)
    existing_df, existing_ids = load_existing(parquet_path)

    print(f"=== Scraping {creator_name} ===")
    print(f"Output: {parquet_path}")
    print(f"Already have {len(existing_ids)} videos recorded.")

    video_ids = get_recent_video_ids(channel_username, limit=MAX_VIDEOS_PER_CHANNEL)
    print(f"Found {len(video_ids)} video IDs from channel (max {MAX_VIDEOS_PER_CHANNEL}).")

    buffer_records: list[dict] = []

    with tqdm(video_ids, desc=f"{creator_name} videos", unit="video") as pbar:
        for vid in pbar:
            if vid in existing_ids:
                pbar.set_postfix(status="skip_existing")
                continue

            # --- metadata ---
            meta = fetch_video_metadata(youtube, vid)

            # --- transcript via yt-dlp auto-subs ---
            transcript_text, failure_reason = fetch_transcript_via_ytdlp(vid)

            record = {
                **meta,
                "transcript_text": transcript_text,
                "transcript_failure_reason": failure_reason,
            }
            buffer_records.append(record)
            existing_ids.add(vid)
            pbar.set_postfix(status="ok")

            # Flush to disk periodically so long runs can be salvaged
            if len(buffer_records) >= FLUSH_EVERY:
                existing_df = save_parquet(parquet_path, existing_df, buffer_records)
                buffer_records.clear()

            # --- cooldown between scrapes ---
            cooldown = random.randint(COOLDOWN_MIN_S, COOLDOWN_MAX_S)
            for remaining in range(cooldown, 0, -1):
                pbar.set_postfix(status="cooldown", pause_s=remaining)
                time.sleep(1)

    # Final flush for anything left in the buffer
    if buffer_records:
        existing_df = save_parquet(parquet_path, existing_df, buffer_records)
        buffer_records.clear()



def main():
    # Ensure yt-dlp exists BEFORE doing anything else
    ytdlp_path = _resolve_ytdlp_path()
    if not ytdlp_path:
        raise SystemExit("FATAL ERROR: yt-dlp not found on PATH. Install it before running this scraper.")

    youtube = build_youtube_client()

    for creator_name, channel_username in CHANNELS.items():
        scrape_creator(creator_name, channel_username, youtube)


if __name__ == "__main__":
    main()
