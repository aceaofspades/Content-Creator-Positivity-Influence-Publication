#%%
"""
Unified vLLM Text Analysis - Streaming Two-Pass Processor

Memory-efficient streaming processor for both Reddit data (submissions + comments)
and standalone text files (YouTube transcripts). Uses a remote vLLM server with
async HTTP requests.

Supports two input modes:
- FILE PAIRS: Reddit submissions + comments (JSONL) — submissions get two-pass
  modular analysis, comments get single-pass lightweight analysis with submission context
- SINGLE FILES: Standalone text (parquet or JSONL) — processed identically to
  submissions using the same two-pass modular prompt system

Processing Strategy:
- PRIMARY FILES (submissions / transcripts): Two-pass modular prompt system
  - Pass 1 (cheap): Detect which of 11 topic categories are discussed → JSON object
  - Pass 2 (targeted): Build prompt with only relevant topic sections → full analysis
  - Tiktoken token counting drives dynamic context window selection

- COMMENTS (file pairs only): Single-pass lightweight prompt
  - Batched (up to 15 comments per submission per API call)
  - Uses dedicated comment prompt (reddit_comments_prompt.txt)
  - Returns JSON object keyed by comment ID

Processing flow per input file set:
  Phase 1: Stream primary file → two-pass process → sidecar (+ sub_lookup for pairs)
  Phase 2: Stream comments → single-pass batch process (only for file pairs)
  Phase 3: Stream-merge source data + sidecar → final parquet
"""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from rich.console import Console
from rich.traceback import install
from tqdm.auto import tqdm

install()
console = Console()

#%%
# =============================================================================
#MARK: NETWORK CONFIGURATION
# =============================================================================

NETWORK_URL = os.getenv("SPARK_LLM_NETWORK_URL")
if not NETWORK_URL:
    raise ValueError("Environment variable SPARK_LLM_NETWORK_URL is not set.")

parsed_url = urlparse(NETWORK_URL)
HOST_IP = parsed_url.hostname if parsed_url.hostname else "127.0.0.1"
MANAGER_URL = f"http://{HOST_IP}:6061"
VLLM_API_URL = f"http://{HOST_IP}:6060/v1"


#%%
# =============================================================================
#MARK: TOKEN RANGE CONFIGURATION
# =============================================================================


@dataclass
class TokenRange:
    """Defines an input token count range and its corresponding total model context."""

    name: str
    min_input_tokens: int  # Inclusive
    max_input_tokens: int  # Exclusive
    max_total_model_tokens: int


TOKEN_RANGES = [
    TokenRange(name="small", min_input_tokens=0, max_input_tokens=10_000, max_total_model_tokens=24_000),
    TokenRange(name="medium", min_input_tokens=10_000, max_input_tokens=45_000, max_total_model_tokens=65_000),
    TokenRange(name="large", min_input_tokens=45_000, max_input_tokens=float("inf"), max_total_model_tokens=200_000),
]


#%%
# =============================================================================
#MARK: CONFIGURATION
# =============================================================================


@dataclass
class InputFileSet:
    """An input file set: either paired submissions+comments or a single standalone file."""

    submissions: Path                    # Primary file (submissions or standalone)
    comments: Optional[Path] = None      # None for single-file mode
    text_columns: list[str] = field(default_factory=lambda: ["title", "selftext"])
    id_column: str = "id"
    input_format: str = "jsonl"          # "jsonl" or "parquet"
    skip_qualification: bool = False     # True for transcripts (no score/bot filtering)
    output_dir: Optional[Path] = None    # Output dir for _llm.parquet and sidecar; defaults to input parent

    def __post_init__(self):
        self.submissions = Path(self.submissions)
        if self.comments is not None:
            self.comments = Path(self.comments)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self.submissions.stem.rsplit("_", 2)[0]

    @property
    def is_single_file(self) -> bool:
        return self.comments is None


@dataclass
class Config:
    """Central configuration for the unified LLM text analysis processor."""

    # Model
    model_name: str = "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"

    # Remote vLLM Server
    vllm_api_url: str = VLLM_API_URL
    manager_url: str = MANAGER_URL

    # vLLM Engine Settings
    gpu_memory_utilization: float = 0.85
    default_max_total_model_tokens: int = 200_000
    model_load_timeout: int = 600

    # Generation
    temperature: float = 0.3
    max_output_tokens: int = 8192
    first_pass_max_output_tokens: int = 512  # First pass just returns a short JSON object
    comment_max_output_tokens: int = 4096  # Comments are simpler, need less output

    # Processing
    batch_size: int = 320  # Submissions read per chunk
    num_parallel_requests: int = 64  # Concurrent HTTP requests
    request_timeout: Optional[int] = None
    comment_batch_size: int = 15  # Comments per single API call

    # Tiktoken
    tiktoken_encoding_name: str = "cl100k_base"
    tiktoken_buffer: float = 1.2  # 20% safety buffer for encoding differences

    # Final parquet compression (zstd level 1-22)
    final_compression_level: int = 22

    # Assembly chunk size (rows read at a time during final parquet merge)
    assembly_chunk_size: int = 10_000

    # Execution Control
    test_mode: bool = False
    row_limit: Optional[int] = None
    names_to_process: list[str] = None
    
    # Reprocessing Control - if True, rows with llm_error will be reprocessed
    reprocess_errors: bool = True

    # Prompts — Submissions use two-pass modular system
    system_prompt: str = "You are a helpful assistant. Please output your response in JSON format."
    first_pass_prompt_path: Path = Path("other_data/destiny2/first_pass_prompt.txt")
    modular_prompt_components_path: Path = Path("other_data/destiny2/modular_prompt_categories.txt")

    # Prompts — Comments use dedicated lightweight prompt
    comment_prompt_path: Path = Path("other_data/destiny2/reddit_comments_prompt.txt")

    # Submission filtering thresholds
    filter_always_include_words: int = 20
    filter_medium_min_words: int = 10
    filter_medium_min_score: int = 5
    filter_medium_min_comments: int = 3
    filter_short_min_score: int = 20
    filter_short_min_comments: int = 10

    # Selftext truncation (-1 = unlimited)
    max_selftext_words: int = -1

    # Input file sets (reddit pairs and/or standalone transcript files)
    file_pairs: list = field(default_factory=lambda: [
        # Reddit subreddit pairs (submissions + comments)
        InputFileSet(
            submissions="reddit_posts/destiny2/filtered/Destiny_2_2025_submissions_filtered.jsonl",
            comments="reddit_posts/destiny2/filtered/Destiny_2_2025_comments_filtered.jsonl",
            output_dir="reddit_posts/destiny2/processed",
        ),
        InputFileSet(
            submissions="reddit_posts/destiny2/filtered/destiny2_2025_submissions_filtered.jsonl",
            comments="reddit_posts/destiny2/filtered/destiny2_2025_comments_filtered.jsonl",
            output_dir="reddit_posts/destiny2/processed",
        ),
        InputFileSet(
            submissions="reddit_posts/destiny2/filtered/DestinyTheGame_2025_submissions_filtered.jsonl",
            comments="reddit_posts/destiny2/filtered/DestinyTheGame_2025_comments_filtered.jsonl",
            output_dir="reddit_posts/destiny2/processed",
        ),
        # YouTube transcript files — combined & token-sorted by 2b_combine_punctuated_yt_transcripts.py
        # InputFileSet(
        #     submissions="channels_parquet/destiny2/combined/combined_punctuated.parquet",
        #     text_columns=["title", "transcript_text"], id_column="video_id",
        #     input_format="parquet", skip_qualification=True,
        #     output_dir="channels_parquet/destiny2/processed",
        # ),
    ])

    def __post_init__(self):
        self.first_pass_prompt_path = Path(self.first_pass_prompt_path)
        self.modular_prompt_components_path = Path(self.modular_prompt_components_path)
        self.comment_prompt_path = Path(self.comment_prompt_path)
        if self.names_to_process is None:
            self.names_to_process = []

    def get_target_file_pairs(self) -> list[InputFileSet]:
        if not self.names_to_process:
            return self.file_pairs
        return [fp for fp in self.file_pairs if fp.name in self.names_to_process]

    @property
    def comment_chunk_size(self) -> int:
        """Comments read per chunk = enough to fill one cycle of API calls."""
        return self.batch_size * self.comment_batch_size


CONFIG = Config()

_current_max_total_model_tokens: Optional[int] = None


#%%
# =============================================================================
#MARK: TOKEN RANGE UTILITIES
# =============================================================================


def get_token_range(input_token_count: int) -> TokenRange:
    for tr in TOKEN_RANGES:
        if tr.min_input_tokens <= input_token_count < tr.max_input_tokens:
            return tr
    return TOKEN_RANGES[-1]


def get_token_range_by_total(max_total_model_tokens: int) -> Optional[TokenRange]:
    for tr in TOKEN_RANGES:
        if tr.max_total_model_tokens == max_total_model_tokens:
            return tr
    return None


def get_max_total_model_tokens_for_input(input_token_count: int) -> int:
    return get_token_range(input_token_count).max_total_model_tokens


#%%
# =============================================================================
#MARK: TIKTOKEN UTILITIES
# =============================================================================


def get_tiktoken_encoding(config: Config) -> tiktoken.Encoding:
    """Get the tiktoken encoding once for reuse."""
    return tiktoken.get_encoding(config.tiktoken_encoding_name)


def count_message_tokens(
    messages: list[dict],
    encoding: tiktoken.Encoding,
    buffer: float = 1.2,
) -> int:
    """
    Count tokens across all messages with a safety buffer.
    Uses cl100k_base as a proxy — not exact for Qwen but close enough for range selection.
    """
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg.get("content", "")))
        total += 4  # per-message overhead (role, separators)
    total += 2  # priming tokens
    return int(total * buffer)


#%%
# =============================================================================
#MARK: PATH UTILITIES
# =============================================================================


def get_output_path(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Generate final output parquet path. Uses output_dir if set, else input's parent."""
    base = output_dir if output_dir is not None else input_path.parent
    return base / f"{input_path.stem}_llm.parquet"


def get_sidecar_path(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Generate sidecar JSONL path. Uses output_dir if set, else input's parent."""
    base = output_dir if output_dir is not None else input_path.parent
    return base / f"{input_path.stem}_llm_sidecar.jsonl"


def count_jsonl_lines(filepath: Path) -> int:
    """Fast line count for a JSONL file (for tqdm progress bars)."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def count_rows(file_set: InputFileSet) -> int:
    """Count total rows in the primary file, handling both JSONL and parquet formats."""
    if file_set.input_format == "parquet":
        return pq.read_metadata(file_set.submissions).num_rows
    return count_jsonl_lines(file_set.submissions)


def iter_parquet_chunks(path: Path, chunksize: int):
    """Yield DataFrames of chunksize rows from a parquet file."""
    df = pd.read_parquet(path)
    for start in range(0, len(df), chunksize):
        yield df.iloc[start:start + chunksize].copy()


#%%
# =============================================================================
#MARK: HELPER FUNCTIONS
# =============================================================================


def count_words(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def truncate_text(text: str, max_words: int) -> str:
    if max_words <= 0 or not text or not isinstance(text, str):
        return text or ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "... [truncated]"


def is_valid_content(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    invalid_values = ["[deleted]", "[removed]", ""]
    return text.strip().lower() not in [v.lower() for v in invalid_values]


def is_bot_account(author: str) -> bool:
    if not author or not isinstance(author, str):
        return False
    bot_patterns = [
        "automoderator", "botdefense", "bot", "automod",
        "remindmebot", "sneakpeekbot", "repostsleuthbot",
    ]
    author_lower = author.lower()
    return any(pattern in author_lower for pattern in bot_patterns)


def check_submission_qualification(row, config: Config) -> tuple[bool, str]:
    """
    Check if a submission qualifies for LLM analysis.
    Accepts dict or pd.Series (both support .get()).
    """
    title = row.get("title", "") or ""
    selftext = row.get("selftext", "") or ""
    author = row.get("author", "")
    score = row.get("score", 0) or 0
    num_comments = row.get("num_comments", 0) or 0

    if not is_valid_content(title):
        return False, "title_deleted_or_empty"

    if selftext and not is_valid_content(selftext):
        selftext = ""

    if is_bot_account(author):
        return False, "bot_account"

    total_words = count_words(title) + count_words(selftext)

    if total_words >= config.filter_always_include_words:
        return True, ""

    if total_words >= config.filter_medium_min_words:
        if score >= config.filter_medium_min_score or num_comments >= config.filter_medium_min_comments:
            return True, ""
        return False, f"10-19_words_low_engagement(score={score},comments={num_comments})"

    if score >= config.filter_short_min_score or num_comments >= config.filter_short_min_comments:
        return True, ""

    return False, f"under_10_words_low_engagement(score={score},comments={num_comments})"


#%%
# =============================================================================
#MARK: JSON / RESPONSE PARSING
# =============================================================================


def extract_json_from_response(response: str) -> tuple[Optional[dict], Optional[str]]:
    """Extract JSON (dict or list) from model response text."""
    if not response:
        return None, "empty_response"

    try:
        return json.loads(response), None
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1)), None
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object
    start_idx = response.find("{")
    end_idx = response.rfind("}") + 1
    if start_idx != -1 and end_idx > 0:
        try:
            return json.loads(response[start_idx:end_idx]), None
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array
    start_idx = response.find("[")
    end_idx = response.rfind("]") + 1
    if start_idx != -1 and end_idx > 0:
        try:
            return json.loads(response[start_idx:end_idx]), None
        except json.JSONDecodeError:
            pass

    return None, "json_parse_failed"


def parse_model_output(generated_text: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (json_string, None) on success, (None, raw_text) on failure."""
    parsed, error = extract_json_from_response(generated_text)
    if error:
        return None, generated_text
    return json.dumps(parsed), None


def get_submission_summary(llm_response: Optional[str]) -> str:
    """Extract overall_state_of_game.summary from a submission's LLM response JSON."""
    if not llm_response or not isinstance(llm_response, str):
        return ""
    try:
        data = json.loads(llm_response)
        return data.get("overall_state_of_game", {}).get("summary", "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


VALID_CATEGORIES = {
    "content_activities", "pvp", "sandbox_balance", "progression_economy",
    "rewards_loot", "inventory_ui", "monetization", "technical",
    "developer_communication", "social_onboarding", "narrative",
}


def extract_categories_from_string(text: str) -> list[str]:
    """Extract valid category names from a string (comma/space separated or embedded)."""
    if not text or not isinstance(text, str):
        return []
    text_lower = text.lower()
    found = []
    for cat in VALID_CATEGORIES:
        # Check for category as whole word (with word boundaries)
        if re.search(rf'\b{re.escape(cat)}\b', text_lower):
            found.append(cat)
        # Also check snake_case as space-separated words (e.g., "sandbox balance")
        elif '_' in cat:
            space_version = cat.replace('_', ' ')
            if space_version in text_lower:
                found.append(cat)
    return found


def parse_categories_response(response_text: str) -> tuple[list[str], Optional[str]]:
    """
    Parse the first-pass category detection response.
    Expects: {"categories": ["cat1", "cat2"]}
    Returns (categories_list, error_or_None).
    """
    if not response_text:
        return [], "empty_first_pass_response"

    parsed, error = extract_json_from_response(response_text)
    
    if error:
        # Last resort: scan raw text for category names
        categories = extract_categories_from_string(response_text)
        if categories:
            return categories, None
        return [], f"first_pass_parse_error: {error}"

    # Handle list directly (legacy format)
    if isinstance(parsed, list):
        categories = [c.strip().lower() for c in parsed if isinstance(c, str) and c.strip().lower() in VALID_CATEGORIES]
        return categories, None

    # Handle dict - this is the expected format: {"categories": [...]}
    if isinstance(parsed, dict):
        # Try to find categories in any reasonable location
        for key in ("categories", "topics", "result", "answer", "response", "relevant_categories", "detected_categories"):
            if key not in parsed:
                continue
            val = parsed[key]
            
            # Value is a list of strings (expected)
            if isinstance(val, list):
                categories = [c.strip().lower() for c in val if isinstance(c, str) and c.strip().lower() in VALID_CATEGORIES]
                return categories, None  # Return even if empty - that's valid (no categories detected)
            
            # Value is a string (model didn't follow format exactly)
            if isinstance(val, str):
                categories = extract_categories_from_string(val)
                return categories, None
        
        # No recognized key found - try scanning all values
        # First check if any value is a list
        for val in parsed.values():
            if isinstance(val, list):
                categories = [c.strip().lower() for c in val if isinstance(c, str) and c.strip().lower() in VALID_CATEGORIES]
                if categories:
                    return categories, None
        
        # Last resort: scan all string values
        all_text = " ".join(str(v) for v in parsed.values() if isinstance(v, str))
        categories = extract_categories_from_string(all_text)
        return categories, None  # Return even if empty
    
    # Truly unexpected type (int, None, etc.)
    return [], f"unexpected_first_pass_format: {type(parsed).__name__}"


#%%
# =============================================================================
#MARK: SIDECAR I/O (APPEND-ONLY CRASH-SAFE RESULTS)
# =============================================================================

# Columns for submissions (includes first_pass_response)
SUBMISSION_LLM_COLUMNS = [
    "llm_first_pass_response",
    "llm_response", 
    "llm_disqualified_reason", 
    "llm_error", 
    "llm_processed_at"
]

# Columns for comments (no first pass)
COMMENT_LLM_COLUMNS = [
    "llm_response", 
    "llm_disqualified_reason", 
    "llm_error", 
    "llm_processed_at"
]


def load_processed_ids(sidecar_path: Path) -> set[str]:
    """Load just the IDs from a sidecar file. Lightweight for resume checks."""
    if not sidecar_path.exists():
        return set()
    ids = set()
    with open(sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def load_processed_ids_excluding_errors(sidecar_path: Path) -> set[str]:
    """
    Load IDs from sidecar, EXCLUDING rows that have llm_error set.
    Use this when reprocess_errors=True to retry failed rows.
    """
    if not sidecar_path.exists():
        return set()
    ids = set()
    with open(sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # Only include if no error (success or disqualified)
                    if not item.get("llm_error"):
                        ids.add(str(item["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def load_sidecar(sidecar_path: Path) -> dict[str, dict]:
    """Load full sidecar file into dict keyed by ID. Used for assembly and submission lookup."""
    if not sidecar_path.exists():
        return {}
    results = {}
    with open(sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    results[str(item["id"])] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    return results


def append_to_sidecar(entries: list[dict], sidecar_path: Path) -> None:
    """Append result entries to sidecar file with fsync for durability."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


#%%
# =============================================================================
#MARK: PROMPT LOADING
# =============================================================================


def load_prompt_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt_sections(components_path: Path) -> dict[str, str]:
    """
    Parse modular_prompt_categories.txt into named sections.
    Cached per-file-processing (read once, reuse for all rows in that file).
    """
    content = components_path.read_text(encoding="utf-8")
    return {
        name: text.strip()
        for name, text in re.findall(r"===SECTION:(\w+)===\n(.*?)===END===", content, re.DOTALL)
    }


def build_modular_prompt(
    categories: list[str],
    transcript: str,
    sections: dict[str, str],
) -> str:
    """
    Build analysis prompt with only relevant topic sections.
    If categories is empty, returns only core sections (no topic-specific content).
    """
    included = [f"topic_{cat}" for cat in categories]

    parts = [sections["header"], sections["core_fields"]]

    if included:
        parts.append(sections["topic_sentiments_header"])
        for section_name in included:
            if section_name in sections:
                parts.append(sections[section_name])

    parts.append(sections["creator_engagement"])
    parts.append(sections["meta"])
    parts.append(f"\n## Transcript:\n<transcript>\n{transcript}\n</transcript>")

    return "\n".join(parts)


#%%
# =============================================================================
#MARK: TRANSCRIPT TEXT BUILDERS
# =============================================================================


def build_submission_transcript(row_data: dict, text_columns: list[str], config: Config) -> Optional[str]:
    """Build the transcript text from configured columns. Returns None if no valid content."""
    # Reddit submissions: ["title", "selftext"]
    if text_columns == ["title", "selftext"]:
        title = row_data.get("title", "") or ""
        if not is_valid_content(title):
            return None
        selftext = row_data.get("selftext", "") or ""
        selftext = truncate_text(selftext, config.max_selftext_words)
        full_text = f"Title: {title}"
        if selftext and is_valid_content(selftext):
            full_text += f"\n\nBody:\n{selftext}"
        return full_text

    # YouTube transcripts: ["title", "transcript_text"]
    if text_columns == ["title", "transcript_text"]:
        title = row_data.get("title", "") or ""
        transcript = row_data.get("transcript_text", "") or ""
        if not is_valid_content(transcript):
            return None
        full_text = f"Title: {title}" if is_valid_content(title) else ""
        if full_text:
            full_text += f"\n\nTranscript:\n{transcript}"
        else:
            full_text = transcript
        return full_text

    # Generic fallback: join all non-empty columns
    parts = []
    for col in text_columns:
        val = row_data.get(col, "") or ""
        if is_valid_content(val):
            parts.append(val)
    return "\n\n".join(parts) if parts else None


def build_comments_text(comments: list[dict]) -> str:
    """Build the comments portion for the comment batch prompt."""
    lines = []
    for i, comment in enumerate(comments, 1):
        body = comment.get("body", "") or ""
        comment_id = comment.get("id", f"unknown_{i}")
        lines.append(f"{i}. (id: {comment_id}) {body}")
    return "\n\n".join(lines)


def build_comment_batch_prompt(
    template: str,
    comments: list[dict],
    submission_title: str,
    submission_summary: str,
) -> str:
    """
    Build the full comment batch prompt by substituting placeholders.
    
    Expected placeholders in template:
    - {{POST_TITLE}}
    - {{POST_SUMMARY}}
    - {{COMMENTS}}
    """
    comments_text = build_comments_text(comments)
    
    prompt = template.replace("{{POST_TITLE}}", submission_title or "")
    prompt = prompt.replace("{{POST_SUMMARY}}", submission_summary or "")
    prompt = prompt.replace("{{COMMENTS}}", comments_text)
    
    return prompt


#%%
# =============================================================================
#MARK: ASYNC HTTP CLIENT FOR VLLM SERVER
# =============================================================================


async def make_chat_completion_request(
    session: aiohttp.ClientSession,
    messages: list[dict],
    config: Config,
    semaphore: asyncio.Semaphore,
    max_output_tokens: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Make a single chat completion request to the vLLM server.
    Returns (generated_text, error_message).
    max_output_tokens overrides config.max_output_tokens if provided.
    """
    url = f"{config.vllm_api_url}/chat/completions"

    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_output_tokens or config.max_output_tokens,
    }

    async with semaphore:
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=config.request_timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return None, f"HTTP {response.status}: {error_text[:500]}"

                data = await response.json()
                generated_text = data["choices"][0]["message"]["content"]
                return generated_text, None

        except asyncio.TimeoutError:
            return None, f"Request timeout after {config.request_timeout}s"
        except aiohttp.ClientError as e:
            return None, f"Client error: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"


#%%
# =============================================================================
#MARK: GENERIC ASYNC BATCH RUNNER
# =============================================================================


async def run_api_batch(
    items: list[tuple[int, list[dict]]],
    config: Config,
    max_output_tokens: Optional[int] = None,
) -> list[tuple[int, Optional[str], Optional[str]]]:
    """
    Run a batch of API calls concurrently.
    items: list of (idx, messages)
    Returns: list of (idx, response_text, error)
    """
    if not items:
        return []

    semaphore = asyncio.Semaphore(config.num_parallel_requests)
    connector = aiohttp.TCPConnector(
        limit=config.num_parallel_requests, limit_per_host=config.num_parallel_requests
    )

    async with aiohttp.ClientSession(connector=connector) as session:

        async def _single_call(idx: int, messages: list[dict]):
            text, err = await make_chat_completion_request(
                session, messages, config, semaphore, max_output_tokens
            )
            return idx, text, err

        tasks = [_single_call(idx, msgs) for idx, msgs in items]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, result in enumerate(gathered):
            if isinstance(result, Exception):
                idx = items[i][0]
                results.append((idx, None, f"Task exception: {str(result)}"))
            else:
                results.append(result)

        return results


#%%
# =============================================================================
#MARK: TWO-PASS SUBMISSION PROCESSING
# =============================================================================


@dataclass
class SubmissionResult:
    """Result of processing a single submission."""
    idx: int
    first_pass_response: Optional[str]  # Raw first-pass response for debugging
    llm_response: Optional[str]
    llm_disqualified_reason: Optional[str]
    llm_error: Optional[str]


async def process_submission_chunk_two_pass(
    chunk_df: pd.DataFrame,
    unprocessed_indices: list[int],
    first_pass_template: str,
    sections_cache: dict[str, str],
    config: Config,
    tiktoken_enc: tiktoken.Encoding,
    file_set: InputFileSet = None,
) -> list[SubmissionResult]:
    """
    Two-pass processing for a chunk of submissions or standalone text rows.

    Phase A: Pre-qualify all rows (sync) — skipped if file_set.skip_qualification
    Phase B: First pass — category detection for all qualified (concurrent API calls)
    Phase C: Build modular prompts + count tokens + group by token range (sync)
    Phase D: Second pass — full analysis per token range group (concurrent API calls)

    Returns list of SubmissionResult.
    """
    text_columns = file_set.text_columns if file_set else ["title", "selftext"]
    skip_qual = file_set.skip_qualification if file_set else False

    all_results: list[SubmissionResult] = []

    # ── Phase A: Pre-qualify ──────────────────────────────────────────────────
    qualified_items: list[tuple[int, str]] = []  # (idx, transcript_text)

    for idx in unprocessed_indices:
        row_data = chunk_df.loc[idx].to_dict()

        if not skip_qual:
            qualified, reason = check_submission_qualification(row_data, config)
            if not qualified:
                all_results.append(SubmissionResult(
                    idx=idx,
                    first_pass_response=None,
                    llm_response=None,
                    llm_disqualified_reason=reason,
                    llm_error=None,
                ))
                continue

        transcript = build_submission_transcript(row_data, text_columns, config)
        if transcript is None:
            all_results.append(SubmissionResult(
                idx=idx,
                first_pass_response=None,
                llm_response=None,
                llm_disqualified_reason="invalid_content",
                llm_error=None,
            ))
            continue

        qualified_items.append((idx, transcript))

    if not qualified_items:
        return all_results

    # ── Phase B: First pass — category detection (concurrent) ─────────────────
    first_pass_items = []
    for idx, transcript in qualified_items:
        user_prompt = first_pass_template.replace("{{TRANSCRIPT}}", transcript)
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        first_pass_items.append((idx, messages))

    first_pass_results = await run_api_batch(
        first_pass_items, config, max_output_tokens=config.first_pass_max_output_tokens
    )

    # Parse first-pass results into categories per idx
    # Store raw response for all, regardless of parse success
    idx_to_categories: dict[int, list[str]] = {}
    idx_to_transcript: dict[int, str] = {idx: t for idx, t in qualified_items}
    idx_to_first_pass_response: dict[int, Optional[str]] = {}

    for idx, response_text, api_error in first_pass_results:
        idx_to_first_pass_response[idx] = response_text  # Store raw response
        
        if api_error:
            all_results.append(SubmissionResult(
                idx=idx,
                first_pass_response=response_text,
                llm_response=None,
                llm_disqualified_reason=None,
                llm_error=f"first_pass_error: {api_error}",
            ))
            continue

        categories, parse_error = parse_categories_response(response_text)
        if parse_error:
            all_results.append(SubmissionResult(
                idx=idx,
                first_pass_response=response_text,
                llm_response=None,
                llm_disqualified_reason=None,
                llm_error=f"first_pass_error: {parse_error}",
            ))
            continue

        idx_to_categories[idx] = categories

    if not idx_to_categories:
        return all_results

    # ── Phase C: Build modular prompts + tiktoken count + group by range ──────
    range_groups: dict[int, list[tuple[int, list[dict]]]] = {}  # max_total → [(idx, messages)]

    for idx, categories in idx_to_categories.items():
        transcript = idx_to_transcript[idx]
        modular_prompt = build_modular_prompt(categories, transcript, sections_cache)
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": modular_prompt},
        ]
        token_count = count_message_tokens(messages, tiktoken_enc, config.tiktoken_buffer)
        max_total = get_max_total_model_tokens_for_input(token_count)

        if max_total not in range_groups:
            range_groups[max_total] = []
        range_groups[max_total].append((idx, messages))

    # ── Phase D: Second pass — full analysis per token range (concurrent) ─────
    for max_total in sorted(range_groups.keys()):
        group_items = range_groups[max_total]

        if not await ensure_model_loaded_for_range(config, max_total):
            for idx, _ in group_items:
                all_results.append(SubmissionResult(
                    idx=idx,
                    first_pass_response=idx_to_first_pass_response.get(idx),
                    llm_response=None,
                    llm_disqualified_reason=None,
                    llm_error=f"model_load_failed_for_range_{max_total}",
                ))
            continue

        second_pass_results = await run_api_batch(group_items, config)

        for idx, response_text, api_error in second_pass_results:
            if api_error:
                all_results.append(SubmissionResult(
                    idx=idx,
                    first_pass_response=idx_to_first_pass_response.get(idx),
                    llm_response=None,
                    llm_disqualified_reason=None,
                    llm_error=api_error,
                ))
            else:
                llm_response, parse_error = parse_model_output(response_text)
                all_results.append(SubmissionResult(
                    idx=idx,
                    first_pass_response=idx_to_first_pass_response.get(idx),
                    llm_response=llm_response,
                    llm_disqualified_reason=None,
                    llm_error=parse_error,
                ))

    return all_results


#%%
# =============================================================================
#MARK: COMMENT BATCH PROCESSING (SINGLE-PASS, LIGHTWEIGHT)
# =============================================================================


@dataclass
class CommentBatchTask:
    """All info needed to process one batch of comments via a single API call."""

    comment_indices: list[int]
    comment_data: list[dict]
    submission_title: str
    submission_summary: str


def prepare_comment_batches_from_chunk(
    chunk_df: pd.DataFrame,
    sub_lookup: dict[str, tuple[str, str]],
    config: Config,
) -> list[CommentBatchTask]:
    """Group a chunk of comments by submission and batch them."""
    groups_by_submission: dict[str, list[int]] = {}

    for idx in chunk_df.index:
        link_id = chunk_df.at[idx, "link_id"] if "link_id" in chunk_df.columns else ""
        if isinstance(link_id, str) and link_id.startswith("t3_"):
            sub_id = link_id[3:]
        else:
            sub_id = str(link_id)

        if sub_id not in groups_by_submission:
            groups_by_submission[sub_id] = []
        groups_by_submission[sub_id].append(idx)

    batches = []
    for sub_id, indices in groups_by_submission.items():
        title, summary = sub_lookup.get(sub_id, ("", ""))

        for batch_start in range(0, len(indices), config.comment_batch_size):
            batch_indices = indices[batch_start : batch_start + config.comment_batch_size]
            batch_data = [chunk_df.loc[idx].to_dict() for idx in batch_indices]
            batches.append(
                CommentBatchTask(
                    comment_indices=batch_indices,
                    comment_data=batch_data,
                    submission_title=title,
                    submission_summary=summary,
                )
            )

    return batches


def parse_comment_batch_response(
    response: str,
    valid_comments: list[dict],
) -> list[tuple[str, Optional[str], Optional[str]]]:
    """
    Parse batched comment response. Returns list of (comment_id, llm_response, llm_error).
    
    Handles multiple response formats:
    - Dict keyed by comment ID: {"abc123": {...}, "def456": {...}}
    - Dict keyed by numeric index (1-based): {"1": {...}, "2": {...}}
    - Dict keyed by string index: {"comment_1": {...}, "comment_2": {...}}
    - List in order: [{...}, {...}]
    """
    results = []

    parsed, parse_error = extract_json_from_response(response)

    if parse_error:
        for comment in valid_comments:
            results.append((comment["id"], None, f"batch_parse_error: {parse_error}"))
        return results

    if isinstance(parsed, dict):
        # Build lookup maps for different key formats
        id_to_comment = {comment["id"]: comment for comment in valid_comments}
        idx_to_comment = {str(i+1): comment for i, comment in enumerate(valid_comments)}
        idx_to_comment_zero = {str(i): comment for i, comment in enumerate(valid_comments)}
        
        # Track which comments we've matched
        matched_ids = set()
        
        for key, result_data in parsed.items():
            comment_id = None
            
            # Strategy 1: Key is exact comment ID
            if key in id_to_comment:
                comment_id = key
            # Strategy 2: Key is 1-based numeric index ("1", "2", etc.)
            elif key in idx_to_comment:
                comment_id = idx_to_comment[key]["id"]
            # Strategy 3: Key is 0-based numeric index ("0", "1", etc.)
            elif key in idx_to_comment_zero:
                comment_id = idx_to_comment_zero[key]["id"]
            # Strategy 4: Key is formatted like "comment_1", "comment_2", etc.
            elif key.startswith("comment_"):
                num = key.replace("comment_", "")
                if num in idx_to_comment:
                    comment_id = idx_to_comment[num]["id"]
                elif num in idx_to_comment_zero:
                    comment_id = idx_to_comment_zero[num]["id"]
            # Strategy 5: Key contains the comment ID somewhere
            else:
                for cid in id_to_comment:
                    if cid in key or key in cid:
                        comment_id = cid
                        break
            
            if comment_id and comment_id not in matched_ids:
                matched_ids.add(comment_id)
                if result_data is not None:
                    results.append((comment_id, json.dumps(result_data), None))
                else:
                    results.append((comment_id, None, "null_result_in_batch"))
        
        # Add errors for any comments we didn't match
        for comment in valid_comments:
            if comment["id"] not in matched_ids:
                results.append((comment["id"], None, "not_found_in_batch_response"))

    elif isinstance(parsed, list):
        for i, comment in enumerate(valid_comments):
            if i < len(parsed) and parsed[i] is not None:
                results.append((comment["id"], json.dumps(parsed[i]), None))
            elif i < len(parsed):
                results.append((comment["id"], None, "null_result_in_batch"))
            else:
                results.append((comment["id"], None, "missing_from_batch_response"))

    else:
        for comment in valid_comments:
            results.append((comment["id"], None, f"unexpected_response_format: {type(parsed).__name__}"))

    return results


async def process_single_comment_batch(
    session: aiohttp.ClientSession,
    batch: CommentBatchTask,
    comment_prompt_template: str,
    config: Config,
    semaphore: asyncio.Semaphore,
) -> list[tuple[int, Optional[str], Optional[str], Optional[str]]]:
    """
    Single-pass processing for one batch of comments.
    Uses the dedicated lightweight comment prompt.
    
    Returns list of (df_index, llm_response, llm_disqualified_reason, llm_error).
    """
    results = []
    valid_comments = []
    valid_indices = []

    # Pre-filter comments
    for df_idx, comment_data in zip(batch.comment_indices, batch.comment_data):
        body = comment_data.get("body", "")
        author = comment_data.get("author", "")

        if not is_valid_content(body):
            results.append((df_idx, None, "body_deleted_or_empty", None))
        elif is_bot_account(author):
            results.append((df_idx, None, "bot_account", None))
        else:
            valid_comments.append(comment_data)
            valid_indices.append(df_idx)

    if not valid_comments:
        return results

    # Build the prompt using the comment template
    prompt = build_comment_batch_prompt(
        template=comment_prompt_template,
        comments=valid_comments,
        submission_title=batch.submission_title,
        submission_summary=batch.submission_summary,
    )
    
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Single API call
    response_text, api_error = await make_chat_completion_request(
        session, messages, config, semaphore,
        max_output_tokens=config.comment_max_output_tokens,
    )

    if api_error:
        for df_idx in valid_indices:
            results.append((df_idx, None, None, api_error))
        return results

    # Parse batch response and map back to DataFrame indices
    id_to_idx = {cd["id"]: df_idx for df_idx, cd in zip(valid_indices, valid_comments)}
    parsed_results = parse_comment_batch_response(response_text, valid_comments)

    for comment_id, llm_response, llm_error in parsed_results:
        df_idx = id_to_idx[comment_id]
        results.append((df_idx, llm_response, None, llm_error))

    return results


async def process_comment_batches_async(
    batch_tasks: list[CommentBatchTask],
    comment_prompt_template: str,
    config: Config,
) -> list[tuple[int, Optional[str], Optional[str], Optional[str]]]:
    """Process multiple comment batches concurrently."""
    semaphore = asyncio.Semaphore(config.num_parallel_requests)
    connector = aiohttp.TCPConnector(
        limit=config.num_parallel_requests, limit_per_host=config.num_parallel_requests
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_single_comment_batch(
                session=session,
                batch=batch,
                comment_prompt_template=comment_prompt_template,
                config=config,
                semaphore=semaphore,
            )
            for batch in batch_tasks
        ]

        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for i, result in enumerate(gathered):
            if isinstance(result, Exception):
                batch = batch_tasks[i]
                for df_idx in batch.comment_indices:
                    all_results.append((df_idx, None, None, f"Task exception: {str(result)}"))
            else:
                all_results.extend(result)

        return all_results


#%%
# =============================================================================
#MARK: STREAMING SUBMISSION PROCESSOR
# =============================================================================


async def process_submissions_streaming(
    file_set: InputFileSet,
    config: Config,
) -> dict[str, tuple[str, str]]:
    """
    Stream-process a primary file chunk by chunk with two-pass modular prompts.

    Works with both JSONL (reddit submissions) and parquet (transcripts) inputs.
    Memory profile: ~1 chunk DataFrame + sub_lookup dict + processed_ids set.
    Returns sub_lookup: {row_id: (title, summary)} for comment processing (empty for single files).
    """
    submissions_path = file_set.submissions
    id_col = file_set.id_column
    sidecar_path = get_sidecar_path(submissions_path, file_set.output_dir)

    # --- Resume support ---
    existing_sidecar = load_sidecar(sidecar_path)

    # Choose ID loading strategy based on reprocess_errors setting
    if config.reprocess_errors:
        processed_ids = load_processed_ids_excluding_errors(sidecar_path)
        console.print(f"[yellow]Reprocess errors mode: will retry rows with llm_error[/yellow]")
    else:
        processed_ids = set(existing_sidecar.keys())

    # Build sub_lookup from existing sidecar (only needed for file pairs)
    sidecar_summaries: dict[str, str] = {}
    if not file_set.is_single_file:
        for sub_id, data in existing_sidecar.items():
            summary = get_submission_summary(data.get("llm_response"))
            if summary:
                sidecar_summaries[sub_id] = summary
    del existing_sidecar

    console.print(f"[blue]Streaming from {submissions_path.name}...[/blue]")
    console.print(f"[dim]  Already processed: {len(processed_ids):,}[/dim]")
    console.print(f"[dim]  Chunk size: {config.batch_size} rows[/dim]")

    sub_lookup: dict[str, tuple[str, str]] = {}

    # Load prompt assets (cached for all chunks)
    first_pass_template = load_prompt_template(config.first_pass_prompt_path)
    sections_cache = load_prompt_sections(config.modular_prompt_components_path)
    tiktoken_enc = get_tiktoken_encoding(config)

    # Pre-count rows and compute unprocessed chunk count for tqdm
    total_rows = count_rows(file_set)
    total_chunks = math.ceil(total_rows / config.batch_size)
    unprocessed_rows = total_rows - len(processed_ids)
    unprocessed_chunks = math.ceil(unprocessed_rows / config.batch_size) if unprocessed_rows > 0 else 0
    console.print(f"[dim]  Total rows: {total_rows:,} ({total_chunks} chunks)[/dim]")
    console.print(f"[dim]  Unprocessed: ~{unprocessed_rows:,} rows ({unprocessed_chunks} chunks)[/dim]")

    total_processed = 0
    total_skipped = 0

    # Choose reader based on input format
    if file_set.input_format == "parquet":
        reader = iter_parquet_chunks(submissions_path, config.batch_size)
    else:
        reader = pd.read_json(submissions_path, lines=True, chunksize=config.batch_size)

    chunks_with_work = 0
    for chunk_df in tqdm(reader, desc="Chunks", total=total_chunks):
        # --- Capture titles for ALL rows (needed for comment processing in file pairs) ---
        if not file_set.is_single_file:
            for idx in chunk_df.index:
                sub_id = str(chunk_df.at[idx, id_col])
                title = str(chunk_df.at[idx, "title"]) if "title" in chunk_df.columns else ""
                title = title if title and title != "nan" else ""
                summary = sidecar_summaries.get(sub_id, "")
                sub_lookup[sub_id] = (title, summary)

        # --- Filter to unprocessed rows ---
        unprocessed_mask = ~chunk_df[id_col].astype(str).isin(processed_ids)
        unprocessed_indices = chunk_df[unprocessed_mask].index.tolist()
        total_skipped += int((~unprocessed_mask).sum())

        if not unprocessed_indices:
            del chunk_df
            continue

        chunks_with_work += 1

        # Respect row_limit
        if config.row_limit is not None:
            remaining = config.row_limit - total_processed
            if remaining <= 0:
                del chunk_df
                break
            unprocessed_indices = unprocessed_indices[:remaining]

        # --- Two-pass processing (handles model loading per token range internally) ---
        results = await process_submission_chunk_two_pass(
            chunk_df=chunk_df,
            unprocessed_indices=unprocessed_indices,
            first_pass_template=first_pass_template,
            sections_cache=sections_cache,
            config=config,
            tiktoken_enc=tiktoken_enc,
            file_set=file_set,
        )

        # --- Build sidecar entries and update lookup ---
        sidecar_entries = []
        timestamp = datetime.now().isoformat()

        for result in results:
            row_id = str(chunk_df.at[result.idx, id_col])
            sidecar_entries.append({
                "id": row_id,  # sidecar always uses "id" as key
                "llm_first_pass_response": result.first_pass_response,
                "llm_response": result.llm_response,
                "llm_disqualified_reason": result.llm_disqualified_reason,
                "llm_error": result.llm_error,
                "llm_processed_at": timestamp,
            })
            if not file_set.is_single_file and result.llm_response:
                summary = get_submission_summary(result.llm_response)
                sub_lookup[row_id] = (sub_lookup.get(row_id, ("", ""))[0], summary)

        append_to_sidecar(sidecar_entries, sidecar_path)
        processed_ids.update(e["id"] for e in sidecar_entries)
        total_processed += len(sidecar_entries)

        del chunk_df

    del sidecar_summaries

    console.print(
        f"[green]Processing complete: {total_processed:,} processed, {total_skipped:,} skipped[/green]"
    )
    return sub_lookup


#%%
# =============================================================================
#MARK: STREAMING COMMENT PROCESSOR
# =============================================================================


async def process_comments_streaming(
    comments_path: Path,
    sub_lookup: dict[str, tuple[str, str]],
    config: Config,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Stream-process a comments JSONL file chunk by chunk.
    Uses dedicated lightweight comment prompt (single-pass, batched).

    Memory profile: ~1 chunk DataFrame + sub_lookup (passed in) + processed_ids set.
    """
    sidecar_path = get_sidecar_path(comments_path, output_dir)

    # Choose ID loading strategy based on reprocess_errors setting
    if config.reprocess_errors:
        processed_ids = load_processed_ids_excluding_errors(sidecar_path)
        console.print(f"[yellow]Reprocess errors mode: will retry rows with llm_error[/yellow]")
    else:
        processed_ids = load_processed_ids(sidecar_path)

    console.print(f"[blue]Streaming comments from {comments_path.name}...[/blue]")
    console.print(f"[dim]  Already processed: {len(processed_ids):,}[/dim]")
    console.print(f"[dim]  Chunk size: {config.comment_chunk_size:,} rows[/dim]")

    # Load comment prompt template
    comment_prompt_template = load_prompt_template(config.comment_prompt_path)
    console.print(f"[dim]  Using comment prompt: {config.comment_prompt_path}[/dim]")

    # Pre-count lines for tqdm
    total_lines = count_jsonl_lines(comments_path)
    total_chunks = math.ceil(total_lines / config.comment_chunk_size)
    unprocessed_rows = total_lines - len(processed_ids)
    unprocessed_chunks = math.ceil(unprocessed_rows / config.comment_chunk_size) if unprocessed_rows > 0 else 0
    console.print(f"[dim]  Total rows: {total_lines:,} ({total_chunks} chunks)[/dim]")
    console.print(f"[dim]  Unprocessed: ~{unprocessed_rows:,} rows ({unprocessed_chunks} chunks)[/dim]")

    # Use smallest token range for comments (batches are short)
    max_total_model_tokens = TOKEN_RANGES[0].max_total_model_tokens

    if not await ensure_model_loaded_for_range(config, max_total_model_tokens):
        console.print("[red]Failed to load model — aborting comments[/red]")
        return False

    total_processed = 0
    total_skipped = 0

    reader = pd.read_json(comments_path, lines=True, chunksize=config.comment_chunk_size)

    for chunk_df in tqdm(reader, desc="Comment chunks", total=total_chunks):
        # --- Filter to unprocessed rows ---
        unprocessed_mask = ~chunk_df["id"].astype(str).isin(processed_ids)
        unprocessed_df = chunk_df[unprocessed_mask]
        total_skipped += int((~unprocessed_mask).sum())

        if unprocessed_df.empty:
            del chunk_df, unprocessed_df
            continue

        # Respect row_limit
        if config.row_limit is not None:
            remaining = config.row_limit - total_processed
            if remaining <= 0:
                del chunk_df, unprocessed_df
                break
            if len(unprocessed_df) > remaining:
                unprocessed_df = unprocessed_df.head(remaining)

        # Prepare batches
        batches = prepare_comment_batches_from_chunk(unprocessed_df, sub_lookup, config)

        if not batches:
            del chunk_df, unprocessed_df
            continue

        # Process with single-pass comment prompt
        results = await process_comment_batches_async(
            batch_tasks=batches,
            comment_prompt_template=comment_prompt_template,
            config=config,
        )

        # Build sidecar entries
        sidecar_entries = []
        timestamp = datetime.now().isoformat()

        for df_idx, llm_response, llm_disqualified, llm_error in results:
            comment_id = str(unprocessed_df.at[df_idx, "id"])
            sidecar_entries.append({
                "id": comment_id,
                "llm_response": llm_response,
                "llm_disqualified_reason": llm_disqualified,
                "llm_error": llm_error,
                "llm_processed_at": timestamp,
            })

        append_to_sidecar(sidecar_entries, sidecar_path)
        processed_ids.update(e["id"] for e in sidecar_entries)
        total_processed += len(sidecar_entries)

        del chunk_df, unprocessed_df, batches

    console.print(
        f"[green]Comments complete: {total_processed:,} processed, {total_skipped:,} skipped[/green]"
    )
    return True


#%%
# =============================================================================
#MARK: FINAL PARQUET ASSEMBLY (STREAMING MERGE)
# =============================================================================


def assemble_final_parquet(
    file_set: InputFileSet,
    source_path: Path,
    sidecar_path: Path,
    output_path: Path,
    config: Config,
    llm_columns: list[str],
) -> bool:
    """
    Stream-merge original data with sidecar LLM results into final parquet.
    Handles both JSONL and parquet source formats.
    Uses pyarrow incremental writer — only one chunk in memory at a time.
    """
    if not sidecar_path.exists():
        console.print(f"[yellow]No sidecar found for {source_path.name} — skipping assembly[/yellow]")
        return False

    console.print(f"[blue]Assembling parquet: {output_path.name}...[/blue]")

    sidecar_data = load_sidecar(sidecar_path)
    console.print(f"[dim]  Sidecar entries: {len(sidecar_data):,}[/dim]")

    # Determine which column in the source maps to the sidecar "id" key
    id_col = file_set.id_column

    writer = None
    total_rows = 0

    try:
        # Choose reader based on source format
        if file_set.input_format == "parquet":
            reader = iter_parquet_chunks(source_path, config.assembly_chunk_size)
        else:
            reader = pd.read_json(source_path, lines=True, chunksize=config.assembly_chunk_size)

        for chunk in reader:
            for col in llm_columns:
                chunk[col] = None

            for idx in chunk.index:
                row_id = str(chunk.at[idx, id_col])
                if row_id in sidecar_data:
                    result = sidecar_data[row_id]
                    for col in llm_columns:
                        chunk.at[idx, col] = result.get(col)

            for col in llm_columns:
                chunk[col] = chunk[col].astype(object)

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression="zstd",
                    compression_level=config.final_compression_level,
                )

            writer.write_table(table)
            total_rows += len(chunk)
            del chunk

    except Exception as e:
        console.print(f"[red]Assembly error for {output_path.name}: {e}[/red]")
        return False
    finally:
        if writer:
            writer.close()

    del sidecar_data

    console.print(
        f"[green]Parquet saved: {output_path.name} ({total_rows:,} rows, zstd-{config.final_compression_level})[/green]"
    )
    return True


#%%
# =============================================================================
#MARK: SUBREDDIT FILE PAIR PROCESSOR
# =============================================================================


async def process_file_set(file_set: InputFileSet, config: Config) -> dict[str, bool]:
    """
    Process a single input file set: primary file → optional comments → assemble parquets.
    Memory-safe: each phase streams data and frees it before the next.
    """
    results = {}

    # --- Phase 1: Stream-process primary file (submissions or standalone) ---
    console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
    console.print(f"[bold magenta]Phase 1: Primary — {file_set.submissions.name}[/bold magenta]")
    console.print(f"[bold magenta]{'='*60}[/bold magenta]")

    sub_lookup = await process_submissions_streaming(file_set, config)
    results[file_set.submissions.name] = True

    # --- Phase 2: Stream-process comments (only for file pairs) ---
    if not file_set.is_single_file:
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]Phase 2: Comments — {file_set.comments.name}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        comments_success = await process_comments_streaming(file_set.comments, sub_lookup, config, file_set.output_dir)
        results[file_set.comments.name] = comments_success

    del sub_lookup

    # --- Phase 3: Assemble final parquet files ---
    console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
    console.print(f"[bold magenta]Phase 3: Parquet Assembly — {file_set.name}[/bold magenta]")
    console.print(f"[bold magenta]{'='*60}[/bold magenta]")

    # Primary file uses SUBMISSION_LLM_COLUMNS (includes first_pass_response)
    sub_output = get_output_path(file_set.submissions, file_set.output_dir)
    sub_sidecar = get_sidecar_path(file_set.submissions, file_set.output_dir)
    if not assemble_final_parquet(
        file_set, file_set.submissions, sub_sidecar, sub_output, config, SUBMISSION_LLM_COLUMNS
    ):
        results[file_set.submissions.name] = False

    # Comments don't have first_pass_response
    if not file_set.is_single_file:
        com_output = get_output_path(file_set.comments, file_set.output_dir)
        com_sidecar = get_sidecar_path(file_set.comments, file_set.output_dir)
        if not assemble_final_parquet(
            file_set, file_set.comments, com_sidecar, com_output, config, COMMENT_LLM_COLUMNS
        ):
            results[file_set.comments.name] = False

    return results


#%%
# =============================================================================
#MARK: MANAGER & SERVER OPERATIONS
# =============================================================================


async def check_manager_status(config: Config) -> Optional[dict]:
    """Check the manager status to see if a model is loaded."""
    url = f"{config.manager_url}/manager/status"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    console.print(f"[red]Manager returned status {response.status}[/red]")
                    return None
    except Exception as e:
        console.print(f"[red]Cannot reach manager at {config.manager_url}: {e}[/red]")
        return None


async def load_model_via_manager(config: Config, max_total_model_tokens: int) -> bool:
    """Request the manager to load the specified model with given max_total_model_tokens."""
    global _current_max_total_model_tokens

    url = f"{config.manager_url}/manager/load"
    payload = {
        "model_name": config.model_name,
        "gpu_utilization": config.gpu_memory_utilization,
        "max_model_len": max_total_model_tokens,
    }

    console.print(f"[blue]Requesting manager to load model: {config.model_name}[/blue]")
    console.print(f"[dim]  GPU Utilization: {config.gpu_memory_utilization}[/dim]")
    console.print(f"[dim]  Max Total Model Tokens: {max_total_model_tokens:,}[/dim]")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=config.model_load_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    console.print("[green]Model loaded successfully![/green]")
                    console.print(f"[dim]  vLLM URL: {data.get('url', 'N/A')}[/dim]")
                    _current_max_total_model_tokens = max_total_model_tokens
                    return True
                else:
                    error_text = await response.text()
                    console.print(f"[red]Failed to load model: {error_text}[/red]")
                    return False
    except asyncio.TimeoutError:
        console.print(f"[red]Model loading timed out after {config.model_load_timeout}s[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return False


async def check_vllm_server_health(config: Config) -> bool:
    """Check if the vLLM server is reachable and healthy."""
    url = f"{config.vllm_api_url}/models"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("id") for m in data.get("data", [])]
                    console.print(f"[green]vLLM server healthy. Available models: {models}[/green]")
                    return True
                else:
                    console.print(f"[red]vLLM server returned status {response.status}[/red]")
                    return False
    except Exception as e:
        console.print(f"[red]Cannot reach vLLM server at {config.vllm_api_url}: {e}[/red]")
        return False


async def ensure_model_loaded(config: Config, max_total_model_tokens: int) -> bool:
    """
    Ensure the correct model is loaded via the manager with the specified max_total_model_tokens.

    Workflow:
    1. Check manager status
    2. If model not loaded, different model, or different max_total_model_tokens -> reload
    3. Verify vLLM server is healthy
    """
    global _current_max_total_model_tokens

    console.print("\n[bold]Checking manager status...[/bold]")

    status = await check_manager_status(config)
    if status is None:
        console.print("[red]Cannot connect to manager. Is it running?[/red]")
        return False

    current_model = status.get("model")
    server_status = status.get("status")

    console.print(f"[dim]  Manager status: {server_status}[/dim]")
    console.print(f"[dim]  Current model: {current_model or 'None'}[/dim]")
    console.print(f"[dim]  Current max_total_model_tokens: {_current_max_total_model_tokens or 'Unknown'}[/dim]")
    console.print(f"[dim]  Required max_total_model_tokens: {max_total_model_tokens:,}[/dim]")

    needs_reload = False

    if current_model != config.model_name:
        if current_model:
            console.print(f"[yellow]Model mismatch: {current_model} != {config.model_name}[/yellow]")
        else:
            console.print("[yellow]No model currently loaded[/yellow]")
        needs_reload = True
    elif _current_max_total_model_tokens is not None and _current_max_total_model_tokens != max_total_model_tokens:
        console.print(
            f"[yellow]max_total_model_tokens change: {_current_max_total_model_tokens:,} -> {max_total_model_tokens:,}[/yellow]"
        )
        needs_reload = True

    if needs_reload:
        if not await load_model_via_manager(config, max_total_model_tokens):
            return False
    else:
        console.print(f"[green]Model {config.model_name} already loaded with compatible config[/green]")
        _current_max_total_model_tokens = max_total_model_tokens

    console.print("\n[bold]Verifying vLLM server health...[/bold]")
    return await check_vllm_server_health(config)


async def ensure_model_loaded_for_range(config: Config, max_total_model_tokens: int) -> bool:
    """Ensure model is loaded with appropriate max_total_model_tokens. Only reloads if different."""
    global _current_max_total_model_tokens

    if _current_max_total_model_tokens == max_total_model_tokens:
        return True

    return await ensure_model_loaded(config, max_total_model_tokens)


def verify_server_connection(config: Config, max_total_model_tokens: Optional[int] = None) -> bool:
    """Sync wrapper for ensuring model is loaded and server is ready."""
    if max_total_model_tokens is None:
        max_total_model_tokens = config.default_max_total_model_tokens
    return asyncio.run(ensure_model_loaded(config, max_total_model_tokens))


#%%
# =============================================================================
#MARK: MAIN PIPELINE
# =============================================================================


def get_execution_scope(config: Config) -> tuple[list[InputFileSet], Optional[int]]:
    if config.test_mode:
        all_pairs = config.get_target_file_pairs()
        if not all_pairs:
            console.print("[yellow]No file sets found for testing.[/yellow]")
            return [], 1
        console.print(f"[yellow]TEST MODE: Processing {all_pairs[0].name} for 1 row[/yellow]")
        return [all_pairs[0]], 1

    return config.get_target_file_pairs(), config.row_limit


def run_pipeline(config: Config) -> None:
    """Main pipeline execution."""
    console.print("\n[bold]vLLM Configuration:[/bold]")
    console.print(f"  Manager URL: {config.manager_url}")
    console.print(f"  vLLM API URL: {config.vllm_api_url}")
    console.print(f"  Model: {config.model_name}")
    console.print(f"  GPU Memory Utilization: {config.gpu_memory_utilization}")
    console.print(f"  Max Output Tokens: {config.max_output_tokens:,}")
    console.print(f"  First Pass Max Output Tokens: {config.first_pass_max_output_tokens}")
    console.print(f"  Comment Max Output Tokens: {config.comment_max_output_tokens}")
    console.print(f"  Parallel Requests: {config.num_parallel_requests}")
    console.print(f"  Comment Batch Size: {config.comment_batch_size}")
    console.print(f"  Reprocess Errors: {config.reprocess_errors}")

    console.print("\n[bold]Prompt System:[/bold]")
    console.print(f"  Primary files: Two-pass modular system")
    console.print(f"    First pass: {config.first_pass_prompt_path}")
    console.print(f"    Modular components: {config.modular_prompt_components_path}")
    console.print(f"  Comments: Single-pass lightweight prompt")
    console.print(f"    Prompt: {config.comment_prompt_path}")
    console.print(f"  Tiktoken encoding: {config.tiktoken_encoding_name} (buffer: {config.tiktoken_buffer}x)")

    console.print("\n[bold]Streaming Configuration:[/bold]")
    console.print(f"  Primary chunk size: {config.batch_size} rows")
    console.print(f"  Comment chunk size: {config.comment_chunk_size:,} rows")
    console.print(f"  Assembly chunk size: {config.assembly_chunk_size:,} rows")

    console.print("\n[bold]Token Ranges (primary files):[/bold]")
    for tr in TOKEN_RANGES:
        max_display = f"{tr.max_input_tokens:,}" if tr.max_input_tokens != float("inf") else "∞"
        console.print(
            f"  {tr.name}: {tr.min_input_tokens:,} - {max_display} input tokens → max_total={tr.max_total_model_tokens:,}"
        )

    console.print("\n[bold]Output Columns:[/bold]")
    console.print(f"  Primary: {SUBMISSION_LLM_COLUMNS}")
    console.print(f"  Comments: {COMMENT_LLM_COLUMNS}")

    # Initial model load with smallest context
    initial_max_total = TOKEN_RANGES[0].max_total_model_tokens
    console.print(f"\n[dim]Initial load with smallest context: {initial_max_total:,}[/dim]")

    if not verify_server_connection(config, initial_max_total):
        console.print("[bold red]Failed to initialize vLLM server. Aborting.[/bold red]")
        return

    # Get execution scope
    target_sets, row_limit = get_execution_scope(config)
    config.row_limit = row_limit

    if not target_sets:
        console.print("[yellow]No file sets to process.[/yellow]")
        return

    console.print(f"\n[bold]File sets to process: {len(target_sets)}[/bold]")
    for fs in target_sets:
        if fs.is_single_file:
            console.print(f"  {fs.name}: {fs.submissions.name} (standalone, {fs.input_format})")
        else:
            console.print(f"  {fs.name}: {fs.submissions.name} + {fs.comments.name}")

    # Process each file set
    all_results: dict[str, bool] = {}
    for file_set in target_sets:
        set_results = asyncio.run(process_file_set(file_set, config))
        all_results.update(set_results)

    # Summary
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print("[bold green]Processing Complete![/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print("\n[bold]Results Summary:[/bold]")
    for filename, success in all_results.items():
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {status} {filename}")


#%%
# =============================================================================
#MARK: UTILITY: ANALYSIS FUNCTIONS
# =============================================================================


def analyze_output(output_path: Path, is_submission: bool = True) -> None:
    """Analyze LLM processing results in an output parquet file."""
    if not output_path.exists():
        console.print(f"[yellow]File not found: {output_path}[/yellow]")
        return

    columns = ["llm_response", "llm_disqualified_reason", "llm_error"]
    if is_submission:
        columns.append("llm_first_pass_response")
    
    df = pd.read_parquet(output_path, columns=columns)

    total = len(df)
    has_response = df["llm_response"].notna().sum()
    has_disqualification = df["llm_disqualified_reason"].notna().sum()
    has_error = df["llm_error"].notna().sum()
    unprocessed = total - has_response - has_disqualification

    console.print(f"\n[bold]Analysis: {output_path.name}[/bold]")
    console.print(f"  Total rows: {total:,}")
    console.print(f"  Successful responses: {has_response:,}")
    console.print(f"  Disqualified: {has_disqualification:,}")
    console.print(f"  Errors: {has_error:,}")
    console.print(f"  Unprocessed: {unprocessed:,}")

    if is_submission and "llm_first_pass_response" in df.columns:
        has_first_pass = df["llm_first_pass_response"].notna().sum()
        console.print(f"  Has first pass response: {has_first_pass:,}")

    if has_disqualification > 0:
        console.print("\n[bold]Disqualification reasons:[/bold]")
        reasons = df["llm_disqualified_reason"].dropna().value_counts()
        for reason, count in reasons.items():
            console.print(f"  {reason}: {count:,}")

    if has_error > 0:
        console.print("\n[bold]Error types (top 10):[/bold]")
        errors = df["llm_error"].dropna().value_counts().head(10)
        for error, count in errors.items():
            console.print(f"  {error[:80]}: {count:,}")


def analyze_first_pass_responses(output_path: Path, sample_size: int = 10) -> None:
    """Analyze first-pass responses for debugging category detection issues."""
    if not output_path.exists():
        console.print(f"[yellow]File not found: {output_path}[/yellow]")
        return

    df = pd.read_parquet(output_path, columns=["id", "llm_first_pass_response", "llm_error"])

    console.print(f"\n[bold]First-Pass Analysis: {output_path.name}[/bold]")

    # Show samples of first-pass responses that led to errors
    errored = df[df["llm_error"].notna() & df["llm_error"].str.contains("first_pass", na=False)]
    if not errored.empty:
        console.print(f"\n[bold]Sample first-pass responses with errors ({min(sample_size, len(errored))}):[/bold]")
        for _, row in errored.head(sample_size).iterrows():
            console.print(f"\n  ID: {row['id']}")
            console.print(f"  Error: {row['llm_error']}")
            fp_response = row.get('llm_first_pass_response', '')
            if fp_response:
                console.print(f"  First pass response: {fp_response[:500]}...")

    # Show samples of successful first-pass responses
    successful = df[df["llm_first_pass_response"].notna() & df["llm_error"].isna()]
    if not successful.empty:
        console.print(f"\n[bold]Sample successful first-pass responses ({min(sample_size, len(successful))}):[/bold]")
        for _, row in successful.head(sample_size).iterrows():
            console.print(f"\n  ID: {row['id']}")
            fp_response = row.get('llm_first_pass_response', '')
            if fp_response:
                console.print(f"  First pass response: {fp_response[:300]}")


def validate_responses(output_path: Path) -> None:
    """Validate and sample model responses in an output parquet file."""
    if not output_path.exists():
        console.print(f"[yellow]File not found: {output_path}[/yellow]")
        return

    df = pd.read_parquet(output_path, columns=["id", "llm_response", "llm_error"])

    console.print(f"\n[bold]Validation: {output_path.name}[/bold]")

    successful = df[df["llm_response"].notna()]
    if not successful.empty:
        console.print(f"\n[bold]Sample responses ({min(3, len(successful))}):[/bold]")
        for _, row in successful.head(3).iterrows():
            response_preview = str(row["llm_response"])[:200]
            console.print(f"  ID {row.get('id', '?')}: {response_preview}...")

    errored = df[df["llm_error"].notna()]
    if not errored.empty:
        console.print(f"\n[bold]Sample errors ({min(3, len(errored))}):[/bold]")
        for _, row in errored.head(3).iterrows():
            console.print(f"  ID {row.get('id', '?')}: {row['llm_error']}")


#%%
# =============================================================================
#MARK: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Modify CONFIG here for different runs
    # Example overrides:
    # CONFIG.test_mode = True
    # CONFIG.num_parallel_requests = 16
    # CONFIG.names_to_process = ["DestinyTheGame"]
    # CONFIG.row_limit = 100
    # CONFIG.reprocess_errors = True  # <-- Set this to retry rows with errors

    run_pipeline(CONFIG)


#%%
# =============================================================================
#MARK: OPTIONAL: RUN ANALYSIS
# =============================================================================

# Uncomment to run analysis on specific output files:
# analyze_output(Path("reddit_posts/destiny2/processed/DestinyTheGame_2017-2024_submissions_filtered_20_llm.parquet"), is_submission=True)
# analyze_first_pass_responses(Path("reddit_posts/destiny2/processed/DestinyTheGame_2017-2024_submissions_filtered_20_llm.parquet"))
# validate_responses(Path("reddit_posts/destiny2/processed/DestinyTheGame_2017-2024_submissions_filtered_20_llm.parquet"))