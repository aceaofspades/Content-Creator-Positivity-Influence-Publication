#%%
#!/usr/bin/env python3
"""
6_data_validation.py
====================
LLM pipeline validation: manages the human-LLM agreement test set lifecycle.

Phases (detected automatically):
  1. No test_set.csv → stratified sample saved to validation/, print labeling instructions
  2. test_set.csv exists, not fully labeled → print progress
  3. test_set.csv fully labeled (all human_validated=True) → compute and report agreement stats

Extending an existing set (run once per batch):
  python 6_data_validation.py --extend
  Appends N_PER_TYPE new rows per content type, sampled with the same stratified-by-year
  procedure but a new seed (EXTEND_SEED) and excluding every id already in the file.
  New rows get empty human__ columns and human_validated=False -- they are NOT graded.
  Existing rows are untouched except for the added `sample_batch` column (1 = original
  May-2026 batch). Until the new rows are graded, phase 3 will not run, so the existing
  validation_report.csv stays as it is.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

BASE_PATH = Path(__file__).parent
DATA_DIR  = BASE_PATH / 'analysis' / 'destiny2' / 'data'
VALID_DIR = BASE_PATH / 'analysis' / 'destiny2' / 'validation'
TEST_SET  = VALID_DIR / 'test_set.csv'
REPORT    = VALID_DIR / 'validation_report.csv'

GAME         = 'destiny2'
N_PER_TYPE   = 25   # 25 submissions + 25 comments + 25 creators = 75 total
RANDOM_SEED  = 42
EXTEND_SEED  = 43   # seed for the --extend batch (batch 2); bump for any later batch

# Raw LLM JSON column (same for all sources in current pipeline)
LLM_COL = 'llm_response'

# Allowed categorical values (matches modular_prompt_categories.txt)
# emotional_tone agreement is computed at group level, not term level.
# 'satisfied' vs 'hopeful' is not held against the model — only polarity matters.
EMOTIONAL_TONE_GROUPS = {
    'hopeful': 'positive', 'enthusiastic': 'positive', 'satisfied': 'positive',
    'optimistic': 'positive', 'impressed': 'positive',
    'indifferent': 'neutral', 'conflicted': 'neutral', 'analytical': 'neutral',
    'measured': 'neutral', 'nostalgic': 'neutral',
    'frustrated': 'negative', 'angry': 'negative', 'resigned': 'negative',
    'disappointed': 'negative', 'bitter': 'negative',
    # accept group names directly from the human labeler
    'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative',
}

ALLOWED = {
    'emotional_tone': [
        'hopeful', 'enthusiastic', 'satisfied', 'optimistic', 'impressed',
        'indifferent', 'conflicted', 'analytical', 'measured', 'nostalgic',
        'frustrated', 'angry', 'resigned', 'disappointed', 'bitter', 'unclear',
    ],
    'game_relevance':  ['fully_relevant', 'mostly_relevant', 'partially_relevant', 'not_relevant'],
    'player_status':   ['active', 'inactive', 'NA'],
    'still_playing':   ['yes', 'no', 'barely', 'taking_a_break', 'returning', 'unclear'],
    'burned_out':      ['yes', 'no', 'partially', 'recovering', 'unclear'],
    'would_recommend': ['yes', 'no', 'conditional', 'unclear'],
}

# =============================================================================
# HELPERS
# =============================================================================

def find_text_column(df: pd.DataFrame, content_type: str) -> str | None:
    """Return the best text column name for the given content type."""
    candidates = {
        'submission': ['selftext', 'body', 'text', 'content', 'title'],
        'comment':    ['body', 'selftext', 'text', 'content'],
        'creator':    ['transcript', 'combined_text', 'text', 'body', 'description', 'title'],
    }
    for col in candidates.get(content_type, []):
        if col in df.columns:
            return col
    return None


def extract_creator_engagement_raw(raw_json) -> dict:
    """Pull categorical creator engagement strings from raw LLM JSON.

    Returns dict with keys: still_playing, burned_out, would_recommend.
    Values are raw strings from the LLM (e.g. "yes", "no", "partially") or None.
    """
    try:
        parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        if not isinstance(parsed, dict):
            return {}
        ce = parsed.get('creator_engagement', {})
        if not isinstance(ce, dict):
            return {}

        def _norm(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            s = str(v).strip()
            return None if s.lower() in {'unclear', 'na', 'n/a', ''} else s

        return {
            'still_playing':   _norm(ce.get('are_they_still_playing')),
            'burned_out':      _norm(ce.get('are_they_burned_out')),
            'would_recommend': _norm(ce.get('would_they_recommend_the_game')),
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


def stratified_year_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Sample n rows stratified equally across year buckets."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    df['_year'] = df['date'].dt.year
    years = sorted(df['_year'].unique())

    if not years:
        return df.drop(columns=['_year']).sample(min(n, len(df)), random_state=seed)

    n_per_year = max(1, n // len(years))
    parts = []
    for year in years:
        bucket = df[df['_year'] == year]
        take = min(n_per_year, len(bucket))
        parts.append(bucket.sample(take, random_state=seed))

    sampled = pd.concat(parts).drop(columns=['_year'])

    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    elif len(sampled) < n:
        already = set(sampled.index)
        remaining = df[~df.index.isin(already)].drop(columns=['_year'])
        shortfall = n - len(sampled)
        if len(remaining) >= shortfall:
            sampled = pd.concat([sampled, remaining.sample(shortfall, random_state=seed)])

    return sampled.reset_index(drop=True)


def _empty_row(content_type: str) -> dict:
    """Return a fully-populated row dict with all columns present and null."""
    return {
        'id':           None,
        'content_type': content_type,
        'date':         None,
        'game':         GAME,
        'title':        None,
        'text':         None,
        'llm__sentiment_score': None,
        'llm__emotional_tone':  None,
        'llm__optimism':        None,
        'llm__game_of_focus':   None,
        'llm__still_playing':   None,
        'llm__burned_out':      None,
        'llm__would_recommend': None,
        'human__sentiment_score': None,
        'human__emotional_tone':  None,
        'human__optimism':        None,
        'human__still_playing':   None,
        'human__burned_out':      None,
        'human__would_recommend': None,
        'human_validated': False,
    }


# =============================================================================
# PHASE 1 — BUILD TEST SET
# =============================================================================

def _load_and_filter(filename: str, score_col: str, require_game_relevance: bool,
                     source_type: str | None = None) -> pd.DataFrame:
    path = DATA_DIR / filename
    print(f"  Loading {path.name}...")
    df = pd.read_parquet(path)

    if source_type and 'source_type' in df.columns:
        df = df[df['source_type'] == source_type]
    if 'analysis_eligible' in df.columns:
        df = df[df['analysis_eligible'] == True]
    if require_game_relevance and 'meta__game_relevance' in df.columns:
        df = df[df['meta__game_relevance'] != 'not_relevant']
    if score_col in df.columns:
        df = df[df[score_col].notna()]

    print(f"    Eligible rows: {len(df):,}")
    return df


def build_submission_title_lookup() -> dict[str, str]:
    """Build a submission_id → title dict from the content master for comment context."""
    path = DATA_DIR / f'{GAME}_content_master.parquet'
    # Only need id and title columns — read minimally to avoid loading the full parquet
    try:
        df = pd.read_parquet(path, columns=['id', 'title', 'source_type'])
        df = df[df['source_type'] == 'reddit_submission']
        return dict(zip(df['id'].astype(str), df['title'].fillna('')))
    except Exception:
        return {}


def build_test_set(seed: int = RANDOM_SEED, exclude_ids: frozenset = frozenset()) -> pd.DataFrame:
    """Sample stratified rows from master parquets and build the test-set CSV.

    exclude_ids: ids already in the test set (str), removed before sampling so an
    extension batch never re-draws an existing row.
    """
    all_rows = []

    # --- SUBMISSIONS ---
    sub_df = _load_and_filter(
        f'{GAME}_content_master.parquet',
        score_col='overall__score',
        require_game_relevance=True,
        source_type='reddit_submission',
    )
    sub_df = sub_df[~sub_df['id'].astype(str).isin(exclude_ids)]
    sub_sample = stratified_year_sample(sub_df, N_PER_TYPE, seed=seed)
    text_col = find_text_column(sub_df, 'submission')
    for _, row in sub_sample.iterrows():
        r = _empty_row('submission')
        r['id']    = str(row.get('id', ''))
        r['date']  = str(row.get('date', ''))
        r['title'] = str(row['title']) if pd.notna(row.get('title')) else ''
        r['text']  = str(row[text_col]) if text_col and pd.notna(row.get(text_col)) else ''
        r['llm__sentiment_score'] = row.get('overall__score')
        r['llm__emotional_tone']  = row.get('emotional_tone')
        r['llm__optimism']        = row.get('outlook__optimism_about_future')
        r['llm__game_of_focus']   = row.get('meta__game_of_focus')
        all_rows.append(r)
    print(f"    Sampled: {len(sub_sample)}")

    # --- COMMENTS ---
    # Build lookup first so each comment row can show its parent post's title
    print("  Building submission title lookup for comment context...")
    sub_title_lookup = build_submission_title_lookup()
    print(f"    Loaded {len(sub_title_lookup):,} submission titles")

    com_df = _load_and_filter(
        f'{GAME}_comments_master.parquet',
        score_col='sentiment_score',
        require_game_relevance=False,
    )
    com_df = com_df[~com_df['id'].astype(str).isin(exclude_ids)]
    com_sample = stratified_year_sample(com_df, N_PER_TYPE, seed=seed)
    text_col = find_text_column(com_df, 'comment')
    for _, row in com_sample.iterrows():
        r = _empty_row('comment')
        r['id']   = str(row.get('id', ''))
        r['date'] = str(row.get('date', ''))
        # link_id is "t3_<submission_id>" — strip the prefix to get the bare ID
        link_id = str(row.get('link_id', ''))
        sub_id = link_id.removeprefix('t3_')
        r['title'] = sub_title_lookup.get(sub_id, '')
        r['text']  = str(row[text_col]) if text_col and pd.notna(row.get(text_col)) else ''
        r['llm__sentiment_score'] = row.get('sentiment_score')
        r['llm__emotional_tone']  = row.get('emotional_tone')
        r['llm__game_of_focus']   = row.get('meta__game_of_focus')
        all_rows.append(r)
    print(f"    Sampled: {len(com_sample)}")

    # --- CREATORS ---
    cre_df = _load_and_filter(
        f'{GAME}_content_master.parquet',
        score_col='overall__score',
        require_game_relevance=True,
        source_type='youtube_creator',
    )
    cre_id_col = 'video_id' if 'video_id' in cre_df.columns else 'id'
    cre_df = cre_df[~cre_df[cre_id_col].astype(str).isin(exclude_ids)]
    cre_sample = stratified_year_sample(cre_df, N_PER_TYPE, seed=seed)
    text_col = find_text_column(cre_df, 'creator')
    for _, row in cre_sample.iterrows():
        r = _empty_row('creator')
        r['id']    = str(row.get('video_id', row.get('id', '')))
        r['date']  = str(row.get('date', ''))
        r['title'] = str(row['title']) if pd.notna(row.get('title')) else ''
        r['text']  = str(row[text_col]) if text_col and pd.notna(row.get(text_col)) else ''
        r['llm__sentiment_score'] = row.get('overall__score')
        r['llm__emotional_tone']  = row.get('emotional_tone')
        r['llm__optimism']        = row.get('outlook__optimism_about_future')
        r['llm__game_of_focus']   = row.get('meta__game_of_focus')
        # Creator engagement: pull raw categorical text from JSON
        raw_json = row.get(LLM_COL)
        if raw_json is not None and not (isinstance(raw_json, float) and np.isnan(raw_json)):
            eng = extract_creator_engagement_raw(raw_json)
            r['llm__still_playing']   = eng.get('still_playing')
            r['llm__burned_out']      = eng.get('burned_out')
            r['llm__would_recommend'] = eng.get('would_recommend')
        all_rows.append(r)
    print(f"    Sampled: {len(cre_sample)}")

    return pd.DataFrame(all_rows)


def print_instructions(path: Path, n: int):
    print()
    print('=' * 72)
    print('TEST SET CREATED — LABELING INSTRUCTIONS')
    print('=' * 72)
    print(f'\nFile : {path}')
    print(f'Rows : {n}  ({N_PER_TYPE} submissions + {N_PER_TYPE} comments + {N_PER_TYPE} creators)')
    print()
    print('TASK: For each row, read the "title" and "text" columns and fill in')
    print('      the human__ columns based on your own judgment. If "text" is')
    print('      blank or deleted, rely on "title" alone. The llm__ columns show')
    print('      what the model said — do not let them bias you.')
    print()
    print('NOTE ON "title" COLUMN:')
    print('  submission  → the post title')
    print('  comment     → the parent submission title (same context the model had)')
    print('  creator     → the video title')
    print()
    print('NUMERIC FIELDS (-2 to +2; leave blank if unclear or inapplicable):')
    print('  human__sentiment_score  overall sentiment  (-2=very negative, +2=very positive)')
    print('  human__optimism         future outlook     (-2=very pessimistic, +2=very optimistic)')
    print()
    print('CATEGORICAL FIELDS (leave blank if inapplicable for that content type):')
    print('  human__emotional_tone       agreement is measured at polarity level, not term level.')
    print('    You may use a specific term OR just: positive / neutral / negative')
    print('    positive: hopeful / enthusiastic / satisfied / optimistic / impressed')
    print('    neutral:  indifferent / conflicted / analytical / measured / nostalgic')
    print('    negative: frustrated / angry / resigned / disappointed / bitter')
    for field, vals in ALLOWED.items():
        if field in ('emotional_tone', 'game_relevance', 'player_status'):
            continue
        print(f'  human__{field:<18}  {" / ".join(vals)}')
    print()
    print('WHICH COLUMNS APPLY PER CONTENT TYPE:')
    print('  submission  sentiment_score, emotional_tone, optimism')
    print('  comment     sentiment_score, emotional_tone')
    print('  creator     sentiment_score, emotional_tone, optimism,')
    print('              still_playing, burned_out, would_recommend')
    print()
    print('NOTE: Creator engagement fields (still_playing, burned_out, would_recommend)')
    print('      show the raw LLM response text. The model may use vocabulary from an')
    print('      older prompt version. Label using the allowed values listed above.')
    print()
    print('When you finish a row, set human_validated to True.')
    print('Run this script again when all rows are done to compute the agreement report.')
    print('=' * 72)


# =============================================================================
# PHASE 2 — PROGRESS REPORT
# =============================================================================

def _validated_mask(df: pd.DataFrame) -> pd.Series:
    return df['human_validated'].astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})


def report_progress(df: pd.DataFrame):
    done = _validated_mask(df).sum()
    total = len(df)
    print(f'\nValidation in progress: {done} / {total} rows complete.')
    remaining_ids = df.loc[~_validated_mask(df), 'id'].tolist()
    if len(remaining_ids) <= 20:
        print(f'Remaining IDs: {remaining_ids}')
    else:
        print(f'{len(remaining_ids)} rows still need labeling.')


# =============================================================================
# PHASE 3 — AGREEMENT STATISTICS
# =============================================================================

def _safe_pearsonr(a: pd.Series, b: pd.Series):
    mask = pd.notna(a) & pd.notna(b)
    if mask.sum() < 3:
        return None, None
    try:
        r, p = pearsonr(a[mask].astype(float), b[mask].astype(float))
        return float(r), float(p)
    except Exception:
        return None, None


def _safe_mae(a: pd.Series, b: pd.Series):
    mask = pd.notna(a) & pd.notna(b)
    if mask.sum() < 1:
        return None
    try:
        return float(np.mean(np.abs(a[mask].astype(float) - b[mask].astype(float))))
    except Exception:
        return None


def _safe_kappa(a: pd.Series, b: pd.Series):
    a = a.astype(str).str.strip()
    b = b.astype(str).str.strip()
    mask = (a != '') & (b != '') & (a.str.lower() != 'nan') & (b.str.lower() != 'nan')
    if mask.sum() < 3:
        return None
    try:
        return float(cohen_kappa_score(a[mask], b[mask]))
    except Exception:
        return None


def _safe_within1_acc(a: pd.Series, b: pd.Series) -> float | None:
    """Proportion of paired values where |a - b| <= 1 (off-by-one accuracy)."""
    mask = pd.notna(a) & pd.notna(b)
    if mask.sum() < 1:
        return None
    try:
        return float((np.abs(a[mask].astype(float) - b[mask].astype(float)) <= 1).mean())
    except Exception:
        return None


def _grade(value, metric: str) -> str:
    if value is None:
        return 'insufficient data'
    v = abs(value)
    if v < 0.4:
        return 'poor'
    if v < 0.6:
        return 'moderate'
    if v < 0.8:
        return 'substantial'
    return 'near-perfect'


def compute_agreement(df: pd.DataFrame) -> list[dict]:
    results = []

    numeric_fields = [
        ('sentiment_score', 'all'),
        ('optimism',        'submission + creator'),
    ]
    for field, scope in numeric_fields:
        llm_col, hum_col = f'llm__{field}', f'human__{field}'
        if llm_col not in df.columns or hum_col not in df.columns:
            continue
        a = pd.to_numeric(df[llm_col], errors='coerce')
        b = pd.to_numeric(df[hum_col], errors='coerce')
        n = (pd.notna(a) & pd.notna(b)).sum()
        r, _   = _safe_pearsonr(a, b)
        mae    = _safe_mae(a, b)
        w1_acc = _safe_within1_acc(a, b)
        results.append({
            'field':          field,
            'scope':          scope,
            'type':           'numeric',
            'n':              int(n),
            'pearson_r':      round(r, 3)      if r      is not None else None,
            'mae':            round(mae, 3)    if mae    is not None else None,
            'within_1_acc':   round(w1_acc, 3) if w1_acc is not None else None,
            'kappa':          None,
            'interpretation': _grade(r, 'pearson'),
        })

    # (label, column_key, scope)
    categorical_fields = [
        ('emotional_tone (valence-collapsed)', 'emotional_tone', 'all'),
        ('still_playing',                      'still_playing',  'creator'),
        ('burned_out',                         'burned_out',     'creator'),
        ('would_recommend',                    'would_recommend', 'creator'),
    ]
    for label, col_key, scope in categorical_fields:
        llm_col, hum_col = f'llm__{col_key}', f'human__{col_key}'
        if llm_col not in df.columns or hum_col not in df.columns:
            continue
        a = df[llm_col].fillna('').astype(str).str.strip().str.lower()
        b = df[hum_col].fillna('').astype(str).str.strip().str.lower()
        # emotional_tone: compare at polarity-group level, not individual term level
        if col_key == 'emotional_tone':
            a = a.map(EMOTIONAL_TONE_GROUPS)
            b = b.map(EMOTIONAL_TONE_GROUPS)
            mask = a.notna() & b.notna()
        else:
            mask = (a != '') & (b != '') & (a != 'nan') & (b != 'nan')
        n = mask.sum()
        kappa = _safe_kappa(a[mask], b[mask]) if n >= 3 else None
        results.append({
            'field':          label,
            'scope':          scope,
            'type':           'categorical',
            'n':              int(n),
            'pearson_r':      None,
            'mae':            None,
            'within_1_acc':   None,
            'kappa':          round(kappa, 3) if kappa is not None else None,
            'interpretation': _grade(kappa, 'kappa'),
        })

    # Sort worst agreement first so problems surface at the top
    def _sort_key(r):
        v = r['pearson_r'] if r['type'] == 'numeric' else r['kappa']
        return abs(v) if v is not None else -1.0

    results.sort(key=_sort_key)
    return results


def print_report(results: list[dict]):
    print()
    print('=' * 80)
    print('HUMAN-LLM AGREEMENT REPORT')
    print('=' * 80)
    print()
    print('Thresholds (kappa and |pearson|):')
    print('  <0.4 = poor   0.4–0.6 = moderate   0.6–0.8 = substantial   >0.8 = near-perfect')
    print('  MAE > 0.5 on the 4-point scale is flagged with (!)')
    print('  W1-Acc: proportion of -2→+2 pairs within ±1 of each other (numeric fields only)')
    print()
    hdr = (f"{'Field':<22} {'Type':<12} {'N':>4}   {'Pearson':>8}  {'MAE':>7}  "
           f"{'W1-Acc':>7}  {'Kappa':>7}  Interpretation")
    print(hdr)
    print('-' * 90)
    for r in results:
        pearson_s = f"{r['pearson_r']:>8.3f}"    if r['pearson_r']  is not None else '       —'
        mae_s     = f"{r['mae']:>6.3f}"           if r['mae']        is not None else '     —'
        w1_s      = f"{r['within_1_acc']:>7.3f}"  if r['within_1_acc'] is not None else '      —'
        kappa_s   = f"{r['kappa']:>7.3f}"         if r['kappa']      is not None else '      —'
        mae_flag  = '(!)' if r['mae'] is not None and r['mae'] > 0.5 else '   '
        print(f"  {r['field']:<20} {r['type']:<12} {r['n']:>4}   {pearson_s}  {mae_s}{mae_flag}  "
              f"{w1_s}  {kappa_s}  {r['interpretation']}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def extend_test_set():
    """Append one new UNGRADED batch (N_PER_TYPE per content type) to test_set.csv."""
    existing = pd.read_csv(TEST_SET)
    if 'sample_batch' not in existing.columns:
        existing['sample_batch'] = 1
    batch = int(existing['sample_batch'].max()) + 1
    seed = EXTEND_SEED + (batch - 2)          # batch 2 -> 43, batch 3 -> 44, ...
    exclude = frozenset(existing['id'].astype(str))
    print(f'Extending test set: batch {batch}, seed {seed}, excluding {len(exclude)} existing ids\n')

    new = build_test_set(seed=seed, exclude_ids=exclude)
    # Keep the file's existing schema exactly (plus sample_batch); human columns empty.
    new['sample_batch'] = batch
    new = new.reindex(columns=existing.columns)
    for c in [c for c in new.columns if c.startswith('human__')]:
        new[c] = None
    new['human_validated'] = False
    assert not set(new['id'].astype(str)) & exclude, 'extension re-drew an existing id'

    out = pd.concat([existing, new], ignore_index=True)
    out.to_csv(TEST_SET, index=False)
    print(f'\nAppended {len(new)} ungraded rows -> {len(out)} total in:\n  {TEST_SET}')
    print(new['content_type'].value_counts().to_string())


def main():
    if '--extend' in sys.argv:
        extend_test_set()
        return
    if not TEST_SET.exists():
        print('No test set found — building stratified sample...\n')
        VALID_DIR.mkdir(parents=True, exist_ok=True)
        df = build_test_set()
        df.to_csv(TEST_SET, index=False)
        print(f'\nSaved {len(df)} rows to:\n  {TEST_SET}')
        print_instructions(TEST_SET, len(df))
        return

    df = pd.read_csv(TEST_SET)

    if not _validated_mask(df).all():
        report_progress(df)
        return

    print(f'\nAll {len(df)} rows validated. Computing agreement statistics...')
    results = compute_agreement(df)
    print_report(results)

    report_df = pd.DataFrame(results)
    report_df.to_csv(REPORT, index=False)
    print(f'Full report saved to:\n  {REPORT}')


if __name__ == '__main__':
    main()
