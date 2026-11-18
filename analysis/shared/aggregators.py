#%%
import numpy as np
import pandas as pd


# ── Weighted mean helper ──────────────────────────────────────────────────────
def _weighted_mean_by_date(
    df: pd.DataFrame,
    field: str,
    weight: pd.Series,
    date_col: str = 'date',
) -> pd.Series:
    """
    Vectorized weighted mean of `field` grouped by date, NaN-aware.
    Rows where `field` is NaN are excluded from both numerator and denominator.
    """
    valid = df[field].notna()
    num = (df[field].where(valid) * weight).groupby(df[date_col]).sum()
    den = weight.where(valid).groupby(df[date_col]).sum()
    result = num / den
    result.name = field
    return result


# ── Creator aggregation (3 schemes) ──────────────────────────────────────────
def aggregate_creator_daily(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    """
    Aggregate YouTube creator sentiment to daily frequency under 3 weighting schemes.

    Schemes:
      unweighted      — equal weight per video (output volume)
      view_weighted   — weighted by video view_count (reach); NaN view_count → weight=1
      creator_weighted — weighted by creator's total view count across all their videos
                         in the dataset (source credibility/authority proxy)

    Returns DataFrame indexed by date with columns:
      {field}_unweighted, {field}_view_weighted, {field}_creator_weighted
    """
    df = df.copy()
    view_w    = df['view_count'].fillna(1.0)
    creator_w = df.groupby('creator')['view_count'].transform('sum').fillna(1.0)

    parts = {}
    for field in fields:
        if field not in df.columns:
            continue
        parts[f'{field}_unweighted']       = df.groupby('date')[field].mean()
        parts[f'{field}_view_weighted']    = _weighted_mean_by_date(df, field, view_w)
        parts[f'{field}_creator_weighted'] = _weighted_mean_by_date(df, field, creator_w)

    result = pd.DataFrame(parts)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Reddit aggregation (2 schemes) ───────────────────────────────────────────
def aggregate_reddit_daily(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    """
    Aggregate Reddit sentiment to daily frequency under 2 weighting schemes.

    Schemes:
      unweighted          — equal weight per submission (discussion breadth)
      engagement_weighted — weighted by score (upvotes), floored at 1 (discussion salience)

    Returns DataFrame indexed by date with columns:
      {field}_unweighted, {field}_engagement_weighted
    """
    df = df.copy()
    eng_w = df['score'].clip(lower=1)

    parts = {}
    for field in fields:
        if field not in df.columns:
            continue
        parts[f'{field}_unweighted']          = df.groupby('date')[field].mean()
        parts[f'{field}_engagement_weighted'] = _weighted_mean_by_date(df, field, eng_w)

    result = pd.DataFrame(parts)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Comments aggregation (2 schemes, uses comment score) ─────────────────────
def aggregate_comments_daily(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    """
    Aggregate Reddit comment sentiment to daily frequency.
    Same 2-scheme structure as reddit submissions.
    """
    df = df.copy()
    eng_w = df['score'].clip(lower=1)

    parts = {}
    for field in fields:
        if field not in df.columns:
            continue
        parts[f'{field}_unweighted']          = df.groupby('date')[field].mean()
        parts[f'{field}_engagement_weighted'] = _weighted_mean_by_date(df, field, eng_w)

    result = pd.DataFrame(parts)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


# ── Deseasonalization ─────────────────────────────────────────────────────────
def deseasonalize_dow(series: pd.Series) -> pd.Series:
    """
    Remove day-of-week effects by OLS on DOW dummies, return residuals.
    Input series must have a DatetimeIndex.
    Returns residuals aligned to the original index (NaN where input was NaN).
    """
    s = series.dropna()
    if len(s) == 0:
        return series.copy()

    dow = s.index.dayofweek
    dummies = pd.get_dummies(dow, prefix='dow', drop_first=True).astype(float).set_index(s.index)
    X = pd.concat([pd.Series(1.0, index=s.index, name='const'), dummies], axis=1).values
    y = s.values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    residuals = pd.Series(y - fitted, index=s.index, name=series.name)
    return residuals.reindex(series.index)


def rolling_7day(series: pd.Series, min_periods: int = 4) -> pd.Series:
    """7-day centered rolling mean. Use for presentation smoothing only."""
    return series.rolling(7, center=True, min_periods=min_periods).mean()


# ── Daily reindex ─────────────────────────────────────────────────────────────
def reindex_daily(series: pd.Series, start=None, end=None) -> pd.Series:
    """Reindex to a complete daily DatetimeIndex, filling gaps with NaN."""
    idx = pd.date_range(
        start=start or series.index.min(),
        end=end or series.index.max(),
        freq='D',
    )
    return series.reindex(idx)


# ── Weekly resampling ─────────────────────────────────────────────────────────
def resample_weekly(df: pd.DataFrame, agg: str = 'mean') -> pd.DataFrame:
    """
    Resample a daily-indexed DataFrame to weekly (Monday-anchored).
    agg: 'mean' or 'sum'
    """
    resampler = df.resample('W-MON')
    return getattr(resampler, agg)()
