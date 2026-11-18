"""
Calendar-correct lagged correlation helpers (shared by the 7_analysis scripts).

Why this module exists
----------------------
Every lead/lag script used to carry its own copy of `normalised_xcorr`, which did
`pd.concat([x, y]).dropna()` and then cross-correlated the surviving rows BY
POSITION. Wherever either series had missing days/weeks, "lag k" therefore meant
"k observed rows", not "k calendar periods" -- the same gap-collapsing defect
already fixed in 05g (within_block_xcorr) and in 05's Granger test. Destiny 2's
dense overall-score series were barely affected; sparse series (topic fields,
CS/Dota/DBD/Warframe daily creator series) were, sometimes moving the peak lag
by weeks. 14's aligned-pair helper had the same defect plus a reversed sign
convention (its "+lag" paired x_{t+lag} with y_t, i.e. y leading).

Convention everywhere: positive lag = x leads y, i.e. corr(x_t, y_{t+lag}).
"""
import numpy as np
import pandas as pd
from scipy.signal import correlate


def _regularise(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align x and y on one regular index so a row offset equals a calendar offset."""
    both = pd.concat([x.rename('x'), y.rename('y')], axis=1)
    idx = both.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        freq = idx.freq or pd.infer_freq(idx[:min(len(idx), 50)])
        if freq is None:
            # Fall back to the modal spacing (gappy index with a dominant step)
            step = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
            full = pd.date_range(idx.min(), idx.max(), freq=step)
        else:
            full = pd.date_range(idx.min(), idx.max(), freq=freq)
        both = both.reindex(full)
    return both['x'], both['y']


def normalised_xcorr(x: pd.Series, y: pd.Series, max_lag: int,
                     min_pairs: int | None = None) -> pd.Series:
    """
    Normalised cross-correlation with lags in calendar periods. Positive lag = x leads y.

    Each series is centred on its mean over the jointly-observed periods; missing
    periods are then set to zero on the regular index, so they contribute nothing
    to any lag's sum and every offset is a true calendar offset. Normalisation uses
    the jointly-observed periods (the same denominator the old row-based version
    used), so on a gap-free series the result is identical to the old estimator,
    and on a gappy one it agrees with per-lag Pearson to ~0.005.

    min_pairs defaults to 2*max_lag + 10 jointly-observed periods (the old rule).
    """
    x, y = _regularise(x, y)
    joint = x.notna() & y.notna()
    need = (max_lag * 2 + 10) if min_pairs is None else min_pairs
    if joint.sum() < need:
        return pd.Series(dtype=float)
    xm, ym = x[joint].mean(), y[joint].mean()
    xa = (x - xm).where(x.notna(), 0.0).to_numpy(dtype=float)
    ya = (y - ym).where(y.notna(), 0.0).to_numpy(dtype=float)
    norm = np.sqrt(((x[joint] - xm) ** 2).sum() * ((y[joint] - ym) ** 2).sum())
    if norm == 0:
        return pd.Series(dtype=float)
    full = correlate(ya, xa, mode='full')
    lags = np.arange(-(len(xa) - 1), len(xa))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return pd.Series(full[mask] / norm, index=lags[mask])


def aligned_pair(x: pd.Series, y: pd.Series, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Arrays (x_t, y_{t+lag}) over periods where both are observed, with lag in
    calendar periods. Positive lag = x leads y.
    """
    x, y = _regularise(x, y)
    pair = pd.concat([x, y.shift(-lag)], axis=1).dropna()
    return pair.iloc[:, 0].to_numpy(dtype=float), pair.iloc[:, 1].to_numpy(dtype=float)
