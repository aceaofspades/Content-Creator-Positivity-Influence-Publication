#%%
"""
07b_lasso_ar_baseline.py
-------------------------
Establishes naive baselines to contextualise the Lasso's CV R² from script 07.
Without a baseline, CV R² ≈ 0.05 is uninterpretable — the key question is whether
sentiment features add anything beyond what player-count persistence and the release
calendar already explain.

Four baselines, all evaluated with TimeSeriesSplit k=5 on the same target as
script 07 (winsorised weekly % change in player counts):
  AR                    — lagged % change at 1, 2, 4, 8 weeks
  Releases              — binary release-week flag + log(days_since_last_release)
  AR + Releases         — both combined
  AR + Releases + Eras  — above + expansion-year dummies (structural era shifts)

Lasso CV R² is read from 07_lasso_cv_scores.csv rather than re-running the full
pipeline — all five models are compared on the same evaluation conditions.

Plus three NESTED arms that answer the question the side-by-side comparison
cannot. Script 07 fits sentiment with no AR terms, so "Lasso 0.05 vs AR 0.37"
is two separate models, not a nested test — it leaves open that sentiment
carries forward information which merely overlaps with persistence. These arms
close it, using one estimator (ElasticNetCV alpha selection + fixed-alpha
scoring, as in 07), one common valid mask, and script 07's own design matrix:
  AR [enet]              — persistence only
  Sentiment [enet]       — script 07's feature matrix only
  AR + Sentiment [enet]  — both; the delta vs 'AR [enet]' is the incremental
                           predictive content of sentiment over persistence

A fourth, RESIDUALIZED arm settles what the joint arm cannot:
  AR -> Sentiment [residualized]  — fit AR, then Lasso the AR residuals on
                                    sentiment, both stages inside each fold

The joint `AR + Sentiment` arm shares one penalty across both blocks, so its
negative delta confounds "sentiment is useless" with "sentiment is expensive to
regularise alongside." The residualized arm removes that: AR is fitted
unpenalised and first, and sentiment only ever sees what AR could not explain.
Its `resid_r2` is the clean incremental number.

PER-FOLD SCORES are written for every arm, because the mean cannot distinguish
two very different worlds that both average near zero:
  (a) sentiment is empty                    -> folds cluster tightly below 0
  (b) sentiment is real but non-stationary   -> folds swing hard in both signs
Only (b) makes "adding sentiment degrades forecasts" a substantive claim rather
than an estimation cost. Script 05d shows these relationships flipping sign
across expansion eras, and TimeSeriesSplit trains on early folds and tests on
later ones, so (b) is a live possibility and not a hypothetical.

Note also that a negative held-out R² is never evidence that a feature is
"anti-informative" — information is non-negative. It reflects estimation cost
(variance, shared penalty) or a relationship that inverts out of sample.

Depends on script 07 having written tables/07_feature_matrix_{H}w.parquet.

Outputs:
  figures/07b_lasso_baselines.png  — grouped bar chart, one group per horizon
  tables/07b_lasso_baselines.csv   — model, horizon_weeks, cv_r2_mean, cv_r2_std,
                                     n_samples (+ n_features/alpha for enet arms,
                                     resid_r2_mean/std for the residualized arm)
  tables/07b_fold_scores.csv       — long format: horizon, model, metric, fold, r2
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_player_counts, load_releases, ensure_output_dirs,
    FIGURES_DIR, TABLES_DIR, show_figure,
)
from sklearn.metrics import r2_score

HORIZONS    = [2, 4, 6]   # weeks ahead — same as script 07
N_CV_SPLITS = 5           # same as script 07
AR_LAGS     = [1, 2, 4, 8]  # fixed lag grid (weeks); equivalent to AR(4) on a sparse grid
PC_MIN      = 1_000       # same near-zero filter as script 07
# Must match the MIN_CORE_COVERAGE script 07 ran with; a non-default value
# reads and writes that sensitivity run's suffixed files (see script 07).
MIN_CORE_COVERAGE = float(os.environ.get('MIN_CORE_COVERAGE', 0.50))
OUT_SUFFIX = '' if MIN_CORE_COVERAGE == 0.50 else f'_cov{round(MIN_CORE_COVERAGE * 100)}'
L1_RATIO    = 0.8         # same as script 07 — effectively Lasso

#%%
ensure_output_dirs()

pc       = load_player_counts(source='total').set_index('date').sort_index()
releases = load_releases()

# Weekly player counts — identical construction to script 07
pc_weekly = pc['players'].resample('W-MON').mean()
pc_weekly = pc_weekly[pc_weekly >= PC_MIN]

# ── Release features ──────────────────────────────────────────────────────────
weeks         = pc_weekly.index
release_dates = pd.to_datetime(releases['start_date'].dropna()).sort_values().reset_index(drop=True)

# Binary: 1 if any season starts within this week's Monday–Sunday window
release_binary = pd.Series(0.0, index=weeks)
for rd in release_dates:
    in_window = (weeks <= rd) & (rd < weeks + pd.Timedelta(days=7))
    release_binary[in_window] = 1.0

# Continuous: log(1 + days since most recent release start)
def _days_since(week_date):
    past = release_dates[release_dates <= week_date]
    return (week_date - past.max()).days if len(past) > 0 else np.nan

log_days_since = pd.Series(
    [_days_since(w) for w in weeks], index=weeks, dtype=float
)
log_days_since = np.log1p(log_days_since)

# ── Expansion era dummies ─────────────────────────────────────────────────────
# Year 8 excluded — data too recent and incomplete.
era_map = releases[releases['year_number'] <= 7].copy()
era_windows = (
    era_map.groupby(['year_number', 'year_name'])
    .agg(era_start=('start_date', 'min'), era_end=('end_date', 'max'))
    .reset_index()
    .sort_values('year_number')
)

def _assign_era(week_date):
    for _, row in era_windows.iterrows():
        if row['era_start'] <= week_date < row['era_end']:
            return row['year_name']
    return None

era_series = pd.Categorical(
    [_assign_era(w) for w in weeks],
    categories=era_windows['year_name'].tolist(),
)
era_dummies = pd.get_dummies(
    pd.Series(era_series, index=weeks), prefix='era', drop_first=True
).astype(float)

# ── Evaluation helper ─────────────────────────────────────────────────────────
def eval_model(X_df: pd.DataFrame, y: pd.Series, label: str) -> dict | None:
    valid = y.notna() & X_df.notna().all(axis=1)
    Xv, yv = X_df.loc[valid].values, y.loc[valid].values
    n = len(yv)
    if n < N_CV_SPLITS * 5:
        print(f'  {label}: insufficient data ({n} rows). Skipping.')
        return None
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = cross_val_score(pipe, Xv, yv, cv=tscv, scoring='r2')
    print(f'  {label}: CV R² = {scores.mean():.4f} ± {scores.std():.4f}  (n={n})')
    return {'cv_r2_mean': float(scores.mean()), 'cv_r2_std': float(scores.std()),
            'n_samples': n, 'fold_scores': [float(s) for s in scores]}


def eval_enet(X_df: pd.DataFrame, y: pd.Series, label: str) -> dict | None:
    """
    CV R² under the same regularised estimator script 07 uses.

    The four baselines above are plain OLS, which is fine at <=10 features but
    cannot be pointed at a ~370-column sentiment matrix. For the nested
    AR-vs-AR+sentiment contrast every arm has to share one estimator, so all
    three use ElasticNetCV to select alpha and a fixed-alpha refit to score --
    the same two-step as script 07. Rows are pre-filtered by the caller to a
    single common mask so the arms are scored on identical data.
    """
    n = len(y)
    if n < N_CV_SPLITS * 5:
        print(f'  {label}: insufficient data ({n} rows). Skipping.')
        return None
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selector = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNetCV(l1_ratio=L1_RATIO, cv=tscv, max_iter=5000,
                                    n_alphas=100, eps=1e-3, random_state=42)),
        ])
        selector.fit(X_df.values, y.values)
        alpha_ = selector.named_steps['model'].alpha_
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=alpha_, l1_ratio=L1_RATIO, max_iter=5000)),
        ])
        scores = cross_val_score(pipe, X_df.values, y.values, cv=tscv, scoring='r2')
    print(f'  {label}: CV R² = {scores.mean():.4f} ± {scores.std():.4f}  '
          f'(n={n}, features={X_df.shape[1]}, alpha={alpha_:.3f})  '
          f'folds=[{", ".join(f"{s:+.2f}" for s in scores)}]')
    return {'cv_r2_mean': float(scores.mean()), 'cv_r2_std': float(scores.std()),
            'n_samples': n, 'n_features': int(X_df.shape[1]), 'alpha': float(alpha_),
            'fold_scores': [float(s) for s in scores]}


def eval_residualized(X_ar: pd.DataFrame, X_sent: pd.DataFrame, y: pd.Series,
                      label: str) -> dict | None:
    """
    Two-stage arm: fit AR, then Lasso the AR *residuals* on sentiment.

    This is the confound-free version of the incremental test. The joint
    `AR + Sentiment [enet]` arm shares one penalty across both blocks, so
    adding ~400 weak sentiment columns forces a heavier alpha that also shrinks
    the four AR coefficients doing the real work — the AR features get punished
    for the company they keep. Here the AR model is fitted unpenalised and
    first; sentiment only ever sees what AR could not explain.

    Both stages are fitted INSIDE each fold. Residualising on the full series
    before splitting would leak test-period information into the training
    residuals and inflate the result.

    Two numbers are reported:
      resid_r2    — R² of sentiment predicting the held-out AR residual. This
                    is the clean incremental measure. ~0 means sentiment has
                    nothing left to add once momentum is accounted for.
      combined_r2 — R² of (AR prediction + sentiment residual correction)
                    against the original target, comparable to the other arms.

    Per-fold scores matter as much as the mean here. Two very different worlds
    produce a mean near zero:
      (a) sentiment is empty        -> folds cluster tightly a hair below 0
      (b) sentiment is real but its
          sign does not persist     -> folds swing hard in both directions
    Only (b) would make "adding sentiment degrades forecasts" a substantive
    finding rather than an estimation cost. Script 05d already shows these
    relationships flipping sign across expansion eras, so (b) is live.
    """
    n = len(y)
    if n < N_CV_SPLITS * 5:
        print(f'  {label}: insufficient data ({n} rows). Skipping.')
        return None

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    inner_cv = TimeSeriesSplit(n_splits=3)   # nested; keeps alpha selection honest
    resid_scores, combined_scores = [], []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tr, te in tscv.split(X_ar):
            ar_pipe = Pipeline([('scaler', StandardScaler()),
                                ('model', LinearRegression())])
            ar_pipe.fit(X_ar.iloc[tr], y.iloc[tr])
            resid_tr = y.iloc[tr].values - ar_pipe.predict(X_ar.iloc[tr])
            resid_te = y.iloc[te].values - ar_pipe.predict(X_ar.iloc[te])

            sent_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model',  ElasticNetCV(l1_ratio=L1_RATIO, cv=inner_cv, max_iter=5000,
                                        n_alphas=50, eps=1e-3, random_state=42)),
            ])
            sent_pipe.fit(X_sent.iloc[tr], resid_tr)
            resid_pred = sent_pipe.predict(X_sent.iloc[te])

            resid_scores.append(r2_score(resid_te, resid_pred))
            combined_scores.append(
                r2_score(y.iloc[te].values, ar_pipe.predict(X_ar.iloc[te]) + resid_pred))

    resid_scores = np.array(resid_scores)
    combined_scores = np.array(combined_scores)
    print(f'  {label}: residual R² = {resid_scores.mean():+.4f} ± {resid_scores.std():.4f}  '
          f'(combined {combined_scores.mean():+.4f})  '
          f'folds=[{", ".join(f"{s:+.2f}" for s in resid_scores)}]')
    return {'cv_r2_mean': float(combined_scores.mean()),
            'cv_r2_std': float(combined_scores.std()),
            'resid_r2_mean': float(resid_scores.mean()),
            'resid_r2_std': float(resid_scores.std()),
            'n_samples': n, 'n_features': int(X_sent.shape[1]),
            'fold_scores': [float(s) for s in combined_scores],
            'resid_fold_scores': [float(s) for s in resid_scores]}

#%%
records = []

ar_cols      = [f'lag_{k}w' for k in AR_LAGS]
release_cols = ['release', 'log_days_since']
era_cols     = era_dummies.columns.tolist()

for H in HORIZONS:
    print(f'\n── Horizon {H}w ──')

    # Target — same construction as script 07
    target = (pc_weekly.shift(-H) - pc_weekly) / pc_weekly * 100
    lo, hi = target.quantile(0.01), target.quantile(0.99)
    target = target.clip(lo, hi).rename(f'pct_change_{H}w')

    # AR features: lagged values of the % change target itself
    ar_features = pd.DataFrame(
        {f'lag_{k}w': target.shift(k) for k in AR_LAGS},
        index=target.index,
    )

    feat_base = pd.concat(
        [ar_features,
         release_binary.rename('release'),
         log_days_since.rename('log_days_since'),
         era_dummies],
        axis=1,
    ).reindex(target.index)

    configs = [
        ('AR',                   ar_cols),
        ('Releases',             release_cols),
        ('AR + Releases',        ar_cols + release_cols),
        ('AR + Releases + Eras', ar_cols + release_cols + era_cols),
    ]
    for label, cols in configs:
        result = eval_model(feat_base[cols], target, label)
        if result:
            records.append({'horizon_weeks': H, 'model': label, **result})

    # ── Nested arms: does sentiment add anything to persistence? ─────────────
    # The 'Lasso (script 07)' row appended below is NOT an incremental test:
    # script 07 fits sentiment features with no AR terms, so its R² and the AR
    # baseline's R² are two separate models side by side, not a nested
    # comparison. Without this block, "Lasso 0.05 vs AR 0.37" leaves open the
    # possibility that sentiment carries real forward information that simply
    # overlaps with persistence. These three arms settle it: same rows, same
    # estimator, same folds, one common valid mask.
    matrix_path = TABLES_DIR / f'07_feature_matrix_{H}w{OUT_SUFFIX}.parquet'
    if not matrix_path.exists():
        print(f'  {matrix_path.name} not found - run script 07 first for the '
              f'AR + sentiment arms.')
    else:
        mat    = pd.read_parquet(matrix_path)
        y_sent = mat['__target__']
        X_sent = mat.drop(columns='__target__')
        ar_idx = ar_features.reindex(X_sent.index)

        # One mask for all three arms. Scoring the AR arm on rows the sentiment
        # arm dropped (or vice versa) would repeat the sample-mismatch mistake
        # this comparison exists to avoid.
        common = y_sent.notna() & ar_idx.notna().all(axis=1) & X_sent.notna().all(axis=1)
        arms = [
            ('AR [enet]',             ar_idx),
            ('Sentiment [enet]',      X_sent),
            ('AR + Sentiment [enet]', pd.concat([ar_idx, X_sent], axis=1)),
        ]
        arm_scores = {}
        for label, X_arm in arms:
            result = eval_enet(X_arm.loc[common], y_sent.loc[common], label)
            if result:
                arm_scores[label] = result['cv_r2_mean']
                records.append({'horizon_weeks': H, 'model': label, **result})

        if {'AR [enet]', 'AR + Sentiment [enet]'} <= arm_scores.keys():
            delta = arm_scores['AR + Sentiment [enet]'] - arm_scores['AR [enet]']
            print(f'  -> incremental CV R2 from sentiment, over AR (shared penalty): {delta:+.4f}')

        # Confound-free version of the same question.
        resid = eval_residualized(ar_idx.loc[common], X_sent.loc[common],
                                  y_sent.loc[common], 'AR -> Sentiment [residualized]')
        if resid:
            records.append({'horizon_weeks': H,
                            'model': 'AR -> Sentiment [residualized]', **resid})
            # The diagnostic the mean alone cannot give: do the folds agree?
            fs = np.array(resid['resid_fold_scores'])
            n_pos, spread = int((fs > 0).sum()), float(fs.max() - fs.min())
            verdict = ('folds disagree in sign - consistent with a real but '
                       'non-stationary relationship' if n_pos and n_pos < len(fs) and spread > 0.25
                       else 'folds agree - consistent with no signal')
            print(f'     residual folds: {n_pos}/{len(fs)} positive, '
                  f'spread={spread:.2f}  -> {verdict}')

# ── Append Lasso R² from script 07 ───────────────────────────────────────────
lasso_csv = TABLES_DIR / f'07_lasso_cv_scores{OUT_SUFFIX}.csv'
# Script 07 writes a header-less empty file when every horizon was skipped for
# lack of data, so "exists" is not enough — read defensively and carry on with
# the baselines alone rather than taking the whole script down with it.
lasso_df = None
if lasso_csv.exists():
    try:
        lasso_df = pd.read_csv(lasso_csv)
    except pd.errors.EmptyDataError:
        lasso_df = None

if lasso_df is not None and not lasso_df.empty:
    for _, row in lasso_df.iterrows():
        records.append({
            'horizon_weeks': int(row['horizon_weeks']),
            'model':         'Lasso (script 07)',
            'cv_r2_mean':    float(row['cv_r2_mean']),
            'cv_r2_std':     float(row['cv_r2_std']),
            'n_samples':     int(row['n_samples']),
        })
elif lasso_csv.exists():
    print(f'WARNING: {lasso_csv.name} is empty — script 07 skipped every horizon '
          f'for this game. Reporting AR/release baselines without the Lasso row.')
else:
    print(f'WARNING: {lasso_csv} not found — run script 07 first to include Lasso R².')

df_out = pd.DataFrame(records)

# Per-fold scores go to their own long-format table. The mean hides the
# distinction this script exists to draw: uniformly mediocre folds mean "no
# signal", wildly disagreeing folds mean "signal whose sign does not persist".
fold_rows = []
for rec in records:
    for key, kind in [('fold_scores', 'cv_r2'), ('resid_fold_scores', 'resid_r2')]:
        for i, score in enumerate(rec.get(key) or []):
            fold_rows.append({'horizon_weeks': rec['horizon_weeks'],
                              'model': rec['model'], 'metric': kind,
                              'fold': i + 1, 'r2': round(score, 5)})
pd.DataFrame(fold_rows).to_csv(TABLES_DIR / f'07b_fold_scores{OUT_SUFFIX}.csv', index=False)
print(f'\nSaved: {TABLES_DIR / f"07b_fold_scores{OUT_SUFFIX}.csv"}')

df_out = df_out.drop(columns=[c for c in ('fold_scores', 'resid_fold_scores')
                              if c in df_out.columns])
df_out.to_csv(TABLES_DIR / f'07b_lasso_baselines{OUT_SUFFIX}.csv', index=False)
print(f'Saved: {TABLES_DIR / f"07b_lasso_baselines{OUT_SUFFIX}.csv"}')
print(df_out.to_string(index=False))

#%%
# ── Figure: grouped bar chart ─────────────────────────────────────────────────
MODEL_ORDER = [
    'AR',
    'Releases',
    'AR + Releases',
    'AR + Releases + Eras',
    'Lasso (script 07)',
    'AR [enet]',
    'Sentiment [enet]',
    'AR + Sentiment [enet]',
]
MODEL_COLORS = {
    'AR':                    '#95a5a6',
    'Releases':              '#3498db',
    'AR + Releases':         '#2ecc71',
    'AR + Releases + Eras':  '#e67e22',
    'Lasso (script 07)':     '#8e44ad',
    # The nested trio, shaded as one family so the eye compares within it.
    'AR [enet]':             '#5d6d7e',
    'Sentiment [enet]':      '#c39bd3',
    'AR + Sentiment [enet]': '#1a5276',
    'AR -> Sentiment [residualized]': '#148f77',
}
MODEL_ORDER.append('AR -> Sentiment [residualized]')

n_models   = len(MODEL_ORDER)
n_horizons = len(HORIZONS)
x          = np.arange(n_horizons)
width      = 0.095
offsets    = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

fig, ax = plt.subplots(figsize=(13, 5))

for i, model in enumerate(MODEL_ORDER):
    means, errs = [], []
    for H in HORIZONS:
        row = df_out[(df_out['model'] == model) & (df_out['horizon_weeks'] == H)]
        means.append(float(row['cv_r2_mean'].iloc[0]) if len(row) else np.nan)
        errs.append(float(row['cv_r2_std'].iloc[0])   if len(row) else np.nan)
    ax.bar(x + offsets[i], means, width=width * 0.9,
           label=model, color=MODEL_COLORS.get(model, '#333333'),
           yerr=errs, capsize=3, error_kw={'elinewidth': 1.2}, zorder=3)

ax.axhline(0, color='black', lw=0.9, linestyle='--', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f'Horizon = {H}w ahead' for H in HORIZONS], fontsize=11)
ax.set_ylabel('CV R²  (TimeSeriesSplit k=5)', fontsize=11)
ax.set_title(
    'Lasso vs. naive baselines: predictability of weekly player-count % change\n'
    'AR = persistence; Releases = calendar events; Eras = expansion-year level shifts',
    fontsize=12,
)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, axis='y', alpha=0.25, zorder=0)

# Annotation: negative R² = worse than predicting the mean
y_min = ax.get_ylim()[0]
if y_min < 0:
    ax.text(0.01, 0.02, 'Note: negative R² = worse than predicting the mean',
            transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

plt.tight_layout()
out = FIGURES_DIR / f'07b_lasso_baselines{OUT_SUFFIX}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
show_figure()
