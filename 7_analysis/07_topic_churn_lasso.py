#%%
"""
07_topic_churn_lasso.py
------------------------
Regularised regression (Lasso / elastic net) predicting forward player-count
change from the full sentiment field set at multiple forecast horizons.

For each horizon H (2, 4, 6 weeks ahead):
  target = % change in player count from current week to H weeks ahead
  features = weekly mean sentiment for every topic__ field + sentiment dims
             + was_discussed indicator per topic field (1 if any row that week
               had a non-NaN value; 0 otherwise)

NULL handling: topic fields are conditionally populated (sparse by design).
  - Impute missing weekly means with 0 (neutral)
  - Add binary was_discussed__{field} indicator so the model can distinguish
    "discussed neutrally" from "not discussed at all"

Outputs:
  figures/07_lasso_coefficients.png     — bar chart of non-zero Lasso coefficients per horizon
  tables/07_lasso_cv_scores.csv         — CV R² per horizon + alpha
  tables/07_lasso_coefficients.csv      — coefficient values per field per horizon
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LassoCV, ElasticNetCV, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions, load_comments_master,
    load_player_counts, ensure_output_dirs,
    FIGURES_DIR, TABLES_DIR, get_topic_cols, SENTIMENT_DIM_COLS,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, aggregate_comments_daily,
    resample_weekly,
)

HORIZONS        = [2, 4, 6]   # weeks ahead
N_CV_SPLITS     = 5
MIN_NONNAN_ROWS = 20          # minimum non-NaN rows in target to include a feature
# Non-topic ("core") features are required to be non-NaN on every retained week
# (topic features are 0-imputed instead). A core feature the LLM almost never
# populates therefore annihilates the sample via that all-columns AND: CS2
# creators trigger `is_the_monetization_fair` in 8% of weeks, which cut the
# usable panel to 6 of 937 weeks and made script 07 unrunnable, while Dead by
# Daylight silently fitted 32 features to 28 rows. Core features below this
# weekly coverage are dropped before the intersection so a schema field the
# game does not use cannot delete the study. Destiny 2, Warframe and Dota 2 all
# sit far above it (min coverage 98% / 91% / 73%), so their samples are
# unchanged and their published numbers still reproduce.
MIN_CORE_COVERAGE = float(os.environ.get('MIN_CORE_COVERAGE', 0.50))  # min share of weeks a core feature must be non-NaN
# A non-default threshold is a sensitivity run (e.g. Warframe at 0.85): its
# tables and figures get a suffix so they never overwrite the primary outputs.
OUT_SUFFIX = '' if MIN_CORE_COVERAGE == 0.50 else f'_cov{round(MIN_CORE_COVERAGE * 100)}'
L1_RATIO        = 0.8         # elastic net: 1.0 = pure Lasso, 0.5 = equal mix
N_BOOTSTRAP     = 100         # resamples for feature stability check
STABLE_THRESHOLD = 0.80       # min selection rate to report a feature

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
cm = load_comments_master()
pc = load_player_counts(source='total').set_index('date').sort_index()

topic_cols     = get_topic_cols(yt)
all_sent_cols  = ['overall__score'] + SENTIMENT_DIM_COLS + topic_cols

#%%
# Weekly aggregation — unweighted mean for all sources
creator_daily = aggregate_creator_daily(yt, all_sent_cols)
reddit_daily  = aggregate_reddit_daily(rd, all_sent_cols)
comment_daily = aggregate_comments_daily(cm, ['sentiment_score'])

creator_weekly = resample_weekly(creator_daily)
reddit_weekly  = resample_weekly(reddit_daily)
comment_weekly = resample_weekly(comment_daily)
pc_weekly      = pc['players'].resample('W-MON').mean()

# Drop near-zero weeks before computing % change — these cause denominator blow-up.
# Threshold: weeks with fewer than 1 000 mean daily players are data gaps or pre-launch noise.
_pc_min = 1_000
_dropped = (pc_weekly < _pc_min).sum()
if _dropped:
    print(f'Dropping {_dropped} near-zero pc_weekly weeks (< {_pc_min} players) before % change.')
pc_weekly = pc_weekly[pc_weekly >= _pc_min]

#%%
def build_features(creator_w: pd.DataFrame, reddit_w: pd.DataFrame,
                   all_cols: list) -> pd.DataFrame:
    """
    Build feature matrix with:
      - Weekly mean sentiment for each field (both creator and Reddit, unweighted)
      - was_discussed indicator per topic field (from creator data)
    Topic field means are imputed with 0 where missing; was_discussed captures sparsity.
    Non-topic fields (overall, dims) are not imputed — NaN rows are dropped.
    """
    frames = {}
    for col in all_cols:
        c_col = f'{col}_unweighted'
        r_col = f'{col}_unweighted'

        if c_col in creator_w.columns:
            is_topic = col.startswith('topic__')
            frames[f'creator__{col}'] = creator_w[c_col].fillna(0) if is_topic else creator_w[c_col]
            if is_topic:
                # was_discussed: any non-NaN in this week's daily data
                # Approximated from weekly: was 0-imputed value actually observed?
                # Use the raw weekly mean before fillna as the indicator source
                frames[f'creator__was_discussed__{col}'] = creator_w[c_col].notna().astype(float)

        if r_col in reddit_w.columns:
            is_topic = col.startswith('topic__')
            frames[f'reddit__{col}'] = reddit_w[r_col].fillna(0) if is_topic else reddit_w[r_col]
            if is_topic:
                frames[f'reddit__was_discussed__{col}'] = reddit_w[r_col].notna().astype(float)

    return pd.DataFrame(frames)


features = build_features(creator_weekly, reddit_weekly, all_sent_cols)

# Append comment sentiment columns (both weighting schemes).
# Comments use sentiment_score (overall valence), not the topic/dim schema.
for col_suffix in ('unweighted', 'engagement_weighted'):
    src_col = f'sentiment_score_{col_suffix}'
    if src_col in comment_weekly.columns:
        features[f'comment__sentiment_score_{col_suffix}'] = comment_weekly[src_col]

#%%
cv_records   = []
coef_records = []

for H in HORIZONS:
    print(f'\n-- Horizon: {H} weeks ahead --')

    # Target: % change from current week to H weeks ahead.
    # Winsorise at 1st/99th percentile to remove residual outliers after near-zero filter.
    target = (pc_weekly.shift(-H) - pc_weekly) / pc_weekly * 100
    lo, hi = target.quantile(0.01), target.quantile(0.99)
    target = target.clip(lo, hi)
    target.name = f'pct_change_{H}w'

    # Align
    idx = features.index.intersection(target.index)
    X_df = features.loc[idx]
    y    = target.loc[idx]

    # Drop rows where target is NaN (end of series) or non-topic features are NaN
    core_cols = [c for c in X_df.columns if not c.startswith(('creator__topic', 'reddit__topic',
                                                                'creator__was', 'reddit__was'))]

    # Drop core features this game barely populates before applying the
    # all-columns AND below — otherwise one sparse schema field empties the panel.
    coverage = X_df[core_cols].notna().mean()
    sparse_core = sorted(coverage.index[coverage < MIN_CORE_COVERAGE])
    if sparse_core:
        print(f'  Dropping {len(sparse_core)} core feature(s) below '
              f'{MIN_CORE_COVERAGE:.0%} weekly coverage:')
        for c in sparse_core:
            print(f'    {coverage[c]:6.1%}  {c}')
        core_cols = [c for c in core_cols if c not in sparse_core]
        X_df = X_df.drop(columns=sparse_core)

    valid = y.notna() & X_df[core_cols].notna().all(axis=1)
    X_df  = X_df.loc[valid]
    y     = y.loc[valid]

    # Drop feature columns with too few non-zero values
    keep = X_df.columns[(X_df != 0).sum() >= MIN_NONNAN_ROWS]
    X_df = X_df[keep]

    if len(y) < N_CV_SPLITS * 5:
        print(f'  Insufficient data ({len(y)} rows). Skipping.')
        continue

    print(f'  Samples: {len(y)}  Features: {X_df.shape[1]}')
    print(f'  Target stats: mean={y.mean():.2f}  std={y.std():.2f}  '
          f'min={y.min():.2f}  max={y.max():.2f}')

    # Persist the aligned design matrix so 07b can fit AR + sentiment on the
    # identical rows. The CV R2 reported below is NOT incremental over
    # persistence -- nothing in this script controls for the target's own
    # autocorrelation -- so the nested AR vs AR+sentiment contrast has to run
    # somewhere, and it has to use exactly these rows to be a fair comparison.
    matrix_path = TABLES_DIR / f'07_feature_matrix_{H}w{OUT_SUFFIX}.parquet'
    X_df.assign(__target__=y).to_parquet(matrix_path)
    print(f'  Saved design matrix -> {matrix_path.name}')

    # OLS sanity check: if unregularized R2 is also ~0, the issue is X/y construction
    ols_cols = [c for c in X_df.columns
                if c in ('creator__overall__score', 'reddit__overall__score')]
    if ols_cols:
        ols_X = StandardScaler().fit_transform(X_df[ols_cols].values)
        ols_r2 = r2_score(y.values, LinearRegression().fit(ols_X, y.values).predict(ols_X))
        print(f'  OLS in-sample R2 on {ols_cols}: {ols_r2:.4f}')

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNetCV(l1_ratio=L1_RATIO, cv=tscv, max_iter=5000,
                                n_alphas=100, eps=1e-3, random_state=42)),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe.fit(X_df.values, y.values)

    model  = pipe.named_steps['model']
    alpha_ = model.alpha_
    alpha_min, alpha_max = model.alphas_.min(), model.alphas_.max()
    at_edge = alpha_ >= alpha_max * 0.99
    print(f'  Alpha: {alpha_:.3f}  (path: {alpha_min:.3f} - {alpha_max:.3f})'
          f'{"  *** AT TOP - path too restrictive ***" if at_edge else "  (interior)"}')
    print(f'  In-sample R2: {pipe.score(X_df.values, y.values):.4f}  (reference only)')

    # CV R² with the selected fixed alpha — the reportable number
    pipe_fixed = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(alpha=alpha_, l1_ratio=L1_RATIO, max_iter=5000)),
    ])
    cv_scores = cross_val_score(pipe_fixed, X_df.values, y.values,
                                cv=tscv, scoring='r2')
    mean_cv_r2, std_cv_r2 = cv_scores.mean(), cv_scores.std()
    print(f'  CV R2 (TimeSeriesSplit k={N_CV_SPLITS}): {mean_cv_r2:.4f} +/- {std_cv_r2:.4f}')
    cv_records.append({'horizon_weeks': H, 'alpha': alpha_,
                       'alpha_path_min': alpha_min, 'alpha_path_max': alpha_max,
                       'alpha_at_edge': at_edge,
                       'cv_r2_mean': mean_cv_r2, 'cv_r2_std': std_cv_r2,
                       'n_samples': len(y), 'n_features': X_df.shape[1]})

    # Non-zero coefficients from full-data fit
    coefs   = pd.Series(model.coef_, index=X_df.columns)
    nonzero = coefs[coefs != 0].sort_values(key=abs, ascending=False)
    print(f'  Non-zero coefficients (full-data fit): {len(nonzero)}')

    # Bootstrap stability: resample with replacement N_BOOTSTRAP times using the
    # same fixed alpha, count how often each feature survives regularisation.
    rng = np.random.default_rng(42)
    selection_counts = Counter()
    for _ in range(N_BOOTSTRAP):
        idx_b = rng.choice(len(X_df), size=len(X_df), replace=True)
        p_b = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(alpha=alpha_, l1_ratio=L1_RATIO, max_iter=5000)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p_b.fit(X_df.values[idx_b], y.values[idx_b])
        for feat, c in zip(X_df.columns, p_b.named_steps['model'].coef_):
            if c != 0:
                selection_counts[feat] += 1

    stability = {f: selection_counts.get(f, 0) / N_BOOTSTRAP for f in X_df.columns}
    stable_feats = [f for f in nonzero.index if stability[f] >= STABLE_THRESHOLD]

    # Bivariate correlation sign check for stable features
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bivar = X_df.corrwith(y)
    print(f'\n  Stable features (>={STABLE_THRESHOLD:.0%} bootstrap selection):')
    for feat in stable_feats:
        lasso_coef = coefs[feat]
        bv = bivar[feat]
        match = np.sign(lasso_coef) == np.sign(bv)
        flag = '' if match else '  [SIGN MISMATCH vs bivariate — suppressor candidate]'
        print(f'    {feat:<65s} coef={lasso_coef:+.3f}  bivar_r={bv:+.3f}'
              f'  stab={stability[feat]:.0%}{flag}')

    for feat, val in nonzero.items():
        coef_records.append({
            'horizon_weeks':      H,
            'feature':            feat,
            'coefficient':        val,
            'stability':          stability[feat],
            'bivar_corr':         bivar[feat],
            'sign_matches_bivar': np.sign(val) == np.sign(bivar[feat]),
            'is_stable':          stability[feat] >= STABLE_THRESHOLD,
        })

# ── Save tables ───────────────────────────────────────────────────────────────
pd.DataFrame(cv_records).to_csv(TABLES_DIR / f'07_lasso_cv_scores{OUT_SUFFIX}.csv', index=False)
coef_df = pd.DataFrame(coef_records)
coef_df.to_csv(TABLES_DIR / f'07_lasso_coefficients{OUT_SUFFIX}.csv', index=False)

#%%
# ── Figure: Coefficient bar charts per horizon ────────────────────────────────
if not coef_df.empty:
    plot_df = coef_df[coef_df['is_stable']]
    if plot_df.empty:
        print('No features met the bootstrap stability threshold — nothing to plot.')
    else:
        horizons_present = sorted(plot_df['horizon_weeks'].unique())
        fig, axes = plt.subplots(1, len(horizons_present),
                                 figsize=(7 * len(horizons_present), 8), sharey=False)
        if len(horizons_present) == 1:
            axes = [axes]

        for ax, H in zip(axes, horizons_present):
            sub = plot_df[plot_df['horizon_weeks'] == H].sort_values('coefficient')
            bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in sub['coefficient']]
            ax.barh(sub['feature'], sub['coefficient'], color=bar_colors, alpha=0.8)
            # Mark sign mismatches with an asterisk
            for _, row in sub.iterrows():
                if not row['sign_matches_bivar']:
                    x_pos = row['coefficient'] + (0.02 if row['coefficient'] > 0 else -0.02)
                    ax.text(x_pos, row['feature'], '*', va='center', fontsize=9,
                            color='#e67e22',
                            ha='left' if row['coefficient'] > 0 else 'right')
            ax.axvline(0, color='black', lw=0.8)
            ax.set_title(f'H = {H} weeks', fontsize=11)
            ax.set_xlabel('Coefficient (standardised features)', fontsize=9)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='x', alpha=0.25)

        fig.suptitle(
            f'Elastic-net: stable predictors of forward player-count % change\n'
            f'(bootstrap stability >= {STABLE_THRESHOLD:.0%}; * = sign disagrees with bivariate correlation)',
            fontsize=12,
        )
        plt.tight_layout()
        out = FIGURES_DIR / f'07_lasso_coefficients{OUT_SUFFIX}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\nSaved: {out}')
        plt.show()
