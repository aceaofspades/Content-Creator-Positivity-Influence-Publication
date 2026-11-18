#%%
"""
10_correlation_matrix.py
-------------------------
DESCRIPTIVE EDA ONLY — no inference. Clusters the correlation matrix of all
topic sentiment fields to expose the latent structure of which topics move
together. Appendix material that contextualizes 07; it is not a result.

No p-values / FDR. Two reasons the inferential layer was removed:
  1. Weekly sentiment is heavily autocorrelated, so per-pair Pearson p-values
     are anti-conservative (effective n << ~400 weeks) and any downstream FDR q
     inherits that inflation.
  2. The r=0 null is tautologically false here: a single general "game mood"
     factor makes essentially every topic pair positively correlated, so
     "~95% of pairs survive FDR" restates the positive manifold rather than
     discovering structure. Counting significant pairs answers the wrong
     question. See scripts 04/05b for autocorrelation-aware nulls.

What the cluster structure IS good for:
  (a) Interpreting 07. Topics collapse into a few correlated blocks, so a
      Lasso-selected topic must be read as a REPRESENTATIVE of its block, not a
      uniquely identified driver (stability != identifiability). A topic Lasso
      *dropped* may simply be collinear with a kept sibling — not unpredictive.
  (b) Framing the null. Topic sentiment is low-dimensional — a handful of latent
      blocks, not 87 independent channels. So the paper's null is best stated as
      "these ~N latent sentiment factors do not lead/predict," not "100
      independent features each failed." Fewer effective hypotheses = fewer
      places a topic-specific creator lead could have hidden.

Produces:
  figures/10_correlation_clustermap.png   — clustered heatmap (1-|r|, avg linkage)
  tables/10_correlation_matrix.csv        — full pairwise correlation matrix
  tables/10_topic_clusters.csv            — field -> latent-block assignment
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_youtube_creators, load_reddit_submissions,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR, get_topic_cols,
)
from analysis.shared.aggregators import (
    aggregate_creator_daily, aggregate_reddit_daily, resample_weekly,
)

MIN_PAIRS  = 30          # minimum overlapping weeks to compute a correlation
SOURCE     = 'creator'   # 'creator', 'reddit', or 'both' (average)
BLOCK_DIST = 0.75        # 1-|r| cut for the headline latent-block count.
                         # 0.75 == |r| >= 0.25 ~= the mean off-diagonal r (the
                         # general-factor floor), so a block = topics co-moving
                         # ABOVE the ambient general-mood level.

#%%
ensure_output_dirs()

yt = load_youtube_creators()
rd = load_reddit_submissions()
topic_cols = get_topic_cols(yt)

#%%
# Weekly aggregates — unweighted
creator_daily = aggregate_creator_daily(yt, topic_cols)
reddit_daily  = aggregate_reddit_daily(rd, topic_cols)
creator_weekly = resample_weekly(creator_daily)
reddit_weekly  = resample_weekly(reddit_daily)

# Select source
if SOURCE == 'creator':
    weekly = creator_weekly[[f'{c}_unweighted' for c in topic_cols if f'{c}_unweighted' in creator_weekly.columns]]
elif SOURCE == 'reddit':
    weekly = reddit_weekly[[f'{c}_unweighted' for c in topic_cols if f'{c}_unweighted' in reddit_weekly.columns]]
else:
    c_w = creator_weekly[[f'{c}_unweighted' for c in topic_cols if f'{c}_unweighted' in creator_weekly.columns]]
    r_w = reddit_weekly[[f'{c}_unweighted'  for c in topic_cols if f'{c}_unweighted' in reddit_weekly.columns]]
    weekly = (c_w + r_w) / 2

# Clean column names for display
weekly.columns = [c.replace('_unweighted', '').replace('topic__', '') for c in weekly.columns]

# Drop columns with too few observations
keep = [c for c in weekly.columns if weekly[c].notna().sum() >= MIN_PAIRS]
weekly = weekly[keep]
print(f'Fields included: {len(keep)} (of {len(topic_cols)} topic fields)')

#%%
# ── Pairwise correlations (descriptive; pairwise-complete, min overlap) ───────
# pandas .corr uses pairwise-complete observations; min_periods leaves pairs
# with fewer than MIN_PAIRS overlapping weeks as NaN. No p-values by design
# (see module docstring): weekly sentiment is autocorrelated, so p-values are
# anti-conservative, and the r=0 null is tautologically false under a general
# game-mood factor. This matrix is read descriptively, not tested.
corr_mat = weekly.corr(method='pearson', min_periods=MIN_PAIRS)
corr_mat.to_csv(TABLES_DIR / '10_correlation_matrix.csv')

n_sparse = int(np.triu(corr_mat.isna().values, k=1).sum())
print(f'Pairs below {MIN_PAIRS}-week overlap (left NaN): {n_sparse}')

#%%
# ── Distance + linkage ───────────────────────────────────────────────────────
# 1-|r| distance: topics that track the same underlying mood cluster together
# regardless of sign. Average linkage is the conventional pairing for a
# precomputed correlation distance (Ward assumes point coordinates, not a
# distance matrix). Replaces the earlier ward/euclidean-on-raw-r choice.
corr_filled = corr_mat.fillna(0.0)          # no-op on current data (0 sparse pairs);
np.fill_diagonal(corr_filled.values, 1.0)   # guards a sparser future game/source

dist = 1.0 - corr_filled.abs().values
dist = (dist + dist.T) / 2.0                # symmetrize away float asymmetry
np.fill_diagonal(dist, 0.0)
link = linkage(squareform(dist, checks=False), method='average')

#%%
# ── Latent-block count — how few dimensions topic sentiment really spans ──────
# Threshold-dependent and descriptive, not inferential. Reported across cuts so
# the "~N latent factors" framing is honest about its sensitivity.
for t in (0.80, BLOCK_DIST, 0.65, 0.50):
    nb = int(fcluster(link, t=t, criterion='distance').max())
    print(f'Latent blocks at 1-|r| <= {t:.2f}  (|r| >= {1 - t:.2f}): {nb}')

blocks   = fcluster(link, t=BLOCK_DIST, criterion='distance')
n_blocks = int(blocks.max())
(pd.DataFrame({'field': list(corr_filled.columns), 'block': blocks})
   .sort_values(['block', 'field'])
   .to_csv(TABLES_DIR / '10_topic_clusters.csv', index=False))

#%%
# ── Clustered heatmap ────────────────────────────────────────────────────────
fig_h = max(10, len(keep) * 0.28)
fig_w = max(12, len(keep) * 0.28)

cg = sns.clustermap(
    corr_filled,
    row_linkage=link,
    col_linkage=link,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    figsize=(fig_w, fig_h),
    xticklabels=True,
    yticklabels=True,
    linewidths=0,
    cbar_kws={'label': 'Pearson r'},
)
cg.ax_heatmap.tick_params(axis='x', labelsize=6, rotation=90)
cg.ax_heatmap.tick_params(axis='y', labelsize=6)
cg.figure.suptitle(
    f'Topic sentiment co-movement — descriptive EDA (no inference)\n'
    f'Source: {SOURCE}  |  n_fields={len(keep)}  |  '
    f'~{n_blocks} latent blocks at |r| >= {1 - BLOCK_DIST:.2f}  (1-|r|, average linkage)',
    y=1.01, fontsize=12,
)

out = FIGURES_DIR / '10_correlation_clustermap.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
