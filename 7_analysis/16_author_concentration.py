#%%
"""
16_author_concentration.py
---------------------------
Tests whether the submission and comment sentiment signals are driven by a few
vocal power-users ("disinformation dozen" dynamic) or distributed broadly across
the community.

Every Reddit-based measure in this paper collapses thousands of individual
submissions or comments into a single weekly community-sentiment signal; this
check asks whether those weekly aggregates reflect the broad community or are an
artefact of a small number of hyper-prolific accounts.

Four analyses:

  1. Volume concentration
     Distribution of posts per author (histogram + Lorenz curve). Gini
     coefficient and top-decile share for submissions and comments separately.

  2. Top-author profile
     For the 25 most prolific authors in each dataset: post count, mean
     sentiment, and score (engagement footprint). Compared to the rest-of-
     community mean. Tests whether top authors are sentiment outliers.

  3. Posting-rate red flags
     Accounts averaging >= 20 posts/day across their active window — a rate high
     enough to suggest automation rather than an unusually engaged human. Because
     submissions/comments are a ~20% sample, observed rates run ~5x below true
     rates, so this is a conservative bar.

  4. Signal sensitivity
     Recomputes the weekly aggregate sentiment signal after progressively
     removing the top 1%, 5%, 10%, and 20% of authors by post count. Reports
     Pearson r between each trimmed signal and the full signal. If r ≈ 1.0
     throughout, the aggregate is robust to vocal minorities. If r drops, the
     signal is disproportionately shaped by a few accounts.

Excluded from all analyses: '[deleted]', '[removed]', 'AutoModerator',
'destiny2-ModTeam' — these are not real community members.

Outputs:
  figures/16a_author_volume_distribution.png  — histogram + Lorenz curve (2×2)
  figures/16b_top_author_sentiment.png        — top-25 authors by volume vs. community mean
  figures/16c_signal_sensitivity.png          — weekly signal under progressive trimming
  tables/16_author_concentration.csv          — Gini, top-N% shares, top-25 tables
  tables/16_signal_sensitivity.csv            — per-trim-level Pearson r and signal stats
"""

import contextlib
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Suppress FreeType errors from OneDrive cloud-only font placeholder files
import matplotlib.font_manager as _fm
_orig_addfont = _fm.FontManager.addfont
def _silent_addfont(self, path):
    with contextlib.redirect_stderr(io.StringIO()):
        _orig_addfont(self, path)
_fm.FontManager.addfont = _silent_addfont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shared.loaders import (
    load_reddit_submissions, load_comments_master,
    ensure_output_dirs, FIGURES_DIR, TABLES_DIR,
)
from analysis.shared.game_config import get_config

# Non-community accounts, per game. Each game's moderator account differs
# ('deadbydaylight-ModTeam', 'GlobalOffensive-ModTeam', ...), so hard-coding the
# D2 one left thousands of mod posts in the other games' distributions.
EXCLUDED_AUTHORS = set(get_config().get(
    'excluded_authors', ['[deleted]', '[removed]', 'AutoModerator']))
TOP_N       = 25
TRIM_LEVELS = [0.01, 0.05, 0.10, 0.20]
MIN_WEEKS   = 10
BOT_POSTS_PER_DAY = 20   # avg posts/day at/above which an account is flagged as
                         # potentially automated. Observed rate runs ~5x below the
                         # true rate (the ~20% sample), so this is a conservative bar.


#%%
ensure_output_dirs()

rd = load_reddit_submissions()
cm = load_comments_master()

rd = rd[~rd['author'].isin(EXCLUDED_AUTHORS)].copy()
cm = cm[~cm['author'].isin(EXCLUDED_AUTHORS)].copy()

print(f'Submissions after exclusions: {len(rd):,}  ({rd["author"].nunique():,} unique authors)')
print(f'Comments after exclusions:    {len(cm):,}  ({cm["author"].nunique():,} unique authors)')
print('Note: submissions are a 20% sample — raw counts underestimate true activity.')


#%%
# ── Helpers ────────────────────────────────────────────────────────────────────
def gini(values: np.ndarray) -> float:
    """Gini coefficient on a 1-D array of non-negative counts."""
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def lorenz(values: np.ndarray, n_points: int = 1_000):
    """Downsampled Lorenz curve; avoids rendering 200K+ point polygons."""
    v = np.sort(values.astype(float))
    cum_share_full = np.concatenate([[0], np.cumsum(v) / v.sum()])
    cum_pop_full   = np.linspace(0, 1, len(cum_share_full))
    idx = np.linspace(0, len(cum_pop_full) - 1, min(n_points, len(cum_pop_full))).astype(int)
    return cum_pop_full[idx], cum_share_full[idx]


def top_pct_share(counts: pd.Series, pct: float) -> float:
    """Fraction of total posts authored by the top `pct` fraction of authors."""
    k = max(1, int(np.ceil(len(counts) * pct)))
    return counts.nlargest(k).sum() / counts.sum()


def weekly_signal(df: pd.DataFrame, sentiment_col: str) -> pd.Series:
    """Unweighted weekly mean sentiment, resampled to W-MON."""
    return df.set_index('date')[sentiment_col].resample('W-MON').mean()


def ascii_name(name: str) -> str:
    return name.encode('ascii', errors='replace').decode('ascii')


#%%
# ── Analysis 1: Volume concentration ──────────────────────────────────────────
sub_counts = rd.groupby('author').size().sort_values(ascending=False)
cmt_counts = cm.groupby('author').size().sort_values(ascending=False)

sub_gini = gini(sub_counts.values)
cmt_gini = gini(cmt_counts.values)

print(f'\nGini coefficient — submissions: {sub_gini:.4f}  |  comments: {cmt_gini:.4f}')

for label, counts in [('Submissions', sub_counts), ('Comments', cmt_counts)]:
    print(f'\n{label} author concentration:')
    for p in [0.01, 0.05, 0.10]:
        share = top_pct_share(counts, p)
        n_authors = max(1, int(np.ceil(len(counts) * p)))
        print(f'  Top {p*100:.0f}% of authors ({n_authors:,} accounts): '
              f'{share*100:.1f}% of posts')


#%%
# ── Figure 16a: Histogram + Lorenz curve ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (counts, label, color) in enumerate([
    (sub_counts, 'Submissions', '#27ae60'),
    (cmt_counts, 'Comments',    '#e67e22'),
]):
    ax_hist   = axes[0, col]
    ax_lorenz = axes[1, col]

    # Histogram (log-log)
    bins = np.logspace(0, np.log10(counts.max()), 60)
    ax_hist.hist(counts.values, bins=bins, color=color, alpha=0.75, edgecolor='white', lw=0.3)
    ax_hist.set_xscale('log')
    ax_hist.set_yscale('log')
    ax_hist.set_xlabel('Posts per author', fontsize=10)
    ax_hist.set_ylabel('Number of authors', fontsize=10)
    ax_hist.set_title(f'{label} — post volume distribution\n'
                      f'({counts.sum():,} posts, {len(counts):,} authors)', fontsize=11)
    ax_hist.grid(True, alpha=0.25)
    ax_hist.axvline(counts.iloc[0], color='red', lw=1.2, linestyle='--', alpha=0.7)
    ax_hist.text(counts.iloc[0] * 1.05, 1.5,
                 f'Top: {ascii_name(counts.index[0])}\n({counts.iloc[0]} posts)',
                 fontsize=7.5, color='red', va='bottom')

    # Lorenz curve
    cum_pop, cum_share = lorenz(counts.values)
    ax_lorenz.plot(cum_pop * 100, cum_share * 100, color=color, lw=2)
    ax_lorenz.plot([0, 100], [0, 100], color='gray', lw=1, linestyle='--', alpha=0.6)
    ax_lorenz.fill_between(cum_pop * 100, cum_pop * 100, cum_share * 100,
                           alpha=0.12, color=color)
    ax_lorenz.set_xlabel('Cumulative % of authors (least → most prolific)', fontsize=10)
    ax_lorenz.set_ylabel('Cumulative % of posts', fontsize=10)
    ax_lorenz.set_title(f'{label} — Lorenz curve  (Gini = {gini(counts.values):.3f})',
                        fontsize=11)
    ax_lorenz.grid(True, alpha=0.25)
    p1_share = top_pct_share(counts, 0.01)
    ax_lorenz.annotate(f'Top 1% → {p1_share*100:.0f}% of posts',
                       xy=(99, p1_share * 100), xytext=(70, p1_share * 100 - 12),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1),
                       fontsize=8.5)

fig.suptitle('Author post-volume concentration — submissions vs. comments\n'
             f'(excludes {", ".join(sorted(EXCLUDED_AUTHORS))})',
             fontsize=13)
plt.tight_layout()
out_a = FIGURES_DIR / '16a_author_volume_distribution.png'
plt.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_a}')
plt.show()


#%%
# ── Analysis 2: Top-author sentiment profile ──────────────────────────────────
def top_author_profile(df: pd.DataFrame, sentiment_col: str,
                       label: str) -> tuple[pd.DataFrame, float]:
    counts  = df.groupby('author').size().rename('n_posts')
    sent    = df.groupby('author')[sentiment_col].mean().rename('mean_sentiment')
    scores  = df.groupby('author')['score'].mean().rename('mean_score')
    profile = pd.concat([counts, sent, scores], axis=1).sort_values('n_posts', ascending=False)
    profile['pct_of_total'] = profile['n_posts'] / len(df) * 100

    overall_mean = df[sentiment_col].mean()
    print(f'\n{label} — overall mean {sentiment_col}: {overall_mean:.4f}')
    print(f'Top {TOP_N} authors:')
    print(profile.head(TOP_N).to_string())
    return profile, overall_mean

sub_profile, sub_mean = top_author_profile(rd, 'overall__score',  'Submissions')
cmt_profile, cmt_mean = top_author_profile(cm, 'sentiment_score', 'Comments')


#%%
# ── Figure 16b: Top-author sentiment vs. community mean ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (profile, overall_mean, label, color) in zip(axes, [
    (sub_profile, sub_mean, 'Submissions', '#27ae60'),
    (cmt_profile, cmt_mean, 'Comments',    '#e67e22'),
]):
    top = profile.head(TOP_N).copy().reset_index()
    x   = np.arange(len(top))

    bars = ax.bar(x, top['mean_sentiment'], color=color, alpha=0.75,
                  width=0.6, label='Mean sentiment')
    ax.axhline(overall_mean, color='black', lw=1.5, linestyle='--',
               label=f'Community mean ({overall_mean:.3f})')
    ax.axhline(0, color='gray', lw=0.6)

    for bar, val in zip(bars, top['mean_sentiment']):
        if abs(val - overall_mean) > 0.3:
            bar.set_color('#e74c3c')
            bar.set_alpha(0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{ascii_name(row['author'])}\n({int(row['n_posts'])})"
         for _, row in top.iterrows()],
        rotation=55, ha='right', fontsize=7,
    )
    ax.set_ylabel('Mean sentiment score', fontsize=10)
    ax.set_title(f'{label} — top {TOP_N} authors by post count\n'
                 f'(red = |mean − community mean| > 0.3)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Top-author sentiment profiles vs. community mean\n'
             '(post count shown in parentheses under each author name)',
             fontsize=13)
plt.tight_layout()
out_b = FIGURES_DIR / '16b_top_author_sentiment.png'
plt.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_b}')
plt.show()


#%%
# ── Analysis 3: High-frequency posting red flags ──────────────────────────────
def posting_rate_flags(df: pd.DataFrame, date_col: str = 'date',
                       label: str = '') -> pd.DataFrame:
    """Accounts whose average rate meets or exceeds BOT_POSTS_PER_DAY posts/day
    across their active window — a level suggestive of automation, not merely
    heavy human use."""
    grp = df.groupby('author').agg(
        n_posts    = (date_col, 'count'),
        first_post = (date_col, 'min'),
        last_post  = (date_col, 'max'),
    )
    grp['active_days'] = (grp['last_post'] - grp['first_post']).dt.days + 1
    grp['posts_per_day'] = grp['n_posts'] / grp['active_days'].clip(lower=1)
    flagged = grp[grp['posts_per_day'] >= BOT_POSTS_PER_DAY].sort_values('posts_per_day', ascending=False)
    print(f'\n{label} — accounts averaging >= {BOT_POSTS_PER_DAY} posts/day '
          f'(potential automation): {len(flagged)}')
    if len(flagged) > 0:
        print(flagged.head(20).to_string())
    return flagged

# Persist the flags (and how much of each layer they account for) so the paper
# cites a table rather than a printout.
_flag_rows = []
for _df, _label in [(rd, 'submissions'), (cm, 'comments')]:
    _flagged = posting_rate_flags(_df, label=_label.capitalize())
    for _author, _r in _flagged.iterrows():
        _flag_rows.append({'source': _label, 'author': _author, 'n_posts': int(_r['n_posts']),
                           'active_days': int(_r['active_days']),
                           'posts_per_day': round(float(_r['posts_per_day']), 2),
                           'pct_of_layer': round(100 * _r['n_posts'] / len(_df), 3)})
pd.DataFrame(_flag_rows, columns=['source', 'author', 'n_posts', 'active_days', 'posts_per_day',
                                  'pct_of_layer']).to_csv(TABLES_DIR / '16_posting_rate_flags.csv', index=False)


#%%
# ── Analysis 4: Signal sensitivity to top-author removal ─────────────────────
sub_full = weekly_signal(rd, 'overall__score')
cmt_full = weekly_signal(cm, 'sentiment_score')

# Pre-compute all trimmed signals once; reused for both table and figure
trimmed_cache: dict[tuple[str, float], pd.Series] = {}
for trim in TRIM_LEVELS:
    for label, df, profile, sentiment_col in [
        ('submissions', rd, sub_profile, 'overall__score'),
        ('comments',    cm, cmt_profile, 'sentiment_score'),
    ]:
        n_drop = max(1, int(np.ceil(len(profile) * trim)))
        drop_authors = set(profile.head(n_drop).index)
        trimmed_cache[(label, trim)] = weekly_signal(
            df[~df['author'].isin(drop_authors)], sentiment_col
        )

sensitivity_records = []
for trim in TRIM_LEVELS:
    for label, df, profile, full_signal in [
        ('submissions', rd, sub_profile, sub_full),
        ('comments',    cm, cmt_profile, cmt_full),
    ]:
        n_drop = max(1, int(np.ceil(len(profile) * trim)))
        trimmed_signal = trimmed_cache[(label, trim)]

        both = pd.concat([full_signal, trimmed_signal], axis=1).dropna()
        both.columns = ['full', 'trimmed']
        if len(both) < MIN_WEEKS:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(both['full'], both['trimmed'])

        pct_posts = profile.head(n_drop)['n_posts'].sum() / len(df) * 100
        sensitivity_records.append({
            'source':            label,
            'trim_pct_authors':  trim * 100,
            'n_authors_dropped': n_drop,
            'pct_posts_dropped': round(pct_posts, 2),
            'r_vs_full':         round(r, 4),
            'p_value':           round(p, 6) if not np.isnan(p) else np.nan,
            'n_weeks':           len(both),
        })
        print(f'[{label}] trim top {trim*100:.0f}% ({n_drop} authors, '
              f'{pct_posts:.1f}% of posts) -> r={r:.4f}')

sens_df = pd.DataFrame(sensitivity_records)


#%%
# ── Figure 16c: Signal sensitivity ───────────────────────────────────────────
fig, axes = plt.subplots(2, len(TRIM_LEVELS) + 1, figsize=(18, 9), sharey='row')

for row, (label, df, profile, sentiment_col, full_signal, color) in enumerate([
    ('Submissions', rd, sub_profile, 'overall__score', sub_full, '#27ae60'),
    ('Comments',    cm, cmt_profile, 'sentiment_score', cmt_full, '#e67e22'),
]):
    axes[row, 0].plot(full_signal.index, full_signal.values, color=color, lw=1.5)
    axes[row, 0].axhline(0, color='gray', lw=0.6)
    axes[row, 0].set_title('Full signal\n(all authors)', fontsize=9)
    axes[row, 0].set_ylabel(f'{label}\nMean sentiment', fontsize=9)
    axes[row, 0].grid(True, alpha=0.2)

    for col_idx, trim in enumerate(TRIM_LEVELS, start=1):
        n_drop = max(1, int(np.ceil(len(profile) * trim)))
        trimmed_signal = trimmed_cache[(label.lower(), trim)]

        both = pd.concat([full_signal, trimmed_signal], axis=1).dropna()
        both.columns = ['full', 'trimmed']
        r = pearsonr(both['full'], both['trimmed'])[0] if len(both) >= MIN_WEEKS else np.nan
        pct_posts = profile.head(n_drop)['n_posts'].sum() / len(df) * 100

        ax = axes[row, col_idx]
        ax.plot(full_signal.index, full_signal.values,
                color=color, lw=1, alpha=0.35, linestyle='--', label='Full')
        ax.plot(trimmed_signal.index, trimmed_signal.values,
                color=color, lw=1.5, label='Trimmed')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f'Drop top {trim*100:.0f}% authors\n'
                     f'({n_drop} accts, {pct_posts:.1f}% posts)\nr={r:.4f}',
                     fontsize=9)
        ax.grid(True, alpha=0.2)
        if col_idx == 1:
            ax.legend(fontsize=7)

fig.suptitle('Weekly sentiment signal robustness to top-author removal\n'
             '(dashed = full signal, solid = trimmed; r = Pearson correlation with full)',
             fontsize=12)
plt.tight_layout()
out_c = FIGURES_DIR / '16c_signal_sensitivity.png'
plt.savefig(out_c, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_c}')
plt.show()


#%%
# ── Save tables ────────────────────────────────────────────────────────────────
conc_rows = []
for label, counts, gini_val in [
    ('submissions', sub_counts, sub_gini),
    ('comments',    cmt_counts, cmt_gini),
]:
    row = {
        'source':    label,
        'n_authors': len(counts),
        'n_posts':   int(counts.sum()),
        'gini':      round(gini_val, 4),
    }
    for p in [0.01, 0.05, 0.10, 0.20]:
        row[f'top_{int(p*100)}pct_share'] = round(top_pct_share(counts, p), 4)
    conc_rows.append(row)

conc_df = pd.DataFrame(conc_rows)
conc_path = TABLES_DIR / '16_author_concentration.csv'
conc_df.to_csv(conc_path, index=False)
print(f'\nSaved: {conc_path}')

sub_top = sub_profile.head(TOP_N).reset_index()
sub_top['source'] = 'submissions'
cmt_top = cmt_profile.head(TOP_N).reset_index()
cmt_top['source'] = 'comments'
top_path = TABLES_DIR / '16_top_author_profiles.csv'
pd.concat([sub_top, cmt_top], ignore_index=True).to_csv(top_path, index=False)
print(f'Saved: {top_path}')

sens_path = TABLES_DIR / '16_signal_sensitivity.csv'
sens_df.to_csv(sens_path, index=False)
print(f'Saved: {sens_path}')

print(f'\nConcentration summary:\n{conc_df.to_string(index=False)}')
print(f'\nSignal sensitivity:\n{sens_df.to_string(index=False)}')
print('\nDone.')
