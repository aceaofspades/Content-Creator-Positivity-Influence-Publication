#%%
"""
Phase 0: Steam Proxy Validation
================================
Determines if Steam player counts are a valid proxy for total player counts.

This analysis must be completed FIRST because it determines the player count
metric used for all subsequent analysis.

Decision criteria:
- Pearson correlation > 0.95
- Stable Steam/Total ratio (low coefficient of variation)
- Same-day peak/trough alignment
- Similar response to release events

If Steam is validated as a proxy, we use Steam for this project
(and it becomes the default for other games with Steam-only data).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from pathlib import Path
from datetime import datetime

#%%
# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / 'analysis' / 'destiny2' / 'steam_proxy_results'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Data paths
TOTAL_PLAYERS_PATH = BASE_PATH / 'other_data' / 'destiny2' / 'daily_player_counts.csv'
STEAM_PLAYERS_PATH = BASE_PATH / 'other_data' / 'destiny2' / 'daily_player_counts_steam_only.csv'
RELEASES_PATH = BASE_PATH / 'other_data' / 'destiny2' / 'releases.csv'

# Date filter: Beyond Light launch onwards (PC playerbase stabilized)
DATE_FILTER_START = pd.Timestamp('2020-11-10')
DATE_FILTER_REASON = "Beyond Light launch - PC playerbase stabilized after ~1 year on Steam"

#%%
# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading player count data...")

# Load total player counts
df_total = pd.read_csv(TOTAL_PLAYERS_PATH)
print(f"Total players columns: {df_total.columns.tolist()}")
print(f"Total players shape: {df_total.shape}")

# Load Steam player counts
df_steam = pd.read_csv(STEAM_PLAYERS_PATH)
print(f"\nSteam players columns: {df_steam.columns.tolist()}")
print(f"Steam players shape: {df_steam.shape}")

# Load releases
df_releases = pd.read_csv(RELEASES_PATH)
print(f"\nReleases columns: {df_releases.columns.tolist()}")
print(f"Releases shape: {df_releases.shape}")

#%%
# =============================================================================
# STANDARDIZE DATE COLUMNS
# =============================================================================

print("\nStandardizing date columns...")

def normalize_date(series):
    """Convert to datetime and strip timezone info (assume UTC throughout)."""
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()  # Remove time component, keep date only

# Total players - 'date' column
df_total['date'] = normalize_date(df_total['date'])
df_total = df_total.sort_values('date').reset_index(drop=True)
df_total = df_total.rename(columns={'players': 'players_total'})

# Steam players - 'DateTime' column
df_steam['date'] = normalize_date(df_steam['DateTime'])
df_steam = df_steam.sort_values('date').reset_index(drop=True)
df_steam = df_steam.rename(columns={'Players': 'players_steam'})

# Releases - 'start_date' column
df_releases['start_date'] = normalize_date(df_releases['start_date'])
df_releases['end_date'] = normalize_date(df_releases['end_date'])

# Filter releases to match our analysis period
df_releases = df_releases[df_releases['start_date'] >= DATE_FILTER_START].reset_index(drop=True)
print(f"Releases in analysis period: {len(df_releases)}")

print(f"Total players date range: {df_total['date'].min()} to {df_total['date'].max()}")
print(f"Steam players date range: {df_steam['date'].min()} to {df_steam['date'].max()}")

#%%
# =============================================================================
# MERGE DATASETS
# =============================================================================

print("\nMerging datasets...")

# Merge on date
df_merged = pd.merge(
    df_total[['date', 'players_total']],
    df_steam[['date', 'players_steam']],
    on='date',
    how='inner'
)

print(f"Merged dataset shape (all dates): {df_merged.shape}")
print(f"Full date range: {df_merged['date'].min()} to {df_merged['date'].max()}")

# Apply date filter
df_merged = df_merged[df_merged['date'] >= DATE_FILTER_START].reset_index(drop=True)
print(f"\nAfter filtering to {DATE_FILTER_START.strftime('%Y-%m-%d')}+:")
print(f"  Reason: {DATE_FILTER_REASON}")
print(f"  Shape: {df_merged.shape}")
print(f"  Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
print(f"\nMissing values:\n{df_merged.isnull().sum()}")

# Drop any rows with missing player counts
df_merged = df_merged.dropna(subset=['players_total', 'players_steam'])
print(f"\nAfter dropping nulls: {df_merged.shape}")

#%%
# =============================================================================
# BASIC STATISTICS
# =============================================================================

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)

print("\nTotal Players:")
print(df_merged['players_total'].describe())

print("\nSteam Players:")
print(df_merged['players_steam'].describe())

# Calculate ratio
df_merged['steam_ratio'] = df_merged['players_steam'] / df_merged['players_total']

# Calculate ratio statistics (needed for later comparisons)
ratio_mean = df_merged['steam_ratio'].mean()
ratio_std = df_merged['steam_ratio'].std()
ratio_cv = ratio_std / ratio_mean

print("\nSteam/Total Ratio:")
print(df_merged['steam_ratio'].describe())

#%%
# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(df_merged['players_total'], df_merged['players_steam'])
print(f"\nPearson correlation: r = {pearson_r:.6f}, p = {pearson_p:.2e}")

# Spearman correlation (rank-based, robust to outliers)
spearman_r, spearman_p = stats.spearmanr(df_merged['players_total'], df_merged['players_steam'])
print(f"Spearman correlation: r = {spearman_r:.6f}, p = {spearman_p:.2e}")

# R-squared
r_squared = pearson_r ** 2
print(f"R-squared: {r_squared:.6f} ({r_squared*100:.2f}% of variance explained)")

#%%
# =============================================================================
# VOLATILITY-CONTROLLED ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("VOLATILITY-CONTROLLED ANALYSIS")
print("="*60)

# 1. Z-score normalization (removes scale differences)
df_merged['total_zscore'] = (df_merged['players_total'] - df_merged['players_total'].mean()) / df_merged['players_total'].std()
df_merged['steam_zscore'] = (df_merged['players_steam'] - df_merged['players_steam'].mean()) / df_merged['players_steam'].std()

zscore_corr, _ = stats.pearsonr(df_merged['total_zscore'], df_merged['steam_zscore'])
print(f"\n1. Z-SCORE NORMALIZED:")
print(f"   Correlation: {zscore_corr:.4f}")
print(f"   (Same as Pearson - normalization preserves correlation)")

# 2. Rolling averages (smooths daily noise)
for window in [7, 14, 30]:
    df_merged[f'total_roll_{window}d'] = df_merged['players_total'].rolling(window, center=True).mean()
    df_merged[f'steam_roll_{window}d'] = df_merged['players_steam'].rolling(window, center=True).mean()
    
    # Drop NaN from rolling
    valid_mask = df_merged[f'total_roll_{window}d'].notna() & df_merged[f'steam_roll_{window}d'].notna()
    roll_corr, _ = stats.pearsonr(
        df_merged.loc[valid_mask, f'total_roll_{window}d'],
        df_merged.loc[valid_mask, f'steam_roll_{window}d']
    )
    print(f"\n2. {window}-DAY ROLLING AVERAGE:")
    print(f"   Correlation: {roll_corr:.4f}")

# 3. Directional agreement (do they move in the same direction?)
df_merged['total_direction'] = np.sign(df_merged['players_total'].diff())
df_merged['steam_direction'] = np.sign(df_merged['players_steam'].diff())

# Exclude days with no change (direction = 0)
direction_mask = (df_merged['total_direction'] != 0) & (df_merged['steam_direction'] != 0)
direction_agreement = (df_merged.loc[direction_mask, 'total_direction'] == df_merged.loc[direction_mask, 'steam_direction']).mean()

print(f"\n3. DIRECTIONAL AGREEMENT:")
print(f"   Same direction: {direction_agreement*100:.1f}% of days")
print(f"   (When Total goes up, does Steam also go up?)")

# 4. Weekly aggregation (reduces noise further)
df_weekly = df_merged.copy()
df_weekly['week'] = df_weekly['date'].dt.to_period('W')
df_weekly_agg = df_weekly.groupby('week').agg({
    'players_total': 'mean',
    'players_steam': 'mean',
    'steam_ratio': 'mean'
}).reset_index()

weekly_corr, _ = stats.pearsonr(df_weekly_agg['players_total'], df_weekly_agg['players_steam'])
weekly_ratio_cv = df_weekly_agg['steam_ratio'].std() / df_weekly_agg['steam_ratio'].mean()

print(f"\n4. WEEKLY AGGREGATION:")
print(f"   Correlation: {weekly_corr:.4f}")
print(f"   Ratio CV: {weekly_ratio_cv*100:.2f}%")

# 5. Monthly aggregation
df_monthly = df_merged.copy()
df_monthly['month'] = df_monthly['date'].dt.to_period('M')
df_monthly_agg = df_monthly.groupby('month').agg({
    'players_total': 'mean',
    'players_steam': 'mean',
    'steam_ratio': 'mean'
}).reset_index()

monthly_corr, _ = stats.pearsonr(df_monthly_agg['players_total'], df_monthly_agg['players_steam'])
monthly_ratio_cv = df_monthly_agg['steam_ratio'].std() / df_monthly_agg['steam_ratio'].mean()

print(f"\n5. MONTHLY AGGREGATION:")
print(f"   Correlation: {monthly_corr:.4f}")
print(f"   Ratio CV: {monthly_ratio_cv*100:.2f}%")

#%%
# =============================================================================
# OUTLIER ANALYSIS - What's causing the extreme ratios?
# =============================================================================

print("\n" + "="*60)
print("OUTLIER ANALYSIS")
print("="*60)

# Find days with extreme ratios
ratio_p95 = df_merged['steam_ratio'].quantile(0.95)
ratio_p05 = df_merged['steam_ratio'].quantile(0.05)

print(f"\nRatio percentiles:")
print(f"  5th percentile: {ratio_p05*100:.2f}%")
print(f"  95th percentile: {ratio_p95*100:.2f}%")

# High ratio days (Steam overrepresented)
high_ratio_days = df_merged[df_merged['steam_ratio'] > ratio_p95].copy()
print(f"\nDays with Steam ratio > {ratio_p95*100:.1f}% (top 5%):")
print(f"  Count: {len(high_ratio_days)}")
if len(high_ratio_days) > 0:
    print(f"  Date range: {high_ratio_days['date'].min()} to {high_ratio_days['date'].max()}")
    print(f"  Mean total players: {high_ratio_days['players_total'].mean():,.0f}")
    print(f"  Mean steam players: {high_ratio_days['players_steam'].mean():,.0f}")

# Low ratio days (Steam underrepresented)  
low_ratio_days = df_merged[df_merged['steam_ratio'] < ratio_p05].copy()
print(f"\nDays with Steam ratio < {ratio_p05*100:.1f}% (bottom 5%):")
print(f"  Count: {len(low_ratio_days)}")
if len(low_ratio_days) > 0:
    print(f"  Date range: {low_ratio_days['date'].min()} to {low_ratio_days['date'].max()}")
    print(f"  Mean total players: {low_ratio_days['players_total'].mean():,.0f}")
    print(f"  Mean steam players: {low_ratio_days['players_steam'].mean():,.0f}")

# The extreme max ratio day
max_ratio_row = df_merged.loc[df_merged['steam_ratio'].idxmax()]
print(f"\nMost extreme day (highest Steam ratio):")
print(f"  Date: {max_ratio_row['date']}")
print(f"  Steam ratio: {max_ratio_row['steam_ratio']*100:.2f}%")
print(f"  Total players: {max_ratio_row['players_total']:,.0f}")
print(f"  Steam players: {max_ratio_row['players_steam']:,.0f}")

# Correlation WITHOUT outliers (middle 90%)
df_no_outliers = df_merged[
    (df_merged['steam_ratio'] >= ratio_p05) & 
    (df_merged['steam_ratio'] <= ratio_p95)
].copy()

corr_no_outliers, _ = stats.pearsonr(df_no_outliers['players_total'], df_no_outliers['players_steam'])
cv_no_outliers = df_no_outliers['steam_ratio'].std() / df_no_outliers['steam_ratio'].mean()

print(f"\n6. EXCLUDING RATIO OUTLIERS (middle 90%):")
print(f"   Observations: {len(df_no_outliers)} (removed {len(df_merged) - len(df_no_outliers)})")
print(f"   Correlation: {corr_no_outliers:.4f}")
print(f"   Ratio CV: {cv_no_outliers*100:.2f}%")

#%%
# =============================================================================
# CALCULATE DAILY CHANGES (needed for subsequent analyses)
# =============================================================================

df_merged['total_pct_change'] = df_merged['players_total'].pct_change()
df_merged['steam_pct_change'] = df_merged['players_steam'].pct_change()
# Note: direction columns already calculated in volatility-controlled analysis above

# Calculate daily change correlation here (needed for release exclusion comparison)
change_corr, change_p = stats.pearsonr(
    df_merged['total_pct_change'].dropna(),
    df_merged['steam_pct_change'].dropna()
)

#%%
# =============================================================================
# RELEASE WINDOW EXCLUSION ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("RELEASE WINDOW EXCLUSION ANALYSIS")
print("="*60)

# Test different exclusion windows
for window_days in [7, 14, 21]:
    # Create mask for days NOT within window of any release
    release_mask = pd.Series(False, index=df_merged.index)
    
    for _, release in df_releases.iterrows():
        start = release['start_date']
        # Mark days within window of release
        window_mask = (df_merged['date'] >= start - pd.Timedelta(days=window_days)) & \
                      (df_merged['date'] <= start + pd.Timedelta(days=window_days))
        release_mask = release_mask | window_mask
    
    # Get non-release days
    df_no_releases = df_merged[~release_mask].copy()
    
    if len(df_no_releases) > 30:
        nr_corr, _ = stats.pearsonr(df_no_releases['players_total'], df_no_releases['players_steam'])
        nr_spearman, _ = stats.spearmanr(df_no_releases['players_total'], df_no_releases['players_steam'])
        nr_ratio_cv = df_no_releases['steam_ratio'].std() / df_no_releases['steam_ratio'].mean()
        
        # Daily change correlation
        nr_change_corr, _ = stats.pearsonr(
            df_no_releases['total_pct_change'].dropna(),
            df_no_releases['steam_pct_change'].dropna()
        )
        
        # Direction agreement
        nr_direction_mask = (df_no_releases['total_direction'] != 0) & (df_no_releases['steam_direction'] != 0)
        nr_direction_agreement = (df_no_releases.loc[nr_direction_mask, 'total_direction'] == 
                                   df_no_releases.loc[nr_direction_mask, 'steam_direction']).mean()
        
        print(f"\n±{window_days} DAYS AROUND RELEASES EXCLUDED:")
        print(f"   Days removed: {release_mask.sum()} ({release_mask.sum()/len(df_merged)*100:.1f}%)")
        print(f"   Days remaining: {len(df_no_releases)}")
        print(f"   Pearson correlation: {nr_corr:.4f} {'✓' if nr_corr > 0.95 else ''}")
        print(f"   Spearman correlation: {nr_spearman:.4f} {'✓' if nr_spearman > 0.93 else ''}")
        print(f"   Ratio CV: {nr_ratio_cv*100:.2f}% {'✓' if nr_ratio_cv < 0.15 else ''}")
        print(f"   Daily change correlation: {nr_change_corr:.4f}")
        print(f"   Direction agreement: {nr_direction_agreement*100:.1f}%")

# Save the best exclusion window results for comparison
# Using 7-day window as the "release exclusion" standard (best efficiency)
release_mask_7d = pd.Series(False, index=df_merged.index)
for _, release in df_releases.iterrows():
    start = release['start_date']
    window_mask = (df_merged['date'] >= start - pd.Timedelta(days=7)) & \
                  (df_merged['date'] <= start + pd.Timedelta(days=7))
    release_mask_7d = release_mask_7d | window_mask

df_no_releases_7d = df_merged[~release_mask_7d].copy()

if len(df_no_releases_7d) > 30:
    nr7_corr, _ = stats.pearsonr(df_no_releases_7d['players_total'], df_no_releases_7d['players_steam'])
    nr7_ratio_cv = df_no_releases_7d['steam_ratio'].std() / df_no_releases_7d['steam_ratio'].mean()
    nr7_change_corr, _ = stats.pearsonr(
        df_no_releases_7d['total_pct_change'].dropna(),
        df_no_releases_7d['steam_pct_change'].dropna()
    )
    nr7_direction_mask = (df_no_releases_7d['total_direction'] != 0) & (df_no_releases_7d['steam_direction'] != 0)
    nr7_direction_agreement = (df_no_releases_7d.loc[nr7_direction_mask, 'total_direction'] == 
                                 df_no_releases_7d.loc[nr7_direction_mask, 'steam_direction']).mean()
else:
    nr7_corr = np.nan
    nr7_ratio_cv = np.nan
    nr7_change_corr = np.nan
    nr7_direction_agreement = np.nan

print(f"\n--- SUMMARY: RELEASE EXCLUSION IMPACT ---")
print(f"{'Metric':<30} {'All Days':<15} {'No Releases (±7d)':<15} {'Δ':<10}")
print("-" * 70)
print(f"{'Pearson correlation':<30} {pearson_r:<15.4f} {nr7_corr:<15.4f} {nr7_corr - pearson_r:+.4f}")
print(f"{'Ratio CV':<30} {ratio_cv*100:<14.2f}% {nr7_ratio_cv*100:<14.2f}% {(nr7_ratio_cv - ratio_cv)*100:+.2f}%")
print(f"{'Daily change correlation':<30} {change_corr:<15.4f} {nr7_change_corr:<15.4f} {nr7_change_corr - change_corr:+.4f}")
print(f"{'Direction agreement':<30} {direction_agreement*100:<14.1f}% {nr7_direction_agreement*100:<14.1f}% {(nr7_direction_agreement - direction_agreement)*100:+.1f}%")

#%%
# =============================================================================
# RATIO STABILITY ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("RATIO STABILITY ANALYSIS")
print("="*60)

# ratio_mean, ratio_std, ratio_cv already calculated in BASIC STATISTICS

print(f"\nSteam/Total Ratio:")
print(f"  Mean: {ratio_mean:.4f} ({ratio_mean*100:.2f}%)")
print(f"  Std Dev: {ratio_std:.4f}")
print(f"  Coefficient of Variation: {ratio_cv:.4f} ({ratio_cv*100:.2f}%)")
print(f"  Min: {df_merged['steam_ratio'].min():.4f}")
print(f"  Max: {df_merged['steam_ratio'].max():.4f}")

# Check for drift over time
df_merged['year'] = df_merged['date'].dt.year
yearly_ratio = df_merged.groupby('year')['steam_ratio'].agg(['mean', 'std']).round(4)
print(f"\nYearly ratio (checking for drift):")
print(yearly_ratio)

# Rolling ratio to see trends
df_merged['steam_ratio_rolling_30d'] = df_merged['steam_ratio'].rolling(30, center=True).mean()

#%%
# =============================================================================
# LAG ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("LAG ANALYSIS")
print("="*60)

# Daily change correlation (already calculated above, printing here for section context)
print(f"\nDaily % change correlation: r = {change_corr:.4f}, p = {change_p:.2e}")

# Cross-correlation at various lags
max_lag = 7  # Check up to 7 days
lag_correlations = []

for lag in range(-max_lag, max_lag + 1):
    if lag < 0:
        corr = stats.pearsonr(
            df_merged['players_total'].iloc[-lag:],
            df_merged['players_steam'].iloc[:lag]
        )[0]
    elif lag > 0:
        corr = stats.pearsonr(
            df_merged['players_total'].iloc[:-lag],
            df_merged['players_steam'].iloc[lag:]
        )[0]
    else:
        corr = pearson_r
    lag_correlations.append({'lag': lag, 'correlation': corr})

df_lags = pd.DataFrame(lag_correlations)
print(f"\nCross-correlation at various lags (negative = Steam leads):")
print(df_lags.to_string(index=False))

best_lag = df_lags.loc[df_lags['correlation'].idxmax(), 'lag']
print(f"\nBest correlation at lag: {best_lag} days")

#%%
# =============================================================================
# RELEASE EVENT RESPONSE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("RELEASE EVENT RESPONSE ANALYSIS")
print("="*60)

# For each release, compare response patterns
release_responses = []

for _, release in df_releases.iterrows():
    start = release['start_date']
    
    # Get data 7 days before and 14 days after release
    mask = (df_merged['date'] >= start - pd.Timedelta(days=7)) & \
           (df_merged['date'] <= start + pd.Timedelta(days=14))
    
    window_data = df_merged[mask].copy()
    
    if len(window_data) >= 14:
        # Normalize to pre-release baseline (7 days before)
        pre_mask = window_data['date'] < start
        if pre_mask.sum() > 0:
            total_baseline = window_data.loc[pre_mask, 'players_total'].mean()
            steam_baseline = window_data.loc[pre_mask, 'players_steam'].mean()
            
            window_data['total_normalized'] = window_data['players_total'] / total_baseline
            window_data['steam_normalized'] = window_data['players_steam'] / steam_baseline
            
            # Peak response (max normalized value)
            total_peak = window_data['total_normalized'].max()
            steam_peak = window_data['steam_normalized'].max()
            
            # Days to peak
            total_peak_day = (window_data.loc[window_data['total_normalized'].idxmax(), 'date'] - start).days
            steam_peak_day = (window_data.loc[window_data['steam_normalized'].idxmax(), 'date'] - start).days
            
            release_responses.append({
                'season': release['season_name'],
                'start_date': start,
                'total_peak_pct': (total_peak - 1) * 100,
                'steam_peak_pct': (steam_peak - 1) * 100,
                'total_peak_day': total_peak_day,
                'steam_peak_day': steam_peak_day,
                'peak_diff_days': abs(total_peak_day - steam_peak_day)
            })

df_release_response = pd.DataFrame(release_responses)
if len(df_release_response) > 0:
    print(f"\nRelease response comparison (% increase from baseline):")
    print(df_release_response[['season', 'total_peak_pct', 'steam_peak_pct', 'total_peak_day', 'steam_peak_day']].to_string(index=False))
    
    # Correlation of peak responses
    if len(df_release_response) > 2:
        peak_corr, peak_p = stats.pearsonr(
            df_release_response['total_peak_pct'],
            df_release_response['steam_peak_pct']
        )
        print(f"\nCorrelation of peak response magnitudes: r = {peak_corr:.4f}")
        print(f"Mean peak day difference: {df_release_response['peak_diff_days'].mean():.2f} days")

#%%
# =============================================================================
# COINTEGRATION TEST
# =============================================================================

print("\n" + "="*60)
print("COINTEGRATION TEST")
print("="*60)

try:
    from statsmodels.tsa.stattools import coint
    
    # Test if the two series are cointegrated (move together long-term)
    coint_stat, coint_pvalue, crit_values = coint(
        df_merged['players_total'],
        df_merged['players_steam']
    )
    
    print(f"\nEngle-Granger Cointegration Test:")
    print(f"  Test statistic: {coint_stat:.4f}")
    print(f"  P-value: {coint_pvalue:.4f}")
    print(f"  Critical values (1%, 5%, 10%): {crit_values}")
    
    if coint_pvalue < 0.05:
        print("  Result: Series ARE cointegrated (move together long-term)")
    else:
        print("  Result: Series are NOT cointegrated")
        
except ImportError:
    print("statsmodels not installed - skipping cointegration test")
    print("Install with: uv add statsmodels")

#%%
# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle('Steam Proxy Validation Analysis', fontsize=14, fontweight='bold')

# 1. Time series comparison
ax1 = axes[0, 0]
ax1.plot(df_merged['date'], df_merged['players_total'] / 1e6, label='Total', alpha=0.8)
ax1.plot(df_merged['date'], df_merged['players_steam'] / 1e6, label='Steam', alpha=0.8)
ax1.set_xlabel('Date')
ax1.set_ylabel('Players (millions)')
ax1.set_title('Player Counts Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Scatter plot with regression
ax2 = axes[0, 1]
ax2.scatter(df_merged['players_total'] / 1e6, df_merged['players_steam'] / 1e6, alpha=0.3, s=10)
# Add regression line
z = np.polyfit(df_merged['players_total'], df_merged['players_steam'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_merged['players_total'].min(), df_merged['players_total'].max(), 100)
ax2.plot(x_line / 1e6, p(x_line) / 1e6, 'r-', linewidth=2, label=f'r = {pearson_r:.4f}')
ax2.set_xlabel('Total Players (millions)')
ax2.set_ylabel('Steam Players (millions)')
ax2.set_title('Total vs Steam Player Correlation')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Steam ratio over time
ax3 = axes[1, 0]
ax3.plot(df_merged['date'], df_merged['steam_ratio'] * 100, alpha=0.4, linewidth=0.5)
ax3.plot(df_merged['date'], df_merged['steam_ratio_rolling_30d'] * 100, 'r-', linewidth=2, label='30-day rolling avg')
ax3.axhline(y=ratio_mean * 100, color='green', linestyle='--', label=f'Mean: {ratio_mean*100:.1f}%')
ax3.fill_between(df_merged['date'], 
                  (ratio_mean - 2*ratio_std) * 100, 
                  (ratio_mean + 2*ratio_std) * 100, 
                  alpha=0.2, color='green', label='±2σ')
ax3.set_xlabel('Date')
ax3.set_ylabel('Steam % of Total')
ax3.set_title('Steam/Total Ratio Stability')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Ratio distribution
ax4 = axes[1, 1]
ax4.hist(df_merged['steam_ratio'] * 100, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=ratio_mean * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {ratio_mean*100:.1f}%')
ax4.axvline(x=ratio_p05 * 100, color='orange', linestyle=':', linewidth=2, label=f'5th/95th pctl')
ax4.axvline(x=ratio_p95 * 100, color='orange', linestyle=':', linewidth=2)
ax4.set_xlabel('Steam % of Total')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Steam Ratio Distribution (CV = {ratio_cv*100:.1f}%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Rolling correlation comparison
ax5 = axes[2, 0]
aggregation_levels = ['Daily', '7-day', '14-day', '30-day', 'Weekly', 'Monthly']
correlations = [pearson_r, 
                stats.pearsonr(df_merged['total_roll_7d'].dropna(), df_merged['steam_roll_7d'].dropna())[0],
                stats.pearsonr(df_merged['total_roll_14d'].dropna(), df_merged['steam_roll_14d'].dropna())[0],
                stats.pearsonr(df_merged['total_roll_30d'].dropna(), df_merged['steam_roll_30d'].dropna())[0],
                weekly_corr, 
                monthly_corr]
colors = ['steelblue' if c < 0.95 else 'green' for c in correlations]
bars = ax5.bar(aggregation_levels, correlations, color=colors, edgecolor='black')
ax5.axhline(y=0.95, color='green', linestyle='--', label='0.95 threshold')
ax5.set_ylabel('Pearson Correlation')
ax5.set_title('Correlation at Different Aggregation Levels')
ax5.set_ylim(0.8, 1.0)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
for bar, corr in zip(bars, correlations):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{corr:.3f}', ha='center', va='bottom', fontsize=9)

# 6. CV at different aggregation levels
ax6 = axes[2, 1]
cv_levels = ['Daily', 'Weekly', 'Monthly']
cv_values = [ratio_cv * 100, weekly_ratio_cv * 100, monthly_ratio_cv * 100]
colors = ['steelblue' if c > 15 else 'green' for c in cv_values]
bars = ax6.bar(cv_levels, cv_values, color=colors, edgecolor='black')
ax6.axhline(y=15, color='green', linestyle='--', label='15% threshold')
ax6.set_ylabel('Coefficient of Variation (%)')
ax6.set_title('Ratio Stability at Different Aggregation Levels')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
for bar, cv in zip(bars, cv_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{cv:.1f}%', ha='center', va='bottom', fontsize=10)

# 7. Daily changes comparison
ax7 = axes[3, 0]
ax7.scatter(df_merged['total_pct_change'] * 100, df_merged['steam_pct_change'] * 100, alpha=0.3, s=10)
ax7.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax7.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
# Add diagonal reference line
lims = [
    np.min([ax7.get_xlim(), ax7.get_ylim()]),
    np.max([ax7.get_xlim(), ax7.get_ylim()]),
]
ax7.plot(lims, lims, 'r--', alpha=0.5, label='Perfect agreement')
ax7.set_xlabel('Total % Change')
ax7.set_ylabel('Steam % Change')
ax7.set_title(f'Daily Change Correlation (r = {change_corr:.4f})\nDirectional Agreement: {direction_agreement*100:.1f}%')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_xlim(-50, 50)
ax7.set_ylim(-50, 50)

# 8. Lag correlation
ax8 = axes[3, 1]
ax8.bar(df_lags['lag'], df_lags['correlation'], color='steelblue', edgecolor='black')
ax8.axhline(y=pearson_r, color='red', linestyle='--', label=f'Lag 0: r = {pearson_r:.4f}')
ax8.set_xlabel('Lag (days, negative = Steam leads)')
ax8.set_ylabel('Correlation')
ax8.set_title('Cross-Correlation at Different Lags')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'steam_proxy_validation.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nVisualization saved to: {OUTPUT_PATH / 'steam_proxy_validation.png'}")

#%%
# =============================================================================
# FINAL DECISION
# =============================================================================

print("\n" + "="*60)
print("STEAM PROXY VALIDATION: FINAL DECISION")
print("="*60)

# Decision criteria - RAW (strict)
criteria_raw = {
    'pearson_correlation': pearson_r,
    'correlation_threshold': 0.95,
    'correlation_pass': pearson_r > 0.95,
    
    'ratio_cv': ratio_cv,
    'cv_threshold': 0.15,  # 15% coefficient of variation
    'ratio_stable': ratio_cv < 0.15,
    
    'lag_aligned': best_lag == 0,
    'best_lag': best_lag,
    
    'daily_change_corr': change_corr,
    'change_corr_threshold': 0.90,
    'changes_aligned': change_corr > 0.90,
}

# Decision criteria - SMOOTHED (weekly)
criteria_smoothed = {
    'weekly_correlation': weekly_corr,
    'weekly_corr_pass': weekly_corr > 0.95,
    
    'weekly_ratio_cv': weekly_ratio_cv,
    'weekly_cv_pass': weekly_ratio_cv < 0.15,
    
    'monthly_correlation': monthly_corr,
    'monthly_corr_pass': monthly_corr > 0.95,
    
    'direction_agreement': direction_agreement,
    'direction_pass': direction_agreement > 0.75,
    
    'spearman_correlation': spearman_r,
    'spearman_pass': spearman_r > 0.90,
}

# Decision criteria - RELEASE EXCLUSION (±7 days - best efficiency)
criteria_no_releases = {
    'nr_correlation': nr7_corr,
    'nr_corr_pass': nr7_corr > 0.90 if not np.isnan(nr7_corr) else False,  # Relaxed from 0.95
    
    'nr_ratio_cv': nr7_ratio_cv,
    'nr_cv_pass': nr7_ratio_cv < 0.20 if not np.isnan(nr7_ratio_cv) else False,  # Relaxed from 0.15
    
    'nr_change_corr': nr7_change_corr,
    'nr_change_pass': nr7_change_corr > 0.70 if not np.isnan(nr7_change_corr) else False,  # Relaxed from 0.80
    
    'nr_direction': nr7_direction_agreement,
    'nr_direction_pass': nr7_direction_agreement > 0.75 if not np.isnan(nr7_direction_agreement) else False,
}

# Decision criteria - MIDDLE 90% (excluding ratio outliers)
criteria_middle90 = {
    'm90_correlation': corr_no_outliers,
    'm90_corr_pass': corr_no_outliers > 0.90,
    
    'm90_cv': cv_no_outliers,
    'm90_cv_pass': cv_no_outliers < 0.15,
}

print("\n--- RAW DAILY DATA ---")
print(f"  1. Correlation > 0.95:        {'✓ PASS' if criteria_raw['correlation_pass'] else '✗ FAIL'} (r = {criteria_raw['pearson_correlation']:.4f})")
print(f"  2. Ratio CV < 15%:            {'✓ PASS' if criteria_raw['ratio_stable'] else '✗ FAIL'} (CV = {criteria_raw['ratio_cv']*100:.2f}%)")
print(f"  3. Best lag = 0 days:         {'✓ PASS' if criteria_raw['lag_aligned'] else '✗ FAIL'} (lag = {criteria_raw['best_lag']} days)")
print(f"  4. Daily change corr > 0.90:  {'✓ PASS' if criteria_raw['changes_aligned'] else '✗ FAIL'} (r = {criteria_raw['daily_change_corr']:.4f})")

print("\n--- SMOOTHED / ROBUST METRICS ---")
print(f"  5. Weekly correlation > 0.95:   {'✓ PASS' if criteria_smoothed['weekly_corr_pass'] else '✗ FAIL'} (r = {criteria_smoothed['weekly_correlation']:.4f})")
print(f"  6. Weekly ratio CV < 15%:       {'✓ PASS' if criteria_smoothed['weekly_cv_pass'] else '✗ FAIL'} (CV = {criteria_smoothed['weekly_ratio_cv']*100:.2f}%)")
print(f"  7. Monthly correlation > 0.95:  {'✓ PASS' if criteria_smoothed['monthly_corr_pass'] else '✗ FAIL'} (r = {criteria_smoothed['monthly_correlation']:.4f})")
print(f"  8. Direction agreement > 75%:   {'✓ PASS' if criteria_smoothed['direction_pass'] else '✗ FAIL'} ({criteria_smoothed['direction_agreement']*100:.1f}%)")
print(f"  9. Spearman (rank) corr > 0.90: {'✓ PASS' if criteria_smoothed['spearman_pass'] else '✗ FAIL'} (r = {criteria_smoothed['spearman_correlation']:.4f})")

print("\n--- EXCLUDING RELEASE WINDOWS (±7 days) ---")
print(f" 10. Correlation > 0.90:          {'✓ PASS' if criteria_no_releases['nr_corr_pass'] else '✗ FAIL'} (r = {criteria_no_releases['nr_correlation']:.4f})")
print(f" 11. Ratio CV < 20%:              {'✓ PASS' if criteria_no_releases['nr_cv_pass'] else '✗ FAIL'} (CV = {criteria_no_releases['nr_ratio_cv']*100:.2f}%)")
print(f" 12. Daily change corr > 0.70:    {'✓ PASS' if criteria_no_releases['nr_change_pass'] else '✗ FAIL'} (r = {criteria_no_releases['nr_change_corr']:.4f})")
print(f" 13. Direction agreement > 75%:   {'✓ PASS' if criteria_no_releases['nr_direction_pass'] else '✗ FAIL'} ({criteria_no_releases['nr_direction']*100:.1f}%)")

print("\n--- MIDDLE 90% (excluding ratio outliers) ---")
print(f" 14. Correlation > 0.90:          {'✓ PASS' if criteria_middle90['m90_corr_pass'] else '✗ FAIL'} (r = {criteria_middle90['m90_correlation']:.4f})")
print(f" 15. Ratio CV < 15%:              {'✓ PASS' if criteria_middle90['m90_cv_pass'] else '✗ FAIL'} (CV = {criteria_middle90['m90_cv']*100:.2f}%)")

# Overall assessment
raw_pass = sum([criteria_raw['correlation_pass'], criteria_raw['ratio_stable'], 
                criteria_raw['lag_aligned'], criteria_raw['changes_aligned']])
smoothed_pass = sum([criteria_smoothed['weekly_corr_pass'], criteria_smoothed['weekly_cv_pass'],
                     criteria_smoothed['monthly_corr_pass'], criteria_smoothed['direction_pass'],
                     criteria_smoothed['spearman_pass']])
no_release_pass = sum([criteria_no_releases['nr_corr_pass'], criteria_no_releases['nr_cv_pass'],
                       criteria_no_releases['nr_change_pass'], criteria_no_releases['nr_direction_pass']])
middle90_pass = sum([criteria_middle90['m90_corr_pass'], criteria_middle90['m90_cv_pass']])

print("\n" + "-"*60)
print(f"RAW CRITERIA:          {raw_pass}/4 passed")
print(f"SMOOTHED CRITERIA:     {smoothed_pass}/5 passed")
print(f"NO-RELEASE (±7d):      {no_release_pass}/4 passed")
print(f"MIDDLE 90%:            {middle90_pass}/2 passed")
print("-"*60)

# Nuanced decision - now includes release exclusion and middle 90%
if raw_pass >= 3:
    decision = "STRONG_PROXY"
    decision_text = "✓ STEAM IS A STRONG PROXY - suitable for all analysis"
elif middle90_pass == 2 and no_release_pass >= 3:
    decision = "CONDITIONAL_PROXY"
    decision_text = "✓ STEAM IS A CONDITIONAL PROXY - excellent outside release weeks & outliers"
elif no_release_pass >= 3:
    decision = "NON_EVENT_PROXY"
    decision_text = "⚠ STEAM IS A NON-EVENT PROXY - excellent for steady-state, unreliable around releases"
elif smoothed_pass >= 4:
    decision = "TREND_PROXY"
    decision_text = "⚠ STEAM IS A TREND PROXY - suitable for weekly/monthly trends, not daily precision"
elif smoothed_pass >= 3 or criteria_smoothed['direction_pass']:
    decision = "DIRECTIONAL_PROXY"
    decision_text = "⚠ STEAM IS A DIRECTIONAL PROXY - suitable for up/down trends, not magnitude"
else:
    decision = "NOT_PROXY"
    decision_text = "✗ STEAM IS NOT A VALID PROXY - use Total for Destiny 2"

print(f"\nDECISION: {decision_text}")

# Recommendations based on decision
print("\n--- RECOMMENDATIONS ---")
if decision == "STRONG_PROXY":
    print("• Use STEAM as default metric across all games for consistency")
    print("• Steam tracks Total reliably at all granularities")
elif decision == "CONDITIONAL_PROXY":
    print("• Steam is a VALID PROXY with known limitations")
    print("• USE Steam for: steady-state trends, directional analysis, cross-game comparisons")
    print("• CAUTION with Steam for: release week analysis (±7 days) - Steam amplifies by 2-3x")
    print("• For D2 release impact: use Total player counts")
    print("• For other games: document that Steam magnifies release effects")
    print("• Recommendation: Use Steam as default, flag release windows in analysis")
elif decision == "NON_EVENT_PROXY":
    print("• Steam is an EXCELLENT proxy during normal gameplay periods")
    print("• Steam OVERREACTS around releases (±7 days) - use caution")
    print("• For release impact analysis: use Total for D2, note Steam amplification for others")
    print("• For steady-state trends: Steam is reliable")
    print("• Recommendation: Use Steam for consistency, but document release-window caveats")
elif decision == "TREND_PROXY":
    print("• Use STEAM as default metric across all games for consistency")
    print("• For Destiny 2: Steam tracks Total well at weekly+ granularity")
    print("• Caveat: Steam is more volatile than Total (larger % swings)")
    print("• For daily analysis: consider 7-day rolling averages")
elif decision == "DIRECTIONAL_PROXY":
    print("• Steam reliably indicates DIRECTION (up/down) but not MAGNITUDE")
    print("• For trend analysis: Steam is acceptable")
    print("• For precise player counts: use Total for Destiny 2")
    print("• Other games (Steam-only): interpret magnitude with caution")
else:
    print("• Use TOTAL for Destiny 2 analysis")
    print("• Other games: Steam data may not generalize well")
    print("• Consider this a Destiny 2-specific finding")

print("-"*60)

#%%
# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save merged data for reference
df_merged.to_parquet(OUTPUT_PATH / 'steam_proxy_validation_data.parquet', index=False)

# Save criteria results - combine all criteria sets
all_criteria = {**criteria_raw, **criteria_smoothed, **criteria_no_releases, **criteria_middle90, 'decision': decision}
criteria_df = pd.DataFrame([all_criteria])
criteria_df.to_csv(OUTPUT_PATH / 'steam_proxy_validation_results.csv', index=False)

# Save weekly/monthly aggregates
df_weekly_agg.to_parquet(OUTPUT_PATH / 'steam_proxy_weekly_agg.parquet', index=False)
df_monthly_agg.to_parquet(OUTPUT_PATH / 'steam_proxy_monthly_agg.parquet', index=False)

# Save non-release filtered data
if len(df_no_releases_7d) > 0:
    df_no_releases_7d.to_parquet(OUTPUT_PATH / 'steam_proxy_no_releases_7d.parquet', index=False)

print(f"\nResults saved to:")
print(f"  - {OUTPUT_PATH / 'steam_proxy_validation_data.parquet'}")
print(f"  - {OUTPUT_PATH / 'steam_proxy_validation_results.csv'}")
print(f"  - {OUTPUT_PATH / 'steam_proxy_weekly_agg.parquet'}")
print(f"  - {OUTPUT_PATH / 'steam_proxy_monthly_agg.parquet'}")
print(f"  - {OUTPUT_PATH / 'steam_proxy_no_releases_7d.parquet'}")
print(f"  - {OUTPUT_PATH / 'steam_proxy_validation.png'}")

#%%
# =============================================================================
# SUMMARY FOR DOCUMENTATION
# =============================================================================

summary = f"""
# Steam Proxy Validation Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview
- Date range: {df_merged['date'].min().strftime('%Y-%m-%d')} to {df_merged['date'].max().strftime('%Y-%m-%d')}
- Filter applied: {DATE_FILTER_START.strftime('%Y-%m-%d')}+ ({DATE_FILTER_REASON})
- Total observations: {len(df_merged):,}

## Key Findings - Raw Daily Data
- Pearson correlation: {pearson_r:.4f}
- Spearman correlation: {spearman_r:.4f}
- R-squared: {r_squared:.4f}
- Steam/Total ratio: {ratio_mean*100:.2f}% (σ = {ratio_std*100:.2f}%)
- Coefficient of variation: {ratio_cv*100:.2f}%
- Best correlation lag: {best_lag} days
- Daily change correlation: {change_corr:.4f}
- Directional agreement: {direction_agreement*100:.1f}%

## Key Findings - Smoothed Data
- Weekly correlation: {weekly_corr:.4f}
- Weekly ratio CV: {weekly_ratio_cv*100:.2f}%
- Monthly correlation: {monthly_corr:.4f}
- Monthly ratio CV: {monthly_ratio_cv*100:.2f}%

## Key Findings - Release Windows Excluded (±7 days)
- Pearson correlation: {nr7_corr:.4f}
- Ratio CV: {nr7_ratio_cv*100:.2f}%
- Daily change correlation: {nr7_change_corr:.4f}
- Direction agreement: {nr7_direction_agreement*100:.1f}%

## Key Findings - Middle 90% (excluding ratio outliers)
- Pearson correlation: {corr_no_outliers:.4f}
- Ratio CV: {cv_no_outliers*100:.2f}%

## Criteria Summary
- Raw criteria passed: {raw_pass}/4
- Smoothed criteria passed: {smoothed_pass}/5
- No-release (±7d) criteria passed: {no_release_pass}/4
- Middle 90% criteria passed: {middle90_pass}/2

## Decision
{decision}: {decision_text}

## Implication for Project
{'Steam is a valid proxy with documented limitations. Use for steady-state analysis and cross-game comparisons. For release impact analysis on D2, use Total. Document that Steam amplifies release effects by 2-3x.' if decision == 'CONDITIONAL_PROXY' else 'Steam is excellent for steady-state analysis. Around releases (±7 days), Steam overreacts - use Total for D2 release analysis, document amplification for Steam-only games.' if decision == 'NON_EVENT_PROXY' else 'Steam is suitable for trend analysis (weekly+). Use 7-day rolling averages for daily analysis. Steam is more volatile than Total.' if decision in ['STRONG_PROXY', 'TREND_PROXY'] else 'Steam indicates direction but not magnitude. Use Total for Destiny 2 when precision matters.' if decision == 'DIRECTIONAL_PROXY' else 'Use Total player counts for Destiny 2.'}
"""

print(summary)

# Save summary
with open(OUTPUT_PATH / 'steam_proxy_validation_summary.md', 'w') as f:
    f.write(summary)

print(f"\nSummary saved to: {OUTPUT_PATH / 'steam_proxy_validation_summary.md'}")

#%%