# Portfolio Underperformance Analysis

## Critical Issues Found

### 1. **KNN Insufficient History (First 20 months)**
- Code: `if len(cluster_distances) > 20:` else uniform probabilities
- **Problem**: First 20+ months use uniform probabilities (1/3, 1/3, 1/3) regardless of actual regime
- **Impact**: The regime probabilities are not informative in early years, so the portfolio is essentially equally weighted across regimes
- **Fix**: Use `min(len(cluster_distances), 20)` neighbors instead

### 2. **Unstable Regime Returns Estimation (2006-2010)**
- Training size: 36-50 observations (2003-2005 pre-training + first few months of 2006)
- **Problem**: With 36 samples across 3 clusters, each cluster has ~12 observations. Estimating mean returns and covariance from 12 asset return observations is extremely noisy
- **Impact**: Markowitz weights are unreliable, leading to poor allocations
- **Symptom**: Portfolio drops below 1.0 during 2008 crisis, suggesting concentrated/wrong allocations

### 3. **Regime Data Fragmentation**
- Some regimes may have < 3 observations, so they're skipped entirely
- **Problem**: When only 1-2 regimes have enough data, the portfolio concentrates into fewer assets
- **Impact**: Reduces diversification in early years

### 4. **Markowitz Extreme Allocations**
- Using `np.maximum(w, 0)` for long-only, but with noisy mean/cov estimates, weights can still be extreme
- **Problem**: With few samples and potentially negative returns, Markowitz can allocate 100% to a single asset
- **Impact**: Portfolio becomes concentrated and risky

## Why Recent Years Underperform

Even with more data (140+ observations by 2015), the portfolio is still below SP500:
- **SP500 concentration**: The simple buy-and-hold SP500 benefits from market cap weighting (tech boom 2015-2025)
- **Portfolio diversification drag**: The regime portfolio holds bonds, gold, oil, energy - diversifying away from equities
- **Mean reversion strategy**: The regime switching is designed for risk management, not return maximization

## Recommendations

1. **Increase KNN minimum sample requirement** (use 10 neighbors minimum, not 0)
2. **Regularize covariance matrix more aggressively** 
3. **Consider equal-weight or risk-parity within regimes** (less sensitive to estimation error)
4. **Add minimum threshold for regime inclusion** (skip regimes with < 10 observations)
5. **Consider shrinkage estimators** for mean and covariance
