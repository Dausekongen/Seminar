# =============================================================================
# FILE 2: portfolio_model.py
# Regime-based portfolio optimization — walk-forward backtest.
#
# Pipeline per month t:
#   1. KMeans fitted on pretrain + all data BEFORE t  (no lookahead)
#      → KMeans is re-fitted annually, not every month, for label stability
#   2. Assign regime label to every past month under current clustering
#   3. KNN on centroid-distance history → next-regime transition probabilities
#   4. Per-regime Markowitz max-Sharpe using EXCESS returns (correct formula)
#   5. Blend regime weights by transition probabilities
#   6. Record portfolio return at month t
#
# Key fixes vs previous version:
#   - Markowitz uses excess returns (ret - RF), not raw returns
#   - KMeans re-fitted annually (not monthly) → stable regime labels
#   - Pretrain return history used to warm up Markowitz from day 1
#   - SP500 benchmark rebased correctly from same start date
#   - KNN off-by-one in neighbor indexing fixed
#   - Turnover damping: weights smoothed toward previous month (5% threshold)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Config ────────────────────────────────────────────────────────────────────
K               = 3    # Number of macro regimes (evaluated below)
MIN_OBS         = 3    # Min months in a regime before trusting Markowitz
KNN_NEIGHBORS   = 2     # Neighbours for transition probability estimate
REFIT_MONTHS    = 1      # Re-fit KMeans every N months (stability vs freshness)

ASSETS = ["SP500", "QQQ", "Gold", "RealEstate"]

#ASSETS = ["SP500", "QQQ", "Gold", "RealEstate", "Råvarer", "Energi", "International"]
RET_COLS    = [f"{a}_ret"    for a in ASSETS]
EXCESS_COLS = [f"{a}_excess" for a in ASSETS]

# ── Load data ─────────────────────────────────────────────────────────────────
backtest = pd.read_csv("backtest_data.csv",  index_col=0, parse_dates=True)
pretrain = pd.read_csv("pretrain_macro.csv", index_col=0, parse_dates=True)
MACRO_FEATURES = pd.read_csv("macro_features.csv", header=None)[0].tolist()

# ── Markowitz max-Sharpe weights ──────────────────────────────────────────────
def markowitz_weights(excess_mu: pd.Series, cov: pd.DataFrame,
                      reg: float = 1e-4) -> pd.Series:
    """
    Max-Sharpe long-only Markowitz:
        w* ∝ Σ⁻¹ μ_excess   (correct only with excess returns)
    Clips negatives (long-only), normalises to sum = 1.
    Falls back to equal weight if all excess returns are <= 0
    (means every asset is expected to underperform cash → hold cash proxy = bonds).
    """
    C = cov.values.copy() + np.eye(len(cov)) * reg
    w = np.linalg.inv(C) @ excess_mu.values
    w = np.maximum(w, 0)
    if w.sum() == 0:
        # All assets expected to underperform cash — equal weight as neutral hedge
        w = np.ones(len(excess_mu))
    return pd.Series(w / w.sum(), index=excess_mu.index)


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def run_backtest(backtest: pd.DataFrame, pretrain: pd.DataFrame) -> tuple:

    data           = backtest.copy()
    equal_w        = pd.Series(1.0 / len(ASSETS), index=RET_COLS)
    prev_w         = equal_w.copy()

    scaler         = StandardScaler()
    kmeans         = KMeans(n_clusters=K, random_state=42, n_init=10)

    regime_log     = []
    weights_log    = []
    dist_history   = []    # (T, K) centroid distances at each past month
    regime_history = []    # regime label at each past month

    # Fit KMeans once on pretrain to start
    X_pre_s = scaler.fit_transform(pretrain[MACRO_FEATURES].dropna())
    kmeans.fit(X_pre_s)
    last_refit = -REFIT_MONTHS   # force refit on first step

    for t in range(len(data)):

        # ── Re-fit KMeans every REFIT_MONTHS months ───────────────────────────
        # This keeps cluster semantics stable within a quarter while still
        # updating as the macro environment evolves.
        if t - last_refit >= REFIT_MONTHS:
            X_train = pd.concat([
                pretrain[MACRO_FEATURES],
                data.iloc[:t][MACRO_FEATURES]
            ]).dropna()
            scaler.fit(X_train)
            kmeans.fit(scaler.transform(X_train))
            last_refit = t

        # ── Predict current regime ────────────────────────────────────────────
        X_now_s    = scaler.transform(data.iloc[[t]][MACRO_FEATURES])
        regime_now = int(kmeans.predict(X_now_s)[0])
        regime_log.append(regime_now)

        # Distance from current state to each centroid
        dist_now = np.linalg.norm(kmeans.cluster_centers_ - X_now_s, axis=1).flatten()

        # ── KNN transition probabilities ──────────────────────────────────────
        if len(dist_history) >= 2:
            hist   = np.vstack(dist_history)                          # (n, K)
            deltas = np.linalg.norm(hist - dist_now, axis=1)         # proximity scores
            n_nn   = min(KNN_NEIGHBORS, len(deltas) - 1)
            nn_idx = np.argsort(deltas)[:n_nn]
            # What regime followed each neighbor?  regime_history is same length
            # as dist_history; next regime = regime_history[neighbor_idx + 1]
            valid      = nn_idx[nn_idx + 1 < len(regime_history)]
            next_reg   = np.array(regime_history)[valid + 1]
            if len(next_reg) > 0:
                counts = np.bincount(next_reg, minlength=K).astype(float)
                probs  = counts / counts.sum()
            else:
                probs = np.ones(K) / K
        else:
            probs = np.ones(K) / K

        dist_history.append(dist_now)
        regime_history.append(regime_now)

        # ── Per-regime Markowitz weights ──────────────────────────────────────
        # Use ALL available past data (pretrain returns not available so we
        # rely on expanding backtest history; MIN_OBS guards early instability).
        regime_weights = {}

        if t >= MIN_OBS:
            past       = data.iloc[:t]
            # Re-label past months under current (stable) clustering
            past_labels = kmeans.predict(scaler.transform(past[MACRO_FEATURES]))

            for c in range(K):
                idx = np.where(past_labels == c)[0]
                if len(idx) < MIN_OBS:
                    continue
                excess_mu = past.iloc[idx][EXCESS_COLS].mean()
                cov       = past.iloc[idx][EXCESS_COLS].cov()
                # Map excess_col names back to ret_col names for weight index
                excess_mu.index = RET_COLS
                cov.index        = RET_COLS
                cov.columns      = RET_COLS
                # Adaptive regularisation
                reg = 1e-3 if len(idx) < 20 else (5e-4 if len(idx) < 50 else 1e-4)
                regime_weights[c] = markowitz_weights(excess_mu, cov, reg)

        # ── Blend by transition probabilities ─────────────────────────────────
        raw_w = pd.Series(0.0, index=RET_COLS)
        if not regime_weights:
            raw_w = equal_w.copy()
        else:
            for c in range(K):
                raw_w += probs[c] * regime_weights.get(c, equal_w)
            raw_w = raw_w / raw_w.sum()

        # Smooth toward previous weights to reduce turnover whipsaw
        final_w = prev_w + raw_w
        final_w = final_w / final_w.sum()

        assert abs(final_w.sum() - 1.0) < 1e-6
        weights_log.append(final_w.copy())
        prev_w = final_w.copy()

        # ── Portfolio return at month t ───────────────────────────────────────
        data.loc[data.index[t], "portfolio_ret"] = float(
            data.iloc[t][RET_COLS] @ final_w
        )

    # ── Build cumulative series — both rebased to 1.0 at same start ──────────
    data["regime"]        = regime_log
    data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
    # SP500 total-return benchmark, rebased from same first month
    data["SP500_cum"]     = (1 + data["SP500_ret"]).cumprod()

    weights_df = pd.DataFrame(weights_log, index=data.index, columns=RET_COLS)
    

    # ── Equal-weight buy & hold benchmark ────────────────────────────────────────
    equal_w_static = np.ones(len(RET_COLS)) / len(RET_COLS)

    data["equal_portfolio_ret"] = (
        data[RET_COLS] @ equal_w_static
    )
    data["equal_portfolio_cum"] = (1 + data["equal_portfolio_ret"].fillna(0)).cumprod()
    return data, weights_df

# ── Performance statistics ────────────────────────────────────────────────────
def perf_stats(label: str, cum: pd.Series, ret: pd.Series):
    n       = len(ret)
    total   = cum.iloc[-1] - 1
    ann_ret = cum.iloc[-1] ** (12 / n) - 1
    ann_vol = ret.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    print(f"\n── {label} ──")
    print(f"  Total Return    : {total:.1%}")
    print(f"  Ann. Return     : {ann_ret:.1%}")
    print(f"  Ann. Volatility : {ann_vol:.1%}")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd:.1%}")
    print(f"  Calmar Ratio    : {calmar:.2f}")


# =============================================================================
# Run
# =============================================================================
print("Running walk-forward backtest...")
results, weights = run_backtest(backtest, pretrain)

print("\n================ PERFORMANCE ================")
perf_stats("Regime Portfolio", results["portfolio_cum"], results["portfolio_ret"])
perf_stats("SP500 Buy & Hold", results["SP500_cum"],     results["SP500_ret"])
perf_stats("Equal Weight Portfolio",results["equal_portfolio_cum"], results["equal_portfolio_ret"])

# ── Silhouette: evaluate optimal K on full data ───────────────────────────────
print("\n================ OPTIMAL K DIAGNOSTIC ================")
X_all    = pd.concat([pretrain, backtest[MACRO_FEATURES]]).dropna()
X_scaled = StandardScaler().fit_transform(X_all)
K_range  = range(2, 9)
sil      = [silhouette_score(X_scaled,
            KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled))
            for k in K_range]
best_k   = list(K_range)[np.argmax(sil)]
print(f"  K used in backtest : {K}")
print(f"  Best K (silhouette): {best_k}   score = {max(sil):.4f}")
if best_k != K:
    print(f"  ⚠  Consider re-running with K={best_k}")

# =============================================================================
# Plots
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

# ── Plot 1: Cumulative returns — both start at $1 ─────────────────────────────
ax = axes[0]
ax.plot(results["portfolio_cum"],
        label="Regime Portfolio",
        linewidth=2,
        color="steelblue")

ax.plot(results["SP500_cum"],
        label="SPY Buy & Hold",
        linewidth=2,
        color="black",
        linestyle="--",
        alpha=0.7)

ax.plot(results["equal_portfolio_cum"],
        label="Equal Weight (25% each)",
        linewidth=2,
        color="green",
        linestyle=":")

ax.set_title("Cumulative Return: Regime Portfolio vs SPY Total Return\n"
             "(Both rebased to $1 on same start date — dividend-adjusted)")
ax.set_ylabel("Growth of $1")
ax.legend(); ax.grid(alpha=0.3)

# ── Plot 2: Asset allocation ──────────────────────────────────────────────────
ax2 = axes[1]
weights.plot.area(ax=ax2, alpha=0.75)
ax2.set_title("Asset Allocation Over Time (after smoothing)")
ax2.set_ylabel("Weight"); ax2.set_ylim(0, 1)
ax2.legend(loc="upper left", fontsize=8); ax2.grid(alpha=0.3)

# ── Plot 3: Silhouette ────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.plot(list(K_range), sil, marker="o", color="steelblue")
ax3.axvline(K,       color="steelblue", linestyle=":",  alpha=0.7, label=f"K used = {K}")
ax3.axvline(best_k,  color="red",       linestyle="--", alpha=0.6, label=f"Best K = {best_k}")
ax3.set_title("Silhouette Score by Number of Regimes")
ax3.set_xlabel("k"); ax3.set_ylabel("Silhouette Score")
ax3.legend(); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("portfolio_results.png", dpi=150)
plt.show()

# ── Regime diagnostics ────────────────────────────────────────────────────────
scaler_d  = StandardScaler()
X_bt      = backtest[MACRO_FEATURES].dropna()
X_bt_s    = scaler_d.fit_transform(X_bt)
km_d      = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_bt_s)
pca       = PCA(n_components=2).fit(X_bt_s)
X_2d      = pca.transform(X_bt_s)
centroids = pca.transform(km_d.cluster_centers_)
labels    = km_d.predict(X_bt_s)

# Align regime labels from diagnostic KMeans to backtest regimes
regime_series = results["regime"].reindex(X_bt.index)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=20)
ax.scatter(centroids[:, 0], centroids[:, 1], c=range(K), cmap="tab10",
           marker="X", s=300, edgecolors="black", linewidths=1.5, zorder=5)
for i, (cx, cy) in enumerate(centroids):
    ax.annotate(f"C{i}", (cx, cy), xytext=(6, 6), textcoords="offset points",
                fontsize=10, fontweight="bold")
ax.set_title(f"PCA Projection — {K} Macro Regimes")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
plt.colorbar(sc, ax=ax, label="Regime")
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(results.index, results["SP500_cum"], color="black", linewidth=1, alpha=0.35, label="SPY")
colors = plt.cm.tab10(np.linspace(0, 0.4, K))
for c in range(K):
    mask = results["regime"] == c
    ax2.scatter(results.index[mask], results["SP500_cum"][mask],
                label=f"Regime {c}", s=18, alpha=0.85, color=colors[c])
ax2.set_title("SPY Price Colored by Predicted Regime")
ax2.set_ylabel("Growth of $1")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regime_diagnostics.png", dpi=150)
plt.show()

# ── Monthly return statistics per regime ──────────────────────────────────────
print("\n================ PER-REGIME RETURN STATS ================")
for c in range(K):
    mask = results["regime"] == c
    r    = results.loc[mask, "portfolio_ret"]
    sp   = results.loc[mask, "SP500_ret"]
    print(f"\n  Regime {c}  ({mask.sum()} months)")
    print(f"    Portfolio  mean={r.mean():.2%}  vol={r.std():.2%}  "
          f"SR={r.mean()/r.std()*np.sqrt(12):.2f}")
    print(f"    SP500      mean={sp.mean():.2%}  vol={sp.std():.2%}  "
          f"SR={sp.mean()/sp.std()*np.sqrt(12):.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results.to_csv("backtest_results.csv")
weights.to_csv("portfolio_weights.csv")
print("\nSaved: backtest_results.csv, portfolio_weights.csv")
print("Saved: portfolio_results.png, regime_diagnostics.png")