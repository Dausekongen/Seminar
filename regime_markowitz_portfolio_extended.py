import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===================================================
# Load Data
# ===================================================
full_kmeans = pd.read_csv("ml_dataset_ext_kmeans.csv", index_col=0, parse_dates=True)
data = pd.read_csv("ml_dataset_ext.csv", index_col=0, parse_dates=True)

assets = ["SP500_ret", "QQQ_ret", "TLT_ret", "LQD_ret", "Gold_ret", "RealEstate_ret"]

macro_features_base = [
    "Inflation",
    "Inflation_mom",
    "IP_growth",
    "Yield_curve",
    "Rate_level",
    "Unemployment",
    "Unemployment_change",
    "Infl_exp"
]

# Add rolling volatility features on top of base macro
VOL_WINDOW = 12
vol_feature_names = []
for col in ["Inflation", "IP_growth", "Yield_curve", "Unemployment_change", "Infl_exp"]:
    new_col = f"{col}_vol{VOL_WINDOW}"
    data[new_col] = data[col].rolling(VOL_WINDOW).std()
    full_kmeans[new_col] = full_kmeans[col].rolling(VOL_WINDOW).std()
    vol_feature_names.append(new_col)

macro_features_extended = macro_features_base + vol_feature_names

# ── Convert log returns → simple returns ──────────────────────────────────────
# CSVs store log returns. Convert so (1+r).cumprod() compounding is valid.
print("Converting log returns to simple returns...")
for col in assets:
    data[col] = np.exp(data[col]) - 1

data = data.dropna(subset=macro_features_extended + assets)
start_date = "2010-01-01"
data = data.loc[start_date:]

# ===================================================
# Portfolio Optimizer
# ===================================================
def markowitz_weights(mu, cov, reg=1e-4):
    """
    Standard Markowitz. No positive-mu filter applied here anymore.
    The optimizer naturally underweights poor-expected assets through
    the covariance matrix. We only enforce non-negative weights (long only)
    and renormalize to sum to 1.
    """
    cov = cov.copy()
    cov += np.eye(len(cov)) * reg
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    # Long-only constraint: floor at zero
    w = np.maximum(w, 0)
    if w.sum() == 0:
        # All expected returns negative in this regime — equal weight as fallback
        # This means we still hold something, spread evenly as a hedge
        w = np.ones(len(mu))
    w = w / w.sum()
    return pd.Series(w, index=mu.index)

# ===================================================
# Backtest Function
# ===================================================
def run_backtest(data, full_kmeans, macro_features, assets, label="Portfolio"):
    k = 3
    pretrain_cutoff = pd.Timestamp("2009-12-31")

    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=k, random_state=42)

    predicted_clusters = []
    applied_weights = []
    cluster_probabilities = []
    historical_distances = []

    data = data.copy()

    for i in range(len(data)):
        # ── Build expanding training window ───────────────────────────────────
        X_pretrain = full_kmeans.loc[:pretrain_cutoff][macro_features].dropna()
        if i == 0:
            X_train = X_pretrain.copy()
        else:
            prev_date = data.index[i - 1]
            X_backtest = data.loc[pretrain_cutoff:prev_date, macro_features].dropna()
            X_train = pd.concat([X_pretrain, X_backtest])
        X_train = X_train.dropna()

        scaler.fit(X_train)
        X_scaled_train = scaler.transform(X_train)
        kmeans.fit(X_scaled_train)

        # ── Label all past backtest months with current clustering ────────────
        if i > 0:
            asset_data_for_regimes = data.iloc[:i]
            X_asset_for_regimes = asset_data_for_regimes[macro_features]
            cluster_labels = kmeans.predict(scaler.transform(X_asset_for_regimes))
        else:
            asset_data_for_regimes = pd.DataFrame()
            cluster_labels = np.array([])

        # ── Predict current regime ────────────────────────────────────────────
        X_current = data.iloc[[i]][macro_features]
        X_scaled_current = scaler.transform(X_current)
        current_cluster = kmeans.predict(X_scaled_current)[0]
        predicted_clusters.append(current_cluster)

        # ── Nearest-neighbor regime transition probabilities ──────────────────
        distances = np.linalg.norm(
            kmeans.cluster_centers_ - X_scaled_current, axis=1
        ).flatten()

        if len(historical_distances) > 0:
            hist_distances = np.vstack(historical_distances)
            hist_next = (
                np.array(predicted_clusters[1:])
                if len(predicted_clusters) > 1
                else np.array([])
            )
            n_neighbors = min(max(1, len(hist_distances)), 20)
            dists = np.linalg.norm(hist_distances - distances, axis=1)
            nearest = np.argsort(dists)[:n_neighbors]
            next_clusters = hist_next[nearest] if len(hist_next) > 0 else np.array([])
            if len(next_clusters) > 0:
                probs = np.bincount(next_clusters, minlength=k).astype(float)
                probs /= probs.sum()
            else:
                probs = np.ones(k) / k
        else:
            probs = np.ones(k) / k

        cluster_probabilities.append(probs)
        historical_distances.append(distances)

        # ── Build per-regime Markowitz weights ────────────────────────────────
        regime_weights = {}
        min_obs = 5

        if len(cluster_labels) > 0:
            for c in range(k):
                cluster_indices = np.where(cluster_labels == c)[0]
                if len(cluster_indices) < min_obs:
                    # Not enough history for this regime yet — skip, handled below
                    continue

                cluster_returns = asset_data_for_regimes.iloc[cluster_indices][assets]
                mu = cluster_returns.mean()
                cov = cluster_returns.cov()

                if len(cluster_indices) < 20:
                    reg_strength = 1e-3
                elif len(cluster_indices) < 50:
                    reg_strength = 5e-4
                else:
                    reg_strength = 1e-4

                # FIX 1: No positive-mu filter.
                # Let Markowitz decide weights based on risk/return tradeoff.
                # If all mu are negative, markowitz_weights returns equal weight
                # so we still stay 100% invested as a diversified hedge.
                w = markowitz_weights(mu, cov, reg=reg_strength)
                regime_weights[c] = w

        # ── Blend regime weights by transition probabilities ──────────────────
        final_w = pd.Series(0.0, index=assets)

        if len(regime_weights) == 0:
            # No regime has enough history yet → equal weight across all assets
            final_w = pd.Series(1.0 / len(assets), index=assets)
        else:
            # For regimes with no weight estimate, substitute equal weight
            # so their probability mass is not lost
            equal_w = pd.Series(1.0 / len(assets), index=assets)
            for c in range(k):
                if c in regime_weights:
                    final_w += probs[c] * regime_weights[c]
                else:
                    # FIX 2: Don't discard probability mass for unseen regimes.
                    # Use equal weight as a neutral placeholder so probs sum
                    # contributes fully to final_w.
                    final_w += probs[c] * equal_w

            # FIX 3: Always renormalize after blending.
            # Even with the above, floating point can leave sum slightly off 1.
            if final_w.sum() > 0:
                final_w = final_w / final_w.sum()
            else:
                final_w = pd.Series(1.0 / len(assets), index=assets)

        applied_weights.append(final_w)

        # Sanity check — weights should always sum to ~1
        assert abs(final_w.sum() - 1.0) < 1e-6, f"Weights don't sum to 1 at step {i}: {final_w.sum()}"

        data.loc[data.index[i], "portfolio_ret"] = np.dot(
            data.loc[data.index[i], assets], final_w
        )

    weights_df = pd.DataFrame(applied_weights, index=data.index)
    data["predicted_cluster"] = predicted_clusters

    # Verify no gaps in weight sum over full backtest
    weight_sums = weights_df.sum(axis=1)
    print(f"\n[{label}] Weight sum stats:")
    print(f"  Min: {weight_sums.min():.6f}  Max: {weight_sums.max():.6f}  Mean: {weight_sums.mean():.6f}")
    print(f"  Any < 0.999: {(weight_sums < 0.999).sum()} months")

    # ── Cumulative returns — both in simple returns, consistent compounding ───
    data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
    data["SP500_cum"] = (1 + data["SP500_ret"]).cumprod()

    return data, weights_df


# ===================================================
# Run Both Backtests
# ===================================================
print("\nRunning backtest — Base macro features...")
data_base, weights_base = run_backtest(
    data.copy(), full_kmeans, macro_features_base, assets, label="Base Macro"
)

print("\nRunning backtest — Extended macro + volatility features...")
data_ext, weights_ext = run_backtest(
    data.copy(), full_kmeans, macro_features_extended, assets, label="Extended + Vol"
)

# ===================================================
# Plot 1: Cumulative Returns
# ===================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

ax = axes[0]
ax.plot(data_base["portfolio_cum"], label="Regime Portfolio (Base Macro)", linewidth=2, color="steelblue")
ax.plot(data_ext["portfolio_cum"],  label="Regime Portfolio (+ Vol Features)", linewidth=2, color="darkorange")
ax.plot(data_base["SP500_cum"],     label="SP500 Buy & Hold", color="black", alpha=0.5, linewidth=2, linestyle="--")
ax.set_title("Portfolio Cumulative Return vs SP500\n(All in simple returns — fully invested at all times)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylabel("Growth of $1")

# ── Weight allocation — Base model ────────────────────────────────────────────
ax2 = axes[1]
weights_base.plot.area(ax=ax2, alpha=0.75)
ax2.set_title("Asset Allocation Over Time — Base Macro\n(Should be 100% filled at all times)")
ax2.set_ylabel("Weight")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

# ── Weight allocation — Extended model ────────────────────────────────────────
ax3 = axes[2]
weights_ext.plot.area(ax=ax3, alpha=0.75)
ax3.set_title("Asset Allocation Over Time — Extended (+ Vol Features)\n(Should be 100% filled at all times)")
ax3.set_ylabel("Weight")
ax3.set_ylim(0, 1)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("portfolio_v3_allocation.png", dpi=150)
plt.show()

# ===================================================
# Performance Statistics
# ===================================================
def perf_stats(cum_series, ret_series, label):
    total_ret  = cum_series.iloc[-1] - 1
    ann_ret    = (cum_series.iloc[-1]) ** (12 / len(ret_series)) - 1
    ann_vol    = ret_series.std() * np.sqrt(12)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else np.nan
    rolling_max = cum_series.cummax()
    drawdown   = (cum_series - rolling_max) / rolling_max
    max_dd     = drawdown.min()
    print(f"\n── {label} ──")
    print(f"  Total Return    : {total_ret:.2%}")
    print(f"  Ann. Return     : {ann_ret:.2%}")
    print(f"  Ann. Volatility : {ann_vol:.2%}")
    print(f"  Sharpe Ratio    : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd:.2%}")

print("\n================ PERFORMANCE STATISTICS ================")
perf_stats(data_base["portfolio_cum"], data_base["portfolio_ret"], "Base Macro Portfolio")
perf_stats(data_ext["portfolio_cum"],  data_ext["portfolio_ret"],  "Extended + Vol Portfolio")
perf_stats(data_base["SP500_cum"],     data_base["SP500_ret"],     "SP500 Buy & Hold")

# ===================================================
# Clustering Diagnostic — Does vol help?
# ===================================================
print("\n================ CLUSTERING DIAGNOSTIC ================")

X_base = data[macro_features_base].dropna()
X_ext  = data[macro_features_extended].dropna()
common_idx = X_base.index.intersection(X_ext.index)
X_base = X_base.loc[common_idx]
X_ext  = X_ext.loc[common_idx]

X_base_scaled = StandardScaler().fit_transform(X_base)
X_ext_scaled  = StandardScaler().fit_transform(X_ext)

K_range = range(2, 9)
sil_base, sil_ext = [], []

for k_test in K_range:
    km = KMeans(n_clusters=k_test, random_state=42)
    sil_base.append(silhouette_score(X_base_scaled, km.fit_predict(X_base_scaled)))
    sil_ext.append(silhouette_score(X_ext_scaled,  km.fit_predict(X_ext_scaled)))

plt.figure(figsize=(8, 5))
plt.plot(K_range, sil_base, marker='o', label="Base Macro (8 features)", color="steelblue")
plt.plot(K_range, sil_ext,  marker='s', label="+ Vol Features (13 features)", color="darkorange")
plt.title("Silhouette Score: Base vs Extended Features")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("clustering_v3.png", dpi=150)
plt.show()

best_k_base = list(K_range)[np.argmax(sil_base)]
best_k_ext  = list(K_range)[np.argmax(sil_ext)]
print(f"Best k — Base    : k={best_k_base}  score={max(sil_base):.4f}")
print(f"Best k — Extended: k={best_k_ext}  score={max(sil_ext):.4f}")

# ===================================================
# Save outputs
# ===================================================
data_ext.to_csv("backtest_results_v3.csv")
weights_ext.to_csv("weights_v3.csv")
weights_base.to_csv("weights_base_v3.csv")
print("\nSaved: backtest_results_v3.csv, weights_v3.csv, weights_base_v3.csv")

from sklearn.decomposition import PCA

print("\n================ PCA DIAGNOSTIC ================")

# Use extended features (best model)
X = data_ext[macro_features_extended].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA → reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit KMeans again (same k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Plot PCA with clusters
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis",
    alpha=0.7
)

plt.title("PCA Projection of Macro Features (Colored by Cluster)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150)
plt.show()

# ===================================================
# DIAGNOSTICS: PCA + REGIME VISUALIZATION
# ===================================================

from sklearn.decomposition import PCA

print("\n================ PCA + REGIME DIAGNOSTICS ================")

# Use extended dataset (best model)
df = data_ext.copy()

# Drop rows without cluster (early periods)
df = df.dropna(subset=["predicted_cluster"])

# ===================================================
# PCA VISUALIZATION
# ===================================================
X = df[macro_features_extended].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis",
    alpha=0.7
)

plt.title("PCA Projection of Macro Features (Clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150)
plt.show()

print("PCA explained variance:", pca.explained_variance_ratio_)


# ===================================================
# SP500 COLORED BY REGIME
# ===================================================
plt.figure(figsize=(14,6))

# Plot SP500
plt.plot(df.index, df["SP500_cum"], color="black", linewidth=0.1, label="SP500")

# Overlay clusters
for c in sorted(df["predicted_cluster"].unique()):
    mask = df["predicted_cluster"] == c
    plt.scatter(
        df.index[mask],
        df["SP500_cum"][mask],
        label=f"Cluster {int(c)}",
        s=12,
        alpha=0.6
    )

plt.title("SP500 Colored by Predicted Regime")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("sp500_regimes.png", dpi=150)
plt.show()


# ===================================================
# REGIME TIMELINE (CLEAN VIEW)
# ===================================================
plt.figure(figsize=(14,2))

plt.scatter(
    df.index,
    np.ones(len(df)),
    c=df["predicted_cluster"],
    cmap="viridis",
    marker="|",
    s=200
)

plt.title("Regime Timeline")
plt.yticks([])
plt.xlabel("Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("regime_timeline.png", dpi=150)
plt.show()

