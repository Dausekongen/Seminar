import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# 1️⃣ Load full macro dataset for KMeans training
# -----------------------------
full_kmeans = pd.read_csv("ml_dataset_full_kmeans.csv", index_col=0, parse_dates=True)

# Load cleaned dataset for backtest
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

# Pre-cutoff: switch from full_kmeans to ml_dataset at 2006
pretrain_cutoff = pd.Timestamp("2005-12-31")
backtest_start = data.index[0]  # Should be 2006-05-31
# -----------------------------
# 2️⃣ Define assets and features
# -----------------------------
assets = ["SP500_ret", "Bonds_ret", "Gold_ret", "Oil_ret", "Energy_ret"]

features = [
    "Inflation",
    "Inflation_mom",
    "IP_growth",
    "Yield_curve",
    "Rate_level",
    "Unemployment",
    "Unemployment_change",
    "Infl_exp",
    "SP500_vol",
    "Bond_vol"
]

macro_features = [
    "Inflation",
    "Inflation_mom",
    "IP_growth",
    "Yield_curve",
    "Rate_level",
    "Unemployment",
    "Unemployment_change",
    "Infl_exp"
]

# -----------------------------
# 3️⃣ Compute rolling volatilities
# -----------------------------
for asset in assets:
    vol_name = asset.replace("_ret","_vol")
    data[vol_name] = data[asset].rolling(12).std()

# Drop rows only when required features or asset returns are missing.
# We want to keep as much history as possible, so we only require the KMeans input features.
data = data.dropna(subset=macro_features + assets)

# -----------------------------
# 4️⃣ Markowitz optimizer
# -----------------------------
def markowitz_weights(mu, cov):
    inv_cov = np.linalg.inv(cov + np.eye(len(cov)) * reg)
    w = inv_cov @ mu
    # Long-only portfolio. This allows rebalancing an asset lower
    # (e.g. from 20% to 10%) and reallocating that weight to other assets.
    # It does not allow negative short positions.
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)
    return pd.Series(w, index=mu.index)

# -----------------------------
# 5️⃣ Rolling one-step-ahead backtest
# -----------------------------
k = 3
reg = 1e-6

# Storage
predicted_clusters = []
applied_weights = []
cluster_distances = []
cluster_movements = []
train_counts = []
cluster_sizes = []

# NEW: store probability history
cluster_probabilities = []

# Initialize
scaler = StandardScaler()
kmeans = KMeans(n_clusters=k, random_state=42)

for i in range(len(data)):

    # Train KMeans using data up to t-1 only
    X_pretrain = full_kmeans.loc[:pretrain_cutoff].dropna()
    
    if i == 0:
        # First month: only use pre-training data
        X_train = X_pretrain.copy()
    else:
        # Use data up to previous month (t-1 logic)
        prev_date = data.index[i-1]
        X_backtest = data.loc[pretrain_cutoff:prev_date, macro_features].dropna()
        X_train = pd.concat([X_pretrain, X_backtest])
    
    X_train = X_train.dropna()
    if len(X_train) == 0:
        raise ValueError("No KMeans training data at iteration " + str(i))

    scaler.fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    kmeans.fit(X_scaled_train)

    train_counts.append(len(X_train))
    cluster_sizes.append(np.bincount(kmeans.labels_, minlength=k))

    # Predict clusters for historical data (t-1) to define regimes
    if i > 0:
        asset_data_for_regimes = data.iloc[:i]
        X_asset_for_regimes = asset_data_for_regimes[macro_features]
        cluster_labels = kmeans.predict(scaler.transform(X_asset_for_regimes))
    else:
        asset_data_for_regimes = pd.DataFrame()
        cluster_labels = np.array([])

    # Current observation at time t
    X_current = data.iloc[[i]][macro_features]
    X_scaled_current = scaler.transform(X_current)

    current_cluster = kmeans.predict(X_scaled_current)[0]
    predicted_clusters.append(current_cluster)

    # Distances to clusters at time t
    distances = np.linalg.norm(
        kmeans.cluster_centers_ - X_scaled_current,
        axis=1
    ).flatten()

    cluster_distances.append(distances)

    # Movement
    if i > 0:
        prev_distances = cluster_distances[-2]
        delta = prev_distances - distances
        cluster_movements.append(delta)
    else:
        cluster_movements.append(np.zeros(k))

    # -----------------------------
    # NEW: Empirical transition probabilities (Method 3)
    # Empirical transition probabilities using adaptive KNN
    if len(cluster_distances) > 1:
        hist_dist = np.array(cluster_distances[:-1])
        hist_next = np.array(predicted_clusters[1:])
        current = distances

        # Use available history, minimum 1 neighbor
        n_neighbors = min(max(1, len(cluster_distances) - 1), 20)
        dists = np.linalg.norm(hist_dist - current, axis=1)
        nearest_idx = np.argsort(dists)[:n_neighbors]

        next_clusters = hist_next[nearest_idx]
        probs = np.bincount(next_clusters, minlength=k)
        probs = probs / probs.sum()
    else:
        probs = np.ones(k) / k  # fallback for very first month

    cluster_probabilities.append(probs)

    # -----------------------------
    # Compute regime weights
    # -----------------------------
    regime_weights = {}
    if len(cluster_labels) > 0:
        for c in range(k):
            cluster_indices = np.where(cluster_labels == c)[0]

            if len(cluster_indices) < 3:
                continue

            cluster_returns = asset_data_for_regimes.iloc[cluster_indices][assets]
            mu = cluster_returns.mean()
            cov = cluster_returns.cov()

            regime_weights[c] = markowitz_weights(mu, cov)

    # -----------------------------
    # NEW: Probability-weighted portfolio
    # -----------------------------
    w = pd.Series(0, index=assets, dtype=float)

    for c in range(k):
        if c in regime_weights:
            w += probs[c] * regime_weights[c]

    # fallback if empty
    if w.sum() == 0:
        w = pd.Series(np.ones(len(assets))/len(assets), index=assets)

    applied_weights.append(w)

    # -----------------------------
    # Portfolio return
    # -----------------------------
    data.loc[data.index[i], "portfolio_ret"] = np.dot(
        data.loc[data.index[i], assets],
        w
    )

# -----------------------------
# 6️⃣ Post-process
# -----------------------------
data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
data["SP500_cum"] = (1 + data["SP500_ret"]).cumprod()
data["Predicted_cluster"] = predicted_clusters

weights_df = pd.DataFrame(applied_weights, index=data.index)
cluster_sizes_arr = np.array(cluster_sizes)

print("Final KMeans training size:", train_counts[-1])
print("Minimum KMeans training size:", np.min(train_counts))
print("Maximum KMeans training size:", np.max(train_counts))

# -----------------------------
# 7️⃣ Plot training diagnostics
# -----------------------------
plt.figure(figsize=(14,4))
plt.plot(data.index, train_counts, label="Training rows")
plt.title("Number of KMeans training observations over time")
plt.ylabel("Rows")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("training_count.png")
plt.show()

if cluster_sizes_arr.shape[0] > 0:
    plt.figure(figsize=(14,4))
    for c in range(k):
        plt.plot(data.index, cluster_sizes_arr[:,c], label=f"Cluster {c}")
    plt.title("Training cluster sizes over time")
    plt.ylabel("Cluster members")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("cluster_sizes.png")
    plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
final_labels = kmeans.predict(scaler.transform(X_train))
centers_pca = pca.transform(kmeans.cluster_centers_)

print("PCA explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(10,7))
scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=final_labels,
    cmap="viridis",
    s=20,
    alpha=0.6
)
plt.scatter(
    centers_pca[:,0],
    centers_pca[:,1],
    c="red",
    marker="X",
    s=100,
    label="KMeans centers"
)
plt.title("KMeans training data PCA projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(alpha=0.3)
plt.colorbar(scatter, label="Cluster")
plt.savefig("kmeans_pca.png")
plt.show()

# -----------------------------
# 8️⃣ Plot performance
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(data["portfolio_cum"], label="Portfolio")
plt.plot(data["SP500_cum"], label="SP500", color='black', alpha=0.3)
plt.title("Regime Probabilistic Portfolio vs SP500")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("portfolio_performance.png")
plt.show()

# -----------------------------
# 8️⃣ Plot distances
# -----------------------------
dist_df = pd.DataFrame(
    cluster_distances,
    columns=[f"Cluster_{i}" for i in range(k)],
    index=data.index
)

plt.figure(figsize=(14,6))
for c in range(k):
    plt.plot(dist_df.index, dist_df[f"Cluster_{c}"], label=f"Cluster {c}")
plt.title("Distance to Cluster Centers")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("distances.png")
plt.show()

# -----------------------------
# 9️⃣ Plot probabilities
# -----------------------------
prob_df = pd.DataFrame(
    cluster_probabilities,
    columns=[f"Cluster_{i}" for i in range(k)],
    index=data.index
)

plt.figure(figsize=(14,6))
for c in range(k):
    plt.plot(prob_df.index, prob_df[f"Cluster_{c}"], label=f"Cluster {c}")
plt.title("Empirical Regime Probabilities (KNN-based)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("probabilities.png")
plt.show()

plt.figure(figsize=(14,6))

# Plot SP500 cumulative baseline
plt.plot(data.index, data["SP500_cum"], color='lightgray', alpha=0.5)

# Scatter with cluster colors
clusters = data["Predicted_cluster"].dropna()

scatter = plt.scatter(
    clusters.index,
    data.loc[clusters.index, "SP500_cum"],
    c=clusters,
    cmap="viridis",
    s=20
)

plt.title("S&P 500 Colored by KMeans Regime")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Cluster")

plt.savefig("regimes.png")
plt.show()