import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load datasets
full_kmeans = pd.read_csv("ml_dataset_full_kmeans.csv", index_col=0, parse_dates=True)
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

assets = ["SP500_ret", "Bonds_ret", "Gold_ret", "Oil_ret", "Energy_ret"]

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

# Compute rolling volatilities
for asset in assets:
    vol_name = asset.replace("_ret","_vol")
    data[vol_name] = data[asset].rolling(12).std()

data = data.dropna(subset=macro_features + assets)

# Markowitz optimizer
def markowitz_weights(mu, cov):
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)
    return pd.Series(w, index=mu.index)

# Backtest parameters
k = 3
reg = 1e-6
pretrain_cutoff = pd.Timestamp("2005-12-31")

# Storage
predicted_clusters = []
applied_weights = []
cluster_distances = []
cluster_movements = []
train_counts = []
cluster_sizes = []
cluster_probabilities = []

# Initialize
scaler = StandardScaler()
kmeans = KMeans(n_clusters=k, random_state=42)

for i in range(len(data)):

    # Train KMeans using data up to t-1 only
    X_pretrain = full_kmeans.loc[:pretrain_cutoff].dropna()
    
    if i == 0:
        X_train = X_pretrain.copy()
    else:
        prev_date = data.index[i-1]
        X_backtest = data.loc[pretrain_cutoff:prev_date, macro_features].dropna()
        X_train = pd.concat([X_pretrain, X_backtest])
    
    X_train = X_train.dropna()

    scaler.fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    kmeans.fit(X_scaled_train)

    train_counts.append(len(X_train))
    cluster_sizes.append(np.bincount(kmeans.labels_, minlength=k))

    # Predict clusters for historical data (t-1)
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

    # IMPROVED: Adaptive KNN - use min 1 neighbor when history is short
    if len(cluster_distances) > 1:
        hist_dist = np.array(cluster_distances[:-1])
        hist_next = np.array(predicted_clusters[1:])
        current = distances

        n_neighbors = min(max(1, len(cluster_distances) - 1), 20)
        dists = np.linalg.norm(hist_dist - current, axis=1)
        nearest_idx = np.argsort(dists)[:n_neighbors]

        next_clusters = hist_next[nearest_idx]
        probs = np.bincount(next_clusters, minlength=k)
        probs = probs / probs.sum()
    else:
        probs = np.ones(k) / k

    cluster_probabilities.append(probs)

    # IMPROVED: Regime weights with higher minimum cluster size + adaptive regularization
    regime_weights = {}
    min_cluster_obs = 5  # Increased from 3
    
    if len(cluster_labels) > 0:
        for c in range(k):
            cluster_indices = np.where(cluster_labels == c)[0]

            if len(cluster_indices) < min_cluster_obs:
                continue

            cluster_returns = asset_data_for_regimes.iloc[cluster_indices][assets]
            mu = cluster_returns.mean()
            cov = cluster_returns.cov()
            
            # IMPROVED: Adaptive regularization based on sample size
            if len(cluster_indices) < 20:
                reg_strength = reg * 10.0  # Heavy
            elif len(cluster_indices) < 50:
                reg_strength = reg * 3.0   # Moderate
            else:
                reg_strength = reg         # Standard
            
            cov += np.eye(len(cov)) * reg_strength
            regime_weights[c] = markowitz_weights(mu, cov)

    # Probability-weighted portfolio for time t
    w = pd.Series(0, index=assets, dtype=float)

    for c in range(k):
        if c in regime_weights:
            w += probs[c] * regime_weights[c]

    if w.sum() == 0:
        w = pd.Series(np.ones(len(assets))/len(assets), index=assets)

    applied_weights.append(w)

    # Portfolio return at time t
    data.loc[data.index[i], "portfolio_ret"] = np.dot(
        data.loc[data.index[i], assets],
        w
    )

# Post-process
data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
data["SP500_cum"] = (1 + data["SP500_ret"]).cumprod()
data["Predicted_cluster"] = predicted_clusters

weights_df = pd.DataFrame(applied_weights, index=data.index)
cluster_sizes_arr = np.array(cluster_sizes)

print("Final KMeans training size:", train_counts[-1])
print("Minimum KMeans training size:", np.min(train_counts))
print("Maximum KMeans training size:", np.max(train_counts))

# Plots
plt.figure(figsize=(14,6))
plt.plot(data["portfolio_cum"], label="Portfolio (IMPROVED)", linewidth=2)
plt.plot(data["SP500_cum"], label="SP500", color='black', alpha=0.3, linewidth=2)
plt.title("Regime Portfolio (IMPROVED) vs SP500")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("portfolio_performance_IMPROVED.png")
plt.show()

print("\nBacktest complete. Check portfolio_performance_IMPROVED.png")
print("Key improvements: higher min cluster (5 obs), adaptive regularization, adaptive KNN")
