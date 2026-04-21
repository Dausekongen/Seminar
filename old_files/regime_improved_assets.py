import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load datasets
full_kmeans = pd.read_csv("ml_dataset_full_kmeans.csv", index_col=0, parse_dates=True)
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

assets = ["SP500_ret", "Bonds_ret"]  # Only good performers
macro_features = [
    "Inflation", "Inflation_mom", "IP_growth", "Yield_curve",
    "Rate_level", "Unemployment", "Unemployment_change", "Infl_exp"
]

data = data.dropna(subset=macro_features + assets)

def markowitz_weights(mu, cov):
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)
    return pd.Series(w, index=mu.index)

k = 3
reg = 1e-6
pretrain_cutoff = pd.Timestamp("2005-12-31")

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

    # Regime weights with higher standard
    regime_weights = {}
    min_cluster_obs = 10  # More stringent
    
    if len(cluster_labels) > 0:
        for c in range(k):
            cluster_indices = np.where(cluster_labels == c)[0]

            if len(cluster_indices) < min_cluster_obs:
                continue

            cluster_returns = asset_data_for_regimes.iloc[cluster_indices][assets]
            mu = cluster_returns.mean()
            cov = cluster_returns.cov()
            
            # Heavy regularization for small samples
            if len(cluster_indices) < 20:
                reg_strength = reg * 20.0  
            elif len(cluster_indices) < 50:
                reg_strength = reg * 5.0   
            else:
                reg_strength = reg
            
            cov += np.eye(len(cov)) * reg_strength
            regime_weights[c] = markowitz_weights(mu, cov)

    # Probability-weighted portfolio (simplified to just blended approach)
    if regime_weights:
        w = sum(regime_weights.values()) / len(regime_weights)
    else:
        w = pd.Series(np.ones(len(assets))/len(assets), index=assets)

    # Portfolio return
    data.loc[data.index[i], "portfolio_ret"] = np.dot(
        data.loc[data.index[i], assets],
        w
    )

# Post-process
data["portfolio_cum"] = (1 + data["portfolio_ret"].fillna(0)).cumprod()
data["SP500_cum"] = (1 + data["SP500_ret"]).cumprod()
data["Bonds_cum"] = (1 + data["Bonds_ret"]).cumprod()
avg_ret = (data["SP500_ret"] + data["Bonds_ret"]) / 2
data["simple50_50_cum"] = (1 + avg_ret).cumprod()

plt.figure(figsize=(14,6))
plt.plot(data["portfolio_cum"], label="Regime (SP500+Bonds)", linewidth=2)
plt.plot(data["SP500_cum"], label="SP500 only", color='black', alpha=0.5, linewidth=2)
plt.plot(data["simple50_50_cum"], label="50/50 SP500+Bonds", color='green', alpha=0.5, linewidth=2)
plt.title("Optimized Regime Portfolio (Excluding Oil & Energy)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("portfolio_reduced_assets.png")
plt.show()

print("Final returns:")
print(f"Regime Portfolio: {data['portfolio_cum'].iloc[-1]:.2f}x")
print(f"SP500: {data['SP500_cum'].iloc[-1]:.2f}x")
print(f"50/50 SP500+Bonds: {data['simple50_50_cum'].iloc[-1]:.2f}x")
