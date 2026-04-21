import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, PowerTransformer
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from scipy.stats.mstats import winsorize

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

# -----------------------------
# 2️⃣ Define assets and features
# -----------------------------
assets = ["SP500_ret", "Bonds_ret", "Gold_ret", "Oil_ret", "Energy_ret"]

# 🔥 Reduced features (Fix 1)
features = [
    "SP500_vol",
    "Bond_vol",
    "Inflation",
    "Yield_curve"
]

# -----------------------------
# 3️⃣ Compute rolling volatilities
# -----------------------------
for asset in assets:
    vol_name = asset.replace("_ret", "_vol")
    data[vol_name] = data[asset].rolling(12).std()

# -----------------------------
# 4️⃣ Cleanup
# -----------------------------
data = data.dropna(subset=features + assets)

# -----------------------------
# 5️⃣ Make data more Gaussian
# -----------------------------

# Log vol
for col in ["SP500_vol", "Bond_vol"]:
    data[col] = np.log(data[col] + 1e-6)

# Winsorize
for col in features:
    data[col] = winsorize(data[col], limits=[0.01, 0.01])

# Power transform
pt = PowerTransformer(method="yeo-johnson")
data[features] = pt.fit_transform(data[features])

# -----------------------------
# 6️⃣ Normality diagnostics
# -----------------------------
X = data[features].dropna()

X.hist(figsize=(12,8), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(12,8))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    stats.probplot(X[col], dist="norm", plot=plt)
    plt.title(col)
plt.tight_layout()
plt.show()

results = []
for col in features:
    series = X[col].dropna()
    sample = series.sample(min(500, len(series)), random_state=42)

    shapiro_p = stats.shapiro(sample)[1]
    jb_p = stats.jarque_bera(series)[1]

    results.append({
        "Feature": col,
        "Shapiro_p": shapiro_p,
        "JB_p": jb_p,
        "Skew": series.skew(),
        "Kurtosis": series.kurtosis()
    })

print("\nNORMALITY TEST RESULTS")
print(pd.DataFrame(results).round(4))

# -----------------------------
# 7️⃣ Markowitz
# -----------------------------
def markowitz_weights(mu, cov):
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mu

    if abs(w.sum()) < 1e-8:
        w = np.ones(len(mu)) / len(mu)
    else:
        w = w / w.sum()

    return pd.Series(w, index=mu.index)

# -----------------------------
# 8️⃣ HMM Backtest
# -----------------------------
k = 3
window = 120
start_idx = window

applied_weights = []
state_probs_history = []
predicted_states = []
portfolio_returns = []

scaler = StandardScaler()

for i in range(start_idx, len(data)):

    train = data.iloc[i-window:i-1]
    X_train = train[features]

    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    model = GaussianHMM(
        n_components=k,
        covariance_type="diag",
        n_iter=500,
        min_covar=1e-3,
        random_state=42
    )

    model.fit(X_scaled)

    # 🔥 Fix 4: Increase persistence
    transmat = model.transmat_
    transmat = 0.9 * np.eye(k) + 0.1 * transmat
    model.transmat_ = transmat

    # Current state
    X_current = scaler.transform(data.iloc[[i-1]][features])
    probs = model.predict_proba(X_current)[0]

    # 🔥 Fix 3: stronger smoothing
    if len(state_probs_history) > 0:
        probs = 0.1 * probs + 0.9 * state_probs_history[-1]

    probs = probs / probs.sum()
    state_probs_history.append(probs)
    predicted_states.append(np.argmax(probs))

    # Past states
    hidden_states = model.predict(X_scaled)

    # Regime portfolios
    regime_weights = {}

    for s in range(k):
        idx = np.where(hidden_states == s)[0]
        if len(idx) < 8:
            continue

        returns = train[assets].iloc[idx]

        mu = returns.mean()
        mu = 0.5 * mu + 0.5 * mu.mean()

        lw = LedoitWolf().fit(returns)
        cov = pd.DataFrame(lw.covariance_, index=assets, columns=assets)

        regime_weights[s] = markowitz_weights(mu, cov)

    # Combine
    w = pd.Series(0.0, index=assets)

    for s in range(k):
        if s in regime_weights:
            w += probs[s] * regime_weights[s]

    if abs(w.sum()) < 1e-8:
        w = pd.Series(np.ones(len(assets))/len(assets), index=assets)

    # 🔥 Fix 2: weight caps
    w = w.clip(-0.3, 0.5)
    w = w / w.sum()

    # 🔥 Fix 5: reduce turnover
    if len(applied_weights) > 0:
        w = 0.05 * w + 0.95 * applied_weights[-1]

    # 🔥 Fix 6: volatility targeting
    if len(portfolio_returns) > 12:
        realized_vol = pd.Series(portfolio_returns).rolling(12).std().iloc[-1]
        if realized_vol > 0:
            scale = min(2.0, max(0.5, 0.10 / realized_vol))
            w = w * scale

    applied_weights.append(w)

    # Return
    ret = np.dot(data.iloc[i][assets], w)
    portfolio_returns.append(ret)
    data.loc[data.index[i], "portfolio_ret"] = ret

# -----------------------------
# 9️⃣ Results
# -----------------------------
data["portfolio_ret"] = data["portfolio_ret"].fillna(0)
data["portfolio_cum"] = (1 + data["portfolio_ret"]).cumprod()
data["SP500_cum"] = (1 + data["SP500_ret"]).cumprod()

weights_df = pd.DataFrame(applied_weights, index=data.index[start_idx:])
prob_df = pd.DataFrame(state_probs_history, index=data.index[start_idx:])

# Performance
ret = data["portfolio_ret"].dropna()

ann_ret = ret.mean()*12
ann_vol = ret.std()*np.sqrt(12)
sharpe = ann_ret / ann_vol
dd = data["portfolio_cum"] / data["portfolio_cum"].cummax() - 1

print("\nPERFORMANCE")
print(f"Return: {ann_ret:.2%}")
print(f"Vol: {ann_vol:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max DD: {dd.min():.2%}")

# -----------------------------
# 🔟 Plots
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(data["portfolio_cum"], label="Portfolio")
plt.plot(data["SP500_cum"], label="SP500", alpha=0.3)
plt.legend()
plt.title("Final Strategy")
plt.grid()
plt.show()

plt.figure(figsize=(14,6))
for i in range(k):
    plt.plot(prob_df.index, prob_df[i], label=f"State {i}")
plt.legend()
plt.title("Smoothed Regime Probabilities")
plt.grid()
plt.show()

plt.figure(figsize=(14,6))
for a in assets:
    plt.plot(weights_df.index, weights_df[a], label=a)
plt.legend()
plt.title("Weights")
plt.grid()
plt.show()