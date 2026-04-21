import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

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

data = data.dropna(subset=features + assets)

# -----------------------------
# 2️⃣ Markowitz
# -----------------------------
def markowitz_weights(mu, cov):
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    return w / w.sum()

# -----------------------------
# 3️⃣ Parameters
# -----------------------------
k = 3
start_idx = 36
reg = 1e-5

scaler = StandardScaler()

portfolio_returns = []
weights_list = []
prob_list = []

# -----------------------------
# 4️⃣ Rolling HMM strategy
# -----------------------------
for i in range(start_idx, len(data)):

    # TRAIN DATA
    X_train = data.iloc[:i-1][features]
    X_scaled = scaler.fit_transform(X_train)

    # HMM MODEL
    hmm = GaussianHMM(
        n_components=k,
        covariance_type="diag",   # more stable
        n_iter=200,
        random_state=42
    )
    hmm.fit(X_scaled)

    # REGIME PROBABILITIES
    probs_all = hmm.predict_proba(X_scaled)
    probs = probs_all[-1]
    prob_list.append(probs)

    # HIDDEN STATES
    states = hmm.predict(X_scaled)

    # -----------------------------
    # REGIME PORTFOLIOS
    # -----------------------------
    regime_weights = {}

    for c in range(k):
        idx = np.where(states == c)[0]

        if len(idx) < 12:
            continue

        returns_c = data.iloc[idx][assets]

        mu = returns_c.mean()
        cov = returns_c.cov()
        cov += np.eye(len(cov)) * reg

        try:
            regime_weights[c] = markowitz_weights(mu, cov)
        except:
            continue

    # -----------------------------
    # COMBINE WEIGHTS
    # -----------------------------
    w = pd.Series(0, index=assets, dtype=float)

    for c in range(k):
        if c in regime_weights:
            w += probs[c] * regime_weights[c]

    if w.sum() == 0:
        w = pd.Series(np.ones(len(assets)) / len(assets), index=assets)

    weights_list.append(w)

    # -----------------------------
    # RETURN
    # -----------------------------
    ret = np.dot(data.iloc[i][assets], w)
    portfolio_returns.append(ret)

# -----------------------------
# 5️⃣ Build results
# -----------------------------
dates = data.index[start_idx:]

results = pd.DataFrame({
    "portfolio_ret": portfolio_returns
}, index=dates)

results["portfolio_cum"] = (1 + results["portfolio_ret"]).cumprod()
results["SP500_cum"] = (1 + data.loc[dates, "SP500_ret"]).cumprod()

weights_df = pd.DataFrame(weights_list, index=dates)
prob_df = pd.DataFrame(prob_list, index=dates,
                       columns=[f"Regime_{i}" for i in range(k)])

# -----------------------------
# 6️⃣ Performance stats
# -----------------------------
ret = results["portfolio_ret"]

ann_return = (1 + ret.mean())**12 - 1
ann_vol = ret.std() * np.sqrt(12)
sharpe = ann_return / ann_vol

cum = results["portfolio_cum"]
roll_max = cum.cummax()
drawdown = cum / roll_max - 1
max_dd = drawdown.min()

print("\n📊 PERFORMANCE")
print(f"Annual Return: {ann_return:.2%}")
print(f"Annual Vol:    {ann_vol:.2%}")
print(f"Sharpe Ratio:  {sharpe:.2f}")
print(f"Max Drawdown:  {max_dd:.2%}")

# -----------------------------
# 7️⃣ Plots
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(results["portfolio_cum"], label="HMM Portfolio")
plt.plot(results["SP500_cum"], label="SP500", alpha=0.4)
plt.title("HMM Portfolio vs SP500")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# Regime probabilities
# -----------------------------
plt.figure(figsize=(14,6))
for c in range(k):
    plt.plot(prob_df.index, prob_df[f"Regime_{c}"], label=f"Regime {c}")

plt.title("HMM Regime Probabilities")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# Weights
# -----------------------------
weights_df.plot(figsize=(14,6), title="Portfolio Weights")
plt.grid(alpha=0.3)
plt.show()