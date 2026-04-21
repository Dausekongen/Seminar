import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
data = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)

# =========================
# Settings
# =========================
initial_investment = 1000
monthly_contribution = 10  # set to 0 if you don't want contributions

# =========================
# Basic portfolio growth
# =========================
data["portfolio_value"] = initial_investment * (1 + data["portfolio_ret"].fillna(0)).cumprod()
data["sp500_value"] = initial_investment * (1 + data["SP500_ret"]).cumprod()

# =========================
# Portfolio with contributions
# =========================
values = []
current_value = initial_investment

for r in data["portfolio_ret"].fillna(0):
    current_value = current_value * (1 + r) + monthly_contribution
    values.append(current_value)

data["portfolio_value_contrib"] = values

# =========================
# Drawdown calculation
# =========================
data["peak"] = data["portfolio_value"].cummax()
data["drawdown"] = (data["portfolio_value"] - data["peak"]) / data["peak"]

# =========================
# Plot growth
# =========================
plt.figure(figsize=(14, 6))
plt.plot(data["portfolio_value"], label="Portfolio ($1000 start)", linewidth=2)
plt.plot(data["sp500_value"], label="SP500 ($1000 start)", alpha=0.6)
plt.plot(data["portfolio_value_contrib"], label="Portfolio + Monthly $10", linestyle="--")

plt.title("Portfolio Growth Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================
# Plot drawdowns
# =========================
plt.figure(figsize=(14, 4))
plt.plot(data["drawdown"], label="Drawdown")
plt.title("Portfolio Drawdown")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# =========================
# Final stats
# =========================
print("Final portfolio value (no contributions): $", round(data["portfolio_value"].iloc[-1], 2))
print("Final SP500 value: $", round(data["sp500_value"].iloc[-1], 2))
print("Final portfolio value (with contributions): $", round(data["portfolio_value_contrib"].iloc[-1], 2))

print("\nMax drawdown:", round(data["drawdown"].min() * 100, 2), "%")