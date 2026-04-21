import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime

# ---------------------------------------------------
# 1️⃣ Settings
# ---------------------------------------------------

start = datetime.datetime(1980,1,1)
end = datetime.datetime(2026,3,3)

# Financial assets
tickers = [
    "SPY",   # Equities
    "TLT",   # Bonds
    "GLD",   # Gold
    "USO",   # Oil
    "XLE"    # Energy stocks
]

# Macro variables from FRED
fred_series = [
    "CPIAUCSL",   # CPI
    "GDP",        # GDP (quarterly)
    "DFF",        # Fed funds rate
    "DGS2",       # 2Y yield
    "DGS10",      # 10Y yield
    "T10YIE",     # Inflation expectations
    "INDPRO",     # Industrial production
    "UNRATE"      # Unemployment
]

# ---------------------------------------------------
# 2️⃣ Download financial data
# ---------------------------------------------------

print("Downloading market data...")

data = yf.download(
    tickers,
    start=start,
    end=end
)

adj_close = data["Close"]

adj_close.columns = [
    "SP500",
    "Bonds",
    "Gold",
    "Oil",
    "Energy"
]

# ---------------------------------------------------
# 3️⃣ Download macro data
# ---------------------------------------------------

print("Downloading macro data...")

macro = pdr.DataReader(
    fred_series,
    "fred",
    start,
    end
)

# Forward fill missing macro values
macro = macro.ffill()

# ---------------------------------------------------
# 4️⃣ Merge datasets
# ---------------------------------------------------

merged = adj_close.merge(
    macro,
    left_index=True,
    right_index=True,
    how="left"
)

merged = merged.ffill()

# ---------------------------------------------------
# 5️⃣ Convert to monthly frequency (IMPORTANT FIX)
# ---------------------------------------------------

monthly = merged.resample("ME").last()   # <-- FIXED (M → ME)

# ---------------------------------------------------
# 6️⃣ Financial returns
# ---------------------------------------------------

asset_cols = ["SP500", "Bonds", "Gold", "Oil", "Energy"]

for col in asset_cols:
    monthly[col + "_ret"] = np.log(
        monthly[col] / monthly[col].shift(1)
    )

# ---------------------------------------------------
# 7️⃣ Macro feature engineering (MAJOR FIX)
# ---------------------------------------------------

# Inflation (YoY)
monthly["Inflation"] = np.log(
    monthly["CPIAUCSL"] / monthly["CPIAUCSL"].shift(12)
)

# Inflation momentum
monthly["Inflation_mom"] = monthly["Inflation"].diff()

# GDP growth (quarterly → use 3-month lag)
monthly["GDP_growth"] = np.log(
    monthly["GDP"] / monthly["GDP"].shift(3)
)

# Industrial production (better monthly growth signal)
monthly["IP_growth"] = np.log(
    monthly["INDPRO"] / monthly["INDPRO"].shift(12)
)

# Yield curve
monthly["Yield_curve"] = (
    monthly["DGS10"] - monthly["DGS2"]
)

# Interest rate level
monthly["Rate_level"] = monthly["DFF"]

# Interest rate change
monthly["Rate_change"] = monthly["DFF"].diff()

# Unemployment (level + change)
monthly["Unemployment"] = monthly["UNRATE"]
monthly["Unemployment_change"] = monthly["UNRATE"].diff()

# Inflation expectations
monthly["Infl_exp"] = monthly["T10YIE"]

# ---------------------------------------------------
# 8️⃣ (OPTIONAL BUT STRONGLY RECOMMENDED)
# Add market-based signals
# ---------------------------------------------------

monthly["SP500_vol"] = monthly["SP500_ret"].rolling(6).std()
monthly["Bond_vol"] = monthly["Bonds_ret"].rolling(6).std()

# ---------------------------------------------------
# 9️⃣ Final feature list (CLEAN VERSION)
# ---------------------------------------------------

feature_cols = [
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

kmeans_features = [
    "Inflation",
    "Inflation_mom",
    "IP_growth",
    "Yield_curve",
    "Rate_level",
    "Unemployment",
    "Unemployment_change",
    "Infl_exp"
]

# ---------------------------------------------------
# 🔟 Clean dataset
# ---------------------------------------------------

final_data = monthly.dropna(subset=feature_cols + [c + "_ret" for c in asset_cols])
full_kmeans = monthly[kmeans_features].dropna()

# Filter full_kmeans to end of 2005
full_kmeans = full_kmeans.loc[:"2005-12-31"]

# ---------------------------------------------------
# 1️⃣1️⃣ Save dataset
# ---------------------------------------------------
monthly.to_csv("ml_dataset_full.csv")
full_kmeans.to_csv("ml_dataset_full_kmeans.csv")
final_data.to_csv("ml_dataset.csv")

# ---------------------------------------------------
# 1️⃣2️⃣ Debug output
# ---------------------------------------------------

print("\nDataset created successfully.")
print("Shape:", final_data.shape)
print("Full KMeans training shape (filtered to 2005):", full_kmeans.shape)
print("Earliest KMeans training date:", full_kmeans.index.min())
print("Latest KMeans training date:", full_kmeans.index.max())

print("\nFeatures used:")
print(feature_cols)

print("\nKMeans features used:")
print(kmeans_features)

print("\nColumns:")
print(final_data.columns)

print("\nPreview:")
print(final_data.head())
# ---------------------------------------------------
# 1️⃣2️⃣ Debug output
# ---------------------------------------------------

print("\nDataset created successfully.")
print("Shape:", final_data.shape)

print("\nFeatures used:")
print(feature_cols)

print("\nColumns:")
print(final_data.columns)

print("\nPreview:")
print(final_data.head())