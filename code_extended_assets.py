import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime
import matplotlib.pyplot as plt
# ---------------------------------------------------
# 1️⃣ Settings
# ---------------------------------------------------
start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2026, 3, 31)

# Financial assets chosen for the US market
tickers = [
    "SPY", "QQQ", "TLT", "LQD", "GLD", "VNQ"
]

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

print("Downloading market data...")
data = yf.download(tickers, start=start, end=end)

adj_close = data["Close"].copy()
adj_close.columns = [
    "SP500", "QQQ", "TLT", "LQD", "Gold", "RealEstate"
]

print("Downloading macro data...")
macro = pdr.DataReader(fred_series, "fred", start, end)
macro = macro.ffill()

merged = adj_close.merge(macro, left_index=True, right_index=True, how="left")
merged = merged.ffill()

monthly = merged.resample("ME").last()
monthly["SP500_raw"] = monthly["SP500"]
asset_cols = ["SP500", "QQQ", "TLT", "LQD", "Gold", "RealEstate"]
for col in asset_cols:
    monthly[col + "_ret"] = np.log(monthly[col] / monthly[col].shift(1))

monthly["Inflation"] = np.log(monthly["CPIAUCSL"] / monthly["CPIAUCSL"].shift(12))
monthly["Inflation_mom"] = monthly["Inflation"].diff()
monthly["GDP_growth"] = np.log(monthly["GDP"] / monthly["GDP"].shift(3))
monthly["IP_growth"] = np.log(monthly["INDPRO"] / monthly["INDPRO"].shift(12))
monthly["Yield_curve"] = monthly["DGS10"] - monthly["DGS2"]
monthly["Rate_level"] = monthly["DFF"]
monthly["Rate_change"] = monthly["DFF"].diff()
monthly["Unemployment"] = monthly["UNRATE"]
monthly["Unemployment_change"] = monthly["UNRATE"].diff()
monthly["Infl_exp"] = monthly["T10YIE"]


monthly["SP500_vol"] = monthly["SP500_ret"].rolling(6).std()
monthly["Bonds_vol"] = monthly["TLT_ret"].rolling(6).std()

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
    "Bonds_vol"
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

final_data = monthly.dropna(subset=feature_cols + [c + "_ret" for c in asset_cols])
full_kmeans = monthly[kmeans_features].dropna()
full_kmeans = full_kmeans.loc[:"2005-12-31"]

final_data.to_csv("ml_dataset_ext.csv")
full_kmeans.to_csv("ml_dataset_ext_kmeans.csv")

print("Dataset created successfully.")
print("Final dataset shape:", final_data.shape)
print("KMeans training shape:", full_kmeans.shape)
print("Date range:", final_data.index.min(), "to", final_data.index.max())
