# =============================================================================
# FILE 1: data_pipeline.py
# Collect, transform, and save all data needed for the portfolio model.
# Run this once — produces backtest_data.csv and pretrain_macro.csv
# =============================================================================

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# ── Settings ──────────────────────────────────────────────────────────────────
START        = datetime.datetime(1980, 1, 1)
END          = datetime.datetime(2026, 3, 31)
PRETRAIN_END = "2008-12-31"   # KMeans pre-train cutoff — nothing after this
                               # date is seen during pre-training
TICKERS = [
    # Core (keep)
    "SPY", "QQQ", "TLT", "LQD", "GLD", "VNQ",
    # Global equities
    "VEA", "VWO",
    # Commodities / inflation
    "DBC", "XLE", "TIP",
    # Credit / risk
    "HYG",
    # Defensive sectors
    "XLP", "XLU",
    # Currency
    "UUP"
]


TICKER_RENAME = {
    "SPY": "SP500",
    "QQQ": "QQQ",
    "TLT": "TLT",
    "LQD": "LQD",
    "GLD": "Gold",
    "VNQ": "RealEstate",

    "VEA": "DevelopedEquity",
    "VWO": "EmergingEquity",

    "DBC": "Commodities",
    "XLE": "Energy",
    "TIP": "TIPS",

    "HYG": "HighYield",

    "XLP": "Staples",
    "XLU": "Utilities",

    "UUP": "USD"
}

FRED_SERIES = {
    "CPIAUCSL"  : "CPI",
    "DFF"       : "FedFunds",    # Risk-free rate proxy
    "DGS2"      : "Yield2Y",
    "DGS10"     : "Yield10Y",
    "T10YIE"    : "InflExp",
    "INDPRO"    : "IndProd",
    "UNRATE"    : "Unemployment",
    "DCOILWTICO": "Oil",
}

ASSETS = [
    "SP500", "QQQ", "TLT", "LQD", "Gold", "RealEstate",
    "DevelopedEquity", "EmergingEquity",
    "Commodities", "Energy", "TIPS",
    "HighYield",
    "Staples", "Utilities",
    "USD"
]

# ── 1. Asset prices (dividend + split adjusted) ───────────────────────────────
print("Downloading asset prices...")
raw    = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = raw["Close"].rename(columns=TICKER_RENAME)

# ── 2. Macro data from FRED ───────────────────────────────────────────────────
print("Downloading macro data from FRED...")
macro = pdr.DataReader(list(FRED_SERIES.keys()), "fred", START, END)
macro = macro.rename(columns=FRED_SERIES).ffill()

# ── 3. Merge → resample to month-end ─────────────────────────────────────────
m = prices.merge(macro, left_index=True, right_index=True, how="left").ffill()
m = m.resample("ME").last()

# ── 4. Asset returns — simple %, used by Markowitz ───────────────────────────
# Excess returns = asset return minus risk-free rate.
# DFF is annualised %, divide by 1200 for a monthly rate.
m["RF_monthly"] = m["FedFunds"] / 1200

for col in ASSETS:
    m[f"{col}_ret"]    = m[col].pct_change()
    m[f"{col}_excess"] = m[f"{col}_ret"] - m["RF_monthly"]

# ── 5. Macro features — used by KMeans only ──────────────────────────────────
# All features are stationary (no raw price levels in clustering).
m["Inflation"]      = np.log(m["CPI"]     / m["CPI"].shift(12))
m["Inflation_mom"]  = m["Inflation"].diff()
m["IP_growth"]      = np.log(m["IndProd"] / m["IndProd"].shift(12))
m["Yield_curve"]    = m["Yield10Y"] - m["Yield2Y"]
m["Rate_level"]     = m["FedFunds"]
m["Rate_chg"]       = m["FedFunds"].diff()
m["Real_rate"]      = m["FedFunds"] - m["InflExp"]
m["Unemployment"]   = m["Unemployment"]
m["Unemp_chg"]      = m["Unemployment"].diff()
m["Oil_ret"]        = m["Oil"].pct_change()
m["Oil_trend"]      = np.log(m["Oil"] / m["Oil"].shift(12))
m["SP500_mom_6m"] = m["SP500"].pct_change(6)
m["SP500_mom_3m"] = m["SP500"].pct_change(3)
m["SP500_mom_12m"] = m["SP500"].pct_change(12)
m["SP500_vol_3m"] = m["SP500_ret"].rolling(3).std()
m["SP500_vol_6m"] = m["SP500_ret"].rolling(6).std()
m["Credit_spread"] = m["HighYield"] - m["LQD"]
m["Credit_spread_chg"] = m["Credit_spread"].diff()
#m["Inflation_mom"] = m["CPI"].pct_change(1)

for col in ["Inflation", "IP_growth", "Yield_curve", "Rate_chg",
            "Real_rate", "Unemp_chg", "Oil_ret"]:
    m[f"{col}_vol12"] = m[col].rolling(12).std()

# ── 6. Column lists ───────────────────────────────────────────────────────────
MACRO_FEATURES = [
    #"Inflation", 
    "Inflation_mom",
    #"IP_growth",
    #"Yield_curve",
    #"Rate_level",
    #"Rate_chg",
    #"Real_rate",
    #"Unemployment",
    #"Unemp_chg",
    #"Oil_ret",
    #"Oil_trend",
    #"Inflation_vol12",
    #"IP_growth_vol12",
    #"Yield_curve_vol12",
    #"Rate_chg_vol12", 
    #"Real_rate_vol12",
    #"Unemp_chg_vol12",
    "Oil_ret_vol12",
    "SP500_mom_6m",
    "SP500_mom_3m",
    "SP500_mom_12m",
    "SP500_vol_3m",
    "SP500_vol_6m",
    "Credit_spread",
    "Credit_spread_chg"
]

ASSET_RETS   = [f"{a}_ret"    for a in ASSETS]
ASSET_EXCESS = [f"{a}_excess" for a in ASSETS]
ALL_NEEDED   = MACRO_FEATURES + ASSET_RETS + ASSET_EXCESS + ["RF_monthly"]



# ── 7. Drop incomplete rows, split, save ─────────────────────────────────────
full = m.dropna(subset=ALL_NEEDED)

backtest_data  = full.loc["2009-01-01":]
pretrain_macro = full.loc[:PRETRAIN_END][MACRO_FEATURES]

backtest_data.to_csv("backtest_data.csv")
pretrain_macro.to_csv("pretrain_macro.csv")
pd.Series(MACRO_FEATURES).to_csv("macro_features.csv", index=False, header=False)

print(f"\nDone.")
print(f"  backtest_data  : {backtest_data.shape}  "
      f"({backtest_data.index[0].date()} → {backtest_data.index[-1].date()})")
print(f"  pretrain_macro : {pretrain_macro.shape}  "
      f"({pretrain_macro.index[0].date()} → {pretrain_macro.index[-1].date()})")
print(f"  Macro features : {len(MACRO_FEATURES)}")
print(f"  Assets         : {len(ASSETS)}")

