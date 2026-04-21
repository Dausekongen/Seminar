from pandas_datareader import data as pdr
import datetime

start = datetime.datetime(2007,1,1)
end = datetime.datetime(2026,3,3)

fred_series = [
    "DFF",      # Federal Funds Rate
    "DGS2",     # 2Y Treasury
    "DGS10",    # 10Y Treasury
    "T10YIE",   # Inflation expectations
    "INDPRO",   # Industrial production
    "UNRATE"    # Unemployment
]

macro = pdr.DataReader(fred_series, "fred", start, end)

macro.to_csv("macro_data.csv")