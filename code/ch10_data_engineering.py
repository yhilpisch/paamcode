"""
Python & AI in Asset Management
Chapter 10 · Data Engineering and Cleaning for Financial Time Series

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

# %% cell 6
prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
summary = pd.DataFrame({
    "min": prices.min(),
    "max": prices.max(),
    "pct_missing": prices.isna().mean(),
})
summary

# %% cell 8
filled = prices.ffill()
fill_flags = prices.isna().cumsum().eq(1).any()
fill_flags

# %% cell 10
log_rets = np.log(filled / filled.shift(1)).dropna()
zscores = (log_rets - log_rets.rolling(63).mean()) / log_rets.rolling(63).std()
outlier_mask = zscores.abs() > 4
outlier_counts = outlier_mask.sum().sort_values(ascending=False)
outlier_counts

# %% cell 12
ticker = "BTC-USD"
fig, ax = plt.subplots(figsize=(12, 6))
filled[ticker].plot(ax=ax, alpha=0.7)
mask = outlier_mask[ticker].reindex(filled.index, fill_value=False)
ax.scatter(
    filled.index[mask],
    filled[ticker][mask],
    color="tomato",
    label="Outlier",
)
ax.set_title(f"Outlier Flags for {ticker}")
ax.legend()
plt.show()

# %% cell 14
def build_features(price_frame: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    log_ret = np.log(price_frame / price_frame.shift(1))
    vol = log_ret.rolling(window).std() * np.sqrt(252)
    momentum = price_frame.pct_change(window)
    rolling_mean = price_frame.rolling(window).mean()
    rolling_std = price_frame.rolling(window).std()
    zscore = (price_frame - rolling_mean) / rolling_std
    out = pd.concat(
        {
            "vol": vol,
            "momentum": momentum,
            "zscore": zscore,
        },
        axis=1,
    )
    return out.dropna()

feature_panel = build_features(filled)
print(feature_panel.head().round(4))

# %% cell 16
OUTPUT_DIR = Path("../data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
path = OUTPUT_DIR / "pyaiam_features.parquet"
feature_panel.to_parquet(path)
path

# %% cell 18
def data_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pct_missing_before": raw.isna().mean(),
            "pct_missing_after": cleaned.isna().mean(),
            "up_to_date": cleaned.iloc[-1].notna(),
        }
    )

dqr = data_quality_report(prices, filled)
dqr
