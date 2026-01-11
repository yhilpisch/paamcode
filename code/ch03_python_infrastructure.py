"""
Python & AI in Asset Management
Chapter 3 · Python Infrastructure for Asset Management Research

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

# %% cell 6
env = pd.Series(
    {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
)
env

# %% cell 8
prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
prices.head()

# %% cell 10
missing = prices.isna().sum()
share_missing = missing.div(len(prices))
pd.DataFrame({"missing": missing, "pct_missing": share_missing})

# %% cell 11
prices_clean = prices.ffill()
log_rets = np.log(prices_clean / prices_clean.shift(1)).dropna()
log_rets.head()

# %% cell 13
panel_long = (
    prices_clean.reset_index()
    .melt(id_vars="Date", var_name="ticker", value_name="price")
    .sort_values(["Date", "ticker"])
)
panel_long.head()

# %% cell 15
def resample_prices(price_frame: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    return price_frame.resample(freq).last()

weekly = resample_prices(prices_clean)
weekly.head()

# %% cell 17
def rolling_features(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    vol = log_returns.rolling(window).std() * np.sqrt(252)
    mom = log_returns.rolling(window).sum()
    out = pd.concat({"vol": vol, "mom": mom}, axis=1)
    return out.dropna()

features = rolling_features(log_rets)
features.head()

# %% cell 19
fig, ax = plt.subplots(figsize=(12, 5))
(prices_clean[["AAPL", "JPM", "SPY"]] / prices_clean[["AAPL", "JPM",
"SPY"]].iloc[0]).plot(ax=ax)
ax.set_title("Indexed Price Paths (start = 1.0)")
ax.set_ylabel("Index level")
plt.show()

# %% cell 21
PROJECT_ROOT = Path("../")
STRUCTURE = {
    "data": ["raw", "intermediate", "features"],
    "notebooks": [],
    "reports": ["figures", "tables"],
}

def print_structure(root: Path, structure: dict) -> None:
    for folder, subfolders in structure.items():
        print(f"{root / folder}")
        for sub in subfolders:
            print(f"    {root / folder / sub}")

print_structure(PROJECT_ROOT, STRUCTURE)

# %% cell 23
CONFIG = {
    "data_file": DATA_PATH,
    "feature_window": 21,
    "plots_dir": PROJECT_ROOT / "reports" / "figures",
}
CONFIG
