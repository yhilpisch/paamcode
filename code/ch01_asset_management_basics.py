"""
Python & AI in Asset Management
Chapter 1 · Asset Management Basics and Problem Landscape

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
# sns.set_context("notebook")

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

plt.rcParams.update({'font.family': 'serif', 'figure.dpi': 300})

try:
    from IPython.display import display as _display  # type: ignore
except Exception:  # pragma: no cover
    _display = None


def display_frame(frame: pd.DataFrame) -> None:
    if _display is not None:
        _display(frame)
        return
    with pd.option_context("display.max_rows", 40, "display.width", 140):
        print(frame.to_string())


# %% cell 6
raw = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
raw.head()
# --> remove final empty rows

# %% cell 7
stats = raw.describe().T.assign(non_null=raw.notna().sum(), missing=raw.isna().sum())
display_frame(stats)
print(f"Date range: {raw.index.min().date()} → {raw.index.max().date()} ({len(raw):,} rows)")

# %% cell 9
def prepare_return_panels(price_frame: pd.DataFrame) -> dict:
    filled = price_frame.ffill()
    log_ret = np.log(filled / filled.shift(1)).dropna()
    simple_ret = filled.pct_change().dropna()
    return {"prices": filled, "log": log_ret, "simple": simple_ret}

panels = prepare_return_panels(raw)
log_rets = panels["log"]
simple_rets = panels["simple"]
log_rets.head()

# %% cell 11
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
selected = ["AAPL", "NVDA", "GLD", "TLT", "BTC-USD", "EURUSD"]
(panels["prices"][selected] / panels["prices"][selected].iloc[0]).plot(ax=axes[0], linewidth=1.1, alpha=0.9)
axes[0].set_title("Indexed Price Paths (start = 1.0)")
axes[0].set_ylabel("Index Level")
log_rets[selected].rolling(63).std().mul(np.sqrt(252)).plot(ax=axes[1], linewidth=1.1, alpha=0.9)
axes[1].set_title("Rolling 3-Month Annualized Volatility")
axes[1].set_ylabel("Annualized Volatility")
# plt.tight_layout()
plt.show()

# %% cell 13
ASSET_CLASSES = {
    "AAPL": "US Equities",
    "NVDA": "US Equities",
    "JPM": "Financials",
    "SPY": "Multi-Asset Benchmark",
    "GLD": "Commodities",
    "TLT": "US Treasuries",
    "EURUSD": "FX",
    "BTC-USD": "Digital Assets",
}

base_universe = list(ASSET_CLASSES.keys())
constraints = {
    "long_only": True,
    "max_weight_per_asset": 0.35,
    "min_weight_per_asset": 0.0,
    "cluster_caps": {
        "US Equities": 0.6,
        "Digital Assets": 0.15,
    },
}
risk_budget = {"target_vol": 0.12, "max_drawdown": 0.2}

mandate = {
    "name": "Diversified Multi-Asset Mandate",
    "universe": base_universe,
    "constraints": constraints,
    "risk_budget": risk_budget,
}
mandate

# %% cell 15
latest_prices = panels["prices"].iloc[-1]
annualized_vol = log_rets.std() * np.sqrt(252)
sample_weights = pd.Series(1 / len(base_universe), index=base_universe, name="eq_weight")
alloc_overview = pd.DataFrame({
    "price": latest_prices,
    "annualized_vol": annualized_vol,
    "asset_class": pd.Series(ASSET_CLASSES),
    "weight": sample_weights,
}).dropna()
alloc_overview

# %% cell 16
subset = alloc_overview.drop(index="BTC-USD", errors="ignore")
fig, ax = plt.subplots(figsize=(12, 5))
scatter = ax.scatter(
    subset["annualized_vol"],
    subset["price"],
    s=subset["weight"] * 3000,
    c=subset["annualized_vol"],
    cmap="viridis",
)
for idx, (asset, row) in enumerate(subset.iterrows()):
    offset = (8, 12 if idx % 2 == 0 else -14)
    ax.annotate(
        asset,
        (row["annualized_vol"], row["price"]),
        textcoords="offset points",
        xytext=offset,
        fontweight="bold",
    )
ax.set_xlabel("Annualized Volatility")
ax.set_ylabel("Latest Price")
ax.set_title("Volatility vs. Price with Sample Weights")
fig.colorbar(scatter, label="Annualized Volatility")
plt.show()

# %% cell 18
def describe_portfolio(log_returns: pd.DataFrame, weights: pd.Series, risk_free_rate: float = 0.02):
    aligned = log_returns[weights.index].dropna()
    port_log = aligned.mul(weights).sum(axis=1)
    ann_return = port_log.mean() * 252
    ann_vol = port_log.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    max_dd = (np.exp(port_log.cumsum()) / np.exp(port_log.cumsum()).cummax() - 1).min()
    return pd.Series(
        {
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }
    )

describe_portfolio(log_rets, sample_weights)
