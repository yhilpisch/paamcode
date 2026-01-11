"""
Python & AI in Asset Management
Chapter 11 · Performance Measurement, Backtesting, and Pitfalls

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
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()
assets = ["AAPL", "NVDA", "SPY"]
log_rets = np.log(prices[assets] / prices[assets].shift(1)).dropna()
summary = pd.DataFrame(
    {
        "avg_return": log_rets.mean() * 252,
        "vol": log_rets.std() * np.sqrt(252),
    }
)
summary

# %% cell 8
def performance_stats(returns: pd.Series, risk_free: float = 0.02):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + returns).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    return pd.Series(
        {
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }
    )

perf_example = performance_stats(log_rets["AAPL"])
perf_example

# %% cell 10
def generate_signal(data: pd.DataFrame) -> pd.Series:
    momentum = data.pct_change(21).iloc[:, 0]
    return np.sign(momentum).shift(1).dropna()

def backtest(prices: pd.Series) -> pd.Series:
    sig = generate_signal(prices.to_frame())
    aligned_prices = prices.loc[sig.index]
    rets = aligned_prices.pct_change().dropna()
    strategy_rets = rets * sig.loc[rets.index]
    return strategy_rets

strategy_returns = backtest(prices["AAPL"])
strategy_returns.head()

# %% cell 12
def plot_performance(returns: pd.Series, title: str):
    wealth = (1 + returns).cumprod()
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    wealth.plot(ax=axes[0])
    axes[0].set_title(f"Equity Curve – {title}")
    axes[0].set_ylabel("Growth of $1")
    rolling_sharpe.plot(ax=axes[1])
    axes[1].set_title("Rolling 3M Sharpe (daily approximation)")
    axes[1].set_ylabel("Sharpe")
    plt.show()

plot_performance(strategy_returns, "Momentum Tilt")

# %% cell 14
def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1

fig, ax = plt.subplots(figsize=(12, 6))
drawdown_series(strategy_returns).plot(ax=ax)
ax.set_title("Drawdown Profile – Momentum Tilt")
ax.set_ylabel("Drawdown")
plt.show()
