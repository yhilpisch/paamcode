"""
Python & AI in Asset Management
Chapter 4 · Mean–Variance Portfolio Theory

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
prices = prices.ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
assets = ["AAPL", "NVDA", "JPM", "SPY", "GLD", "TLT"]
log_rets = log_rets[assets]
exp_returns = log_rets.mean() * 252
cov_matrix = log_rets.cov() * 252
exp_returns

# %% cell 8
def portfolio_stats(weights: np.ndarray, mean_vec: np.ndarray, cov_mat: np.ndarray) -> tuple[float, float]:
    port_ret = weights @ mean_vec
    port_vol = np.sqrt(weights @ cov_mat @ weights)
    return port_ret, port_vol

risk_free = 0.02

# %% cell 10
rng = np.random.default_rng(7)
n_ports = 8000
weights = rng.dirichlet(np.ones(len(assets)), size=n_ports)
port_metrics = np.array([portfolio_stats(w, exp_returns.values, cov_matrix.values)
for w in weights])
port_returns = port_metrics[:, 0]
port_vols = port_metrics[:, 1]
sharpe = (port_returns - risk_free) / port_vols
fig, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter(port_vols, port_returns, c=sharpe, cmap="viridis", s=6)
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected return")
ax.set_title("Monte Carlo Portfolio Cloud")
fig.colorbar(sc, label="Sharpe ratio")
plt.show()

# %% cell 12
cov_inv = np.linalg.inv(cov_matrix.values)
ones = np.ones(len(assets))
A = ones @ cov_inv @ ones
B = ones @ cov_inv @ exp_returns.values
gmv_weights = (cov_inv @ ones) / A
gmv_ret, gmv_vol = portfolio_stats(gmv_weights, exp_returns.values,
cov_matrix.values)
ms_weights = (cov_inv @ (exp_returns.values - risk_free * ones))
ms_weights = ms_weights / np.sum(ms_weights)
ms_ret, ms_vol = portfolio_stats(ms_weights, exp_returns.values, cov_matrix.values)
pd.DataFrame(
    {
        "gmv": gmv_weights,
        "max_sharpe": ms_weights,
    },
    index=assets,
)

# %% cell 14
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(port_vols, port_returns, c="lightgray", s=6, alpha=0.5)
ax.scatter(gmv_vol, gmv_ret, color="tomato", label="GMV", s=80)
ax.scatter(ms_vol, ms_ret, color="navy", label="Max Sharpe", s=80)
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected return")
ax.set_title("Efficient Frontier Highlights")
ax.legend()
plt.show()

# %% cell 16
weights_df = pd.DataFrame(
    {
        "GMV": gmv_weights,
        "Max Sharpe": ms_weights,
    },
    index=assets,
)
weights_df

# %% cell 17
ax = weights_df.plot(kind="bar", figsize=(12, 6))
ax.set_ylabel("Weight")
ax.set_title("GMV vs. Max-Sharpe Weights")
plt.show()

# %% cell 19
shrink_factors = np.linspace(0.2, 1.0, 5)
rows = []
for shrink in shrink_factors:
    shrunk_mean = exp_returns.values * shrink
    w = cov_inv @ (shrunk_mean - risk_free * ones)
    w /= np.sum(w)
    ret, vol = portfolio_stats(w, shrunk_mean, cov_matrix.values)
    rows.append({"shrink": shrink, "ret": ret, "vol": vol})
pd.DataFrame(rows)
