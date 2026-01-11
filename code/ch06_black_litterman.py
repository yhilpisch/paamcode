"""
Python & AI in Asset Management
Chapter 6 · Black–Litterman and Bayesian Portfolio Construction

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
RISK_AVERSION = 3.0
TAU = 0.05

# %% cell 6
assets = ["AAPL", "NVDA", "JPM", "SPY", "GLD", "TLT"]
prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
prices = prices.ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()[assets]
mean_returns = log_rets.mean() * 252
cov_matrix = log_rets.cov() * 252
w_benchmark = np.repeat(1 / len(assets), len(assets))
mean_returns

# %% cell 8
pi = RISK_AVERSION * cov_matrix.values @ w_benchmark
pd.Series(pi, index=assets)

# %% cell 10
P = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, 0],
    ]
)
Q = np.array([0.01, 0.005])
OMEGA = np.diag([0.0004, 0.0009])

# %% cell 12
def black_litterman(cov: np.ndarray, tau: float, pi_vec: np.ndarray, P: np.ndarray,
Q: np.ndarray, omega: np.ndarray):
    tau_cov = tau * cov
    inv_tau_cov = np.linalg.inv(tau_cov)
    middle = P.T @ np.linalg.inv(omega) @ P
    posterior_cov = np.linalg.inv(inv_tau_cov + middle)
    posterior_mean = posterior_cov @ (inv_tau_cov @ pi_vec + P.T @ np.linalg.inv(omega) @
        Q)
    return posterior_mean, posterior_cov

post_mean, post_cov = black_litterman(
    cov_matrix.values,
    TAU,
    pi,
    P,
    Q,
    OMEGA,
)
pd.Series(post_mean, index=assets)

# %% cell 14
comparison = pd.DataFrame({"pi": pi, "posterior": post_mean}, index=assets)
comparison

# %% cell 16
cov_inv = np.linalg.inv(post_cov)
ones = np.ones(len(assets))
posterior_weights = cov_inv @ (post_mean - 0.02 * ones)
posterior_weights /= posterior_weights.sum()
weights_df = pd.DataFrame(
    {
        "Benchmark": w_benchmark,
        "Posterior": posterior_weights,
    },
    index=assets,
)
weights_df

# %% cell 17
ax = weights_df.plot(kind="bar", figsize=(12, 6))
ax.set_ylabel("Weight")
ax.set_title("Benchmark vs. Black–Litterman Posterior Weights")
plt.show()

# %% cell 19
rng = np.random.default_rng(123)
n_ports = 4000
weights = rng.dirichlet(np.ones(len(assets)), size=n_ports)
prior_returns = weights @ pi
prior_vols = np.sqrt(np.einsum("bi,ij,bj->b", weights, cov_matrix.values, weights))
post_returns = weights @ post_mean
post_vols = np.sqrt(np.einsum("bi,ij,bj->b", weights, post_cov, weights))
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    prior_vols,
    prior_returns,
    s=6,
    alpha=0.6,
    color=plt.cm.coolwarm(0.2),
    label="Prior",
)
ax.scatter(
    post_vols,
    post_returns,
    s=6,
    alpha=0.7,
    color=plt.cm.coolwarm(0.8),
    label="Posterior",
)
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected return")
ax.set_title("Prior vs. Posterior Monte Carlo Frontiers")
ax.legend()
plt.show()
