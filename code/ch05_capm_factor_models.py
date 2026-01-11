"""
Python & AI in Asset Management
Chapter 5 · CAPM, Multifactor Models, and APT

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"
RISK_FREE_ANNUAL = 0.02
RISK_FREE_DAILY = RISK_FREE_ANNUAL / 252

# %% cell 6
prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
prices = prices.ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
excess_rets = log_rets.subtract(RISK_FREE_DAILY)
excess_rets.head()

# %% cell 8
def regression_frame(asset: str, market: str = "SPY") -> pd.DataFrame:
    frame = pd.DataFrame({"asset": excess_rets[asset], "market": excess_rets[market]})
    return frame.dropna()

capm_data = regression_frame("AAPL")
capm_data.head()

# %% cell 10
X = sm.add_constant(capm_data["market"])
y = capm_data["asset"]
model = sm.OLS(y, X).fit()
pd.DataFrame({"param": model.params, "tvalue": model.tvalues})

# %% cell 12
window = 252
cov = log_rets[["AAPL", "SPY"]].rolling(window).cov().dropna()
rolling_beta = (
    cov.xs("AAPL", level=1)["SPY"] /
    cov.xs("SPY", level=1)["SPY"]
)
fig, ax = plt.subplots(figsize=(12, 6))
rolling_beta.plot(ax=ax)
ax.set_title("Rolling 1Y Beta (AAPL vs. SPY)")
ax.set_ylabel("Beta")
plt.show()

# %% cell 14
factors = excess_rets[["SPY", "GLD", "TLT"]]
asset = excess_rets["NVDA"]
X = sm.add_constant(factors)
multi_model = sm.OLS(asset, X).fit()
pd.DataFrame({"param": multi_model.params, "tvalue": multi_model.tvalues})

# %% cell 16
factor_means = factors.mean() * 252
loadings = {}
for ticker in excess_rets.columns:
    X = sm.add_constant(factors)
    res = sm.OLS(excess_rets[ticker], X).fit()
    loadings[ticker] = res.params[1:]
loadings_df = pd.DataFrame(loadings).T
expected_returns = loadings_df @ factor_means
expected_returns.head()
