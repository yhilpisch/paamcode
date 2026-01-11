"""
Python & AI in Asset Management
Appendix B · NumPy and pandas Reference

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
parse_dates=['Date']).set_index('Date').sort_index().ffill()

# %% cell 8
w = np.array([0.3, 0.2, 0.5])
mu = np.array([0.08, 0.05, 0.06])
Sigma = np.array(
    [
        [0.10, 0.02, 0.01],
        [0.02, 0.05, 0.015],
        [0.01, 0.015, 0.04],
    ]
)
port_ret = w @ mu
port_var = w @ Sigma @ w
float(port_ret), float(np.sqrt(port_var))

# %% cell 10
subset = prices[['AAPL', 'SPY']].loc['2023-01-01': '2023-06-30']
subset.head()

# %% cell 12
weekly = prices.resample('W-FRI').last()
monthly = prices.resample('ME').last()
weekly.head(), monthly.head()

# %% cell 14
log_rets = np.log(prices / prices.shift(1)).dropna()
rolling_vol = log_rets['AAPL'].rolling(63).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(12, 6))
rolling_vol.plot(ax=ax)
ax.set_title('AAPL 3M Rolling Volatility (annualized)')
ax.set_ylabel('Volatility')
plt.show()

# %% cell 16
vec = np.array([1.0, -2.0, 3.5, -4.2])
positive_mask = vec > 0
vec_squared = vec**2
vec[positive_mask], vec_squared

# %% cell 18
wide = prices[['AAPL', 'SPY']].iloc[-5:]
long = wide.reset_index().melt(id_vars='Date', var_name='ticker', value_name='price')
long.head()
