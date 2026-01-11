"""
Python & AI in Asset Management
Chapter 14 · Tree-Based Methods and Ensembles

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

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
feature_components = {
    "momentum": prices.pct_change(20),
    "volatility": log_rets.rolling(20).std(),
}
features = pd.concat(feature_components, axis=1).dropna()
labels = log_rets.shift(-1)["AAPL"].rename("label")
common_index = features.index.intersection(labels.dropna().index)
features = features.loc[common_index]
labels = labels.loc[common_index]
features.head()

# %% cell 8
split = int(len(features) * 0.7)
X_train, X_test = features.iloc[:split], features.iloc[split:]
y_train, y_test = labels.iloc[:split], labels.iloc[split:]
tree = DecisionTreeRegressor(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
mean_squared_error(y_test, tree_pred)

# %% cell 10
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=50,
    random_state=0,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
gbt = GradientBoostingRegressor(random_state=0)
gbt.fit(X_train, y_train)
gbt_pred = gbt.predict(X_test)
pd.Series(
    {
        "tree_mse": mean_squared_error(y_test, tree_pred),
        "rf_mse": mean_squared_error(y_test, rf_pred),
        "gbt_mse": mean_squared_error(y_test, gbt_pred),
    }
)

# %% cell 12
importances = pd.DataFrame(
    {
        "RF": rf.feature_importances_,
        "GBT": gbt.feature_importances_,
    },
    index=features.columns,
)
importances

# %% cell 14
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

# %% cell 16
signal = pd.Series(rf_pred, index=X_test.index).rank(pct=True) - 0.5
strategy_returns = signal * y_test
performance_stats(strategy_returns)
