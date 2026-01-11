"""
Python & AI in Asset Management
Chapter 12 · Machine Learning Foundations and Workflow

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

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# %% cell 6
prices = pd.read_csv(
    DATA_PATH, parse_dates=["Date"]
).set_index("Date").sort_index().ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
feature_window = 20
momentum = prices["AAPL"].pct_change(feature_window)
volatility = log_rets["AAPL"].rolling(feature_window).std()
features = pd.concat(
    {"momentum": momentum, "volatility": volatility}, axis=1
).dropna()
labels = log_rets["AAPL"].shift(-1).rename("label")
frame = pd.concat([features, labels], axis=1).dropna()
frame.head()

# %% cell 8
def performance_stats(returns: pd.Series, risk_free: float = 0.02) -> pd.Series:
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


# %% cell 10
def rolling_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X):
        yield (
            X.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[train_idx],
            y.iloc[val_idx],
        )


# %% cell 12
feature_cols = ["momentum", "volatility"]
X = frame[feature_cols]
y = frame["label"]


# %% cell 14
def run_ridge_workflow(X: pd.DataFrame, y: pd.Series, alpha: float = 10.0):
    metrics = []
    preds = pd.Series(index=y.index, dtype=float)
    for X_train, X_val, y_train, y_val in rolling_split(X, y):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        fold_pred = model.predict(X_val)
        preds.loc[y_val.index] = fold_pred
        mse = mean_squared_error(y_val, fold_pred)
        metrics.append(mse)
    return preds.dropna(), np.mean(metrics)


predictions, avg_mse = run_ridge_workflow(
    frame[["momentum", "volatility"]], frame["label"]
)
print(f"Average fold MSE: {avg_mse:.6f}")


# %% cell 16
ic = predictions.corr(frame["label"].loc[predictions.index])
print(f"Information Coefficient: {ic:.3f}")
portfolio_returns = np.sign(predictions) * frame["label"].loc[predictions.index]
perf = performance_stats(portfolio_returns)
print(perf)
