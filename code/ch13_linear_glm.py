"""
Python & AI in Asset Management
Chapter 13 · Linear and Generalized Linear Models for Return Prediction

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

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

# %% cell 7
prices = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()

# %% cell 8
panel = prices.ffill().pct_change()
features = panel.rolling(20).mean()
vol = panel.rolling(20).std()
X = pd.concat({"mom": features, "vol": vol}, axis=1).stack().dropna()
y = panel.shift(-1).stack().reindex(X.index).dropna()
X = X.loc[y.index]
data = pd.DataFrame({"mom": X["mom"], "vol": X["vol"], "label": y})
data.head()

# %% cell 10
dates = data.index.get_level_values(0)
split_idx = int(len(dates) * 0.7)
split_date = dates.sort_values()[split_idx]
train = data.loc[dates <= split_date]
test = data.loc[dates > split_date]
X_train, y_train = train[["mom", "vol"]], train["label"]
X_test, y_test = test[["mom", "vol"]], test["label"]

# %% cell 12
ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=5.0)),
])
lasso = Pipeline([
    ("scale", StandardScaler()),
    ("model", Lasso(alpha=0.001)),
])
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)
pd.Series({"ridge_mse": mean_squared_error(y_test, ridge_pred), "lasso_mse":
mean_squared_error(y_test, lasso_pred)})

# %% cell 14
y_train_cls = (y_train > 0).astype(int)
y_test_cls = (y_test > 0).astype(int)
logit = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
logit.fit(X_train, y_train_cls)
logit_prob = logit.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test_cls, logit_prob)
auc

# %% cell 16
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

# %% cell 18
weights = pd.Series(logit_prob - 0.5, index=y_test_cls.index).clip(-0.5, 0.5)
asset_returns = pd.Series(y_test.values, index=y_test.index)
strategy = weights * asset_returns
performance_stats(strategy)
