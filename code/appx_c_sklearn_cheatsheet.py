"""
Python & AI in Asset Management
Appendix C · scikit-learn Cheat Sheet

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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=['Date']).set_index('Date').sort_index().ffill()

# %% cell 8
log_rets = np.log(prices / prices.shift(1)).dropna()
X = pd.DataFrame(
    {
        'momentum': prices['AAPL'].pct_change(20),
        'volatility': log_rets['AAPL'].rolling(20).std(),
    }
).dropna()
y = log_rets['AAPL'].shift(-1).reindex(X.index).dropna()
X = X.loc[y.index]
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=10.0)),
])
X_train, X_test = X.iloc[:-252], X.iloc[-252:]
y_train, y_test = y.iloc[:-252], y.iloc[-252:]
pipe.fit(X_train, y_train)
float(mean_squared_error(y_test, pipe.predict(X_test)))

# %% cell 10
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {'model__alpha': [1.0, 10.0, 100.0]}
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
)
search.fit(X, y)
search.best_params_

# %% cell 12
y_cls = (y > 0).astype(int)
logit = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])
logit.fit(X_train, y_cls.iloc[:-252])
probs = logit.predict_proba(X_test)[:, 1]
float(roc_auc_score(y_cls.iloc[-252:], probs))
