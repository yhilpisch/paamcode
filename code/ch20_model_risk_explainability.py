"""
Python & AI in Asset Management
Chapter 20 · Model Risk Management and Explainability

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

import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
features = pd.concat(
    {"momentum": prices.pct_change(20),
     "volatility": log_rets.rolling(20).std(),
     "skew": log_rets.rolling(60).skew()},
    axis=1,
).dropna()
label = log_rets.shift(-1)["AAPL"].rename("label")
common = features.index.intersection(label.dropna().index)
features = features.loc[common]
label = label.loc[common]
split = int(len(features) * 0.8)
X_train, X_test = features.iloc[:split], features.iloc[split:]
y_train, y_test = label.iloc[:split], label.iloc[split:]
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

# %% cell 8
experiment = {
    "model": "RandomForestRegressor",
    "params": rf.get_params(),
    "train_start": str(X_train.index.min()),
    "train_end": str(X_train.index.max()),
}
log_path = Path("../reports") / "model_risk_rf.json"
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text(json.dumps(experiment, indent=2))
log_path

# %% cell 10
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=1)
importance_df = pd.DataFrame(
    {"feature": X_test.columns,
     "importance": perm.importances_mean,
     "std": perm.importances_std},
)
importance_df.sort_values("importance", ascending=False)

# %% cell 12
def what_if(row: pd.Series, adjustments: dict) -> pd.Series:
    outputs = {}
    for feat, delta in adjustments.items():
        modified = row.copy()
        modified[feat] += delta
        outputs[feat] = rf.predict(modified.values.reshape(1, -1))[0]
    return pd.Series(outputs)

sample_row = X_test.iloc[-1]
what_if(sample_row, {"momentum": 0.02, "volatility": 0.01})

# %% cell 14
pred_series = pd.Series(rf.predict(X_test), index=X_test.index)
rolling_ic = pred_series.rolling(63).corr(y_test)
fig, ax = plt.subplots(figsize=(12, 6))
rolling_ic.plot(ax=ax)
ax.set_title("Rolling Information Coefficient")
ax.set_ylabel("IC")
plt.show()
