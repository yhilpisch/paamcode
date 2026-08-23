from pathlib import Path  # filesystem paths

import numpy as np  # numerical tools
import pandas as pd  # time-indexed data
import matplotlib.pyplot as plt  # plotting library

plt.style.use("seaborn-v0_8")  # base plotting style
plt.rcParams["font.family"] = "serif"  # use serif font
plt.rcParams["lines.linewidth"] = 1.0  # thin line width

rng = np.random.default_rng(seed=42)  # reproducible random generator

dates = pd.date_range("2026-01-01", periods=50, freq="B")  # 50 business days

log_ret = pd.Series(
    rng.normal(0.0005, 0.01, size=len(dates)),  # small drift, some noise
    index=dates,
)  # toy daily log-returns

portfolio_value_raw = np.exp(log_ret.cumsum())  # unscaled portfolio path
portfolio_value = (
    portfolio_value_raw / portfolio_value_raw.iloc[0]
)  # start at 1.0

fig, ax = plt.subplots(figsize=(8, 3))  # create figure and axis
ax.plot(
    portfolio_value.index,
    portfolio_value.values,
    label="Equity portfolio",
)  # portfolio value curve
ax.set_title("Simulated Equity Portfolio Value")  # informative title
ax.set_xlabel("Date")  # x-axis label
ax.set_ylabel("Portfolio value (start = 1.0)")  # y-axis label
ax.legend()  # show legend

root_dir = Path(__file__).resolve().parents[2]  # repository root
fig_dir = root_dir / "figures"  # figures directory at repo root
fig_dir.mkdir(parents=True, exist_ok=True)  # ensure figures/ exists

output_path = fig_dir / "ch03_equity_curve.png"  # output file path
fig.savefig(output_path, dpi=300, bbox_inches="tight")  # save high-res PNG
