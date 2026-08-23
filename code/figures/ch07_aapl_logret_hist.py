from pathlib import Path  # filesystem paths

import matplotlib.pyplot as plt  # plotting
import numpy as np  # numerical tools
import pandas as pd  # data handling

plt.style.use("seaborn-v0_8")  # base plotting style
plt.rcParams["font.family"] = "serif"  # use serif font
plt.rcParams["lines.linewidth"] = 1.0  # thin line width

root_dir = Path(__file__).resolve().parents[2]  # repository root
data_path = root_dir / "data" / "eod_tech.csv"  # path to EOD data

eod = pd.read_csv(data_path)  # raw end-of-day prices for tech stocks
eod["Date"] = pd.to_datetime(eod["Date"])  # ensure Date is datetime

prices = eod.set_index("Date")  # wide price panel

log_ret = np.log(prices).diff()  # daily log-returns
series = log_ret["AAPL"].dropna()  # select one ticker and drop NaNs

fig, ax = plt.subplots(figsize=(5, 3))  # create figure and axis
ax.hist(
    series.values,
    bins=50,
    density=False,
    alpha=0.8,
    color="tab:blue",
)  # histogram of log-returns
ax.set_title("Histogram of AAPL Daily Log-Returns")  # informative title
ax.set_xlabel("Daily log-return")  # x-axis label
ax.set_ylabel("Frequency")  # y-axis label

fig_dir = root_dir / "figures"  # figures directory at repo root
fig_dir.mkdir(parents=True, exist_ok=True)  # ensure figures/ exists

output_path = fig_dir / "ch07_aapl_logret_hist.png"  # output file path
fig.savefig(output_path, dpi=300, bbox_inches="tight")  # save PNG at 300 dpi
