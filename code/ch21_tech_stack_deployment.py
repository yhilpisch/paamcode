"""
Python & AI in Asset Management
Chapter 21 · Technology Stack and Deployment Patterns

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

import datetime as dt
import logging

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()

# %% cell 8
CONFIG = {
    "data_file": DATA_PATH,
    "features_file": Path("../data/features/pyaiam_features.parquet"),
    "reports_dir": Path("../reports"),
    "recipients": ["research@tpq.io"],
}
CONFIG

# %% cell 10
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pipeline")
logger.info("Logger initialized")

# %% cell 12
def ingest(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering rolling features")
    log_ret = np.log(df / df.shift(1))
    vol = log_ret.rolling(20).std()
    momentum = df.pct_change(20, fill_method=None)
    return pd.concat({"vol": vol, "mom": momentum}, axis=1).dropna()

def score(panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("Scoring placeholder model")
    return panel.groupby(level=0).mean()

def daily_pipeline() -> None:
    raw = ingest(CONFIG["data_file"])
    feats = engineer(raw)
    scores = score(feats)
    out = CONFIG["reports_dir"] / f"scores_{dt.date.today():%Y%m%d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out)
    logger.info("Saved scores to %s", out)

daily_pipeline()

# %% cell 14
def send_alert(message: str) -> None:
    logger.warning("ALERT: %s", message)

send_alert("Pipeline completed (demo)")

# %% cell 16
import textwrap
crontab = textwrap.dedent("""
# Run pipeline weekdays at 06:00 UTC
0 6 * * 1-5 /usr/bin/python /repo/scripts/run_pipeline.py >> /repo/logs/pipeline.log
2>&1
""")
print(crontab)
