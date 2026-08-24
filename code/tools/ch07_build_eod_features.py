from pathlib import Path  # filesystem paths

import numpy as np  # numerical tools
import pandas as pd  # data handling

ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root
DATA_DIR = ROOT_DIR / "data"  # data directory

input_path = DATA_DIR / "eod_tech.csv"  # raw EOD prices
output_path = DATA_DIR / "eod_tech_features.csv"  # features file (long format)

eod = pd.read_csv(input_path)  # raw end-of-day prices
eod["Date"] = pd.to_datetime(eod["Date"])  # ensure Date is datetime

prices = eod.set_index("Date")  # wide price panel with tickers as columns

log_ret = np.log(prices).diff()  # daily log-returns

window = 20  # look-back window in trading days
mom_20d = log_ret.rolling(window).sum()  # 20-day cumulative log-return
vol_20d = log_ret.rolling(window).std() * np.sqrt(252)  # annualized volatility
fwd_1d_ret = log_ret.shift(-1)  # next-day log-return
fwd_1d_excess_ret = fwd_1d_ret.sub(fwd_1d_ret.mean(axis=1), axis=0)

mom_20d_cs = mom_20d.sub(mom_20d.mean(axis=1), axis=0)
mom_20d_cs = mom_20d_cs.div(mom_20d.std(axis=1), axis=0)

panel = pd.DataFrame(
    {
        "log_ret": log_ret.stack(),
        "mom_20d_cs": mom_20d_cs.stack(),
        "vol_20d": vol_20d.stack(),
        "fwd_1d_excess_ret": fwd_1d_excess_ret.stack(),
    }
).dropna()  # long format with MultiIndex (date, ticker)

panel.index = panel.index.set_names(["date", "ticker"])  # name index levels
panel = panel.reset_index()  # move index to columns for CSV output

output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure data dir exists
panel.to_csv(output_path, index=False)  # write features to CSV
