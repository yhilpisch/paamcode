"""
Python & AI in Asset Management
Chapter 23 · LLMs and Agents in the Asset Management Value Chain

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

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=['Date']).set_index('Date').sort_index().ffill()

# %% cell 8
def performance_stats(returns: pd.Series, risk_free: float = 0.02) -> pd.Series:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + returns).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    return pd.Series(
        {
            'annualized_return': ann_ret,
            'annualized_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }
    )

# %% cell 10
def load_prices() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH,
        parse_dates=['Date']).set_index('Date').sort_index().ffill()


def simple_strategy(asset: str = 'AAPL') -> pd.Series:
    px = load_prices()[asset]
    rets = px.pct_change().dropna()
    signal = np.sign(rets.rolling(20).mean()).shift(1).dropna()
    strat_rets = rets.loc[signal.index] * signal
    return strat_rets


def strategy_report(asset: str = 'AAPL') -> dict:
    sr = simple_strategy(asset)
    stats = performance_stats(sr)
    out = {'asset': asset}
    out.update(stats.to_dict())
    return out

# %% cell 12
class SimpleAgent:
    def __init__(self):
        self.history = []

    def run(self, task: str, **kwargs) -> dict:
        if task == 'diagnose_strategy':
            asset = kwargs.get('asset', 'AAPL')
            report = strategy_report(asset)
            self.history.append({'task': task, 'asset': asset, 'report': report})
            return report
        msg = {'error': f'Unknown task: {task}'}
        self.history.append(msg)
        return msg


agent = SimpleAgent()

# %% cell 14
from pprint import pprint

# %% cell 15
report = agent.run('diagnose_strategy', asset='AAPL')
pprint(json.dumps(report, indent=2))
