"""
Python & AI in Asset Management
Appendix F · Practical Tools

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

import logging
import time

# %% cell 6
CONFIG = {
    'data_file': DATA_PATH,
    'reports_dir': Path('../reports'),
    'feature_window': 20,
}
CONFIG

# %% cell 8
def make_logger(name: str = 'pyaiam') -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(name)

logger = make_logger()
logger.info('Logger initialized')

# %% cell 10
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info('Function %s took %.3fs', func.__name__, elapsed)
        return result
    return wrapper

@timed
def demo_sleep():
    time.sleep(0.1)

_ = demo_sleep()
