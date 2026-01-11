"""
Python & AI in Asset Management
Chapter 17 · Reinforcement Learning Foundations

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

from dataclasses import dataclass

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date").sort_index().ffill()

# %% cell 8
@dataclass
class PortfolioEnv:
    prices: pd.DataFrame
    cost: float = 0.0005

    def reset(self):
        self.t = 0
        self.wealth = 1.0
        return self.state()

    def state(self):
        window = self.prices.iloc[max(0, self.t - 20): self.t]
        return window.pct_change().mean().fillna(0).values

    def step(self, action):  # action in {-1, 0, 1}
        ret = self.prices.iloc[self.t + 1]["AAPL"] / self.prices.iloc[self.t]["AAPL"] - 1
        reward = action * ret - self.cost * abs(action)
        self.wealth *= (1 + reward)
        self.t += 1
        done = self.t >= len(self.prices) - 2
        return self.state(), reward, done

env = PortfolioEnv(prices[["AAPL"]])
state0 = env.reset()
state0

# %% cell 10
actions = [-1, 0, 1]
policy = {a: 1 / len(actions) for a in actions}
value = 0.0
gamma = 0.95
n_episodes = 250

for episode in range(n_episodes):
    env.reset()
    done = False
    G = 0.0
    step = 0
    while not done:
        action = np.random.choice(actions)
        _, reward, done = env.step(action)
        G += (gamma ** step) * reward
        step += 1
    value += G
    if (episode + 1) % 50 == 0:
        avg_so_far = value / (episode + 1)
        print(f"Episode {episode + 1}/{n_episodes}: avg return {avg_so_far:.6f}")

value / n_episodes
