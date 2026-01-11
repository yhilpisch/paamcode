"""
Python & AI in Asset Management
Chapter 18 · RL Algorithms for Asset Management

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
class SimpleEnv:
    prices: pd.Series
    cost: float = 0.0005

    def reset(self):
        self.t = 0
        self.position = 0
        return self.state()

    def state(self):
        window = self.prices.iloc[max(0, self.t - 20): self.t]
        if len(window):
            momentum = window.pct_change().mean()
        else:
            momentum = 0.0
        if momentum is None or not np.isfinite(momentum):
            momentum = 0.0
        return np.array([float(momentum), float(self.position)])

    def step(self, action):
        price_now = self.prices.iloc[self.t]
        price_next = self.prices.iloc[self.t + 1]
        ret = price_next / price_now - 1
        reward = action * ret - self.cost * abs(action - self.position)
        self.position = action
        self.t += 1
        done = self.t >= len(self.prices) - 2
        return self.state(), reward, done

series = pd.read_csv(DATA_PATH,
parse_dates=["Date"]).set_index("Date")["AAPL"].ffill()
env = SimpleEnv(series)
_ = env.reset()

# %% cell 10
actions = [-1, 0, 1]
q_values = {}
alpha = 0.1
gamma = 0.95
epsilon = 0.1


def discretize(state):
    clean = np.nan_to_num(state, nan=0.0)
    return tuple((clean * 50).astype(int))


n_episodes = 200
for episode in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        key = discretize(state)
        q_values.setdefault(key, np.zeros(len(actions)))
        if np.random.rand() < epsilon:
            idx = np.random.randint(len(actions))
        else:
            idx = np.argmax(q_values[key])
        action = actions[idx]
        next_state, reward, done = env.step(action)
        next_key = discretize(next_state)
        q_values.setdefault(next_key, np.zeros(len(actions)))
        target = reward + gamma * q_values[next_key].max()
        q_values[key][idx] += alpha * (target - q_values[key][idx])
        state = next_state
    if (episode + 1) % 50 == 0:
        print(f"Completed Q-learning episode {episode + 1}/{n_episodes}")

len(q_values)

# %% cell 12
def run_policy(policy_vals, episodes=20):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            key = discretize(state)
            action = actions[np.argmax(policy_vals.get(key, np.zeros(len(actions))))]
            state, reward, done = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards)

avg_reward = run_policy(q_values)
avg_reward

# %% cell 14
theta = np.zeros(2)
eta = 0.05

def softmax_policy(state):
    logits = np.array([np.dot(theta, [state[0], action]) for action in actions])
    exp_logits = np.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()

for episode in range(120):
    state = env.reset()
    done = False
    states, actions_taken, rewards = [], [], []
    while not done:
        probs = softmax_policy(state)
        action = np.random.choice(actions, p=probs)
        states.append(state)
        actions_taken.append(action)
        state, reward, done = env.step(action)
        rewards.append(reward)
    G = sum(rewards)
    grad = np.zeros_like(theta)
    for s, a in zip(states, actions_taken):
        probs = softmax_policy(s)
        idx = actions.index(a)
        grad += (1 - probs[idx]) * np.array([s[0], a])
    theta += eta * G * grad

theta
