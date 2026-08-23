"""
Python & AI in Asset Management
Lab · Covariance Geometry

(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "lines.linewidth": 1.0})

OUT_DIR = Path("labs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

states = np.array(["Boom", "Good", "Normal", "Soft", "Recession", "Crisis"])

x1 = np.array([1.40, 1.20, 1.05, 0.95, 0.80, 0.60])
x2 = np.array([1.15, 1.10, 1.02, 0.98, 0.90, 0.75])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x1, x2, s=45, color="#1f77b4")

for i, name in enumerate(states):
    ax.annotate(name, (x1[i], x2[i]), textcoords="offset points", xytext=(6, 6))

ax.set_xlabel("Asset 1 payoff")
ax.set_ylabel("Asset 2 payoff")
ax.set_title("State cloud in payoff space")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_state_cloud.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
