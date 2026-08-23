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
p = np.array([0.10, 0.20, 0.30, 0.20, 0.15, 0.05])

x1 = np.array([1.40, 1.20, 1.05, 0.95, 0.80, 0.60])
x2 = np.array([1.15, 1.10, 1.02, 0.98, 0.90, 0.75])

x = np.column_stack([x1, x2])

rf = 1.02
w_market = np.array([0.5, 0.5])
R_market = x @ w_market

m_raw = 1.0 / R_market
k = 1.0 / (rf * (p * m_raw).sum())
m = k * m_raw

idx = np.argsort(R_market)
states_sorted = states[idx]
R_sorted = R_market[idx]
m_sorted = m[idx]

x_pos = np.arange(len(states_sorted))

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(x_pos, R_sorted, width=0.6, color="#1f77b4", label="Market return")
ax1.set_ylabel("Gross return")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(states_sorted, rotation=30, ha="right")
ax1.grid(True, axis="y", alpha=0.35)
ax1.set_ylim(0.0, R_sorted.max() * 1.25)

ax2 = ax1.twinx()
ax2.plot(x_pos, m_sorted, color="#ff7f0e", marker="o", label="SDF")
ax2.set_ylabel("SDF level")
ax2.set_ylim(m_sorted.min() * 0.95, m_sorted.max() * 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

ax1.set_title("SDF levels across states")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_sdf_states.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
