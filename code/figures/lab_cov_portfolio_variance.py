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

p = np.array([0.10, 0.20, 0.30, 0.20, 0.15, 0.05])

x1 = np.array([1.40, 1.20, 1.05, 0.95, 0.80, 0.60])
x2 = np.array([1.15, 1.10, 1.02, 0.98, 0.90, 0.75])

x = np.column_stack([x1, x2])
mu = (p[:, None] * x).sum(axis=0)
xc = x - mu
sigma = xc.T @ (p[:, None] * xc)

a, b, c = sigma[0, 0], sigma[1, 1], sigma[0, 1]

w1 = np.linspace(-0.5, 1.5, 501)
w2 = 1.0 - w1

var_p = (w1**2) * a + (w2**2) * b + 2.0 * w1 * w2 * c

w1_star = (b - c) / (a + b - 2.0 * c)
w2_star = 1.0 - w1_star
var_star = w1_star**2 * a + w2_star**2 * b + 2.0 * w1_star * w2_star * c

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(w1, var_p, color="#1f77b4")
ax.axvline(w1_star, linewidth=1.0, color="#d62728")
ax.scatter([w1_star], [var_star], color="#d62728", s=45)

ax.set_title(r"Portfolio variance vs weight $w_1$")
ax.set_xlabel(r"$w_1$")
ax.set_ylabel("Variance")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_portfolio_variance.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
