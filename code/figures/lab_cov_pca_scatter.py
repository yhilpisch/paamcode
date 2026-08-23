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
mu = (p[:, None] * x).sum(axis=0)
xc = x - mu
sigma = xc.T @ (p[:, None] * xc)

vals, vecs = np.linalg.eigh(sigma)
idx = np.argsort(vals)[::-1]
vecs = vecs[:, idx]

y = xc @ vecs

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y[:, 0], y[:, 1], s=45, color="#1f77b4")

for i, name in enumerate(states):
    ax.annotate(
        name, (y[i, 0], y[i, 1]),
        textcoords="offset points", xytext=(6, 6),
    )

ax.axhline(0.0, linewidth=0.8, color="black")
ax.axvline(0.0, linewidth=0.8, color="black")
ax.set_xlabel("PC1 coordinate")
ax.set_ylabel("PC2 coordinate")
ax.set_title("Centered states in the eigenbasis")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_pca_scatter.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
