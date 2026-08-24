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
vals, vecs = np.linalg.eigh(sigma)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

sqrt_sigma = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

theta = np.linspace(0.0, 2.0 * np.pi, 400)
circle = np.column_stack([np.cos(theta), np.sin(theta)])
ellipse = circle @ sqrt_sigma.T

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(xc[:, 0], xc[:, 1], s=45, label="Centered states")
ax.plot(
    ellipse[:, 0],
    ellipse[:, 1],
    linewidth=1.6,
    label="Mahalanobis radius 1",
)
ax.axhline(0.0, linewidth=0.8, color="black")
ax.axvline(0.0, linewidth=0.8, color="black")
ax.set_xlabel("Centered payoff 1")
ax.set_ylabel("Centered payoff 2")
ax.set_title("Centered states and covariance ellipse")
ax.grid(True, alpha=0.35)
ax.legend(frameon=True)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_ellipse.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
