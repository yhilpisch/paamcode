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

rng = np.random.default_rng(42)

z = rng.normal(size=(20000, 2))
z = z / np.linalg.norm(z, axis=1, keepdims=True)
q = np.einsum("ni,ij,nj->n", z, sigma, z)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(q, bins=60, color="#1f77b4", alpha=0.8)
ax.set_title(r"PSD check: $z^\top \Sigma z$ across random directions")
ax.set_xlabel(r"$z^\top \Sigma z$")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_psd_hist.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
