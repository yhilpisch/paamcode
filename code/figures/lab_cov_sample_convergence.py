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

A = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))

rng = np.random.default_rng(0)

ns = np.array([10, 20, 50, 100, 200, 500, 1000, 3000, 10000])
errs = []

for n in ns:
    z = rng.normal(size=(int(n), 2))
    x_sim = z @ A.T
    s = np.cov(x_sim.T, bias=False)
    errs.append(np.linalg.norm(s - sigma))

errs = np.array(errs)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ns, errs, marker="o", color="#1f77b4")
ax.set_xscale("log")
ax.set_title("Sample covariance convergence")
ax.set_xlabel("Sample size")
ax.set_ylabel(r"$\|S_n - \Sigma\|$")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_sample_convergence.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
