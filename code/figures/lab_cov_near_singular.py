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

sigma1 = 0.20
sigma2 = 0.15
rho_target = 0.999
c_near = rho_target * sigma1 * sigma2

sigma = np.array([[sigma1**2, c_near], [c_near, sigma2**2]])

vals, vecs = np.linalg.eigh(sigma)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

sqrt_sigma = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

theta = np.linspace(0.0, 2.0 * np.pi, 400)
circle = np.column_stack([np.cos(theta), np.sin(theta)])
ellipse = circle @ sqrt_sigma.T

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ellipse[:, 0], ellipse[:, 1], linewidth=1.6, color="#1f77b4")
ax.axhline(0.0, linewidth=0.8, color="black")
ax.axvline(0.0, linewidth=0.8, color="black")
ax.set_title("Near-singular covariance ellipse")
ax.set_xlabel("Axis 1")
ax.set_ylabel("Axis 2")
ax.grid(True, alpha=0.35)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_near_singular.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
