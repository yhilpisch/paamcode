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

rf = 1.02
w_market = np.array([0.5, 0.5])
R_market = x @ w_market
mu_market = (p * R_market).sum()
var_market = (p * (R_market - mu_market) ** 2).sum()

rng = np.random.default_rng(123)
W = rng.normal(size=(300, 2))
W = W / W.sum(axis=1, keepdims=True)

mu_port = W @ mu
var_port = np.einsum("ni,ij,nj->n", W, sigma, W)
sigma_port = np.sqrt(var_port)

asset_sigma = np.sqrt(np.diag(sigma))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sigma_port, mu_port, alpha=0.5, s=18, label="Portfolios")
ax.scatter(asset_sigma, mu, color="#ffbf00", s=45, label="Assets")
ax.scatter(np.sqrt(var_market), mu_market, color="#d62728",
           s=45, label="Market")

ax.set_xlabel("Standard deviation")
ax.set_ylabel("Expected gross return")
ax.set_title("Mean-variance cloud")
ax.grid(True, alpha=0.35)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_mean_variance.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
