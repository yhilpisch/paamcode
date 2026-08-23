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

w_market = np.array([0.5, 0.5])
R_market = x @ w_market
mu_market = (p * R_market).sum()
var_market = (p * (R_market - mu_market) ** 2).sum()


def factor_reg(R):
    mu_R = (p * R).sum()
    cov_RM = (p * (R - mu_R) * (R_market - mu_market)).sum()
    beta = cov_RM / var_market
    alpha = mu_R - beta * mu_market
    return alpha, beta


alpha1, beta1 = factor_reg(x1)
alpha2, beta2 = factor_reg(x2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

for ax, R, alpha, beta, title in [
    (axes[0], x1, alpha1, beta1, "Asset 1"),
    (axes[1], x2, alpha2, beta2, "Asset 2"),
]:
    ax.scatter(R_market, R, s=45, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Market return")
    ax.grid(True, alpha=0.35)

    Rm_line = np.linspace(R_market.min(), R_market.max(), 60)
    R_line = alpha + beta * Rm_line
    ax.plot(Rm_line, R_line, color="#ff7f0e", linewidth=1.6)

axes[0].set_ylabel("Asset return")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cov_factor_regression.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
