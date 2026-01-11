"""
Python & AI in Asset Management
Chapter 7 · Coherent and Convex Risk Measures for Portfolios

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Case study computations for Chapter 7: risk measures.
This script simulates daily portfolio returns from a small
multivariate normal model and computes sample estimates of
volatility, Value-at-Risk, and Expected Shortfall.
"""

from __future__ import annotations

import numpy as np


def simulate_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w: np.ndarray,
    n_sims: int = 50_000,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate asset and portfolio returns from a multivariate normal model."""
    rng = np.random.default_rng(seed=seed)
    rets = rng.multivariate_normal(mu, Sigma, size=n_sims)
    port_ret = rets @ w
    port_loss = -port_ret
    return port_ret, port_loss


def risk_measures_from_losses(
    port_ret: np.ndarray,
    port_loss: np.ndarray,
    alpha: float = 0.99,
) -> tuple[float, float, float]:
    """Compute volatility, VaR, and ES from simulated losses."""
    vol = float(port_ret.std(ddof=1))
    var_alpha = float(np.quantile(port_loss, alpha))
    tail_losses = port_loss[port_loss >= var_alpha]
    es_alpha = float(tail_losses.mean())
    return vol, var_alpha, es_alpha


def main() -> None:
    """Run the Chapter 7 case study and print summary numbers."""
    mu = np.array([0.0004, 0.0003, 0.0005])
    # daily covariance matrix for three assets
    Sigma = np.array(
        [
            [0.0004, 0.0002, 0.0001],
            [0.0002, 0.0003, 0.00015],
            [0.0001, 0.00015, 0.0005],
        ]
    )
    w = np.array([0.4, 0.3, 0.3])

    port_ret, port_loss = simulate_portfolio(mu, Sigma, w)
    vol, var_99, es_99 = risk_measures_from_losses(port_ret, port_loss, alpha=0.99)

    print(f"Daily volatility estimate:      {vol:.5f}")
    print(f"99% one-day VaR estimate:       {var_99:.5f}")
    print(f"99% one-day Expected Shortfall: {es_99:.5f}")


if __name__ == "__main__":
    main()
