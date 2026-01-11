"""
Python & AI in Asset Management
Chapter 8 · Risk Decomposition, Risk Parity, and Risk Budgeting

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Case study computations for Chapter 8: risk decomposition.
This script computes portfolio volatility and volatility-based
risk contributions for simple equal-weight, minimum-variance,
and risk-parity-style portfolios in a three-asset universe.
"""

from __future__ import annotations

import numpy as np


def vol_and_rcov(Sigma: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute portfolio volatility and component risk contributions."""
    sigma = float(np.sqrt(w @ Sigma @ w))
    marg = Sigma @ w  # marginal contributions (up to scaling)
    rc = w * marg / sigma
    return sigma, rc


def main() -> None:
    """Run the Chapter 8 case study and print summary numbers."""
    Sigma = np.array(
        [
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16],
        ]
    )
    ones = np.ones(3)

    # Equal-weight portfolio
    w_ew = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    # Minimum-variance portfolio with budget constraint sum w_i = 1
    inv_Sigma = np.linalg.inv(Sigma)
    w_mv_raw = inv_Sigma @ ones
    w_mv = w_mv_raw / w_mv_raw.sum()

    # Simple risk-parity-style portfolio (fixed for exposition)
    w_rp = np.array([0.5, 0.3, 0.2])

    vol_ew, rc_ew = vol_and_rcov(Sigma, w_ew)
    vol_mv, rc_mv = vol_and_rcov(Sigma, w_mv)
    vol_rp, rc_rp = vol_and_rcov(Sigma, w_rp)

    np.set_printoptions(precision=5, suppress=True)

    print("Equal-weight portfolio:")
    print(f"  weights: {w_ew}")
    print(f"  volatility: {vol_ew:.5f}")
    print(f"  risk contributions: {rc_ew}")

    print("\nMinimum-variance portfolio:")
    print(f"  weights: {w_mv}")
    print(f"  volatility: {vol_mv:.5f}")
    print(f"  risk contributions: {rc_mv}")

    print("\nRisk-parity-style portfolio:")
    print(f"  weights: {w_rp}")
    print(f"  volatility: {vol_rp:.5f}")
    print(f"  risk contributions: {rc_rp}")


if __name__ == "__main__":
    main()
