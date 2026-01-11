"""
Python & AI in Asset Management
Chapter 9 · Active Portfolio Risk Management Beyond Diversification

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh

Case study computations for Chapter 9: active risk management overlays.
This script simulates a simple daily return series, applies a volatility-
targeting rule and a drawdown-aware overlay, and reports annualized
performance and risk statistics for the baseline and overlay variants.
"""

from __future__ import annotations

import numpy as np


def simulate_baseline(n_days: int = 2500, seed: int = 9) -> np.ndarray:
    """Simulate a simple baseline daily return series."""
    rng = np.random.default_rng(seed=seed)
    # Centered around a small positive mean with modest volatility
    mu = 0.0004  # about 10% annualized
    sigma = 0.0125  # about 20% annualized
    ret = rng.normal(loc=mu, scale=sigma, size=n_days)
    return ret


def rolling_vol(ret: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple rolling volatility estimate."""
    vol = np.full_like(ret, fill_value=np.nan, dtype=float)
    for t in range(window, len(ret)):
        window_slice = ret[t - window : t]
        vol[t] = window_slice.std(ddof=1)
    return vol


def apply_vol_target(
    ret: np.ndarray,
    vol_target: float = 0.12,
    window: int = 20,
    lam_min: float = 0.0,
    lam_max: float = 2.0,
) -> np.ndarray:
    """Apply a simple volatility-targeting rule."""
    sigma_hat = rolling_vol(ret, window=window)
    lam = np.empty_like(ret)
    lam[:] = 1.0

    mask = ~np.isnan(sigma_hat) & (sigma_hat > 0.0)
    lam[mask] = vol_target / (sigma_hat[mask] * np.sqrt(252.0))
    lam = np.clip(lam, lam_min, lam_max)

    return lam * ret


def equity_curve(ret: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Compute an equity curve from returns."""
    return start * np.cumprod(1.0 + ret)


def max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from an equity curve."""
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - equity / running_max
    return float(dd.max())


def apply_drawdown_overlay(
    ret: np.ndarray,
    dd_threshold: float = 0.10,
    scale_in_drawdown: float = 0.5,
) -> np.ndarray:
    """Apply a simple drawdown-aware overlay on top of given returns."""
    eq = equity_curve(ret)
    running_max = np.maximum.accumulate(eq)
    dd = 1.0 - eq / running_max

    kappa = np.ones_like(ret)
    kappa[dd > dd_threshold] = scale_in_drawdown

    return kappa * ret


def summary_stats(ret: np.ndarray) -> dict[str, float]:
    """Compute annualized return, volatility, and max drawdown."""
    ann_factor = 252.0
    eq = equity_curve(ret)
    ann_ret = float(eq[-1] ** (ann_factor / len(ret)) - 1.0)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(ann_factor))
    md = max_drawdown(eq)
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "max_dd": md}


def main() -> None:
    """Run the Chapter 9 case study and print summary statistics."""
    ret = simulate_baseline()

    # Volatility targeting and combined overlay
    ret_vol = apply_vol_target(ret, vol_target=0.12, window=20)
    ret_comb = apply_drawdown_overlay(
        ret_vol, dd_threshold=0.10, scale_in_drawdown=0.5
    )

    base_stats = summary_stats(ret)
    vol_stats = summary_stats(ret_vol)
    comb_stats = summary_stats(ret_comb)

    print("Baseline strategy:")
    print(
        f"  ann_ret={base_stats['ann_ret']:.4f}, "
        f"ann_vol={base_stats['ann_vol']:.4f}, "
        f"max_dd={base_stats['max_dd']:.4f}"
    )

    print("\nVolatility-targeted strategy:")
    print(
        f"  ann_ret={vol_stats['ann_ret']:.4f}, "
        f"ann_vol={vol_stats['ann_vol']:.4f}, "
        f"max_dd={vol_stats['max_dd']:.4f}"
    )

    print("\nVol-targeted + drawdown overlay:")
    print(
        f"  ann_ret={comb_stats['ann_ret']:.4f}, "
        f"ann_vol={comb_stats['ann_vol']:.4f}, "
        f"max_dd={comb_stats['max_dd']:.4f}"
    )


if __name__ == "__main__":
    main()
