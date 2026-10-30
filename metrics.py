"""Private-markets performance metrics from LP cash flows."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def irr(cashflows, periods_per_year: int = 4) -> float:
    """Annualised IRR of periodic cash flows (negative = paid in)."""
    cf = np.asarray(cashflows, dtype=float)
    if not (cf < 0).any() or not (cf > 0).any():
        return np.nan
    t = np.arange(len(cf))
    npv = lambda r: np.sum(cf / (1 + r) ** t)
    try:
        r_q = brentq(npv, -0.99, 10.0)
    except ValueError:
        return np.nan
    return (1 + r_q) ** periods_per_year - 1


def multiples(calls, dists, nav_end: float) -> dict:
    paid_in, distributed = float(np.sum(calls)), float(np.sum(dists))
    if paid_in == 0:
        return dict(TVPI=np.nan, DPI=np.nan, RVPI=np.nan)
    return dict(TVPI=(distributed + nav_end) / paid_in,
                DPI=distributed / paid_in,
                RVPI=nav_end / paid_in)


def ks_pme(calls, dists, nav_end: float, index_levels) -> float:
    """Kaplan-Schoar PME: cash flows compounded with a public index to the end date.
    > 1.0 means the private investment beat the index."""
    idx = np.asarray(index_levels, dtype=float)
    growth_to_end = idx[-1] / idx
    fv_d = np.sum(np.asarray(dists) * growth_to_end) + nav_end
    fv_c = np.sum(np.asarray(calls) * growth_to_end)
    return fv_d / fv_c if fv_c else np.nan


def lp_cashflows(calls, dists, nav_end: float):
    """Net LP cash-flow vector with the residual NAV as terminal value."""
    cf = -np.asarray(calls, dtype=float) + np.asarray(dists, dtype=float)
    cf = cf.copy()
    cf[-1] += nav_end
    return cf
