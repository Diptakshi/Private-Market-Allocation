"""Takahashi-Alexander cash-flow model (Yale model), quarterly implementation.

Reference: Takahashi, D. & Alexander, S. (2002), "Illiquid Alternative Asset
Fund Modeling", Journal of Portfolio Management 28(2).

Annual recurrences:
    C_t   = RC_t * (CC - PIC_{t-1})                      contributions (capital calls)
    D_t   = RD_t * NAV_{t-1} * (1 + G)                   distributions
    RD_t  = max(Y, (t / L) ** B)
    NAV_t = NAV_{t-1} * (1 + G) + C_t - D_t
Annual rates are converted to quarterly equivalents so a fund produces
quarterly cash flows; RD reaches 100% at the end of the fund's life.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .assumptions import TA_PARAMS


def _q_rate(annual: float) -> float:
    """Convert an annual drawdown/payout fraction into a quarterly one."""
    annual = min(max(annual, 0.0), 1.0)
    return 1.0 if annual >= 1.0 else 1.0 - (1.0 - annual) ** 0.25


def ta_fund(commitment: float, asset_class: str, growth_path=None, n_quarters=None) -> pd.DataFrame:
    """Project quarterly calls, distributions and NAV for one fund.

    growth_path: optional array of quarterly NAV returns (stochastic case).
                 If None, the deterministic growth G of the asset class is used.
    """
    p = TA_PARAMS[asset_class]
    life_q = p["life"] * 4
    n = n_quarters or life_q
    g_q = (1 + p["growth"]) ** 0.25 - 1
    rows, pic, nav = [], 0.0, 0.0
    for q in range(1, n + 1):
        year = (q - 1) // 4 + 1
        rc_a = p["rc"][min(year, len(p["rc"])) - 1]
        call = _q_rate(rc_a) * (commitment - pic) if q <= life_q else 0.0
        r = g_q if growth_path is None else growth_path[q - 1]
        rd_a = max(p["yld"], (q / 4 / p["life"]) ** p["bow"])
        rd_q = 1.0 if q >= life_q else _q_rate(rd_a)
        grown = nav * (1 + r)
        dist = rd_q * grown
        nav = grown + call - dist
        pic += call
        rows.append((q, call, dist, nav))
    return pd.DataFrame(rows, columns=["fund_quarter", "call", "distribution", "nav"])


def unit_profile(asset_class: str) -> pd.DataFrame:
    """Deterministic cash-flow profile of a 1.0 commitment (used for pacing)."""
    return ta_fund(1.0, asset_class)
