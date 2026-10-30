"""Commitment pacing: annual commitments per asset class that steer private NAV
towards a target allocation of a growing total portfolio.

Because the Takahashi-Alexander model is linear in the commitment, the NAV at
any date is a linear combination of unit profiles. We stack those profiles into
a matrix and solve a non-negative least-squares problem against the target NAV
path, with a small smoothness penalty so commitments do not swing year to year."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .assumptions import ASSET_CLASSES
from .cashflows import unit_profile


def pacing_plan(total_portfolio: float = 2000.0, growth: float = 0.04, private_target: float = 0.25,
                class_weights=None, years: int = 12, ramp_years: int = 5, smooth_penalty: float = 1.0):
    class_weights = class_weights or {"Private Equity": 0.40, "Private Debt": 0.20,
                                      "Real Estate": 0.20, "Infrastructure": 0.20}
    yrs = np.arange(1, years + 1)
    tpv = total_portfolio * (1 + growth) ** yrs
    ramp = np.minimum(yrs / ramp_years, 1.0)
    plans, navs = [], []
    for ac in ASSET_CLASSES:
        target = tpv * private_target * class_weights[ac] * ramp
        prof = unit_profile(ac)
        unit_nav_yearend = prof.nav.values[3::4]          # NAV at each fund year-end
        A = np.zeros((years, years))
        for v in range(years):                          # commitment made at start of year v+1
            k = min(years - v, len(unit_nav_yearend))
            A[v:v + k, v] = unit_nav_yearend[:k]
        D = smooth_penalty * (np.eye(years) - np.eye(years, k=1))[:-1]
        commits, _ = nnls(np.vstack([A, D]), np.r_[target, np.zeros(years - 1)])
        nav = A @ commits
        plans.append(pd.DataFrame({"year": yrs, "asset_class": ac, "commitment": commits,
                                   "projected_nav": nav, "target_nav": target}))
    plan = pd.concat(plans, ignore_index=True)
    plan["total_portfolio"] = np.tile(tpv, len(ASSET_CLASSES))
    return plan
