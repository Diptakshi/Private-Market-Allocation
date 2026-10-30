"""Total-portfolio allocation with and without desmoothing private-asset returns.

Appraisal-based private-asset returns look far less volatile than they are.
Feeding them straight into a mean-variance optimiser over-allocates to private
markets. We compare the optimal portfolio built from reported returns with the
one built from Geltner-desmoothed returns, under an illiquidity cap."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .assumptions import ALL_ASSETS, ASSET_CLASSES, RISK_FREE
from .simulate import annual_mu_vol, return_panel


def annual_cov(panel: pd.DataFrame, kind: str) -> np.ndarray:
    data = pd.DataFrame({a: panel[(a, kind)] for a in ALL_ASSETS}).dropna()
    return data.cov().values * 4


def max_sharpe(mu, cov, max_private=0.35, max_single=0.60):
    n = len(mu)
    priv = np.array([a in ASSET_CLASSES for a in ALL_ASSETS], float)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: max_private - w @ priv}]
    neg_sr = lambda w: -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w)
    res = minimize(neg_sr, np.full(n, 1 / n), bounds=[(0, max_single)] * n, constraints=cons, method="SLSQP")
    return res.x


def frontier(mu, cov, n_points=25, max_private=0.35, max_single=0.60):
    n = len(mu); priv = np.array([a in ASSET_CLASSES for a in ALL_ASSETS], float)
    pts = []
    for tgt in np.linspace(mu.min(), mu.max() * 0.98, n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t},
                {"type": "ineq", "fun": lambda w: max_private - w @ priv}]
        res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), bounds=[(0, max_single)] * n,
                       constraints=cons, method="SLSQP")
        if res.success:
            pts.append((np.sqrt(res.x @ cov @ res.x), tgt))
    return pd.DataFrame(pts, columns=["vol", "ret"])


def run_allocation(n_quarters: int = 120, seed: int = 7):
    panel = return_panel(n_quarters, seed)
    mu, _ = annual_mu_vol()
    vols = pd.DataFrame({k: [panel[(a, k)].std() * 2 for a in ALL_ASSETS]
                         for k in ["reported", "desmoothed", "true"]}, index=ALL_ASSETS)
    results = {}
    for kind in ["reported", "desmoothed"]:
        cov = annual_cov(panel, kind)
        w = max_sharpe(mu, cov)
        results[kind] = dict(weights=w, ret=w @ mu, vol=np.sqrt(w @ cov @ w), frontier=frontier(mu, cov))
    weights = pd.DataFrame({k: v["weights"] for k, v in results.items()}, index=ALL_ASSETS)
    return panel, vols, weights, results
