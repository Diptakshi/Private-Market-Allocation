"""Liquidity simulation of a semi-liquid (evergreen) private-markets fund.

Quarterly Monte Carlo over a two-regime world (normal / stress, Markov switching).
Each quarter:
  1. the private sleeve earns a smoothed (appraisal-based) return and pays
     distributions into the liquid sleeve; the liquid sleeve earns a money-market-like return;
  2. investors subscribe and request redemptions (requests jump in stress);
  3. redemptions are paid up to a gate of `gate` x NAV; unpaid requests queue;
  4. if the liquid sleeve cannot cover the payout, private assets are sold on the
     secondary market at a discount (a "forced sale");
  5. liquidity above target is deployed into private assets (capped per quarter).
The question answered: how large should the liquidity sleeve be, trading return
drag against gate activations and forced sales?"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .assumptions import ALL_ASSETS
from .simulate import correlated_quarterly_returns


def _private_mix_returns(r):
    w = np.array([0.40, 0.20, 0.20, 0.20])               # PE, PD, RE, Infra
    return r[..., :4] @ w


def simulate_evergreen(liquid_target: float, n_paths: int = 4000, quarters: int = 40, seed: int = 1,
                       gate: float = 0.05, sale_discount: float = 0.12, deploy_cap: float = 0.04,
                       p_stress: float = 0.04, p_stay_stress: float = 0.70, smoothing: float = 0.5):
    rng = np.random.default_rng(seed)
    r = correlated_quarterly_returns(quarters, rng, n_paths=n_paths)
    priv_true = _private_mix_returns(r)
    eq = r[..., ALL_ASSETS.index("Global Equities")]
    nav0 = 500.0
    priv = np.full(n_paths, nav0 * (1 - liquid_target)); liq = np.full(n_paths, nav0 * liquid_target)
    queue = np.zeros(n_paths); obs = np.zeros(n_paths); stress = np.zeros(n_paths, bool)
    gated = np.zeros(n_paths, bool); forced = np.zeros(n_paths, bool)
    min_liq_ratio = np.full(n_paths, liquid_target); nav_path = np.empty((n_paths, quarters))
    for t in range(quarters):
        u = rng.random(n_paths)
        stress = np.where(stress, u < p_stay_stress, u < p_stress)
        shock = np.where(stress, np.minimum(eq[:, t], 0.0) * 0.6, 0.0)   # extra private markdown in stress
        obs = (1 - smoothing) * (priv_true[:, t] + shock) + smoothing * obs
        priv *= 1 + obs
        dist = priv * np.where(stress, 0.01, 0.03)
        priv -= dist; liq = liq * (1 + 0.0075) + dist
        nav = priv + liq
        subs = nav * np.where(stress, 0.005, 0.03) * rng.lognormal(0, 0.3, n_paths)
        reqs = nav * np.where(stress, 0.08, 0.02) * rng.lognormal(0, 0.3, n_paths) + queue
        liq += subs
        cap = gate * nav
        paid = np.minimum(reqs, cap)
        gated |= reqs > cap + 1e-9
        queue = reqs - paid
        shortfall = np.maximum(paid - liq, 0.0)
        forced |= shortfall > 1e-9
        liq -= paid - shortfall
        priv -= shortfall / (1 - sale_discount)
        nav = priv + liq
        excess = np.maximum(liq - liquid_target * nav, 0.0)
        deploy = np.minimum(excess, deploy_cap * nav)
        liq -= deploy; priv += deploy
        min_liq_ratio = np.minimum(min_liq_ratio, liq / nav)
        nav_path[:, t] = nav
    # NAV per unit: approximate with flow-adjusted returns via total-return index
    return dict(gated=gated, forced=forced, min_liq_ratio=min_liq_ratio, nav_path=nav_path)


def liquidity_tradeoff(targets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30), **kw) -> pd.DataFrame:
    rows = []
    for lt in targets:
        res = simulate_evergreen(lt, **kw)
        rows.append(dict(liquid_target=lt,
                         prob_gate_10y=res["gated"].mean(),
                         prob_forced_sale_10y=res["forced"].mean(),
                         median_min_liquidity=np.median(res["min_liq_ratio"]),
                         p5_min_liquidity=np.percentile(res["min_liq_ratio"], 5)))
    df = pd.DataFrame(rows)
    df["expected_return_drag"] = expected_drag(df.liquid_target)
    return df


def expected_drag(liquid_target):
    """Annual return given up by holding the liquid sleeve instead of the private mix."""
    from .assumptions import TA_PARAMS
    w = {"Private Equity": .4, "Private Debt": .2, "Real Estate": .2, "Infrastructure": .2}
    private_mu = sum(w[a] * TA_PARAMS[a]["growth"] for a in w)
    return np.asarray(liquid_target) * (private_mu - 0.03)
