"""Correlated return generation (Cholesky) and appraisal smoothing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .assumptions import ALL_ASSETS, ASSET_CLASSES, CORR, PUBLIC, SMOOTHING, TA_PARAMS, TRUE_VOL


def annual_mu_vol():
    mu, vol = [], []
    for a in ALL_ASSETS:
        if a in TA_PARAMS:
            mu.append(TA_PARAMS[a]["growth"]); vol.append(TRUE_VOL[a])
        else:
            mu.append(PUBLIC[a]["mu"]); vol.append(PUBLIC[a]["vol"])
    return np.array(mu), np.array(vol)


def correlated_quarterly_returns(n_quarters: int, rng: np.random.Generator, n_paths: int | None = None):
    """Quarterly log-normal-ish returns for all assets, correlated through a
    Cholesky factor of CORR. Returns array (n_quarters, n_assets) or
    (n_paths, n_quarters, n_assets)."""
    mu, vol = annual_mu_vol()
    L = np.linalg.cholesky(CORR)
    shape = (n_quarters, len(ALL_ASSETS)) if n_paths is None else (n_paths, n_quarters, len(ALL_ASSETS))
    z = rng.standard_normal(shape) @ L.T
    q_vol = vol / 2.0
    q_mu = np.log1p(mu) / 4 - 0.5 * q_vol ** 2
    return np.expm1(q_mu + q_vol * z)


def smooth(true_returns: np.ndarray, alpha: float) -> np.ndarray:
    """Appraisal smoothing: r_obs_t = (1 - a) * r_true_t + a * r_obs_{t-1}."""
    out = np.empty_like(true_returns)
    prev = true_returns[0]
    for t, r in enumerate(true_returns):
        prev = (1 - alpha) * r + alpha * prev
        out[t] = prev
    return out


def geltner_desmooth(observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Geltner (1993) desmoothing with the AR(1) coefficient estimated from the data:
    r_true_t = (r_obs_t - rho * r_obs_{t-1}) / (1 - rho)."""
    x = np.asarray(observed, dtype=float)
    xc = x - x.mean()
    rho = float(np.dot(xc[1:], xc[:-1]) / np.dot(xc[:-1], xc[:-1]))
    rho = min(max(rho, 0.0), 0.95)
    des = (x[1:] - rho * x[:-1]) / (1 - rho)
    return des, rho


def return_panel(n_quarters: int, seed: int = 7) -> pd.DataFrame:
    """Panel of true, reported (smoothed) and desmoothed quarterly returns."""
    rng = np.random.default_rng(seed)
    true = correlated_quarterly_returns(n_quarters, rng)
    frames = {}
    for i, a in enumerate(ALL_ASSETS):
        frames[(a, "true")] = true[:, i]
        if a in ASSET_CLASSES:
            rep = smooth(true[:, i], SMOOTHING[a])
            des, _ = geltner_desmooth(rep)
            frames[(a, "reported")] = rep
            frames[(a, "desmoothed")] = np.r_[np.nan, des]
        else:
            frames[(a, "reported")] = true[:, i]
            frames[(a, "desmoothed")] = true[:, i]
    return pd.DataFrame(frames)
