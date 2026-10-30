"""Illustrative modelling assumptions.

All figures are illustrative inputs chosen to be broadly plausible for each
asset class. They are NOT forecasts and are not sourced from any manager's
capital market assumptions. Change them here; every module reads from this file.
"""
import numpy as np

ASSET_CLASSES = ["Private Equity", "Private Debt", "Real Estate", "Infrastructure"]

# Takahashi-Alexander (2002) parameters, annual.
#   rc   : rate of contribution per fund year (applied to unfunded commitment)
#   life : fund life in years (L)
#   bow  : bow factor (B) - higher = later distributions
#   yld  : minimum distribution yield (Y)
#   growth: expected NAV growth rate (G)
TA_PARAMS = {
    "Private Equity": dict(rc=[0.25, 0.333, 0.50], life=12, bow=2.5, yld=0.00, growth=0.12),
    "Private Debt":   dict(rc=[0.40, 0.50, 0.60], life=7,  bow=1.5, yld=0.08, growth=0.075),
    "Real Estate":    dict(rc=[0.30, 0.40, 0.50], life=9,  bow=2.0, yld=0.04, growth=0.08),
    "Infrastructure": dict(rc=[0.25, 0.333, 0.50], life=12, bow=2.0, yld=0.05, growth=0.09),
}

# Annualised "true" (unsmoothed) volatility of NAV growth, and the appraisal
# smoothing coefficient used to generate reported (smoothed) returns.
TRUE_VOL = {"Private Equity": 0.22, "Private Debt": 0.09, "Real Estate": 0.15, "Infrastructure": 0.13}
SMOOTHING = {"Private Equity": 0.55, "Private Debt": 0.45, "Real Estate": 0.70, "Infrastructure": 0.60}

# Listed assets used as benchmark (PME) and in the total-portfolio allocation.
PUBLIC = {
    "Global Equities": dict(mu=0.070, vol=0.16),
    "Global Bonds":    dict(mu=0.035, vol=0.06),
}

ALL_ASSETS = ASSET_CLASSES + list(PUBLIC)

# Correlation of TRUE returns: PE, PD, RE, Infra, Equities, Bonds
CORR = np.array([
    [1.00, 0.55, 0.50, 0.55, 0.80, 0.00],
    [0.55, 1.00, 0.40, 0.45, 0.55, 0.20],
    [0.50, 0.40, 1.00, 0.55, 0.50, 0.10],
    [0.55, 0.45, 0.55, 1.00, 0.55, 0.15],
    [0.80, 0.55, 0.50, 0.55, 1.00, 0.05],
    [0.00, 0.20, 0.10, 0.15, 0.05, 1.00],
])

RISK_FREE = 0.025
