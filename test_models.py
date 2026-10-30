import numpy as np
from pmalloc.assumptions import ASSET_CLASSES
from pmalloc.cashflows import unit_profile
from pmalloc.metrics import irr, multiples, ks_pme
from pmalloc.simulate import smooth, geltner_desmooth


def test_irr_known_value():
    # -100 today, +110 in four quarters -> 10% annual
    assert abs(irr([-100, 0, 0, 0, 110]) - 0.10) < 1e-6


def test_fund_calls_full_commitment_and_winds_down():
    for ac in ASSET_CLASSES:
        p = unit_profile(ac)
        assert abs(p.call.sum() - 1.0) < 0.01
        assert p.nav.iloc[-1] < 1e-3


def test_multiples():
    m = multiples([100], [60], 70)
    assert np.isclose(m["TVPI"], 1.3) and np.isclose(m["DPI"], 0.6)


def test_pme_equals_one_when_tracking_index():
    idx = np.array([100, 110, 121])
    # invest 100 at t0, nothing else, NAV tracks index -> PME = 1
    assert np.isclose(ks_pme([100, 0, 0], [0, 0, 0], 121, idx), 1.0)


def test_desmoothing_restores_volatility():
    rng = np.random.default_rng(0)
    true = rng.normal(0.02, 0.05, 2000)
    obs = smooth(true, 0.6)
    des, rho = geltner_desmooth(obs)
    assert abs(rho - 0.6) < 0.05
    assert obs.std() < 0.6 * true.std()
    assert abs(des.std() / true.std() - 1) < 0.1
