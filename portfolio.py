"""Stochastic simulation of an existing fund-of-funds programme (illustrative)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .assumptions import ALL_ASSETS
from .cashflows import ta_fund
from .metrics import irr, ks_pme, lp_cashflows, multiples
from .simulate import correlated_quarterly_returns

START, ASOF = "2017-03-31", "2026-06-30"

# Fictional programme: (fund name, asset class, vintage, commitment EUR m)
FUNDS = [
    ("Buyout Fund A", "Private Equity", 2017, 60), ("Growth Fund B", "Private Equity", 2019, 50),
    ("Buyout Fund C", "Private Equity", 2021, 70), ("Mid-Market Fund D", "Private Equity", 2023, 60),
    ("Direct Lending Fund E", "Private Debt", 2018, 50), ("Senior Debt Fund F", "Private Debt", 2020, 45),
    ("Unitranche Fund G", "Private Debt", 2022, 55), ("Value-Add RE Fund H", "Real Estate", 2017, 40),
    ("Core-Plus RE Fund I", "Real Estate", 2020, 40), ("Logistics RE Fund J", "Real Estate", 2023, 35),
    ("Core Infra Fund K", "Infrastructure", 2018, 50), ("Energy Transition Fund L", "Infrastructure", 2021, 55),
    ("Digital Infra Fund M", "Infrastructure", 2024, 45),
]


def simulate_programme(seed: int = 42, idio_vol: float = 0.10):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, ASOF, freq="QE")
    n = len(dates)
    mkt = correlated_quarterly_returns(n, rng)
    eq_idx = 100 * np.cumprod(1 + mkt[:, ALL_ASSETS.index("Global Equities")])
    benchmark = pd.DataFrame({"date": dates, "index_level": eq_idx})

    rows, dims = [], []
    for k, (name, ac, vintage, commit) in enumerate(FUNDS):
        fid = f"F{k + 1:02d}"
        start = dates.get_loc(pd.Timestamp(f"{vintage}-03-31"))
        span = n - start
        path = mkt[start:, ALL_ASSETS.index(ac)] + rng.normal(0, idio_vol / 2, span)
        cf = ta_fund(commit, ac, growth_path=path, n_quarters=span)
        cf.insert(0, "date", dates[start:])
        cf.insert(0, "fund_id", fid)
        rows.append(cf)
        dims.append(dict(fund_id=fid, fund_name=name, asset_class=ac, vintage=vintage, commitment_eur_m=commit))
    cfs = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(dims), cfs, benchmark


def fund_metrics(dim: pd.DataFrame, cfs: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    idx = bench.set_index("date")["index_level"]
    out = []
    for _, f in dim.iterrows():
        c = cfs[cfs.fund_id == f.fund_id]
        nav_end = c.nav.iloc[-1]
        m = multiples(c.call, c.distribution, nav_end)
        out.append(dict(fund_id=f.fund_id,
                        paid_in=c.call.sum(), distributed=c.distribution.sum(), nav=nav_end,
                        unfunded=f.commitment_eur_m - c.call.sum(),
                        irr=irr(lp_cashflows(c.call, c.distribution, nav_end)),
                        pme=ks_pme(c.call, c.distribution, nav_end, idx.loc[c.date].values), **m))
    return dim.merge(pd.DataFrame(out), on="fund_id")
