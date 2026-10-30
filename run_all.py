"""Run every model, export CSVs (Power BI-ready) and charts.

    python run_all.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmalloc.assumptions import ALL_ASSETS, ASSET_CLASSES
from pmalloc.cashflows import unit_profile
from pmalloc.portfolio import simulate_programme, fund_metrics, ASOF
from pmalloc.pacing import pacing_plan
from pmalloc.evergreen import liquidity_tradeoff
from pmalloc.allocation import run_allocation, annual_cov
from pmalloc.metrics import irr, lp_cashflows, multiples, ks_pme

OUT = Path("outputs"); DATA = OUT / "data"; CH = OUT / "charts"
DATA.mkdir(parents=True, exist_ok=True); CH.mkdir(parents=True, exist_ok=True)

COLORS = dict(zip(ALL_ASSETS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK,
                     "legend.frameon": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=160); plt.close(fig)


# 1. J-curves ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
jc = []
for ac in ASSET_CLASSES:
    p = unit_profile(ac)
    net = (p.distribution - p.call).cumsum()
    ax.plot(p.fund_quarter / 4, net, lw=2, color=COLORS[ac], label=ac)
    jc.append(pd.DataFrame({"asset_class": ac, "fund_year": p.fund_quarter / 4,
                            "cum_net_cashflow": net, "nav": p.nav}))
ax.axhline(0, color="#c3c2b7", lw=1)
ax.set(title="J-curve: cumulative net cash flow per 1.0 committed", xlabel="Fund year", ylabel="Multiple of commitment")
ax.legend(ncol=4, loc="upper left", fontsize=8)
save(fig, "01_j_curves.png")
pd.concat(jc).to_csv(DATA / "j_curves.csv", index=False)

# 2. Existing programme -----------------------------------------------------
dim, cfs, bench = simulate_programme()
metrics = fund_metrics(dim, cfs, bench)
dim.to_csv(DATA / "dim_fund.csv", index=False)
cfs.assign(net_cashflow=cfs.distribution - cfs.call).to_csv(DATA / "fact_cashflows.csv", index=False)
bench.to_csv(DATA / "fact_benchmark.csv", index=False)
metrics.to_csv(DATA / "fund_metrics.csv", index=False)

agg = cfs.groupby("date")[["call", "distribution", "nav"]].sum()
cf_all = (agg.distribution - agg.call).values.copy()
prog = multiples(agg.call, agg.distribution, agg.nav.iloc[-1])
prog_irr = irr(lp_cashflows(agg.call, agg.distribution, agg.nav.iloc[-1]))
prog_pme = ks_pme(agg.call, agg.distribution, agg.nav.iloc[-1], bench.index_level.values)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
a1.bar(agg.index, -agg.call, width=60, color=COLORS["Private Equity"], label="Capital calls")
a1.bar(agg.index, agg.distribution, width=60, color=COLORS["Real Estate"], label="Distributions")
a1.axhline(0, color="#c3c2b7", lw=1); a1.set(title="Programme cash flows (EUR m per quarter)"); a1.legend(ncol=2, loc="upper left")
navc = cfs.merge(dim, on="fund_id").pivot_table(index="date", columns="asset_class", values="nav", aggfunc="sum").fillna(0)[ASSET_CLASSES]
a2.stackplot(navc.index, navc.T.values, colors=[COLORS[a] for a in ASSET_CLASSES], labels=ASSET_CLASSES,
             edgecolor="#fcfcfb", linewidth=1)
a2.set(title="Net asset value by asset class (EUR m)"); a2.legend(ncol=4, loc="upper left", fontsize=8)
save(fig, "02_programme.png")

# 3. Pacing -----------------------------------------------------------------
plan = pacing_plan()
plan.to_csv(DATA / "pacing_plan.csv", index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
pv = plan.pivot(index="year", columns="asset_class", values="commitment")[ASSET_CLASSES]
bottom = np.zeros(len(pv))
for ac in ASSET_CLASSES:
    a1.bar(pv.index, pv[ac], bottom=bottom, color=COLORS[ac], label=ac, edgecolor="#fcfcfb", linewidth=1.5)
    bottom += pv[ac].values
a1.set(title="Annual commitments by asset class (EUR m)", xlabel="Year"); a1.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
tot = plan.groupby("year")[["projected_nav", "target_nav"]].sum()
a2.plot(tot.index, tot.target_nav, color=MUTED, lw=2, ls="--", label="Target private NAV")
a2.plot(tot.index, tot.projected_nav, color=COLORS["Private Equity"], lw=2, marker="o", ms=4, label="Projected private NAV")
a2.set(title="Private NAV vs target (25% of a growing EUR 2bn portfolio)", xlabel="Year"); a2.legend(loc="upper left")
save(fig, "03_pacing.png")

# 4. Evergreen liquidity ----------------------------------------------------
eg = liquidity_tradeoff()
eg.to_csv(DATA / "evergreen_tradeoff.csv", index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
x = eg.liquid_target * 100
a1.plot(x, eg.prob_forced_sale_10y * 100, color=COLORS["Private Debt"], lw=2, marker="o", ms=5)
for xi, yi in zip(x, eg.prob_forced_sale_10y * 100):
    a1.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK2)
a1.set(title="Probability of a forced secondary sale within 10 years", xlabel="Liquidity sleeve (% of NAV)", ylabel="%")
a2.plot(x, eg.expected_return_drag * 1e4, color=COLORS["Private Equity"], lw=2, marker="o", ms=5)
a2.set(title="Expected return drag of the liquidity sleeve", xlabel="Liquidity sleeve (% of NAV)", ylabel="bp per year")
save(fig, "04_evergreen_liquidity.png")

# 5-6. Desmoothing & allocation ---------------------------------------------
panel, vols, weights, res = run_allocation()
vols.to_csv(DATA / "volatility_comparison.csv"); weights.to_csv(DATA / "allocation_weights.csv")
ct = annual_cov(panel, "true")
summary = pd.DataFrame({k: dict(expected_return=v["ret"], perceived_vol=v["vol"],
                                true_vol=np.sqrt(v["weights"] @ ct @ v["weights"])) for k, v in res.items()}).T
summary.to_csv(DATA / "allocation_summary.csv")

fig, ax = plt.subplots(figsize=(7, 3.6))
pv_ = vols.loc[ASSET_CLASSES]; xi = np.arange(len(pv_)); w = 0.26
for i, (k, c) in enumerate([("reported", "#86b6ef"), ("desmoothed", COLORS["Private Equity"]), ("true", "#184f95")]):
    ax.bar(xi + (i - 1) * w, pv_[k] * 100, w * 0.92, color=c, label=k.capitalize())
ax.set_xticks(xi, ASSET_CLASSES); ax.set(title="Annualised volatility: reported vs Geltner-desmoothed vs true", ylabel="%")
ax.legend(ncol=3)
save(fig, "05_desmoothing.png")

fig, ax = plt.subplots(figsize=(7, 3.6))
xi = np.arange(len(ALL_ASSETS))
ax.bar(xi - 0.2, weights["reported"] * 100, 0.38, color="#86b6ef", label="Optimised on reported returns")
ax.bar(xi + 0.2, weights["desmoothed"] * 100, 0.38, color=COLORS["Private Equity"], label="Optimised on desmoothed returns")
ax.set_xticks(xi, [a.replace(" ", "\n") for a in ALL_ASSETS]); ax.set(title="Max-Sharpe allocation (private markets capped at 35%)", ylabel="Weight (%)")
ax.legend()
save(fig, "06_allocation.png")

# 7. One-page investment committee summary ----------------------------------
fig = plt.figure(figsize=(11.7, 8.3))
fig.suptitle("Private Markets Programme - Investment Committee Summary (illustrative, simulated data)",
             x=0.03, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.text(0.03, 0.925, f"As of {pd.Timestamp(ASOF):%d %b %Y}  |  13 funds  |  4 asset classes  |  figures in EUR m", color=INK2)
kpis = [("Committed", f"{dim.commitment_eur_m.sum():,.0f}"), ("Paid-in", f"{agg.call.sum():,.0f}"),
        ("Distributed", f"{agg.distribution.sum():,.0f}"), ("NAV", f"{agg.nav.iloc[-1]:,.0f}"),
        ("TVPI", f"{prog['TVPI']:.2f}x"), ("DPI", f"{prog['DPI']:.2f}x"), ("Net IRR", f"{prog_irr:.1%}"),
        ("KS-PME", f"{prog_pme:.2f}")]
for i, (k, v) in enumerate(kpis):
    fig.text(0.03 + i * 0.12, 0.865, k, color=INK2, fontsize=9)
    fig.text(0.03 + i * 0.12, 0.83, v, color=INK, fontsize=16, fontweight="bold")
gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.78, bottom=0.07, hspace=0.45, wspace=0.22)
ax = fig.add_subplot(gs[0, 0])
ax.stackplot(navc.index, navc.T.values, colors=[COLORS[a] for a in ASSET_CLASSES], labels=ASSET_CLASSES, edgecolor="#fcfcfb", linewidth=1)
ax.set(title="NAV by asset class"); ax.legend(ncol=2, fontsize=7, loc="upper left")
ax = fig.add_subplot(gs[0, 1])
m = metrics.sort_values("TVPI")
ax.barh(m.fund_name, m.TVPI, color=[COLORS[a] for a in m.asset_class], height=0.7)
ax.axvline(1, color="#c3c2b7", lw=1); ax.set(title="TVPI by fund"); ax.tick_params(axis="y", labelsize=7)
ax = fig.add_subplot(gs[1, 0])
ax.plot(tot.index, tot.target_nav, color=MUTED, lw=2, ls="--", label="Target")
ax.plot(tot.index, tot.projected_nav, color=COLORS["Private Equity"], lw=2, marker="o", ms=3, label="Projected")
ax.set(title="Pacing plan: private NAV vs target", xlabel="Year"); ax.legend()
ax = fig.add_subplot(gs[1, 1])
ax.plot(eg.liquid_target * 100, eg.prob_forced_sale_10y * 100, color=COLORS["Private Debt"], lw=2, marker="o", ms=4)
ax.set(title="Evergreen fund: forced-sale probability (10y) vs liquidity sleeve", xlabel="Liquidity sleeve (% NAV)", ylabel="%")
fig.savefig(CH / "00_ic_summary.png", dpi=150); fig.savefig(OUT / "ic_summary.pdf"); plt.close(fig)

# Console summary -----------------------------------------------------------
pd.set_option("display.width", 140)
print("Programme:", {k: round(v, 3) for k, v in prog.items()}, "IRR", round(prog_irr, 4), "PME", round(prog_pme, 3))
print(metrics[["fund_name", "asset_class", "vintage", "TVPI", "DPI", "irr", "pme"]].round(2).to_string(index=False))
print(eg.round(3).to_string(index=False))
print(vols.round(3)); print(weights.round(3)); print(summary.round(4))
