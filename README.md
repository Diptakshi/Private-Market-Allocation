# Private Markets Allocation & Liquidity Toolkit

A Python toolkit for the core quantitative questions in a **multi-asset private markets** portfolio
(private equity, private debt, real estate, infrastructure):

1. **How will a fund's cash flows and NAV behave?** Takahashi–Alexander (Yale) cash-flow model
2. **How is an existing programme performing?** IRR, TVPI, DPI, RVPI and Kaplan–Schoar PME
3. **How much should we commit each year?** Commitment pacing towards a target allocation
4. **How much liquidity does a semi-liquid / evergreen fund need?** Monte Carlo redemption stress test
5. **How much private markets should a portfolio hold?** Mean–variance allocation on Geltner-desmoothed returns

![IC summary](outputs/charts/00_ic_summary.png)

> **Data notice:** all fund cash flows, returns and the benchmark index are **simulated** from the
> illustrative assumptions in `pmalloc/assumptions.py`. Fund names are fictional. No numbers here
> are forecasts or real fund data. The point of the project is the methodology.

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py      # runs every model, writes CSVs to outputs/data and charts to outputs/charts
pytest -q              # unit tests
```

## Project structure

```
pmalloc/
  assumptions.py   # every input in one place: TA parameters, volatilities, correlations, smoothing
  cashflows.py     # Takahashi–Alexander model (quarterly)
  metrics.py       # IRR, TVPI / DPI / RVPI, Kaplan–Schoar PME
  simulate.py      # correlated returns (Cholesky), appraisal smoothing, Geltner desmoothing
  portfolio.py     # stochastic simulation of a 13-fund programme, fund-level metrics
  pacing.py        # commitment pacing (non-negative least squares)
  evergreen.py     # liquidity Monte Carlo for a semi-liquid fund
  allocation.py    # max-Sharpe allocation: reported vs desmoothed covariance
run_all.py         # end-to-end run, exports and charts
tests/             # pytest unit tests
outputs/data/      # CSVs, structured as a star schema for Power BI
outputs/charts/    # PNG charts + one-page IC summary
```

---

## 1. Cash-flow model (Takahashi–Alexander)

For a commitment `CC`, with paid-in capital `PIC`, growth `G`, yield `Y`, bow `B` and life `L`:

| | Formula |
|---|---|
| Capital calls | `C_t = RC_t × (CC − PIC_{t−1})` |
| Distributions | `D_t = RD_t × NAV_{t−1} × (1 + G)`, with `RD_t = max(Y, (t/L)^B)` |
| NAV | `NAV_t = NAV_{t−1} × (1 + G) + C_t − D_t` |

Annual rates are converted into quarterly ones. Each asset class has its own parameters. Private
debt draws fast and distributes early (short J-curve). Private equity distributes late (deep J-curve).
In the stochastic version, `G` is replaced by a simulated quarterly return path.

![J-curves](outputs/charts/01_j_curves.png)

## 2. Programme monitoring

The simulation covers 13 fictional funds with vintages from 2017 to 2024. Each fund follows the TA
model, driven by asset-class returns that are correlated through a **Cholesky** factor, plus
fund-specific noise. Metrics:

- **IRR** of the LP cash flows, with residual NAV as terminal value (Brent root-finding, annualised)
- **TVPI / DPI / RVPI**: total, realised and unrealised value multiples
- **Kaplan–Schoar PME**: cash flows compounded with a listed equity index. Above 1.0 means the
  private investment beat the public market.

![Programme](outputs/charts/02_programme.png)

## 3. Commitment pacing

The goal is to bring private markets to **25% of a EUR 2bn portfolio growing 4% a year**, ramped in
over 5 years, split 40/20/20/20 across asset classes. The TA model is linear in commitments, so
NAV = `A × commitments`, where each column of `A` is a vintage's unit NAV profile. Solving a
**non-negative least-squares** problem (with a smoothness penalty) gives the yearly commitment plan.
The chart shows the classic pattern: commitments are front-loaded to build NAV, ease off while early
vintages are fully invested, then rise again as distributions start returning capital.

![Pacing](outputs/charts/03_pacing.png)

## 4. Evergreen / semi-liquid fund liquidity

A quarterly Monte Carlo (4,000 paths × 10 years) runs in a two-regime world (normal / stress,
Markov switching). Subscriptions, redemption requests, distributions and appraisal-smoothed private
returns are simulated together. Redemptions are capped by a **5% quarterly gate**. If the liquidity
sleeve cannot pay, private assets are sold on the secondary market at a 12% discount.

**Finding:** the gate is triggered whenever stress-period redemption requests exceed 5% of NAV, so its
probability depends on investor behaviour, not on the size of the sleeve. What the sleeve controls is
the **probability of a forced secondary sale**. That falls from ~65% with a 5% sleeve to ~11% with a
25% sleeve, at a cost of about 170 bp a year in return drag. This is the trade-off an investment
committee has to choose.

![Evergreen](outputs/charts/04_evergreen_liquidity.png)

## 5. Allocation: why desmoothing matters

Reported private-asset returns are based on appraisals, so they are smoothed. They understate
volatility and correlation. Here, "true" returns are generated and then smoothed
(`r_obs_t = (1−α) r_true_t + α r_obs_{t−1}`), and the **Geltner** method recovers them by estimating
the AR(1) coefficient from the data.

![Desmoothing](outputs/charts/05_desmoothing.png)

A max-Sharpe optimiser, with private markets capped at 35%, gives very different portfolios depending
on the covariance input:

| Input covariance | Expected return | Volatility the optimiser believes | Volatility under true covariance |
|---|---|---|---|
| Reported (smoothed) | 6.4% | 5.4% | **8.1%** |
| Desmoothed | 6.0% | 6.8% | **7.1%** |

With reported returns, the optimiser concentrates in private equity and understates risk by about a
third. With desmoothed returns, it spreads the allocation across the four private asset classes and
gives a realistic risk estimate.

![Allocation](outputs/charts/06_allocation.png)

---

## Power BI

`outputs/data/` is laid out as a star schema (`dim_fund`, `fact_cashflows`, `fact_benchmark`, plus
summary tables) for a fund reporting dashboard. See the companion repository
**private-markets-reporting-dashboard**.

## Limitations and next steps

- All data is simulated. A natural extension is calibrating the TA parameters to real vintage data
  (e.g. Preqin / Burgiss / Cambridge Associates benchmarks).
- The TA model is deterministic in its rates. Stochastic call and distribution rates
  (e.g. Buchner, Kaserer & Wagner) would give more realistic dispersion.
- PME uses a single equity index. Direct Alpha and asset-class-specific benchmarks would refine it.
- The evergreen model ignores management fees, carried interest and FX.

## References

- Takahashi, D. & Alexander, S. (2002). *Illiquid Alternative Asset Fund Modeling*. Journal of Portfolio Management.
- Kaplan, S. & Schoar, A. (2005). *Private Equity Performance: Returns, Persistence, and Capital Flows*. Journal of Finance.
- Geltner, D. (1993). *Estimating Market Values from Appraised Values without Assuming an Efficient Market*. Journal of Real Estate Research.

---
*Author: Diptakshi Sen. MSc Banking & Finance (Financial Markets), KEDGE Business School.*
