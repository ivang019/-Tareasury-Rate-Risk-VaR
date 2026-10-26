"""Interest-rate market risk of an illustrative treasury: VaR, ES and backtesting.

Steps: (1) daily rate changes in basis points, (2) portfolio -> sensitivity (DV01) by rate,
(3) 1-day VaR and Expected Shortfall with 4 methods, (4) backtesting.
Conventions: rates in %, changes in bp, losses as positive numbers.

Main assumptions and limitations:
- The portfolio (portfolio.yaml) is illustrative. Loss is linear in rate changes (delta) and
  today's portfolio is applied to the whole history (hypothetical P&L).
- The swap is valued with the Mbono curve as a stand-in for the TIIE swap curve, which is not
  public: the swap-vs-government basis risk is NOT captured.
- Days with a missing rate are dropped (never filled).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import xlogy
from scipy.stats import chi2, norm

from mbonos import BASIS, PERIOD, dirty_price, mbono_rates

ROOT = Path(__file__).parent
WINDOW = 250             # business days of history for each VaR
MIN_DAYS = 125           # minimum valid days inside the window
CONFIDENCES = (95, 99)
LAMBDA = 0.94            # RiskMetrics EWMA decay; not calibrated
FACTORS = ["tiie_28", "cetes_28", "cetes_91", "mbono_3y", "mbono_5y"]
METHODS = {"historical": "Historical", "parametric": "Parametric normal",
           "ewma": "EWMA normal", "fhs": "Filtered historical (FHS)"}
# Cetes 28 outliers: the level moves away and comes back with no other rate following it
CETES28_OUTLIERS = ["2016-11-14", "2016-11-15", "2017-01-02", "2017-01-03", "2025-07-08",
                    "2025-07-09", "2026-09-25", "2026-09-28", "2026-09-29"]


# ------------------------------------------------------------------ 1. daily changes
def load_changes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Levels (%) and daily changes (bp) of the 5 rates: `raw` and `clean` (without outliers)."""
    data = pd.read_csv(ROOT / "data/sie_series.csv", index_col="date", parse_dates=["date"])
    mbonos, exclude = mbono_rates(data)
    levels = pd.concat([data[["tiie_28", "cetes_28", "cetes_91"]], mbonos], axis=1)[FACTORS]
    raw = levels.diff() * 100                      # NaN if either of the two days is missing
    for factor in exclude:                         # reference-bond switch days
        raw.loc[exclude[factor], factor] = np.nan
    clean = raw.copy()
    clean.loc[clean.index.isin(pd.to_datetime(CETES28_OUTLIERS)), "cetes_28"] = np.nan
    return levels, raw, clean


# -------------------------------------------------------- 2. portfolio and sensitivities
def position_value(p: dict, base: pd.Series, level: dict) -> float:
    """Value in MXN (signed for the treasury). `base`: today's rates; `level`: shocked rates."""
    notional = p["notional_mxn_mm"] * 1e6
    if p["type"] == "cetes":
        return notional / (1 + level[p["factor"]] / 100 * p["term_days"] / BASIS)
    if p["type"] == "mbono":
        return notional / 100 * dirty_price(level[p["factor"]], p["term_days"], p["coupon_pct"])
    if p["type"] == "floating_debt":               # liability: worth par on the reset date
        coupon = (base[p["factor"]] + p["margin_pct"]) / 100 * p["reset_period_days"] / BASIS
        discount = (level[p["factor"]] + p["margin_pct"]) / 100 * p["days_to_next_reset"] / BASIS
        return -notional * (1 + coupon) / (1 + discount)
    if p["type"] == "pay_fixed_swap":              # value = floating leg (receive) - fixed leg (pay)
        period, next_pay, term = p["payment_period_days"], p["days_to_next_payment"], p["term_days"]
        floating_factor = p["floating_leg_factor"]
        floating = (notional * (1 + base[floating_factor] / 100 * period / BASIS)
                    / (1 + level[floating_factor] / 100 * next_pay / BASIS))
        i = (1 + level[p["factor"]] / 100 * PERIOD / BASIS) ** (period / PERIOD) - 1
        times = np.arange(next_pay, term + 1, period)
        coupon = notional * p["fixed_rate_pct"] / 100 * period / BASIS
        fixed = coupon * np.sum((1 + i) ** (-times / period)) + notional * (1 + i) ** (-term / period)
        return floating - fixed
    raise ValueError(f"Unknown position type: {p['type']}")


def dv01_by_factor(portfolio: list[dict], base: pd.Series) -> pd.Series:
    """MXN gained (+) or lost (-) by the portfolio if each rate rises 1 bp (central difference)."""
    def value(shock_bp: dict) -> float:
        level = {f: base[f] + shock_bp.get(f, 0) / 100 for f in FACTORS}
        return sum(position_value(p, base, level) for p in portfolio)
    return pd.Series({f: (value({f: 1}) - value({f: -1})) / 2 for f in FACTORS})


# ------------------------------------------------------------------- 3. 1-day VaR and ES
def hist_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR = k-th worst loss (k = ceil((1-alpha)n)); ES = Acerbi-Tasche estimator."""
    ordered = np.sort(losses)[::-1]
    n = len(ordered)
    var = ordered[math.ceil(round((1 - alpha) * n, 9)) - 1]
    m = round((1 - alpha) * n, 9)
    whole = int(m)
    rest = m - whole
    es = (ordered[:whole].sum() + (rest * ordered[whole] if rest > 1e-9 else 0.0)) / m
    return var, es


def normal_var_es(sigma: float, alpha: float) -> tuple[float, float]:
    z = norm.ppf(alpha)
    return z * sigma, sigma * norm.pdf(z) / (1 - alpha)


def ewma_sigma(series: pd.Series, lam: float = LAMBDA, n_init: int = WINDOW) -> pd.Series:
    """Forecast of the standard deviation (zero mean) for each date using information up to t-1.

    The first `n_init` valid days only initialise the recursion; days without data do not update it.
    """
    s2 = (series.dropna().iloc[:n_init] ** 2).mean()
    out = pd.Series(np.nan, index=series.index)
    seen = 0
    for date, x in series.items():
        if seen >= n_init:
            out[date] = math.sqrt(s2)
        if not np.isnan(x):
            seen += 1
            if seen > n_init:
                s2 = lam * s2 + (1 - lam) * x * x
    return out


def rolling_var(raw: pd.DataFrame, clean: pd.DataFrame, dv01: pd.Series) -> pd.DataFrame:
    """1-day VaR/ES for each date using information up to the previous day (4 methods).

    The last row is the forecast for the day after the last data point. Column `loss` is the
    realised loss of the day (it includes the confirmed outliers with their observed change).
    """
    next_day = clean.index[-1] + pd.offsets.BDay(1)
    idx = clean.index.append(pd.DatetimeIndex([next_day]))
    X = clean.reindex(idx)[FACTORS]
    d = dv01[FACTORS].values
    portfolio_loss = -(X * dv01[FACTORS]).sum(axis=1, min_count=len(FACTORS))   # NaN if a rate is missing
    Xj = X.loc[portfolio_loss.notna()].reindex(idx)                              # days with all 5 rates
    sigmas = Xj.apply(ewma_sigma)                   # EWMA per rate (for filtered historical)
    portfolio_sigma = ewma_sigma(portfolio_loss)    # EWMA of the portfolio loss

    rows = {}
    for i in range(WINDOW, len(idx)):
        window = Xj.iloc[i - WINDOW:i].dropna()
        if len(window) < MIN_DAYS:
            continue
        losses = -(window.values @ d)               # loss of today's portfolio with past changes
        sigma = math.sqrt(np.mean(losses ** 2))
        sigma_today = sigmas.iloc[i].values
        sigma_window = sigmas.reindex(window.index).values
        good = ~np.isnan(sigma_window).any(axis=1)
        filtered = None
        if np.isfinite(sigma_today).all() and good.sum() >= MIN_DAYS:
            filtered = -((window.values[good] / sigma_window[good] * sigma_today) @ d)
        row = {}
        for q in CONFIDENCES:
            a = q / 100
            row[f"var_historical_{q}"], row[f"es_historical_{q}"] = hist_var_es(losses, a)
            row[f"var_parametric_{q}"], row[f"es_parametric_{q}"] = normal_var_es(sigma, a)
            if np.isfinite(portfolio_sigma.iloc[i]):
                row[f"var_ewma_{q}"], row[f"es_ewma_{q}"] = normal_var_es(portfolio_sigma.iloc[i], a)
            if filtered is not None:
                row[f"var_fhs_{q}"], row[f"es_fhs_{q}"] = hist_var_es(filtered, a)
        rows[idx[i]] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    realised = -(raw[FACTORS] * dv01[FACTORS]).sum(axis=1, min_count=len(FACTORS))
    out["loss"] = realised.reindex(out.index)
    return out


# ------------------------------------------------------------------------ 4. backtesting
def kupiec_lr(n: int, x: int, p: float) -> float:
    """Kupiec: is the failure frequency x/n compatible with the expected p?"""
    pi = x / n
    return -2 * ((xlogy(n - x, 1 - p) + xlogy(x, p)) - (xlogy(n - x, 1 - pi) + xlogy(x, pi)))


def christoffersen_lr(n00: int, n01: int, n10: int, n11: int, p: float) -> tuple[float, float]:
    """Christoffersen: independence of failures (LR_ind) and conditional coverage (LR_cc)."""
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    ll_pi = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_p = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    return -2 * (ll_pi - ll_alt), -2 * (ll_p - ll_alt)


def backtest(rolling: pd.DataFrame) -> pd.DataFrame:
    """Failures (loss > VaR) on the common sample of the 4 methods, with Kupiec/Christoffersen tests.

    Streaks are cut where data is missing: only pairs of consecutive business days are counted.
    """
    columns = [f"var_{m}_{q}" for m in METHODS for q in CONFIDENCES] + ["loss"]
    common = rolling[columns].notna().all(axis=1)
    rows = []
    for q in CONFIDENCES:
        p = 1 - q / 100
        for m, name in METHODS.items():
            ind = (rolling["loss"] > rolling[f"var_{m}_{q}"]).astype(float).where(common)
            n, x = int(ind.notna().sum()), int(ind.sum())
            pairs = pd.DataFrame({"a": ind.shift(1), "b": ind}).dropna()
            n00, n01 = int(((pairs.a == 0) & (pairs.b == 0)).sum()), int(((pairs.a == 0) & (pairs.b == 1)).sum())
            n10, n11 = int(((pairs.a == 1) & (pairs.b == 0)).sum()), int(((pairs.a == 1) & (pairs.b == 1)).sum())
            lr_ind, lr_cc = christoffersen_lr(n00, n01, n10, n11, p)
            failure = ind == 1
            rows.append({
                "Confidence": f"{q}%", "Method": name, "Days": n, "Failures": x, "Expected": round(n * p, 1),
                "Kupiec p": chi2.sf(kupiec_lr(n, x, p), 1),
                "Independence p": chi2.sf(lr_ind, 1), "Cond. coverage p": chi2.sf(lr_cc, 2),
                "Loss at failure / ES": (rolling.loc[failure, "loss"].mean()
                                         / rolling.loc[failure, f"es_{m}_{q}"].mean()),
                "Mean VaR (MXN mm)": rolling.loc[common, f"var_{m}_{q}"].mean() / 1e6,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- plots
COLORS = {"historical": "#6b7280", "parametric": "#b91c1c", "ewma": "#d97706", "fhs": "#1d4ed8"}


def plots(rolling: pd.DataFrame, dv01: pd.Series, folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    names = {"tiie_28": "TIIE 28", "cetes_28": "Cetes 28", "cetes_91": "Cetes 91",
             "mbono_3y": "Mbono 3Y", "mbono_5y": "Mbono 5Y"}

    fig, ax = plt.subplots(figsize=(8, 3.6))                        # sensitivity by rate
    values = dv01 / 1e3
    bars = ax.bar([names[f] for f in values.index], values.values,
                  color=["#15803d" if v > 0 else "#b91c1c" for v in values])
    ax.bar_label(bars, fmt="%+.0f", padding=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("thousand MXN per +1 bp")
    ax.set_title("Sensitivity by rate: gain (green) or loss (red) if the rate rises 1 bp", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "dv01.png", bbox_inches="tight")

    loss = rolling["loss"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)   # loss against 99% VaR
    for ax, m in zip(axes, ["parametric", "fhs"]):
        var = rolling[f"var_{m}_99"]
        failures = loss[loss > var]
        ax.plot(loss.index, loss / 1e6, color="#9ca3af", lw=0.6, label="Daily loss (negative = gain)")
        ax.plot(var.index, var / 1e6, color=COLORS[m], lw=1.3, label=f"99% VaR - {METHODS[m]}")
        ax.scatter(failures.index, failures / 1e6, color="#dc2626", s=16, zorder=3, label=f"Failures ({len(failures)})")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        ax.set_ylabel("MXN million")
    fig.tight_layout()
    fig.savefig(folder / "backtest_series.png", bbox_inches="tight")

    columns = [f"var_{m}_{q}" for m in METHODS for q in CONFIDENCES] + ["loss"]
    common = rolling[rolling[columns].notna().all(axis=1)]          # failures by year (99%, common sample)
    failures = pd.DataFrame({METHODS[m]: common["loss"] > common[f"var_{m}_99"] for m in METHODS})
    by_year = failures.groupby(common.index.year).sum()
    expected = common.groupby(common.index.year).size() * 0.01
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(by_year))
    for k, m in enumerate(METHODS):
        ax.bar(x + (k - 1.5) * 0.2, by_year[METHODS[m]], 0.2, color=COLORS[m], label=METHODS[m])
    ax.plot(x, expected.values, color="black", lw=1.2, ls="--", marker="_", ms=14, label="Expected (1%)")
    ax.set_xticks(x)
    ax.set_xticklabels(by_year.index)
    ax.set_ylim(0, by_year.values.max() * 1.35)
    ax.set_ylabel("failures in the year")
    ax.set_title("99% VaR failures by year", fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(folder / "failures_by_year.png", bbox_inches="tight")
    plt.close("all")


def main() -> None:
    levels, raw, clean = load_changes()
    portfolio = yaml.safe_load((ROOT / "portfolio.yaml").read_text(encoding="utf-8"))["positions"]
    base = levels.dropna().iloc[-1]                                 # rates of the last complete day
    dv01 = dv01_by_factor(portfolio, base)
    print(f"Valuation date: {base.name:%Y-%m-%d}")
    print("DV01 (thousand MXN per bp):", (dv01 / 1e3).round(1).to_dict())

    rolling = rolling_var(raw, clean, dv01)
    today = rolling.iloc[-1]
    var_today = pd.DataFrame(
        {name: {f"{t} {q}% (MXN mm)": today[f"{t.lower()}_{m}_{q}"] / 1e6 for q in CONFIDENCES for t in ("VaR", "ES")}
         for m, name in METHODS.items()}).T
    print("\n1-day VaR and ES:\n", var_today.round(2).to_string())

    rolling = rolling.iloc[:-1]                                     # drop tomorrow's forecast
    results = backtest(rolling)
    print("\nBacktesting (common sample):\n", results.round(3).to_string(index=False))

    (ROOT / "results").mkdir(exist_ok=True)
    var_today.round(3).to_csv(ROOT / "results/var_today.csv")
    results.round(4).to_csv(ROOT / "results/backtest.csv", index=False)
    plots(rolling, dv01, ROOT / "figures")


if __name__ == "__main__":
    main()
