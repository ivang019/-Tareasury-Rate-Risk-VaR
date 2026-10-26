"""Constant-maturity Mbono yields (3 and 5 years) from Banxico's price vector (SIE, table CF300).

Banxico does not publish a daily constant-maturity Mbono yield. It does publish, for each maturity
bucket, the clean price, coupon and days to maturity of the reference bond. Steps:

1. Yield to maturity of each reference bond (Banxico convention: face value 100, coupon every
   182 days, 360-day basis), solved by bisection from the clean price.
2. 364-day Cetes as the short node: its simple rate is converted to the same convention.
3. Linear interpolation in days to maturity at 1,092 days (3 years) and 1,820 days (5 years),
   without extrapolation and without interpolating between nodes more than 2,300 days apart.

Limitations: linear interpolation with few nodes; the reference bond of each bucket changes from
time to time (those days are flagged so their daily changes can be excluded).
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
FACE, PERIOD, BASIS = 100.0, 182, 360
TARGET_TERMS = {"mbono_3y": 1092, "mbono_5y": 1820}   # days to maturity
BOND_NODES = ["bond_0_3", "bond_3_5", "bond_7_10"]    # the 5-7 year bucket has no data
MAX_NODE_GAP_DAYS = 2300
YTM_MIN, YTM_MAX = -10.0, 100.0                       # bisection bracket (%)


def remaining_coupons(days: int) -> tuple[int, int]:
    """Coupons left to pay and days to the next coupon (coupons counted back from maturity)."""
    n = math.ceil(days / PERIOD)
    return n, days - PERIOD * (n - 1)


def accrued_interest(days: int, coupon_pct: float) -> float:
    _, to_next = remaining_coupons(days)
    return FACE * coupon_pct / 100 * (PERIOD - to_next) / BASIS


def dirty_price(ytm_pct: float, days: int, coupon_pct: float) -> float:
    """Dirty price of an Mbono (face value 100) for a given yield to maturity."""
    n, to_next = remaining_coupons(days)
    coupon = FACE * coupon_pct / 100 * PERIOD / BASIS
    r = ytm_pct / 100 * PERIOD / BASIS
    left = n - 1
    annuity = left if r == 0 else (1 - (1 + r) ** -left) / r
    value = coupon + coupon * annuity + FACE * (1 + r) ** -left
    return value / (1 + r) ** (to_next / PERIOD)


def ytm(clean_price: float, days: int, coupon_pct: float) -> float:
    """Yield to maturity (%) that reproduces the clean price (bisection)."""
    target = clean_price + accrued_interest(days, coupon_pct)
    low, high = YTM_MIN, YTM_MAX
    if dirty_price(low, days, coupon_pct) < target or dirty_price(high, days, coupon_pct) > target:
        return np.nan
    for _ in range(100):
        mid = (low + high) / 2
        if dirty_price(mid, days, coupon_pct) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cetes_to_mbono(rate_pct: float, days: int) -> float:
    """Simple 360-day Cetes rate -> equivalent rate compounded every 182 days."""
    factor = 1 + rate_pct / 100 * days / BASIS
    return (factor ** (PERIOD / days) - 1) * BASIS / PERIOD * 100


def reference_bond_change(term: pd.Series, coupon: pd.Series | None = None) -> pd.Series:
    """True on the day the reference bond of the bucket changes.

    For a given bond, days to maturity fall by exactly the calendar days elapsed; any other change
    in term (or a coupon change) means a different bond.
    """
    t = term.dropna()
    days = pd.Series(t.index, index=t.index).diff().dt.days
    change = t.ne(t.shift() - days) & t.shift().notna()
    if coupon is not None:
        c = coupon.reindex(t.index)
        change |= c.ne(c.shift()) & c.notna() & c.shift().notna()
    return change.reindex(term.index, fill_value=False)


def nodes(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Days to maturity and yield (%, Mbono convention) of each node, by day."""
    terms, ytms = {}, {}
    for n in BOND_NODES:
        t = data[[f"{n}_term", f"{n}_clean", f"{n}_coupon"]].dropna()
        values = [ytm(p, int(d), c) if d > 0 else np.nan
                  for d, p, c in zip(t[f"{n}_term"], t[f"{n}_clean"], t[f"{n}_coupon"])]
        ytms[n] = pd.Series(values, index=t.index)
        terms[n] = data[f"{n}_term"]
    t = data[["cetes_364", "cetes_364_term"]].dropna()
    ytms["cetes_364"] = pd.Series([cetes_to_mbono(r, int(d)) for r, d in zip(t.cetes_364, t.cetes_364_term)],
                                  index=t.index)
    terms["cetes_364"] = data["cetes_364_term"]
    ytms = pd.DataFrame(ytms).reindex(data.index)
    terms = pd.DataFrame(terms).where(ytms.notna())
    return terms, ytms


def interpolate(terms: pd.DataFrame, ytms: pd.DataFrame, target: int) -> pd.DataFrame:
    """Yield at `target` days between the lower and upper node (no extrapolation)."""
    rows = []
    for date, p in terms.iterrows():
        p = p.dropna()
        below, above = p[p <= target], p[p >= target]
        row = {"ytm": np.nan, "lower": None, "upper": None}
        if not below.empty and not above.empty:
            lo, up = below.idxmax(), above.idxmin()
            if p[up] - p[lo] <= MAX_NODE_GAP_DAYS:
                y0, y1 = ytms.at[date, lo], ytms.at[date, up]
                row = {"ytm": y0 if p[up] == p[lo] else y0 + (y1 - y0) * (target - p[lo]) / (p[up] - p[lo]),
                       "lower": lo, "upper": up}
        rows.append(row)
    return pd.DataFrame(rows, index=terms.index)


def mbono_rates(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3- and 5-year Mbono yields and the days to exclude from their daily changes (bond switch)."""
    terms, ytms = nodes(data)
    switches = pd.DataFrame({n: reference_bond_change(data[f"{n}_term"], data[f"{n}_coupon"])
                             for n in BOND_NODES})
    switches["cetes_364"] = reference_bond_change(data["cetes_364_term"])
    levels, exclude = {}, {}
    for name, target in TARGET_TERMS.items():
        t = interpolate(terms, ytms, target)
        flag = pd.Series(False, index=t.index)
        for lag in (0, 1):                             # nodes used today or yesterday
            for col in ("lower", "upper"):
                used = t[col].shift(lag)
                flag |= pd.Series([bool(switches.at[d, n]) if isinstance(n, str) else False
                                   for d, n in zip(t.index, used)], index=t.index)
        levels[name], exclude[name] = t["ytm"], flag
    return pd.DataFrame(levels), pd.DataFrame(exclude)


def plot(data: pd.DataFrame, levels: pd.DataFrame, path: Path) -> None:
    """Derived yield against the yield of the official auctions (auction days only)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tenor, color in zip(axes, ["3y", "5y"], ["#d97706", "#b91c1c"]):
        auction = data[f"auction_{tenor}"].dropna()
        ax.plot(levels.index, levels[f"mbono_{tenor}"], color=color, lw=0.9, label="Derived from prices")
        ax.scatter(auction.index, auction, s=7, color="black", zorder=3, label="Auction (official data)")
        ax.set_title(f"{tenor[0]}-year Mbono")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("% per year")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def main() -> None:
    data = pd.read_csv(ROOT / "data/sie_series.csv", index_col="date", parse_dates=["date"])
    levels, _ = mbono_rates(data)
    for tenor in ("3y", "5y"):
        auction = data[f"auction_{tenor}"].dropna()
        diff = ((levels[f"mbono_{tenor}"].reindex(auction.index) - auction) * 100).dropna()
        print(f"Mbono {tenor}: {len(diff)} auctions, median difference {diff.median():+.1f} bp")
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(data, levels, ROOT / "figures/mbono_vs_auction.png")


if __name__ == "__main__":
    main()
