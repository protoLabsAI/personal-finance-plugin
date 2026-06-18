"""Savings & investment projections (plan §4C). Pure functions over plain
numbers, built on the TVM engine; the @tool shims in tools.py wrap these. Each
returns a typed dict of figures plus a short `summary`; the year-by-year
schedule gates behind a `detail` flag (plan §1/§5).

Sign convention is handled internally: callers pass intuitive positive figures
(a balance, a contribution, a target) and never see the TVM cash-flow signs.
"""

from __future__ import annotations

import math

from . import money, tvm

__all__ = ["compound_growth", "savings_goal", "compare_apy"]


def compound_growth(
    starting_balance: float,
    contribution: float,
    rate: float,
    years: float,
    freq: str = "monthly",
    inflation_rate: float | None = None,
    detail: str = "summary",
) -> dict:
    """Project an investment growing at `rate` with regular `contribution`s.

    starting_balance: amount invested today. contribution: added each period.
    rate: nominal annual return (0.07 == 7%). years: horizon. freq: contribution
    frequency. inflation_rate: if given, also report the inflation-adjusted
    (real) future value in today's dollars. detail='full' adds a yearly schedule.
    """
    if starting_balance < 0:
        raise ValueError("starting_balance must be non-negative")
    if contribution < 0:
        raise ValueError("contribution must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    money.validate_detail(detail)

    ppy = money.periods_per_year(freq)
    periods = round(years * ppy)
    periodic = money.periodic_rate(rate, freq)

    # Contributions and the starting balance are paid in (negative), so the
    # future value comes back positive.
    nominal_fv = tvm.fv(periodic, periods, -contribution, -starting_balance)
    total_contributions = starting_balance + contribution * periods
    total_growth = nominal_fv - total_contributions

    out = {
        "starting_balance": money.round_cents(starting_balance),
        "contribution": money.round_cents(contribution),
        "rate": rate,
        "years": years,
        "frequency": freq,
        "periods": periods,
        "nominal_fv": money.round_cents(nominal_fv),
        "total_contributions": money.round_cents(total_contributions),
        "total_growth": money.round_cents(total_growth),
        "summary": (
            f"${starting_balance:,.0f} plus ${contribution:,.0f} {freq} "
            f"at {rate:.1%} for {years:g} years grows to ${money.round_cents(nominal_fv):,.2f} "
            f"(${money.round_cents(total_growth):,.2f} of it growth)."
        ),
    }

    if inflation_rate is not None:
        real_fv = nominal_fv / money.inflation_factor(inflation_rate, years)
        out["inflation_rate"] = inflation_rate
        out["real_fv"] = money.round_cents(real_fv)
        out["summary"] += f" In today's dollars: ${money.round_cents(real_fv):,.2f}."

    if detail == "full":
        out["schedule"] = _yearly_schedule(starting_balance, contribution, periodic, ppy, periods)
    return out


def savings_goal(
    target: float, years: float, starting_balance: float = 0.0, rate: float = 0.0, freq: str = "monthly"
) -> dict:
    """Required periodic contribution to reach `target` in `years`.

    target: amount to accumulate. years: horizon. starting_balance: amount
    already saved. rate: nominal annual return. freq: contribution frequency.
    If the starting balance already grows past the target on its own, the
    required contribution is zero and `already_met` is True.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if years <= 0:
        raise ValueError("years must be positive")
    if starting_balance < 0:
        raise ValueError("starting_balance must be non-negative")

    periods = round(years * money.periods_per_year(freq))
    periodic = money.periodic_rate(rate, freq)
    # pv is paid in (negative); fv is the target we want to receive (positive).
    required = -tvm.pmt(periodic, periods, -starting_balance, target)
    already_met = required <= 0
    required = max(required, 0.0)

    return {
        "target": money.round_cents(target),
        "years": years,
        "starting_balance": money.round_cents(starting_balance),
        "rate": rate,
        "frequency": freq,
        "periods": periods,
        "required_contribution": money.round_cents(required),
        "already_met": already_met,
        "summary": (
            f"Your ${starting_balance:,.0f} already reaches ${target:,.0f} in {years:g} years at {rate:.1%} — "
            f"no contributions needed."
            if already_met
            else f"Save ${money.round_cents(required):,.2f} {freq} to reach ${target:,.0f} in "
            f"{years:g} years at {rate:.1%} (starting from ${starting_balance:,.0f})."
        ),
    }


def compare_apy(accounts: list[dict]) -> dict:
    """Rank savings options by effective annual yield (APY).

    accounts: list of dicts, each with:
      - rate: nominal annual rate (0.05 == 5%)
      - freq: compounding frequency name, or "continuous"
      - name: optional label (defaults to "option 1", "option 2", ...)

    For "continuous" freq: APY = e**rate - 1.
    For all others: APY = (1 + rate/n)**n - 1 where n = periods_per_year(freq).
    Returns ranked (desc by APY), best (name of top option), and summary.
    """
    if not accounts:
        raise ValueError("accounts must be a non-empty list")

    ranked = []
    for i, account in enumerate(accounts):
        if "rate" not in account:
            raise ValueError(f"account at index {i} is missing required field 'rate'")
        if "freq" not in account:
            raise ValueError(f"account at index {i} is missing required field 'freq'")

        rate = account["rate"]
        if not isinstance(rate, (int, float)):
            raise ValueError(f"account at index {i}: 'rate' must be numeric, got {rate!r}")
        # rate <= -1 makes the compounding base (1 + rate/n) non-positive and the
        # APY meaningless (e.g. rate=-2 monthly silently yields 0%). Match the
        # money.inflation_factor convention and reject it outright.
        if rate <= -1:
            raise ValueError(f"account at index {i}: 'rate' must be greater than -1")

        freq = account["freq"]
        name = account.get("name", f"option {i + 1}")

        if freq == "continuous":
            apy = math.exp(rate) - 1
        else:
            n = money.periods_per_year(freq)
            apy = (1 + rate / n) ** n - 1

        ranked.append({"name": name, "rate": rate, "freq": freq, "apy": round(apy, 6)})

    ranked.sort(key=lambda r: r["apy"], reverse=True)
    best = ranked[0]["name"]
    best_apy = ranked[0]["apy"]

    return {
        "ranked": ranked,
        "best": best,
        "summary": f"{best} offers the highest APY at {best_apy:.4%}.",
    }


def _yearly_schedule(
    starting_balance: float, contribution: float, periodic: float, ppy: int, periods: int
) -> list[dict]:
    """Balance at each year boundary, with cumulative contributions and growth."""
    rows: list[dict] = []
    year = 0
    elapsed = 0
    while elapsed < periods:
        elapsed = min(elapsed + ppy, periods)
        year += 1
        balance = tvm.fv(periodic, elapsed, -contribution, -starting_balance)
        contributions = starting_balance + contribution * elapsed
        rows.append(
            {
                "year": year,
                "balance": money.round_cents(balance),
                "contributions": money.round_cents(contributions),
                "growth": money.round_cents(balance - contributions),
            }
        )
    return rows
