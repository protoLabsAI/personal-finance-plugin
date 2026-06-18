"""Loan & debt tools (plan §4B). Pure functions over plain numbers, built on the
TVM engine; the @tool shims in tools.py wrap these. Every tool returns a typed
dict of figures plus a short `summary` string; large per-period schedules gate
behind a `detail` flag (plan §1/§5).
"""

from __future__ import annotations

import math

from . import money, tvm

__all__ = [
    "loan_payment",
    "loan_amortization",
    "loan_payoff_extra",
    "debt_payoff_strategy",
    "debt_to_income",
    "refinance_breakeven",
]

_MAX_MONTHS = 1200  # 100 years — guards against budgets that never clear the debt


def _validate_loan(principal: float, term_years: float) -> None:
    if principal <= 0:
        raise ValueError("principal must be positive")
    if term_years <= 0:
        raise ValueError("term_years must be positive")


def loan_payment(principal: float, apr: float, term_years: float, freq: str = "monthly") -> dict:
    """Level payment for a fully-amortizing loan, plus lifetime cost.

    principal: amount borrowed. apr: nominal annual rate (0.06 == 6%).
    term_years: length of the loan. freq: payment frequency (default monthly).
    """
    _validate_loan(principal, term_years)
    periods = round(term_years * money.periods_per_year(freq))
    rate = money.periodic_rate(apr, freq)
    payment = -tvm.pmt(rate, periods, principal)
    total_paid = payment * periods
    total_interest = total_paid - principal
    return {
        "principal": money.round_cents(principal),
        "apr": apr,
        "term_years": term_years,
        "frequency": freq,
        "periods": periods,
        "payment": money.round_cents(payment),
        "total_paid": money.round_cents(total_paid),
        "total_interest": money.round_cents(total_interest),
        "summary": (
            f"{freq.capitalize()} payment of ${money.round_cents(payment):,.2f} on a "
            f"${principal:,.0f} loan at {apr:.2%} over {term_years:g} years — "
            f"${money.round_cents(total_interest):,.2f} total interest."
        ),
    }


def _amortization_rows(principal: float, rate: float, periods: int, payment: float) -> list[dict]:
    """Period-by-period schedule via running balance. The final payment is
    adjusted to clear any residual cent so the balance lands exactly on zero."""
    rows: list[dict] = []
    balance = principal
    for period in range(1, periods + 1):
        interest = balance * rate
        principal_part = payment - interest
        if period == periods:  # clear any float residual on the last payment
            principal_part = balance
        balance -= principal_part
        rows.append(
            {
                "period": period,
                "payment": money.round_cents(interest + principal_part),
                "principal": money.round_cents(principal_part),
                "interest": money.round_cents(interest),
                "balance": money.round_cents(max(balance, 0.0)),
            }
        )
    return rows


def loan_amortization(
    principal: float, apr: float, term_years: float, freq: str = "monthly", detail: str = "summary"
) -> dict:
    """Amortization for a loan. detail='summary' returns totals only; detail='full'
    adds a per-period `schedule` (period, payment, principal, interest, balance)."""
    money.validate_detail(detail)
    base = loan_payment(principal, apr, term_years, freq)
    if detail == "full":
        rate = money.periodic_rate(apr, freq)
        payment = -tvm.pmt(rate, base["periods"], principal)
        base["schedule"] = _amortization_rows(principal, rate, base["periods"], payment)
    return base


def loan_payoff_extra(
    principal: float, apr: float, term_years: float, extra_payment: float, freq: str = "monthly"
) -> dict:
    """Effect of paying `extra_payment` on top of the scheduled payment each period:
    a shorter term and less interest. Compares against the baseline schedule."""
    if extra_payment < 0:
        raise ValueError("extra_payment must be non-negative")
    base = loan_payment(principal, apr, term_years, freq)
    rate = money.periodic_rate(apr, freq)
    payment = -tvm.pmt(rate, base["periods"], principal)

    balance = principal
    periods = 0
    interest_total = 0.0
    # Cap iterations at the original term — extra payments can only shorten it.
    while balance > 1e-9 and periods < base["periods"]:
        interest = balance * rate
        principal_part = min(payment + extra_payment - interest, balance)
        balance -= principal_part
        interest_total += interest
        periods += 1

    new_total_interest = money.round_cents(interest_total)
    interest_saved = money.round_cents(base["total_interest"] - new_total_interest)
    return {
        "baseline_periods": base["periods"],
        "new_periods": periods,
        "periods_saved": base["periods"] - periods,
        "baseline_total_interest": base["total_interest"],
        "new_total_interest": new_total_interest,
        "interest_saved": interest_saved,
        "scheduled_payment": base["payment"],
        "extra_payment": money.round_cents(extra_payment),
        "summary": (
            f"Paying ${money.round_cents(extra_payment):,.2f} extra per period pays the loan "
            f"off in {periods} periods instead of {base['periods']} "
            f"({base['periods'] - periods} sooner), saving ${interest_saved:,.2f} in interest."
        ),
    }


def _simulate_payoff(debts: list[dict], budget: float, method: str) -> dict:
    """Month-by-month simulation of the snowball (smallest balance first) or
    avalanche (highest APR first) method, rolling each cleared minimum back into
    the budget. Returns months, total interest, and the order debts were cleared.
    """
    state = [
        {"name": d["name"], "balance": float(d["balance"]), "apr": float(d["apr"]), "min": float(d["min_payment"])}
        for d in debts
    ]
    months = 0
    total_interest = 0.0
    # Debts that start already paid off never enter the simulation loop, but
    # should still be reported as cleared (in input order, before any others).
    order: list[str] = [d["name"] for d in state if d["balance"] <= 1e-9]
    while any(d["balance"] > 1e-9 for d in state):
        months += 1
        if months > _MAX_MONTHS:
            raise ValueError("budget too low — these debts never reach payoff (interest outpaces payments)")
        active = [d for d in state if d["balance"] > 1e-9]
        for d in active:
            interest = d["balance"] * d["apr"] / 12
            d["balance"] += interest
            total_interest += interest
        available = budget
        for d in active:  # minimum payments on every debt first
            pay = min(d["min"], d["balance"], available)
            d["balance"] -= pay
            available -= pay
        if method == "snowball":
            priority = sorted(active, key=lambda d: d["balance"])
        else:  # avalanche
            priority = sorted(active, key=lambda d: -d["apr"])
        for d in priority:  # remaining budget attacks the priority target
            if available <= 1e-9:
                break
            pay = min(available, d["balance"])
            d["balance"] -= pay
            available -= pay
        for d in priority:  # record cleared debts in attack order, not list order
            if d["balance"] <= 1e-9 and d["name"] not in order:
                order.append(d["name"])
    return {"months": months, "total_interest": money.round_cents(total_interest), "payoff_order": order}


def debt_payoff_strategy(debts: list[dict], budget: float, method: str = "avalanche") -> dict:
    """Order multiple debts for payoff and project months + total interest.

    debts: list of {name, balance, apr, min_payment}. budget: total to apply
    across all debts each month (must cover the minimums). method: 'snowball'
    (smallest balance first) or 'avalanche' (highest APR first). Always reports
    both methods in `comparison` so the trade-off is visible.
    """
    if method not in ("snowball", "avalanche"):
        raise ValueError("method must be 'snowball' or 'avalanche'")
    if not debts:
        raise ValueError("debts must be a non-empty list")
    minimums = sum(float(d["min_payment"]) for d in debts)
    if budget < minimums:
        raise ValueError(f"budget ${budget:,.2f} is below the sum of minimum payments ${minimums:,.2f}")

    snowball = _simulate_payoff(debts, budget, "snowball")
    avalanche = _simulate_payoff(debts, budget, "avalanche")
    chosen = snowball if method == "snowball" else avalanche
    return {
        "method": method,
        "months": chosen["months"],
        "total_interest": chosen["total_interest"],
        "payoff_order": chosen["payoff_order"],
        "comparison": {"snowball": snowball, "avalanche": avalanche},
        "summary": (
            f"{method.capitalize()}: clears all debts in {chosen['months']} months with "
            f"${chosen['total_interest']:,.2f} interest. Order: {', '.join(chosen['payoff_order'])}."
        ),
    }


def debt_to_income(gross_monthly_income: float, housing_payment: float, other_debt_payments: float = 0.0) -> dict:
    """Debt-to-income ratios. front_end = housing / income; back_end = all debt
    (housing + other) / income. Returned as decimals (0.36 == 36%)."""
    if gross_monthly_income <= 0:
        raise ValueError("gross_monthly_income must be positive")
    if housing_payment < 0 or other_debt_payments < 0:
        raise ValueError("payments must be non-negative")
    total_debt = housing_payment + other_debt_payments
    front_end = housing_payment / gross_monthly_income
    back_end = total_debt / gross_monthly_income
    return {
        "gross_monthly_income": money.round_cents(gross_monthly_income),
        "housing_payment": money.round_cents(housing_payment),
        "total_debt_payments": money.round_cents(total_debt),
        "front_end": front_end,
        "back_end": back_end,
        "summary": (
            f"Front-end DTI {front_end:.1%} (housing), back-end DTI {back_end:.1%} (all debt). "
            f"Lenders commonly look for back-end DTI at or below 36%."
        ),
    }


def refinance_breakeven(
    balance: float,
    current_apr: float,
    current_remaining_years: float,
    new_apr: float,
    new_term_years: float,
    closing_costs: float,
    freq: str = "monthly",
) -> dict:
    """Compare staying on the current loan vs refinancing the remaining balance.

    Computes the periodic payment under each scenario using the module's own
    loan_payment, reports the per-period savings and the breakeven period to
    recoup closing costs, and surfaces the lifetime interest trade-off so that a
    longer new term's lower payment is weighed against its higher total cost.

    balance: outstanding principal. current_apr/new_apr: nominal annual rates.
    current_remaining_years/new_term_years: remaining/new loan lengths.
    closing_costs: upfront refi costs (must be >= 0). freq: payment frequency.
    """
    if balance <= 0:
        raise ValueError("balance must be positive")
    if current_remaining_years <= 0:
        raise ValueError("current_remaining_years must be positive")
    if new_term_years <= 0:
        raise ValueError("new_term_years must be positive")
    if closing_costs < 0:
        raise ValueError("closing_costs must be non-negative")
    if current_apr < 0:
        raise ValueError("current_apr must be non-negative")
    if new_apr < 0:
        raise ValueError("new_apr must be non-negative")

    ppy = money.periods_per_year(freq)

    # Reuse loan_payment for both scenarios — single source of payment AND
    # lifetime-cost math. Its total_interest is derived from the unrounded
    # payment (then rounded at the edge), so reading it back avoids the drift
    # that recomputing from the already-rounded payment would introduce.
    current_info = loan_payment(balance, current_apr, current_remaining_years, freq)
    new_info = loan_payment(balance, new_apr, new_term_years, freq)

    current_payment = current_info["payment"]
    new_payment = new_info["payment"]
    periodic_savings = money.round_cents(current_payment - new_payment)

    current_total_interest = current_info["total_interest"]
    new_total_interest = new_info["total_interest"]
    lifetime_interest_savings = money.round_cents(current_total_interest - new_total_interest)

    if periodic_savings <= 0:
        breakeven_periods = None
        breakeven_years = None
        summary = (
            f"Refinancing raises the {freq} payment from ${current_payment:,.2f} to ${new_payment:,.2f} "
            f"(no savings — no breakeven). "
            f"New loan total interest: ${new_total_interest:,.2f} vs current: ${current_total_interest:,.2f}."
        )
    else:
        breakeven_periods = math.ceil(closing_costs / periodic_savings)
        breakeven_years = breakeven_periods / ppy
        term_note = (
            " Note: the new term is longer, which raises lifetime interest despite the lower payment."
            if new_term_years > current_remaining_years
            else ""
        )
        summary = (
            f"Refi saves ${periodic_savings:,.2f}/{freq} — closing costs recouped in {breakeven_periods} periods "
            f"({breakeven_years:.1f} years). "
            f"Lifetime interest: current ${current_total_interest:,.2f}, new ${new_total_interest:,.2f} "
            f"(${lifetime_interest_savings:,.2f} {'saved' if lifetime_interest_savings >= 0 else 'more'}).{term_note}"
        )

    return {
        "balance": money.round_cents(balance),
        "frequency": freq,
        "current_payment": money.round_cents(current_payment),
        "new_payment": money.round_cents(new_payment),
        "periodic_savings": periodic_savings,
        "breakeven_periods": breakeven_periods,
        "breakeven_years": breakeven_years,
        "current_total_interest": current_total_interest,
        "new_total_interest": new_total_interest,
        "lifetime_interest_savings": lifetime_interest_savings,
        "summary": summary,
    }
