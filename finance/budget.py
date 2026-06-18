"""Budgeting & household tools (plan §4E). Pure functions over plain numbers;
the @tool shims in tools.py wrap these. Every tool returns a typed dict of
figures plus a short `summary` string; per-item breakdowns gate behind a
`detail` flag (plan §1/§5).

Currency aggregation goes through money.money (Decimal) so sums of many figures
don't drift, then rounds to cents at the edge via money.round_cents.
"""

from __future__ import annotations

import math

from . import money

__all__ = ["budget_breakdown", "emergency_fund_target", "net_worth", "cashflow_forecast"]


def budget_breakdown(
    monthly_income: float,
    expenses: dict[str, float],
    needs_categories: list[str] | None = None,
    detail: str = "summary",
) -> dict:
    """Monthly budget against the 50/30/20 rule.

    monthly_income: take-home (or gross) monthly income. expenses: {category:
    amount}. needs_categories: if given, classify those categories as needs and
    the rest as wants, reporting actual_needs/actual_wants. detail='full' adds a
    per-category breakdown with each category's share of income.
    """
    if monthly_income <= 0:
        raise ValueError("monthly_income must be positive")
    money.validate_detail(detail)

    total = float(sum(money.money(v) for v in expenses.values()))
    surplus_deficit = monthly_income - total
    savings_rate = surplus_deficit / monthly_income

    # Compute the needs classification (if requested) up front so every flag is
    # known before `out` and the summary are assembled.
    actual_needs: float | None = None
    actual_wants: float | None = None
    if needs_categories is not None:
        needs_set = set(needs_categories)
        actual_needs = float(sum(money.money(v) for k, v in expenses.items() if k in needs_set))
        actual_wants = total - actual_needs

    flags: list[str] = []
    if surplus_deficit < 0:
        flags.append("deficit")
    if savings_rate < 0.20:
        flags.append("savings_below_20pct")
    if actual_needs is not None and actual_needs > 0.5 * monthly_income:
        flags.append("needs_over_50pct")

    out = {
        "monthly_income": money.round_cents(monthly_income),
        "total_expenses": money.round_cents(total),
        "surplus_deficit": money.round_cents(surplus_deficit),
        "savings_rate": savings_rate,
        "needs_target": money.round_cents(0.5 * monthly_income),
        "wants_target": money.round_cents(0.3 * monthly_income),
        "savings_target": money.round_cents(0.2 * monthly_income),
        "flags": flags,
    }

    if needs_categories is not None:
        out["actual_needs"] = money.round_cents(actual_needs)
        out["actual_wants"] = money.round_cents(actual_wants)

    out["summary"] = (
        f"On ${monthly_income:,.0f}/mo income, ${money.round_cents(total):,.2f} spent leaves "
        f"${money.round_cents(surplus_deficit):,.2f} ({savings_rate:.0%} savings rate)."
        + (f" Flags: {', '.join(flags)}." if flags else "")
    )

    if detail == "full":
        out["categories"] = [
            {
                "category": k,
                "amount": money.round_cents(v),
                "pct_of_income": v / monthly_income,
            }
            for k, v in expenses.items()
        ]
    return out


def emergency_fund_target(
    essential_monthly_expenses: float,
    months: int = 6,
    current_reserve: float = 0.0,
    monthly_saving: float = 0.0,
) -> dict:
    """How large an emergency fund to hold, and how long to build it.

    essential_monthly_expenses: bare-bones monthly cost of living. months: months
    of runway to cover. current_reserve: what's already saved. monthly_saving:
    amount set aside each month toward the fund. months_to_fund is None when the
    fund is already met or when nothing is being saved.
    """
    if essential_monthly_expenses < 0:
        raise ValueError("essential_monthly_expenses must be non-negative")
    if months <= 0:
        raise ValueError("months must be positive")

    target = essential_monthly_expenses * months
    gap = max(target - current_reserve, 0.0)
    already_funded = gap == 0.0

    if gap > 0 and monthly_saving > 0:
        months_to_fund = math.ceil(gap / monthly_saving)
    else:
        months_to_fund = None

    return {
        "essential_monthly_expenses": money.round_cents(essential_monthly_expenses),
        "months": months,
        "target": money.round_cents(target),
        "current_reserve": money.round_cents(current_reserve),
        "monthly_saving": money.round_cents(monthly_saving),
        "gap": money.round_cents(gap),
        "months_to_fund": months_to_fund,
        "already_funded": already_funded,
        "summary": (
            f"Fully funded: ${money.round_cents(current_reserve):,.2f} already covers {months} months "
            f"of ${essential_monthly_expenses:,.0f} essentials."
            if already_funded
            else (
                f"Target ${money.round_cents(target):,.2f} ({months} months); ${money.round_cents(gap):,.2f} "
                + (
                    f"to go — {months_to_fund} months at ${monthly_saving:,.0f}/mo."
                    if months_to_fund is not None
                    else "to go — set a monthly saving amount to estimate a timeline."
                )
            )
        ),
    }


def net_worth(assets: dict[str, float], liabilities: dict[str, float], detail: str = "summary") -> dict:
    """Net worth = total assets − total liabilities.

    assets / liabilities: {name: amount}. detail='full' adds the per-item
    breakdowns (rounded to cents).
    """
    money.validate_detail(detail)

    total_assets = float(sum(money.money(v) for v in assets.values()))
    total_liabilities = float(sum(money.money(v) for v in liabilities.values()))
    nw = total_assets - total_liabilities

    out = {
        "total_assets": money.round_cents(total_assets),
        "total_liabilities": money.round_cents(total_liabilities),
        "net_worth": money.round_cents(nw),
        "summary": (
            f"Net worth ${money.round_cents(nw):,.2f}: ${money.round_cents(total_assets):,.2f} assets "
            f"less ${money.round_cents(total_liabilities):,.2f} liabilities."
        ),
    }

    if detail == "full":
        out["assets"] = {k: money.round_cents(v) for k, v in assets.items()}
        out["liabilities"] = {k: money.round_cents(v) for k, v in liabilities.items()}
    return out


def cashflow_forecast(
    starting_cash: float,
    monthly_income: float,
    monthly_expenses: float,
    months: int,
    detail: str = "summary",
) -> dict:
    """Walk the cash balance forward month by month.

    Each month the balance moves by (monthly_income − monthly_expenses). Reports
    the ending balance, the lowest balance reached, and how many months end below
    zero. detail='full' adds the per-month rows (month, inflow, outflow, balance).
    """
    if months <= 0:
        raise ValueError("months must be positive")
    money.validate_detail(detail)

    balance = starting_cash
    min_balance: float | None = None
    shortfall_months = 0
    rows: list[dict] = []
    for month in range(1, months + 1):
        balance += monthly_income - monthly_expenses
        min_balance = balance if min_balance is None else min(min_balance, balance)
        if balance < 0:
            shortfall_months += 1
        rows.append(
            {
                "month": month,
                "inflow": money.round_cents(monthly_income),
                "outflow": money.round_cents(monthly_expenses),
                "balance": money.round_cents(balance),
            }
        )

    out = {
        "starting_cash": money.round_cents(starting_cash),
        "monthly_income": money.round_cents(monthly_income),
        "monthly_expenses": money.round_cents(monthly_expenses),
        "months": months,
        "ending_balance": money.round_cents(balance),
        "min_balance": money.round_cents(min_balance),
        "shortfall_months": shortfall_months,
        "summary": (
            f"Over {months} months, balance ends at ${money.round_cents(balance):,.2f} "
            f"(low of ${money.round_cents(min_balance):,.2f})"
            + (f", negative in {shortfall_months} months." if shortfall_months else ", never negative.")
        ),
    }

    if detail == "full":
        out["rows"] = rows
    return out
