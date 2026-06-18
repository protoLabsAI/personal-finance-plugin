"""Income-disruption planning (plan §4G). Pure functions over plain numbers; the
@tool shims in tools.py wrap these. Each returns a typed dict of figures plus a
short `summary`; the month-by-month scenario gates behind a `detail` flag
(plan §1/§5).

These are planning projections under stated assumptions, not financial advice —
each summary says so, in keeping with the plan's honest-framing rule (§1).
"""

from __future__ import annotations

import math

from . import money

__all__ = ["income_disruption_runway", "income_replacement_need"]

_NOT_ADVICE = money.NOT_ADVICE

# Cap the simulation so a perpetually-solvent scenario terminates (plan §4G).
_MAX_MONTHS = 600

# A weekly benefit spread across an average month, and weeks expressed in months.
_WEEKS_PER_MONTH = 52 / 12
_MONTHS_PER_WEEK = 12 / 52


def income_disruption_runway(
    liquid_savings: float,
    essential_monthly_burn: float,
    severance: float = 0.0,
    weekly_benefit: float = 0.0,
    max_benefit_weeks: float = 0,
    other_monthly_income: float = 0.0,
    discretionary_cut_pct: float = 0.0,
    detail: str = "summary",
) -> dict:
    """Simulate how long liquid savings last after an income disruption.

    liquid_savings: cash on hand. essential_monthly_burn: must-pay spending each
    month. severance: lump sum received up front (added to the starting balance).
    weekly_benefit: unemployment benefit per week, paid while eligibility weeks
    remain. max_benefit_weeks: total weeks of benefit eligibility.
    other_monthly_income: any ongoing income (side work, partner, etc.).
    discretionary_cut_pct: fraction (0..1) trimmed from essential burn.
    detail='full' adds a per-month `scenario` table (month, inflow, burn, balance).

    The benefit is converted to a monthly figure (weekly_benefit * 52/12) and paid
    for floor(benefit_months) full months plus one partial month for the fractional
    remainder, where benefit_months = max_benefit_weeks * 12/52.

    `monthly_net_burn` is the steady monthly drawdown while the benefit is active:
    reduced_burn - other_monthly_income - benefit_monthly. `months_of_runway` is
    the number of whole months the balance stays non-negative; None means the
    balance never goes negative within the horizon (indefinite).
    """
    if liquid_savings < 0:
        raise ValueError("liquid_savings must be non-negative")
    if essential_monthly_burn < 0:
        raise ValueError("essential_monthly_burn must be non-negative")
    if not 0 <= discretionary_cut_pct <= 1:
        raise ValueError("discretionary_cut_pct must be between 0 and 1")
    money.validate_detail(detail)

    reduced_burn = essential_monthly_burn * (1 - discretionary_cut_pct)
    benefit_monthly = weekly_benefit * _WEEKS_PER_MONTH
    benefit_months = max_benefit_weeks * _MONTHS_PER_WEEK
    monthly_net_burn = reduced_burn - other_monthly_income - benefit_monthly

    starting_balance = liquid_savings + severance
    full_benefit_months = math.floor(benefit_months)
    fractional = benefit_months - full_benefit_months

    def _benefit_for(month: int) -> float:
        if month <= full_benefit_months:
            return benefit_monthly
        if month == full_benefit_months + 1 and fractional > 0:
            return benefit_monthly * fractional
        return 0.0

    # Walk the running balance to find the runway; the per-month rows are only
    # materialized for detail='full' (mirrors loans.py building its schedule
    # only when asked). The balance walk is identical in both modes, so the
    # figures match byte-for-byte.
    balance = starting_balance
    months_of_runway: int | None = None
    for month in range(1, _MAX_MONTHS + 1):
        inflow = other_monthly_income + _benefit_for(month)
        balance += inflow - reduced_burn
        if balance < 0:
            months_of_runway = month - 1
            break

    if months_of_runway is None:
        runway_phrase = "indefinite — income covers the reduced burn"
    else:
        runway_phrase = f"{months_of_runway} month{'s' if months_of_runway != 1 else ''}"

    out = {
        "months_of_runway": months_of_runway,
        "monthly_net_burn": money.round_cents(monthly_net_burn),
        "reduced_monthly_burn": money.round_cents(reduced_burn),
        "benefit_monthly": money.round_cents(benefit_monthly),
        "benefit_months": benefit_months,
        "starting_balance": money.round_cents(starting_balance),
        "summary": (
            f"Runway: {runway_phrase}. Reduced burn ${money.round_cents(reduced_burn):,.2f}/mo, "
            f"benefit ${money.round_cents(benefit_monthly):,.2f}/mo for {benefit_months:g} months. "
            f"{_NOT_ADVICE}"
        ),
    }
    if detail == "full":
        # Rebuild the same walk, recording one row per simulated month up to and
        # including the month the balance crosses below zero.
        rows: list[dict] = []
        balance = starting_balance
        for month in range(1, _MAX_MONTHS + 1):
            inflow = other_monthly_income + _benefit_for(month)
            balance += inflow - reduced_burn
            rows.append(
                {
                    "month": month,
                    "inflow": money.round_cents(inflow),
                    "burn": money.round_cents(reduced_burn),
                    "balance": money.round_cents(balance),
                }
            )
            if balance < 0:
                break
        out["scenario"] = rows
    return out


def income_replacement_need(
    prior_monthly_takehome: float,
    benefits_monthly: float = 0.0,
    other_monthly_income: float = 0.0,
    desired_coverage_months: float = 6,
) -> dict:
    """Size the income gap and emergency fund needed to cover a disruption.

    prior_monthly_takehome: net pay before the disruption. benefits_monthly: any
    unemployment/disability benefit. other_monthly_income: other ongoing income.
    desired_coverage_months: how many months of the gap to hold in reserve.

    monthly_income_gap is the shortfall after replacement income (floored at 0);
    emergency_fund_needed covers that gap for the desired months; replacement_ratio
    is the share of prior take-home covered by benefits + other income.
    """
    if prior_monthly_takehome <= 0:
        raise ValueError("prior_monthly_takehome must be positive")
    if desired_coverage_months <= 0:
        raise ValueError("desired_coverage_months must be positive")

    replacement_income = benefits_monthly + other_monthly_income
    monthly_income_gap = max(prior_monthly_takehome - replacement_income, 0.0)
    emergency_fund_needed = monthly_income_gap * desired_coverage_months
    replacement_ratio = replacement_income / prior_monthly_takehome

    return {
        "prior_monthly_takehome": money.round_cents(prior_monthly_takehome),
        "monthly_income_gap": money.round_cents(monthly_income_gap),
        "emergency_fund_needed": money.round_cents(emergency_fund_needed),
        "replacement_ratio": replacement_ratio,
        "desired_coverage_months": desired_coverage_months,
        "summary": (
            f"Monthly income gap ${money.round_cents(monthly_income_gap):,.2f}; hold "
            f"${money.round_cents(emergency_fund_needed):,.2f} to cover {desired_coverage_months:g} months. "
            f"Replacement income covers {replacement_ratio:.1%} of prior take-home. {_NOT_ADVICE}"
        ),
    }
