"""Retirement planning tools (plan §4D, v1 subset). Pure functions over plain
numbers, built on the TVM engine; the @tool shims in tools.py wrap these. Each
returns a typed dict of figures plus a short `summary` string.

These are PROJECTIONS under stated assumptions, not financial advice (plan §1).
Every modeling tool echoes its inputs in an `assumptions` dict and the `summary`
carries that caveat so the honest framing travels with the numbers.

Sign convention is handled internally: callers pass intuitive positive figures
(savings, contributions, income) and never see the TVM cash-flow signs.
"""

from __future__ import annotations

from . import money, tvm

__all__ = ["retirement_projection", "safe_withdrawal", "rmd", "social_security_estimate"]

_CAVEAT = money.NOT_ADVICE

# ── IRS Uniform Lifetime Table (effective 2022+, SECURE 2.0) ────────────────────
# Source: IRS Publication 590-B, Table III (Uniform Lifetime).
# Ages 120+ all use 2.0 (captured in the lookup logic below).
_RMD_UNIFORM_TABLE: dict[int, float] = {
    72: 27.4,
    73: 26.5,
    74: 25.5,
    75: 24.6,
    76: 23.7,
    77: 22.9,
    78: 22.0,
    79: 21.1,
    80: 20.2,
    81: 19.4,
    82: 18.5,
    83: 17.7,
    84: 16.8,
    85: 16.0,
    86: 15.2,
    87: 14.4,
    88: 13.7,
    89: 12.9,
    90: 12.2,
    91: 11.5,
    92: 10.8,
    93: 10.1,
    94: 9.5,
    95: 8.9,
    96: 8.4,
    97: 7.8,
    98: 7.3,
    99: 6.8,
    100: 6.4,
    101: 6.0,
    102: 5.6,
    103: 5.2,
    104: 4.9,
    105: 4.6,
    106: 4.3,
    107: 4.1,
    108: 3.9,
    109: 3.7,
    110: 3.5,
    111: 3.4,
    112: 3.3,
    113: 3.1,
    114: 3.0,
    115: 2.9,
    116: 2.8,
    117: 2.7,
    118: 2.5,
    119: 2.3,
    120: 2.0,
}

# ── Social Security PIA bend-point constants (2024 values) ──────────────────────
_SS_BEND_1 = 1174  # 2024 first bend point (monthly AIME, dollars)
_SS_BEND_2 = 7078  # 2024 second bend point (monthly AIME, dollars)
_SS_RATE_1 = 0.90  # 90% of AIME up to first bend point
_SS_RATE_2 = 0.32  # 32% of AIME between first and second bend points
_SS_RATE_3 = 0.15  # 15% of AIME above second bend point


def _sustainable_income(rate: float, years: float, balance: float) -> float:
    """Level annual withdrawal that depletes `balance` over `years` at `rate`,
    returned as a positive figure. The balance is money on hand (pv paid in,
    negative); withdrawals come back positive."""
    return -tvm.pmt(rate, years, balance)


def _depletion_age(rate: float, retirement_age: int, years_in_retirement: float, balance: float, draw: float):
    """Age at which the portfolio runs dry when drawing `draw` per year, or None
    if the funds outlast `life_expectancy`. tvm.nper raises ValueError when the
    draw never depletes the balance (it sits at or below the sustainable level);
    we treat that — and any depletion past the horizon — as "funds outlast"."""
    if draw <= 0:
        return None
    if rate == 0:
        # tvm.nper's zero-rate branch returns a negative number here (a
        # documented numpy quirk); the time to deplete a balance with no growth
        # is simply balance / draw.
        years_to_deplete = balance / draw
    else:
        try:
            years_to_deplete = tvm.nper(rate, draw, -balance)
        except ValueError:
            return None
    if years_to_deplete <= 0 or years_to_deplete >= years_in_retirement:
        return None
    # round to 1 decimal place: this is an AGE (in years), not a currency amount.
    age = round(retirement_age + years_to_deplete, 1)
    # If rounding pushes the age to/past life expectancy, the funds effectively
    # last a lifetime — report None rather than a depletion age tying the death
    # age, which would contradict the funds-outlast framing.
    if age >= retirement_age + years_in_retirement:
        return None
    return age


def retirement_projection(
    current_age: int,
    retirement_age: int,
    life_expectancy: int,
    current_savings: float,
    annual_contribution: float,
    expected_return: float,
    desired_annual_income: float,
    inflation_rate: float = 0.0,
    social_security: dict | None = None,
) -> dict:
    """Project a retirement: accumulate savings to retirement, then draw them down.

    ACCUMULATION grows `current_savings` plus `annual_contribution` each year at
    `expected_return` to a balance at `retirement_age`. DRAWDOWN reports the
    level annual income that balance sustains over the retirement horizon, the
    surplus/shortfall against `desired_annual_income`, and the age the portfolio
    depletes if `desired_annual_income` is drawn (None if funds outlast).

    social_security: optional {annual_benefit, claim_age, include}. When include
    is True and `retirement_age` has reached `claim_age`, the benefit offsets the
    income the portfolio must provide. inflation_rate: if > 0, also report a
    `real` view of the drawdown in today's dollars via the real return.
    """
    if not (current_age < retirement_age < life_expectancy):
        raise ValueError("ages must satisfy current_age < retirement_age < life_expectancy")
    if current_savings < 0:
        raise ValueError("current_savings must be non-negative")
    if annual_contribution < 0:
        raise ValueError("annual_contribution must be non-negative")
    if desired_annual_income < 0:
        raise ValueError("desired_annual_income must be non-negative")

    years_to_retire = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age

    # Contributions and current savings are paid in (negative) → balance positive.
    balance_at_retirement = tvm.fv(expected_return, years_to_retire, -annual_contribution, -current_savings)
    sustainable_annual_income = _sustainable_income(expected_return, years_in_retirement, balance_at_retirement)

    if social_security and social_security.get("include"):
        missing = [k for k in ("annual_benefit", "claim_age") if k not in social_security]
        if missing:
            raise ValueError(f"social_security with include=True requires: {', '.join(missing)}")
    ss_included = bool(
        social_security and social_security.get("include") and retirement_age >= social_security["claim_age"]
    )
    social_security_income = float(social_security["annual_benefit"]) if ss_included else 0.0

    total_sustainable = sustainable_annual_income + social_security_income
    surplus_or_shortfall = total_sustainable - desired_annual_income

    # Depletion: only the portfolio funds the income SS doesn't cover.
    portfolio_draw = desired_annual_income - social_security_income
    depletion_age = _depletion_age(
        expected_return, retirement_age, years_in_retirement, balance_at_retirement, portfolio_draw
    )

    assumptions = {
        "current_age": current_age,
        "retirement_age": retirement_age,
        "life_expectancy": life_expectancy,
        "expected_return": expected_return,
        "inflation_rate": inflation_rate,
        "social_security_included": ss_included,
    }

    if depletion_age is None:
        outlook = f"funds outlast age {life_expectancy} at the desired draw"
    else:
        outlook = f"funds deplete around age {depletion_age:g}"
    if surplus_or_shortfall >= 0:
        gap = f"a ${money.round_cents(surplus_or_shortfall):,.0f} annual surplus"
    else:
        gap = f"a ${money.round_cents(-surplus_or_shortfall):,.0f} annual shortfall"

    out = {
        "balance_at_retirement": money.round_cents(balance_at_retirement),
        "years_in_retirement": years_in_retirement,
        "sustainable_annual_income": money.round_cents(sustainable_annual_income),
        "social_security_income": money.round_cents(social_security_income),
        "desired_annual_income": money.round_cents(desired_annual_income),
        "surplus_or_shortfall": money.round_cents(surplus_or_shortfall),
        "depletion_age": depletion_age,
        "assumptions": assumptions,
        "summary": (
            f"By age {retirement_age}, ${money.round_cents(balance_at_retirement):,.0f} sustains "
            f"${money.round_cents(total_sustainable):,.0f}/yr over {years_in_retirement} years "
            f"(incl. ${money.round_cents(social_security_income):,.0f} Social Security) vs "
            f"${desired_annual_income:,.0f} desired — {gap}; {outlook}. {_CAVEAT}"
        ),
    }

    if inflation_rate > 0:
        real_return = (1 + expected_return) / (1 + inflation_rate) - 1
        real_sustainable = _sustainable_income(real_return, years_in_retirement, balance_at_retirement)
        assumptions["real_return"] = real_return
        out["real"] = {
            "real_return": real_return,
            "sustainable_annual_income": money.round_cents(real_sustainable),
            "surplus_or_shortfall": money.round_cents(
                real_sustainable + social_security_income - desired_annual_income
            ),
            "note": "desired_annual_income treated as today's dollars; figures in today's purchasing power",
        }

    return out


def safe_withdrawal(
    nest_egg: float | None = None,
    target_income: float | None = None,
    rate: float = 0.04,
    horizon: int = 30,
) -> dict:
    """Apply the 4%/Trinity safe-withdrawal rule of thumb.

    Provide EXACTLY ONE of: nest_egg (→ sustainable_income = nest_egg * rate) or
    target_income (→ required_nest_egg = target_income / rate). rate is the
    annual withdrawal rate (0.04 == the classic 4% rule).

    `horizon` is the planning horizon in years the Trinity study assumes, carried
    here as a documented assumption only: this rule of thumb is HORIZON-INDEPENDENT
    as implemented (the figure is rate * balance or income / rate and does not vary
    with horizon). It is echoed under `assumptions` and the summary for honest
    framing, not used in the computation.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if (nest_egg is None) == (target_income is None):
        raise ValueError("provide exactly one of nest_egg or target_income")

    # rate/horizon are echoed ONLY under assumptions (matching retirement_projection).
    assumptions = {"rate": rate, "horizon": horizon, "rule": "4%/Trinity safe-withdrawal rule of thumb"}

    if nest_egg is not None:
        if nest_egg < 0:
            raise ValueError("nest_egg must be non-negative")
        sustainable_income = nest_egg * rate
        mode = "nest_egg"
        result = {
            "nest_egg": money.round_cents(nest_egg),
            "sustainable_income": money.round_cents(sustainable_income),
        }
        summary = (
            f"At a {rate:.1%} withdrawal rate, ${nest_egg:,.0f} supports about "
            f"${money.round_cents(sustainable_income):,.0f}/yr over a ~{horizon}-year horizon "
            f"(4%/Trinity rule, horizon-independent). {_CAVEAT}"
        )
    else:
        if target_income < 0:
            raise ValueError("target_income must be non-negative")
        required_nest_egg = target_income / rate
        mode = "target_income"
        result = {
            "target_income": money.round_cents(target_income),
            "required_nest_egg": money.round_cents(required_nest_egg),
        }
        summary = (
            f"To draw ${target_income:,.0f}/yr at a {rate:.1%} withdrawal rate you need about "
            f"${money.round_cents(required_nest_egg):,.0f} over a ~{horizon}-year horizon "
            f"(4%/Trinity rule, horizon-independent). {_CAVEAT}"
        )

    return {"mode": mode, **result, "assumptions": assumptions, "summary": summary}


def rmd(age: int, account_balance: float) -> dict:
    """Required Minimum Distribution using the IRS Uniform Lifetime Table (2022+).

    RMD = account_balance / distribution_period[age]. Under SECURE 2.0, RMDs begin
    at age 73; for younger ages the function returns rmd=0.0 (not an error). Ages
    above 120 use the distribution period for age 120 (2.0).

    Raises ValueError if age < 0 or account_balance < 0.
    """
    if age < 0:
        raise ValueError("age must be non-negative")
    if account_balance < 0:
        raise ValueError("account_balance must be non-negative")

    if age < 73:
        return {
            "age": age,
            "account_balance": money.round_cents(account_balance),
            "distribution_period": None,
            "rmd": 0.0,
            "summary": (f"No RMD required at age {age}. Under SECURE 2.0, RMDs begin at age 73. {_CAVEAT}"),
        }

    # Ages > 120 use the 120 entry (period 2.0).
    lookup_age = min(age, 120)
    distribution_period = _RMD_UNIFORM_TABLE[lookup_age]
    rmd_amount = money.round_cents(account_balance / distribution_period)

    return {
        "age": age,
        "account_balance": money.round_cents(account_balance),
        "distribution_period": distribution_period,
        "rmd": rmd_amount,
        "summary": (
            f"At age {age} with a ${money.round_cents(account_balance):,.2f} balance, "
            f"your RMD is ${rmd_amount:,.2f} (IRS Uniform Lifetime Table period: {distribution_period}). "
            f"{_CAVEAT}"
        ),
    }


def social_security_estimate(
    aime: float,
    claim_age: int = 67,
    full_retirement_age: int = 67,
) -> dict:
    """Simplified Social Security benefit estimate using 2024 PIA bend-point formula.

    aime: Average Indexed Monthly Earnings (monthly dollars).
    claim_age: age at which you claim (62–70, inclusive).
    full_retirement_age: your FRA (65–67, inclusive).

    PIA (Primary Insurance Amount) is the benefit at FRA:
        PIA = 90% * min(aime, bend1) + 32% * clamp(aime-bend1, 0, bend2-bend1)
              + 15% * max(aime-bend2, 0)

    Claiming before FRA reduces the benefit; claiming after FRA (up to 70) increases
    it by the delayed retirement credit (8%/year, i.e. 2/3% per month).
    """
    if aime < 0:
        raise ValueError("aime must be non-negative")
    if not (62 <= claim_age <= 70):
        raise ValueError("claim_age must be between 62 and 70 (inclusive)")
    if not (65 <= full_retirement_age <= 67):
        raise ValueError("full_retirement_age must be between 65 and 67 (inclusive)")

    # PIA calculation using 2024 bend points.
    pia = (
        _SS_RATE_1 * min(aime, _SS_BEND_1)
        + _SS_RATE_2 * max(0.0, min(aime - _SS_BEND_1, _SS_BEND_2 - _SS_BEND_1))
        + _SS_RATE_3 * max(0.0, aime - _SS_BEND_2)
    )

    # Claim-age adjustment relative to FRA.
    months_diff = (claim_age - full_retirement_age) * 12

    if months_diff == 0:
        # Claiming at FRA: no adjustment.
        adjustment_factor = 1.0
    elif months_diff > 0:
        # Claiming after FRA: delayed retirement credit of 2/3% per month, capped at age 70.
        months_late = min(months_diff, (70 - full_retirement_age) * 12)
        adjustment_factor = 1.0 + months_late * (2 / 3 / 100)
    else:
        # Claiming before FRA: first 36 months at 5/9%, beyond 36 at 5/12%.
        months_early = -months_diff
        reduction_first_36 = min(months_early, 36) * (5 / 9 / 100)
        reduction_beyond_36 = max(0, months_early - 36) * (5 / 12 / 100)
        adjustment_factor = 1.0 - (reduction_first_36 + reduction_beyond_36)

    monthly_benefit = money.round_cents(pia * adjustment_factor)
    annual_benefit = money.round_cents(monthly_benefit * 12)

    if months_diff > 0:
        timing = f"claiming at {claim_age} (delayed, +{(adjustment_factor - 1):.1%} vs FRA)"
    elif months_diff < 0:
        timing = f"claiming at {claim_age} (early, {(adjustment_factor - 1):.1%} vs FRA)"
    else:
        timing = f"claiming at FRA ({claim_age})"

    return {
        "aime": aime,
        "pia": money.round_cents(pia),
        "claim_age": claim_age,
        "full_retirement_age": full_retirement_age,
        "monthly_benefit": monthly_benefit,
        "annual_benefit": annual_benefit,
        "summary": (
            f"With an AIME of ${aime:,.0f}/mo, your estimated PIA is ${money.round_cents(pia):,.2f}/mo; "
            f"{timing} yields ${monthly_benefit:,.2f}/mo (${annual_benefit:,.2f}/yr). "
            f"This is a simplified estimate using 2024 bend points. {_CAVEAT}"
        ),
    }
