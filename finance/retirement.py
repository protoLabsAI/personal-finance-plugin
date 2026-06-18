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

__all__ = ["retirement_projection", "safe_withdrawal"]

_CAVEAT = money.NOT_ADVICE


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
