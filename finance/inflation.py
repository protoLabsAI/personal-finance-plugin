"""Inflation adjustment (plan §4H). Pure function over plain numbers; the @tool
shim in tools.py wraps this. Returns a typed dict of figures plus a short
`summary`.

Two directions: discount a future amount back into today's purchasing power
('to_present'), or grow a present amount into the nominal figure you'd need
later to preserve that purchasing power ('to_future').
"""

from __future__ import annotations

from . import money

__all__ = ["inflation_adjust"]


def inflation_adjust(amount: float, rate: float, years: float, direction: str = "to_present") -> dict:
    """Translate `amount` between today's and future dollars at inflation `rate`.

    amount: the figure to adjust. rate: annual inflation rate (0.03 == 3%).
    years: number of years between now and then. direction: 'to_present' gives
    the real (today's-dollars) value of a future amount; 'to_future' gives the
    nominal amount you'll need later to match today's purchasing power.
    """
    if direction not in ("to_future", "to_present"):
        raise ValueError("direction must be 'to_future' or 'to_present'")
    if years < 0:
        raise ValueError("years must be non-negative")
    if rate <= -1:
        raise ValueError("rate must be greater than -1 (a deflation of -100% or more is undefined)")

    factor = (1 + rate) ** years
    adjusted = amount * factor if direction == "to_future" else amount / factor

    if direction == "to_future":
        summary = (
            f"To match the purchasing power of ${amount:,.2f} today, you'd need "
            f"${money.round_cents(adjusted):,.2f} in {years:g} years at {rate:.1%} inflation."
        )
    else:
        summary = (
            f"${amount:,.2f} in {years:g} years is worth ${money.round_cents(adjusted):,.2f} "
            f"in today's dollars at {rate:.1%} inflation."
        )

    return {
        "amount": amount,
        "rate": rate,
        "years": years,
        "direction": direction,
        "adjusted": money.round_cents(adjusted),
        "summary": summary,
    }
