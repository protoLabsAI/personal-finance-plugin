"""Shared helpers: payment-frequency maps, periodic-rate conversion, and money
rounding. Kept dependency-free and separate from the TVM engine so every
domain module (loans, growth, budget, …) converts units the same way.

Precision policy (plan §3): the TVM engine works in float; currency
*aggregation* uses Decimal to avoid drift when summing many figures; outputs
round to cents only at the edge via `round_cents`.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

__all__ = [
    "PERIODS_PER_YEAR",
    "periods_per_year",
    "periodic_rate",
    "round_cents",
    "money",
    "validate_detail",
    "inflation_factor",
    "NOT_ADVICE",
]

# How many compounding/payment periods a given frequency has in one year.
PERIODS_PER_YEAR: dict[str, int] = {
    "annual": 1,
    "semiannual": 2,
    "quarterly": 4,
    "monthly": 12,
    "biweekly": 26,
    "weekly": 52,
    "daily": 365,
}


def periods_per_year(freq: str) -> int:
    """Periods per year for a named frequency (case-insensitive)."""
    try:
        return PERIODS_PER_YEAR[freq.lower()]
    except (KeyError, AttributeError):
        valid = ", ".join(PERIODS_PER_YEAR)
        raise ValueError(f"unknown frequency {freq!r}; expected one of: {valid}") from None


def periodic_rate(annual_rate: float, freq: str) -> float:
    """Convert a nominal annual rate to the per-period rate for `freq`."""
    return annual_rate / periods_per_year(freq)


def round_cents(value: float) -> float:
    """Round to the nearest cent, half-up, returned as a float for output.

    Uses Decimal(str(value)) so the rounding sees the decimal the caller wrote
    (e.g. 1.005 → 1.01), not the float's binary approximation.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def money(value) -> Decimal:
    """A Decimal for exact currency aggregation. Pass through int/float/str via
    str() so float inputs use their written form, not the binary approximation."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


# The honest-framing disclaimer the modeling tools append to their summaries
# (plan §1). Single-sourced so the wording stays consistent across tools.
NOT_ADVICE = "Projection under stated assumptions, not financial advice."


def validate_detail(detail: str) -> None:
    """Validate a tool's output-verbosity flag. Shared so every tool gates its
    large output the same way (plan §5)."""
    if detail not in ("summary", "full"):
        raise ValueError("detail must be 'summary' or 'full'")


def inflation_factor(rate: float, years: float) -> float:
    """The compounding factor ``(1 + rate) ** years`` used to move money between
    today's and future dollars. Rejects rate <= -1 (a deflation of -100% or more
    is undefined) so callers get a clear error, not a complex/zero result."""
    if rate <= -1:
        raise ValueError("rate must be greater than -1 (a deflation of -100% or more is undefined)")
    return (1 + rate) ** years
