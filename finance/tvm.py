"""Hand-rolled re-implementation of every numpy-financial function.

Zero dependency on numpy / numpy-financial. Every function derives from one
master time-value-of-money identity (see the numpy-financial docs):

    fv + pv*(1+r)^n + pmt*(1 + r*when)/r * ((1+r)^n - 1) = 0      (r != 0)
    fv + pv + pmt*n = 0                                           (r == 0)

`when` is 1 for begin-of-period cash flows (annuity-due) or 0 for end-of-period
(ordinary, the default). Sign convention (matching numpy_financial): money paid
out is negative, money received is positive.

float throughout, matching numpy_financial (agreement < 1e-6). The high-level
tools translate intuitive positive inputs to/from this convention.
"""

from __future__ import annotations

import math

__all__ = ["fv", "pv", "pmt", "nper", "ipmt", "ppmt", "npv", "rate", "irr", "mirr", "ConvergenceError"]


class ConvergenceError(ValueError):
    """Raised when an iterative solver (rate, irr) cannot find a root — instead
    of silently returning NaN the way numpy_financial does (plan §3)."""


def fv(rate: float, nper: float, pmt: float, pv: float, when: int = 0) -> float:
    """Future value of a cash-flow stream. Closed form."""
    if rate == 0:
        return -(pv + pmt * nper)
    growth = (1 + rate) ** nper
    annuity = pmt * (1 + rate * when) / rate * (growth - 1)
    return -(pv * growth + annuity)


def pv(rate: float, nper: float, pmt: float, fv: float = 0.0, when: int = 0) -> float:
    """Present value of a cash-flow stream. Closed form."""
    if rate == 0:
        return -(fv + pmt * nper)
    growth = (1 + rate) ** nper
    annuity = pmt * (1 + rate * when) / rate * (growth - 1)
    return -(fv + annuity) / growth


def pmt(rate: float, nper: float, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """Level payment per period. Closed form."""
    if rate == 0:
        return -(fv + pv) / nper
    growth = (1 + rate) ** nper
    factor = (1 + rate * when) / rate * (growth - 1)
    return -(fv + pv * growth) / factor


def nper(rate: float, pmt: float, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """Number of periods. Closed form via logarithm; linear when rate == 0."""
    if rate == 0:
        # Match numpy_financial's zero-rate branch verbatim: it returns
        # (pv - fv)/pmt, which differs in sign from the master identity's
        # -(fv+pv)/pmt. A known reference quirk; we pin to the reference.
        return (-fv + pv) / pmt
    z = pmt * (1 + rate * when) / rate
    return math.log((-fv + z) / (pv + z)) / math.log(1 + rate)


def _remaining_balance(rate: float, per: int, pmt: float, pv: float, when: int) -> float:
    """Balance still owed just before payment `per` — the future value after
    (per-1) payments. A renamed `fv` so it doesn't clash with ipmt's `fv` arg
    (mirrors numpy_financial's `_rbl`)."""
    return fv(rate, per - 1, pmt, pv, when)


def ipmt(rate: float, per: int, nper: float, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """Interest portion of payment number `per` (1-indexed).

    Computed from the remaining balance just before the period times the rate.
    Mirrors numpy_financial, including its begin-of-period adjustments.
    """
    total = pmt(rate, nper, pv, fv, when)
    interest = _remaining_balance(rate, per, total, pv, when) * rate
    if when == 1:
        interest /= 1 + rate
        if per == 1:
            interest = 0.0
    return interest


def ppmt(rate: float, per: int, nper: float, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """Principal portion of payment number `per` (1-indexed): pmt - ipmt."""
    return pmt(rate, nper, pv, fv, when) - ipmt(rate, per, nper, pv, fv, when)


def npv(rate: float, values) -> float:
    """Net present value of a cash-flow series. The first value is at t=0 and is
    NOT discounted (matching numpy_financial); value[i] is discounted by i."""
    return sum(value / (1 + rate) ** i for i, value in enumerate(values))


def _tvm_residual(r: float, nper: float, pmt: float, pv: float, fv: float, when: int) -> float:
    """The master TVM identity evaluated at rate `r` — zero at the true rate."""
    if r == 0:
        return fv + pv + pmt * nper
    growth = (1 + r) ** nper
    return fv + pv * growth + pmt * (1 + r * when) / r * (growth - 1)


def rate(
    nper: float,
    pmt: float,
    pv: float,
    fv: float = 0.0,
    when: int = 0,
    guess: float = 0.1,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> float:
    """Periodic interest rate that zeroes the TVM identity.

    Newton-Raphson with an analytic derivative (matching numpy_financial's
    iteration so converging cases agree). Falls back to bisection over a bracket
    when Newton diverges, and raises ConvergenceError if no root is found —
    never a silent NaN.
    """

    def g_div_gp(r: float) -> float:
        # g(r) and its derivative g'(r); returns g/g' for the Newton step.
        t1 = (r + 1) ** nper
        t2 = (r + 1) ** (nper - 1)
        g = fv + t1 * pv + pmt * (t1 - 1) * (r * when + 1) / r
        gp = (
            nper * t2 * pv
            - pmt * (t1 - 1) * (r * when + 1) / (r**2)
            + nper * pmt * t2 * (r * when + 1) / r
            + pmt * (t1 - 1) * when / r
        )
        return g / gp

    rn = guess
    for _ in range(maxiter):
        try:
            step = g_div_gp(rn)
        except (ZeroDivisionError, ValueError, OverflowError):
            break
        rnp1 = rn - step
        if not math.isfinite(rnp1):
            break
        if abs(rnp1 - rn) < tol:
            return rnp1
        rn = rnp1

    return _bisect_rate(nper, pmt, pv, fv, when, tol, maxiter)


def _bisect_rate(nper: float, pmt: float, pv: float, fv: float, when: int, tol: float, maxiter: int) -> float:
    """Bracket the TVM residual's sign change and bisect. Raises if no bracket."""

    def residual(r: float) -> float:
        return _tvm_residual(r, nper, pmt, pv, fv, when)

    # Scan a wide plausible band of periodic rates for a sign change.
    grid = [-0.9999 + 0.0001 * i for i in range(0, 20)] + [0.005 * i for i in range(1, 401)]
    lo = None
    for left, right in zip(grid, grid[1:]):
        try:
            fl, fr = residual(left), residual(right)
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        if not (math.isfinite(fl) and math.isfinite(fr)):
            continue
        if fl == 0:
            return left
        if fl * fr < 0:
            lo, hi = left, right
            break
    if lo is None:
        raise ConvergenceError("rate: no sign change found — the cash flows have no real periodic rate")

    flo = residual(lo)
    for _ in range(maxiter * 4):
        mid = (lo + hi) / 2
        fmid = residual(mid)
        if abs(fmid) < tol or (hi - lo) / 2 < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    raise ConvergenceError("rate: bisection did not converge")


def irr(values, guess: float = 0.1, tol: float = 1e-9, maxiter: int = 100) -> float:
    """Internal rate of return: the rate r where npv(r, values) == 0.

    Newton's method with the analytic NPV derivative, falling back to bracketed
    bisection, and raising ConvergenceError when there is no real root (e.g. all
    cash flows share a sign) — never a silent NaN.
    """
    values = list(values)

    def f(r: float) -> float:
        return npv(r, values)

    def fprime(r: float) -> float:
        return sum(-i * v / (1 + r) ** (i + 1) for i, v in enumerate(values))

    rn = guess
    for _ in range(maxiter):
        try:
            denom = fprime(rn)
            if denom == 0:
                break
            rnp1 = rn - f(rn) / denom
        except (ZeroDivisionError, OverflowError, ValueError):
            break
        if not math.isfinite(rnp1) or rnp1 <= -1:
            break
        if abs(rnp1 - rn) < tol:
            return rnp1
        rn = rnp1

    # Bisection fallback: scan for a sign change in NPV(r) above r = -1.
    grid = [-0.999 + 0.001 * i for i in range(0, 100)] + [0.1 * i for i in range(1, 101)]
    lo = hi = None
    for left, right in zip(grid, grid[1:]):
        fl, fr = f(left), f(right)
        if fl == 0:
            return left
        if math.isfinite(fl) and math.isfinite(fr) and fl * fr < 0:
            lo, hi = left, right
            break
    if lo is None:
        raise ConvergenceError("irr: no sign change in NPV — the cash flows have no real internal rate of return")

    flo = f(lo)
    for _ in range(maxiter * 4):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) / 2 < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    raise ConvergenceError("irr: bisection did not converge")


def mirr(values, finance_rate: float, reinvest_rate: float) -> float:
    """Modified internal rate of return.

    Outflows (negative) are discounted to t=0 at `finance_rate`; inflows
    (positive) are compounded to the final period at `reinvest_rate`. Requires
    at least one inflow and one outflow, else there is no MIRR (numpy_financial
    returns NaN; we raise).
    """
    values = list(values)
    n = len(values)
    positives = [v if v > 0 else 0 for v in values]
    negatives = [v if v < 0 else 0 for v in values]
    if not (any(positives) and any(negatives)):
        raise ValueError("mirr: requires at least one positive and one negative cash flow")
    numer = abs(npv(reinvest_rate, positives))
    denom = abs(npv(finance_rate, negatives))
    return (numer / denom) ** (1 / (n - 1)) * (1 + reinvest_rate) - 1
