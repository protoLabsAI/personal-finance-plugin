"""Golden tests for the hand-rolled TVM engine (finance/tvm.py).

Every value is pinned to numpy_financial's documented output (the reference the
engine re-implements from first principles). numpy_financial is a DEV-only
cross-check — never a runtime dep — so we hardcode its numbers as literals here.
Generated via:  python -c "import numpy_financial as nf; print(nf.fv(...))"
"""

from __future__ import annotations

import importlib

import pytest

# Tolerance: numpy_financial uses float; the docs promise agreement < 1e-6.
TOL = 1e-6


@pytest.fixture
def tvm(plugin):
    """The engine module, loaded as a submodule of the plugin package."""
    return importlib.import_module(plugin.__name__ + ".finance.tvm")


# ── fv ────────────────────────────────────────────────────────────────────────
def test_fv_ordinary(tvm):
    assert tvm.fv(0.05 / 12, 10 * 12, -100, -100) == pytest.approx(15692.928894335748, abs=TOL)


def test_fv_zero_rate(tvm):
    assert tvm.fv(0, 12, -100, -100) == pytest.approx(1300.0, abs=TOL)


def test_fv_annuity_due(tvm):
    assert tvm.fv(0.05 / 12, 10 * 12, -100, -100, when=1) == pytest.approx(15757.629844104778, abs=TOL)


def test_fv_rejects_rate_at_or_below_minus_one(tvm):
    # rate <= -1 (periodic rate of -100% or worse) is undefined; with a
    # non-integer nper the old code silently returned a complex number.
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.fv(-1, 2.5, -100, -100)
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.fv(-2, 2.5, -100, -100)


# ── pv ────────────────────────────────────────────────────────────────────────
def test_pv_ordinary(tvm):
    assert tvm.pv(0.05 / 12, 10 * 12, -100, 15692.93) == pytest.approx(-100.00067131625819, abs=TOL)


def test_pv_zero_rate(tvm):
    assert tvm.pv(0, 12, -100, 1200) == pytest.approx(0.0, abs=TOL)


def test_pv_rejects_rate_at_or_below_minus_one(tvm):
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.pv(-1, 2.5, -100, 100)
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.pv(-2, 2.5, -100, 100)


# ── pmt ───────────────────────────────────────────────────────────────────────
def test_pmt_mortgage(tvm):
    assert tvm.pmt(0.075 / 12, 12 * 15, 200000) == pytest.approx(-1854.0247200054619, abs=TOL)


def test_pmt_short_loan(tvm):
    assert tvm.pmt(0.0824 / 12, 12, 2500) == pytest.approx(-217.74859035344943, abs=TOL)


def test_pmt_zero_rate(tvm):
    assert tvm.pmt(0, 12, 2400) == pytest.approx(-200.0, abs=TOL)


def test_pmt_rejects_rate_at_or_below_minus_one(tvm):
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.pmt(-1, 2.5, 2500)
    with pytest.raises(ValueError, match="rate must be greater than -1"):
        tvm.pmt(-2, 2.5, 2500)


# ── nper ──────────────────────────────────────────────────────────────────────
def test_nper_ordinary(tvm):
    assert tvm.nper(0.07 / 12, -150, 8000) == pytest.approx(64.07334877066185, abs=TOL)


def test_nper_zero_rate(tvm):
    assert tvm.nper(0, -100, 1000) == pytest.approx(-10.0, abs=TOL)


def test_nper_unreachable_goal_raises_clean_error(tvm):
    # Payment below the periodic interest never amortizes: must raise a clear,
    # domain-specific error, not a cryptic "math domain error" from math.log.
    with pytest.raises(ValueError, match="never"):
        tvm.nper(0.05 / 12, -10, 10000)


# ── ipmt / ppmt ───────────────────────────────────────────────────────────────
def test_ipmt_mid_period(tvm):
    assert tvm.ipmt(0.05, 3, 10, -1000) == pytest.approx(41.850781066040696, abs=TOL)


def test_ppmt_mid_period(tvm):
    assert tvm.ppmt(0.05, 3, 10, -1000) == pytest.approx(87.65379389941594, abs=TOL)


def test_ipmt_first_period(tvm):
    # First-period interest on an ordinary annuity is just rate * pv (signed).
    assert tvm.ipmt(0.05, 1, 10, -1000) == pytest.approx(50.0, abs=TOL)


def test_ipmt_last_period(tvm):
    assert tvm.ipmt(0.05, 10, 10, -1000) == pytest.approx(6.166884522164594, abs=TOL)


def test_ipmt_plus_ppmt_equals_pmt_every_period(tvm):
    p = tvm.pmt(0.05, 10, -1000)
    for per in range(1, 11):
        total = tvm.ipmt(0.05, per, 10, -1000) + tvm.ppmt(0.05, per, 10, -1000)
        assert total == pytest.approx(p, abs=TOL)


# ── npv ───────────────────────────────────────────────────────────────────────
def test_npv_basic(tvm):
    assert tvm.npv(0.08, [-40000, 5000, 8000, 12000, 30000]) == pytest.approx(3065.2226681795255, abs=TOL)


def test_npv_near_irr_is_near_zero(tvm):
    # The series' IRR is ~0.281, so NPV at 0.281 is ~0 (first flow undiscounted).
    assert tvm.npv(0.281, [-100, 39, 59, 55, 20]) == pytest.approx(-0.00847859163845488, abs=TOL)


def test_npv_zero_rate_is_plain_sum(tvm):
    assert tvm.npv(0, [-100, 39, 59, 55, 20]) == pytest.approx(73.0, abs=TOL)


# ── rate ──────────────────────────────────────────────────────────────────────
def test_rate_savings(tvm):
    assert tvm.rate(12, -100, 1000, 0) == pytest.approx(0.029228540769158533, abs=TOL)


def test_rate_loan(tvm):
    assert tvm.rate(60, -200, 8000, 0) == pytest.approx(0.014394781000914082, abs=TOL)


def test_rate_mortgage(tvm):
    assert tvm.rate(360, -1854.02, 200000, 0) == pytest.approx(0.008886526827653955, abs=TOL)


def test_rate_round_trips_with_pmt(tvm):
    # A payment computed at a known rate should recover that rate.
    monthly = 0.06 / 12
    payment = tvm.pmt(monthly, 240, 150000)
    assert tvm.rate(240, payment, 150000, 0) == pytest.approx(monthly, abs=TOL)


def test_rate_raises_when_no_root(tvm):
    # All same-sign cash flows have no rate that zeroes the TVM equation.
    # numpy_financial returns NaN here; we raise instead (plan: no silent NaN).
    with pytest.raises(ValueError):
        tvm.rate(10, 100, 100, 100)


def test_rate_no_root_with_fractional_nper_raises_cleanly(tvm):
    # A no-root case where Newton can step below -1 and (1+r)**nper goes complex:
    # must surface as ConvergenceError, never an uncaught TypeError.
    with pytest.raises(tvm.ConvergenceError):
        tvm.rate(2.5, 100, 100, 100)


# ── irr ───────────────────────────────────────────────────────────────────────
def test_irr_basic(tvm):
    assert tvm.irr([-100, 39, 59, 55, 20]) == pytest.approx(0.28094842115996066, abs=TOL)


def test_irr_project(tvm):
    assert tvm.irr([-4000, 1200, 1410, 1875, 1050]) == pytest.approx(0.14299344106065282, abs=TOL)


def test_irr_zeroes_npv(tvm):
    values = [-4000, 1200, 1410, 1875, 1050]
    assert tvm.npv(tvm.irr(values), values) == pytest.approx(0.0, abs=TOL)


def test_irr_finds_root_in_former_grid_gap(tvm):
    # [10,-7,1] has two NPV roots (~-0.5 and ~-0.8) that fell in the old grid gap
    # (-0.9, 0.1). The contiguous grid must now bracket one; verify it zeroes NPV.
    r = tvm.irr([10, -7, 1])
    assert tvm.npv(r, [10, -7, 1]) == pytest.approx(0.0, abs=1e-6)
    assert r > -1


def test_irr_raises_when_all_positive(tvm):
    # No sign change → no internal rate of return. Raise, don't return NaN.
    with pytest.raises(ValueError):
        tvm.irr([100, 200, 300])


# ── mirr ──────────────────────────────────────────────────────────────────────
def test_mirr_basic(tvm):
    assert tvm.mirr([-100, 39, 59, 55, 20], 0.10, 0.12) == pytest.approx(0.20437673767455267, abs=TOL)


def test_mirr_two_outflows(tvm):
    assert tvm.mirr([-1000, -4000, 5000, 2000], 0.10, 0.12) == pytest.approx(0.17908568603489283, abs=TOL)


def test_mirr_raises_without_both_signs(tvm):
    with pytest.raises(ValueError):
        tvm.mirr([100, 200, 300], 0.10, 0.12)


# ── property / identity tests ─────────────────────────────────────────────────
def test_pv_fv_round_trip(tvm):
    # fv and pv are inverses: discount the future value back and recover the pv.
    r, n, payment, pv0 = 0.04 / 12, 180, -500, -10000
    future = tvm.fv(r, n, payment, pv0)
    assert tvm.pv(r, n, payment, future) == pytest.approx(pv0, abs=1e-6)


def test_amortization_pays_off_to_zero(tvm):
    # Paying the computed level payment for the full term leaves a ~0 balance.
    r, n, principal = 0.06 / 12, 360, 250000
    payment = tvm.pmt(r, n, principal)
    assert tvm.fv(r, n, payment, principal) == pytest.approx(0.0, abs=1e-6)


def test_principal_portions_sum_to_loan(tvm):
    # The principal parts of every payment must repay exactly the loan amount.
    r, n, principal = 0.06 / 12, 360, 250000
    repaid = sum(tvm.ppmt(r, per, n, principal) for per in range(1, n + 1))
    assert repaid == pytest.approx(-principal, abs=1e-3)


def test_pmt_nper_round_trip(tvm):
    # The number of periods to pay off a loan at its computed payment is the term.
    r, n, principal = 0.05 / 12, 240, 100000
    payment = tvm.pmt(r, n, principal)
    assert tvm.nper(r, payment, principal) == pytest.approx(n, abs=1e-6)
