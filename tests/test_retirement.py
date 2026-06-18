"""Tests for finance/retirement.py — retirement projection & safe withdrawal
(plan §4D, v1 subset). Golden TVM figures cross-checked against numpy_financial.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def retirement(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.retirement")


# ── retirement_projection: accumulation ─────────────────────────────────────────
def test_projection_balance_at_retirement(retirement):
    # 50k now + 10k/yr for 35 yrs at 7% → numpy_financial fv.
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    assert out["balance_at_retirement"] == pytest.approx(1916197.86, abs=0.01)
    assert out["years_in_retirement"] == 25


def test_projection_zero_return_accumulation(retirement):
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.0,
        desired_annual_income=40000,
    )
    # 10000 + 3000*30 with no growth.
    assert out["balance_at_retirement"] == pytest.approx(100000.0, abs=0.01)


# ── retirement_projection: drawdown / sustainable income ────────────────────────
def test_projection_sustainable_income(retirement):
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    # Level withdrawal depleting 1,916,197.86 over 25 yrs at 7%.
    assert out["sustainable_annual_income"] == pytest.approx(164429.93, abs=0.01)
    assert out["sustainable_annual_income"] > 0  # reported as a positive figure


def test_projection_surplus_case(retirement):
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    # Sustainable (164k) far exceeds desired 60k → positive surplus.
    assert out["surplus_or_shortfall"] == pytest.approx(104429.93, abs=0.01)
    assert out["surplus_or_shortfall"] > 0


def test_projection_shortfall_case(retirement):
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.05,
        desired_annual_income=80000,
    )
    assert out["balance_at_retirement"] == pytest.approx(242535.97, abs=0.01)
    assert out["sustainable_annual_income"] == pytest.approx(19461.71, abs=0.01)
    # 19,461.71 sustainable - 80,000 desired → large negative shortfall.
    assert out["surplus_or_shortfall"] == pytest.approx(-60538.29, abs=0.01)
    assert out["surplus_or_shortfall"] < 0


# ── retirement_projection: depletion age ────────────────────────────────────────
def test_projection_depletion_age_when_shortfall(retirement):
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.05,
        desired_annual_income=80000,
    )
    # Drawing 80k/yr from 242,535.97 at 5% depletes ~3.37 yrs in.
    assert out["depletion_age"] == pytest.approx(68.37, abs=0.01)


def test_projection_depletion_age_zero_return(retirement):
    # expected_return=0 is a valid (cash) assumption. Balance simply divides by
    # the draw; depletion must land AFTER retirement, never before it (guards the
    # tvm.nper zero-rate sign quirk that returned a negative number of years).
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=1_000_000,
        annual_contribution=0,
        expected_return=0.0,
        desired_annual_income=100000,
    )
    # 1,000,000 / 100,000 = 10 years → age 75.
    assert out["depletion_age"] == pytest.approx(75.0, abs=0.1)


def test_projection_funds_outlast_reports_none(retirement):
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    # Desired draw (60k) is below sustainable (164k) → funds never deplete.
    assert out["depletion_age"] is None


# ── retirement_projection: social security ──────────────────────────────────────
def test_projection_ss_included_offsets_income(retirement):
    ss = {"annual_benefit": 24000, "claim_age": 65, "include": True}
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
        social_security=ss,
    )
    assert out["social_security_income"] == pytest.approx(24000.0, abs=0.01)
    # total_sustainable = portfolio 164,429.93 + SS 24,000; surplus vs 60k.
    assert out["surplus_or_shortfall"] == pytest.approx(128429.93, abs=0.01)


def test_projection_ss_excluded_contributes_nothing(retirement):
    ss = {"annual_benefit": 24000, "claim_age": 65, "include": False}
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
        social_security=ss,
    )
    assert out["social_security_income"] == 0.0
    assert out["surplus_or_shortfall"] == pytest.approx(104429.93, abs=0.01)


def test_projection_ss_not_yet_claimable_excluded(retirement):
    # claim_age beyond retirement_age → SS not yet available at retirement.
    ss = {"annual_benefit": 24000, "claim_age": 70, "include": True}
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
        social_security=ss,
    )
    assert out["social_security_income"] == 0.0


def test_projection_ss_offsets_portfolio_draw_in_depletion(retirement):
    # Shortfall scenario, but partial SS reduces the portfolio draw, pushing
    # the depletion age later than the no-SS case.
    ss = {"annual_benefit": 20000, "claim_age": 65, "include": True}
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.05,
        desired_annual_income=80000,
        social_security=ss,
    )
    # Net portfolio draw 80k - 20k = 60k/yr → depletes ~4.63 yrs in.
    assert out["depletion_age"] == pytest.approx(69.63, abs=0.01)


# ── retirement_projection: inflation / real view ────────────────────────────────
def test_projection_real_view_with_inflation(retirement):
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.05,
        desired_annual_income=80000,
        inflation_rate=0.03,
    )
    real = out["real"]
    # real_return = 1.05/1.03 - 1 ≈ 0.0194; sustainable in today's dollars.
    assert out["assumptions"]["real_return"] == pytest.approx(0.0194175, abs=1e-6)
    assert real["sustainable_annual_income"] == pytest.approx(14749.45, abs=0.01)


def test_projection_omits_real_view_without_inflation(retirement):
    out = retirement.retirement_projection(
        current_age=35,
        retirement_age=65,
        life_expectancy=85,
        current_savings=10000,
        annual_contribution=3000,
        expected_return=0.05,
        desired_annual_income=80000,
    )
    assert "real" not in out


# ── retirement_projection: assumptions echo & caveat ────────────────────────────
def test_projection_echoes_assumptions(retirement):
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    a = out["assumptions"]
    assert a["expected_return"] == 0.07
    assert a["current_age"] == 30
    assert a["retirement_age"] == 65
    assert a["life_expectancy"] == 90


def test_projection_summary_has_not_advice_caveat(retirement):
    out = retirement.retirement_projection(
        current_age=30,
        retirement_age=65,
        life_expectancy=90,
        current_savings=50000,
        annual_contribution=10000,
        expected_return=0.07,
        desired_annual_income=60000,
    )
    assert "not financial advice" in out["summary"].lower()


# ── retirement_projection: validation ───────────────────────────────────────────
def test_projection_rejects_bad_ages(retirement):
    with pytest.raises(ValueError):
        retirement.retirement_projection(65, 65, 90, 50000, 10000, 0.07, 60000)
    with pytest.raises(ValueError):
        retirement.retirement_projection(70, 65, 90, 50000, 10000, 0.07, 60000)
    with pytest.raises(ValueError):
        retirement.retirement_projection(30, 90, 85, 50000, 10000, 0.07, 60000)


def test_projection_rejects_negative_money(retirement):
    with pytest.raises(ValueError):
        retirement.retirement_projection(30, 65, 90, -1, 10000, 0.07, 60000)
    with pytest.raises(ValueError):
        retirement.retirement_projection(30, 65, 90, 50000, -1, 0.07, 60000)
    with pytest.raises(ValueError):
        retirement.retirement_projection(30, 65, 90, 50000, 10000, 0.07, -1)


# ── safe_withdrawal ─────────────────────────────────────────────────────────────
def test_safe_withdrawal_from_nest_egg(retirement):
    out = retirement.safe_withdrawal(nest_egg=1000000)
    assert out["mode"] == "nest_egg"
    assert out["sustainable_income"] == pytest.approx(40000.0, abs=0.01)
    assert out["rate"] == 0.04


def test_safe_withdrawal_required_nest_egg(retirement):
    out = retirement.safe_withdrawal(target_income=40000)
    assert out["mode"] == "target_income"
    assert out["required_nest_egg"] == pytest.approx(1000000.0, abs=0.01)


def test_safe_withdrawal_custom_rate(retirement):
    out = retirement.safe_withdrawal(nest_egg=1000000, rate=0.035)
    assert out["sustainable_income"] == pytest.approx(35000.0, abs=0.01)


def test_safe_withdrawal_summary_caveat(retirement):
    out = retirement.safe_withdrawal(nest_egg=1000000)
    assert "not financial advice" in out["summary"].lower()


def test_safe_withdrawal_requires_exactly_one_input(retirement):
    with pytest.raises(ValueError):
        retirement.safe_withdrawal()  # neither
    with pytest.raises(ValueError):
        retirement.safe_withdrawal(nest_egg=1000000, target_income=40000)  # both


def test_safe_withdrawal_rejects_bad_rate_and_horizon(retirement):
    with pytest.raises(ValueError):
        retirement.safe_withdrawal(nest_egg=1000000, rate=0)
    with pytest.raises(ValueError):
        retirement.safe_withdrawal(nest_egg=1000000, horizon=0)
