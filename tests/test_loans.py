"""Tests for finance/loans.py — loan & debt tools (plan §4B)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def loans(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.loans")


# ── loan_payment ──────────────────────────────────────────────────────────────
def test_loan_payment_mortgage(loans):
    out = loans.loan_payment(200000, 0.06, 30)
    assert out["payment"] == pytest.approx(1199.10, abs=0.01)
    assert out["total_paid"] == pytest.approx(431676.38, abs=0.01)
    assert out["total_interest"] == pytest.approx(231676.38, abs=0.01)
    assert out["periods"] == 360


def test_loan_payment_auto_default_monthly(loans):
    out = loans.loan_payment(25000, 0.045, 5)
    assert out["payment"] == pytest.approx(466.08, abs=0.01)
    assert out["total_interest"] == pytest.approx(2964.53, abs=0.01)


def test_loan_payment_zero_interest(loans):
    out = loans.loan_payment(1200, 0.0, 1)
    assert out["payment"] == pytest.approx(100.0, abs=0.01)
    assert out["total_interest"] == pytest.approx(0.0, abs=0.01)


def test_loan_payment_has_summary_string(loans):
    out = loans.loan_payment(200000, 0.06, 30)
    assert isinstance(out["summary"], str) and out["summary"]


def test_loan_payment_rejects_bad_inputs(loans):
    with pytest.raises(ValueError):
        loans.loan_payment(0, 0.06, 30)
    with pytest.raises(ValueError):
        loans.loan_payment(200000, 0.06, 0)


# ── loan_amortization ─────────────────────────────────────────────────────────
def test_amortization_summary_omits_schedule(loans):
    out = loans.loan_amortization(25000, 0.045, 5, detail="summary")
    assert "schedule" not in out
    assert out["total_interest"] == pytest.approx(2964.53, abs=0.01)


def test_amortization_full_has_one_row_per_period(loans):
    out = loans.loan_amortization(25000, 0.045, 5, detail="full")
    assert len(out["schedule"]) == 60
    first = out["schedule"][0]
    assert set(first) >= {"period", "payment", "principal", "interest", "balance"}


def test_amortization_first_period_interest(loans):
    out = loans.loan_amortization(25000, 0.045, 5, detail="full")
    # First-period interest is balance * periodic rate = 25000 * 0.045/12 = 93.75
    assert out["schedule"][0]["interest"] == pytest.approx(93.75, abs=0.01)


def test_amortization_balance_reaches_zero(loans):
    out = loans.loan_amortization(25000, 0.045, 5, detail="full")
    assert out["schedule"][-1]["balance"] == pytest.approx(0.0, abs=0.01)


def test_amortization_principal_sums_to_loan(loans):
    out = loans.loan_amortization(25000, 0.045, 5, detail="full")
    principal_paid = sum(row["principal"] for row in out["schedule"])
    assert principal_paid == pytest.approx(25000, abs=0.05)


# ── loan_payoff_extra ─────────────────────────────────────────────────────────
def test_payoff_extra_shortens_term(loans):
    out = loans.loan_payoff_extra(25000, 0.045, 5, extra_payment=100)
    assert out["baseline_periods"] == 60
    assert out["new_periods"] == 49
    assert out["periods_saved"] == 11


def test_payoff_extra_interest_saved(loans):
    out = loans.loan_payoff_extra(25000, 0.045, 5, extra_payment=100)
    assert out["interest_saved"] == pytest.approx(581.57, abs=0.05)
    assert out["new_total_interest"] == pytest.approx(2382.96, abs=0.05)


def test_payoff_extra_zero_extra_matches_baseline(loans):
    out = loans.loan_payoff_extra(25000, 0.045, 5, extra_payment=0)
    assert out["new_periods"] == 60
    assert out["interest_saved"] == pytest.approx(0.0, abs=0.05)


def test_payoff_extra_rejects_negative_extra(loans):
    with pytest.raises(ValueError):
        loans.loan_payoff_extra(25000, 0.045, 5, extra_payment=-50)


# ── debt_payoff_strategy ──────────────────────────────────────────────────────
DEBTS = [
    {"name": "CardA", "balance": 1000, "apr": 0.20, "min_payment": 25},
    {"name": "CardB", "balance": 5000, "apr": 0.15, "min_payment": 100},
    {"name": "Loan", "balance": 3000, "apr": 0.07, "min_payment": 80},
]


def test_debt_snowball(loans):
    out = loans.debt_payoff_strategy(DEBTS, budget=500, method="snowball")
    assert out["months"] == 21
    assert out["total_interest"] == pytest.approx(1099.55, abs=0.10)
    assert out["payoff_order"] == ["CardA", "Loan", "CardB"]


def test_debt_avalanche(loans):
    out = loans.debt_payoff_strategy(DEBTS, budget=500, method="avalanche")
    assert out["months"] == 20
    assert out["total_interest"] == pytest.approx(923.90, abs=0.10)
    assert out["payoff_order"] == ["CardA", "CardB", "Loan"]


def test_debt_strategy_includes_comparison(loans):
    out = loans.debt_payoff_strategy(DEBTS, budget=500, method="snowball")
    # Avalanche should save interest vs snowball in this scenario.
    assert out["comparison"]["avalanche"]["total_interest"] < out["comparison"]["snowball"]["total_interest"]


def test_debt_strategy_rejects_budget_below_minimums(loans):
    with pytest.raises(ValueError):
        loans.debt_payoff_strategy(DEBTS, budget=100, method="snowball")  # mins sum to 205


def test_debt_strategy_rejects_unknown_method(loans):
    with pytest.raises(ValueError):
        loans.debt_payoff_strategy(DEBTS, budget=500, method="laddering")


# ── debt_to_income ────────────────────────────────────────────────────────────
def test_dti_front_and_back_end(loans):
    out = loans.debt_to_income(gross_monthly_income=6000, housing_payment=1500, other_debt_payments=500)
    assert out["front_end"] == pytest.approx(0.25, abs=1e-6)
    assert out["back_end"] == pytest.approx(2000 / 6000, abs=1e-6)


def test_dti_rejects_zero_income(loans):
    with pytest.raises(ValueError):
        loans.debt_to_income(gross_monthly_income=0, housing_payment=1500, other_debt_payments=500)
