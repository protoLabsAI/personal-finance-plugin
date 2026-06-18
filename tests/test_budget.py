"""Tests for finance/budget.py — budgeting & household (plan §4E)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def budget(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.budget")


# ── budget_breakdown ──────────────────────────────────────────────────────────
def test_budget_breakdown_basic_surplus(budget):
    out = budget.budget_breakdown(5000, {"rent": 1500, "food": 600, "fun": 400})
    assert out["total_expenses"] == pytest.approx(2500.0, abs=0.01)
    assert out["surplus_deficit"] == pytest.approx(2500.0, abs=0.01)
    assert out["savings_rate"] == pytest.approx(0.5, abs=1e-9)


def test_budget_breakdown_503020_targets(budget):
    out = budget.budget_breakdown(5000, {"rent": 1500})
    assert out["needs_target"] == pytest.approx(2500.0, abs=0.01)
    assert out["wants_target"] == pytest.approx(1500.0, abs=0.01)
    assert out["savings_target"] == pytest.approx(1000.0, abs=0.01)


def test_budget_breakdown_deficit_flag(budget):
    out = budget.budget_breakdown(3000, {"rent": 2000, "food": 800, "car": 700})
    assert out["surplus_deficit"] == pytest.approx(-500.0, abs=0.01)
    assert "deficit" in out["flags"]
    assert "savings_below_20pct" in out["flags"]


def test_budget_breakdown_savings_below_20pct_flag(budget):
    # Surplus exists but is only 10% of income → below the 20% target.
    out = budget.budget_breakdown(4000, {"rent": 3600})
    assert out["savings_rate"] == pytest.approx(0.10, abs=1e-9)
    assert "savings_below_20pct" in out["flags"]
    assert "deficit" not in out["flags"]


def test_budget_breakdown_healthy_no_flags(budget):
    out = budget.budget_breakdown(5000, {"rent": 2000, "food": 500})
    # surplus 2500 → 50% savings, no deficit, no needs classification.
    assert out["flags"] == []


def test_budget_breakdown_needs_classification(budget):
    out = budget.budget_breakdown(
        5000,
        {"rent": 2000, "food": 600, "dining": 400, "shopping": 300},
        needs_categories=["rent", "food"],
    )
    assert out["actual_needs"] == pytest.approx(2600.0, abs=0.01)
    assert out["actual_wants"] == pytest.approx(700.0, abs=0.01)


def test_budget_breakdown_needs_over_50pct_flag(budget):
    out = budget.budget_breakdown(
        4000,
        {"rent": 2200, "fun": 300},
        needs_categories=["rent"],
    )
    # needs 2200 > 0.5*4000 == 2000 → flag.
    assert "needs_over_50pct" in out["flags"]


def test_budget_breakdown_needs_over_50pct_flag_in_flags_and_summary(budget):
    # The needs_over_50pct flag must appear in out["flags"] AND be reflected in
    # the summary's "Flags:" listing (flags assembled before out/summary built).
    out = budget.budget_breakdown(
        4000,
        {"rent": 2200, "fun": 300},
        needs_categories=["rent"],
    )
    assert "needs_over_50pct" in out["flags"]
    assert "Flags:" in out["summary"]
    assert "needs_over_50pct" in out["summary"]
    # Every flag in the list is mentioned in the summary, and vice versa.
    for flag in out["flags"]:
        assert flag in out["summary"]


def test_budget_breakdown_needs_under_50pct_no_flag(budget):
    out = budget.budget_breakdown(
        5000,
        {"rent": 2000, "fun": 300},
        needs_categories=["rent"],
    )
    # needs 2000 == 0.5*5000, not strictly over → no flag.
    assert "needs_over_50pct" not in out["flags"]


def test_budget_breakdown_full_detail_category_pct(budget):
    out = budget.budget_breakdown(
        4000,
        {"rent": 1200, "food": 400},
        detail="full",
    )
    cats = out["categories"]
    rent = next(c for c in cats if c["category"] == "rent")
    assert rent["amount"] == pytest.approx(1200.0, abs=0.01)
    assert rent["pct_of_income"] == pytest.approx(0.30, abs=1e-9)


def test_budget_breakdown_summary_omits_categories(budget):
    out = budget.budget_breakdown(4000, {"rent": 1200})
    assert "categories" not in out


def test_budget_breakdown_rejects_bad_income(budget):
    with pytest.raises(ValueError):
        budget.budget_breakdown(0, {"rent": 1000})
    with pytest.raises(ValueError):
        budget.budget_breakdown(-100, {"rent": 1000})


# ── emergency_fund_target ─────────────────────────────────────────────────────
def test_emergency_fund_target_basic(budget):
    out = budget.emergency_fund_target(3000, months=6)
    assert out["target"] == pytest.approx(18000.0, abs=0.01)
    assert out["gap"] == pytest.approx(18000.0, abs=0.01)
    assert out["already_funded"] is False


def test_emergency_fund_target_partial_reserve_and_saving(budget):
    out = budget.emergency_fund_target(3000, months=6, current_reserve=6000, monthly_saving=500)
    assert out["gap"] == pytest.approx(12000.0, abs=0.01)
    # 12000 / 500 == 24 months exactly.
    assert out["months_to_fund"] == 24


def test_emergency_fund_target_ceil_partial_month(budget):
    out = budget.emergency_fund_target(3000, months=6, current_reserve=6000, monthly_saving=700)
    # 12000 / 700 == 17.14… → ceil to 18.
    assert out["months_to_fund"] == 18


def test_emergency_fund_target_no_saving_returns_none(budget):
    out = budget.emergency_fund_target(3000, months=6, current_reserve=0, monthly_saving=0)
    assert out["months_to_fund"] is None
    assert out["gap"] == pytest.approx(18000.0, abs=0.01)


def test_emergency_fund_target_already_funded(budget):
    out = budget.emergency_fund_target(3000, months=6, current_reserve=20000, monthly_saving=500)
    assert out["gap"] == 0.0
    assert out["already_funded"] is True
    assert out["months_to_fund"] is None


def test_emergency_fund_target_rejects_bad_inputs(budget):
    with pytest.raises(ValueError):
        budget.emergency_fund_target(-1, months=6)
    with pytest.raises(ValueError):
        budget.emergency_fund_target(3000, months=0)


# ── net_worth ─────────────────────────────────────────────────────────────────
def test_net_worth_basic(budget):
    out = budget.net_worth(
        {"cash": 10000, "home": 300000},
        {"mortgage": 200000, "card": 5000},
    )
    assert out["total_assets"] == pytest.approx(310000.0, abs=0.01)
    assert out["total_liabilities"] == pytest.approx(205000.0, abs=0.01)
    assert out["net_worth"] == pytest.approx(105000.0, abs=0.01)


def test_net_worth_negative(budget):
    out = budget.net_worth({"cash": 1000}, {"loan": 5000})
    assert out["net_worth"] == pytest.approx(-4000.0, abs=0.01)


def test_net_worth_full_detail_breakdowns(budget):
    out = budget.net_worth({"cash": 1000}, {"loan": 5000}, detail="full")
    assert out["assets"] == {"cash": pytest.approx(1000.0, abs=0.01)}
    assert out["liabilities"] == {"loan": pytest.approx(5000.0, abs=0.01)}


def test_net_worth_summary_omits_breakdowns(budget):
    out = budget.net_worth({"cash": 1000}, {"loan": 5000})
    assert "assets" not in out
    assert "liabilities" not in out


# ── cashflow_forecast ─────────────────────────────────────────────────────────
def test_cashflow_forecast_growing_balance(budget):
    out = budget.cashflow_forecast(1000, 5000, 4000, 6)
    # +1000/month for 6 months on top of 1000 starting.
    assert out["ending_balance"] == pytest.approx(7000.0, abs=0.01)
    assert out["min_balance"] == pytest.approx(2000.0, abs=0.01)
    assert out["shortfall_months"] == 0


def test_cashflow_forecast_shortfall_months(budget):
    # Burning 500/month from 1200 → goes negative partway through.
    out = budget.cashflow_forecast(1200, 3000, 3500, 6)
    # balances: 700, 200, -300, -800, -1300, -1800
    assert out["ending_balance"] == pytest.approx(-1800.0, abs=0.01)
    assert out["min_balance"] == pytest.approx(-1800.0, abs=0.01)
    assert out["shortfall_months"] == 4


def test_cashflow_forecast_full_detail_rows(budget):
    out = budget.cashflow_forecast(1000, 5000, 4000, 3, detail="full")
    rows = out["rows"]
    assert len(rows) == 3
    assert rows[0] == {
        "month": 1,
        "inflow": pytest.approx(5000.0, abs=0.01),
        "outflow": pytest.approx(4000.0, abs=0.01),
        "balance": pytest.approx(2000.0, abs=0.01),
    }
    assert rows[-1]["balance"] == pytest.approx(4000.0, abs=0.01)


def test_cashflow_forecast_summary_omits_rows(budget):
    out = budget.cashflow_forecast(1000, 5000, 4000, 3)
    assert "rows" not in out


def test_cashflow_forecast_rejects_bad_months(budget):
    with pytest.raises(ValueError):
        budget.cashflow_forecast(1000, 5000, 4000, 0)
