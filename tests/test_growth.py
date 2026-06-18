"""Tests for finance/growth.py — savings & investment projections (plan §4C)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def growth(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.growth")


# ── compound_growth ───────────────────────────────────────────────────────────
def test_compound_growth_nominal(growth):
    out = growth.compound_growth(10000, 500, 0.07, 30)
    assert out["nominal_fv"] == pytest.approx(691150.47, abs=0.01)
    assert out["total_contributions"] == pytest.approx(190000, abs=0.01)
    assert out["total_growth"] == pytest.approx(501150.47, abs=0.01)
    assert out["periods"] == 360


def test_compound_growth_real_value_with_inflation(growth):
    out = growth.compound_growth(10000, 500, 0.07, 30, inflation_rate=0.03)
    assert out["real_fv"] == pytest.approx(284744.84, abs=0.01)


def test_compound_growth_omits_real_fv_without_inflation(growth):
    out = growth.compound_growth(10000, 500, 0.07, 30)
    assert "real_fv" not in out


def test_compound_growth_zero_rate_is_just_contributions(growth):
    out = growth.compound_growth(1000, 100, 0.0, 10)
    assert out["nominal_fv"] == pytest.approx(13000.0, abs=0.01)
    assert out["total_growth"] == pytest.approx(0.0, abs=0.01)


def test_compound_growth_summary_omits_schedule(growth):
    out = growth.compound_growth(10000, 500, 0.07, 30, detail="summary")
    assert "schedule" not in out


def test_compound_growth_full_has_yearly_schedule(growth):
    out = growth.compound_growth(10000, 500, 0.07, 30, detail="full")
    assert len(out["schedule"]) == 30
    assert set(out["schedule"][0]) >= {"year", "balance", "contributions", "growth"}
    # Final year's balance must reconcile with the headline nominal FV.
    assert out["schedule"][-1]["balance"] == pytest.approx(out["nominal_fv"], abs=0.01)


def test_compound_growth_rejects_bad_inputs(growth):
    with pytest.raises(ValueError):
        growth.compound_growth(-1, 500, 0.07, 30)
    with pytest.raises(ValueError):
        growth.compound_growth(10000, 500, 0.07, 0)


# ── savings_goal ──────────────────────────────────────────────────────────────
def test_savings_goal_required_contribution(growth):
    out = growth.savings_goal(100000, 18, starting_balance=5000, rate=0.06)
    assert out["required_contribution"] == pytest.approx(220.25, abs=0.01)
    assert out["periods"] == 216
    assert out["already_met"] is False


def test_savings_goal_zero_rate(growth):
    out = growth.savings_goal(12000, 1, starting_balance=0, rate=0.0)
    assert out["required_contribution"] == pytest.approx(1000.0, abs=0.01)


def test_savings_goal_already_met_clamps_to_zero(growth):
    # Starting balance already over-funds the goal: no contribution needed.
    out = growth.savings_goal(100000, 18, starting_balance=200000, rate=0.06)
    assert out["already_met"] is True
    assert out["required_contribution"] == 0.0


def test_savings_goal_round_trips_with_compound_growth(growth):
    # Contributing the required amount must reach the target — within a few
    # dollars, since the contribution is rounded to whole cents and that
    # shortfall compounds over the horizon.
    goal = growth.savings_goal(100000, 18, starting_balance=5000, rate=0.06)
    proj = growth.compound_growth(5000, goal["required_contribution"], 0.06, 18)
    assert proj["nominal_fv"] == pytest.approx(100000, abs=5.0)


def test_savings_goal_rejects_bad_inputs(growth):
    with pytest.raises(ValueError):
        growth.savings_goal(0, 18, rate=0.06)
    with pytest.raises(ValueError):
        growth.savings_goal(100000, 0, rate=0.06)


# ── compare_apy ───────────────────────────────────────────────────────────────
def test_compare_apy_golden_monthly(growth):
    """5% nominal monthly -> APY = (1 + 0.05/12)**12 - 1 ≈ 0.051161898"""
    out = growth.compare_apy([{"rate": 0.05, "freq": "monthly"}])
    assert out["ranked"][0]["apy"] == pytest.approx(0.051161898, abs=1e-6)


def test_compare_apy_golden_annual(growth):
    """5% nominal annual -> APY = 0.05 exactly"""
    out = growth.compare_apy([{"rate": 0.05, "freq": "annual"}])
    assert out["ranked"][0]["apy"] == pytest.approx(0.05, abs=1e-6)


def test_compare_apy_golden_daily(growth):
    """5% nominal daily (n=365) -> APY = (1 + 0.05/365)**365 - 1 ≈ 0.051267496"""
    out = growth.compare_apy([{"rate": 0.05, "freq": "daily"}])
    assert out["ranked"][0]["apy"] == pytest.approx(0.051267496, abs=1e-6)


def test_compare_apy_golden_continuous(growth):
    """5% continuous -> APY = e**0.05 - 1 ≈ 0.051271096"""
    out = growth.compare_apy([{"rate": 0.05, "freq": "continuous"}])
    assert out["ranked"][0]["apy"] == pytest.approx(0.051271096, abs=1e-6)


def test_compare_apy_ranking_order(growth):
    """For equal nominal rate, continuous > daily > monthly > annual."""
    accounts = [
        {"name": "annual", "rate": 0.05, "freq": "annual"},
        {"name": "monthly", "rate": 0.05, "freq": "monthly"},
        {"name": "daily", "rate": 0.05, "freq": "daily"},
        {"name": "continuous", "rate": 0.05, "freq": "continuous"},
    ]
    out = growth.compare_apy(accounts)
    names = [r["name"] for r in out["ranked"]]
    assert names == ["continuous", "daily", "monthly", "annual"]


def test_compare_apy_best_matches_top(growth):
    """best field must equal the name of the highest-APY account."""
    accounts = [
        {"name": "low", "rate": 0.03, "freq": "annual"},
        {"name": "high", "rate": 0.05, "freq": "continuous"},
    ]
    out = growth.compare_apy(accounts)
    assert out["best"] == "high"
    assert out["ranked"][0]["name"] == "high"


def test_compare_apy_default_name(growth):
    """Missing name defaults to 'option 1', 'option 2', etc."""
    accounts = [{"rate": 0.05, "freq": "annual"}, {"rate": 0.04, "freq": "monthly"}]
    out = growth.compare_apy(accounts)
    names = {r["name"] for r in out["ranked"]}
    assert "option 1" in names
    assert "option 2" in names


def test_compare_apy_output_keys(growth):
    """Output dict has ranked, best, and summary."""
    out = growth.compare_apy([{"rate": 0.05, "freq": "monthly"}])
    assert "ranked" in out
    assert "best" in out
    assert "summary" in out
    row = out["ranked"][0]
    assert set(row) >= {"name", "rate", "freq", "apy"}


def test_compare_apy_summary_mentions_winner_and_apy(growth):
    """summary string names the winner and includes APY as a percentage."""
    out = growth.compare_apy([{"name": "MySavings", "rate": 0.05, "freq": "monthly"}])
    assert "MySavings" in out["summary"]
    assert "%" in out["summary"]


def test_compare_apy_rejects_empty_list(growth):
    with pytest.raises(ValueError):
        growth.compare_apy([])


def test_compare_apy_rejects_missing_rate(growth):
    with pytest.raises(ValueError):
        growth.compare_apy([{"freq": "monthly"}])


def test_compare_apy_rejects_missing_freq(growth):
    with pytest.raises(ValueError):
        growth.compare_apy([{"rate": 0.05}])


def test_compare_apy_rejects_non_numeric_rate(growth):
    with pytest.raises(ValueError):
        growth.compare_apy([{"rate": "five", "freq": "monthly"}])
