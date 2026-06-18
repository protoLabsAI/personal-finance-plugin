"""Tests for finance/disruption.py — income-disruption planning (plan §4G)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def disruption(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.disruption")


# ── income_disruption_runway ──────────────────────────────────────────────────
def test_runway_no_benefits_plain_drawdown(disruption):
    # $12,000 saved, $3,000/mo burn, nothing else. Balance: 9k, 6k, 3k, 0, then
    # goes negative in month 5 → 4 full months of runway.
    out = disruption.income_disruption_runway(12000, 3000)
    assert out["months_of_runway"] == 4
    assert out["reduced_monthly_burn"] == pytest.approx(3000.0, abs=0.01)
    assert out["benefit_monthly"] == pytest.approx(0.0, abs=0.01)
    assert out["benefit_months"] == pytest.approx(0.0, abs=0.01)


def test_runway_monthly_net_burn_definition(disruption):
    # While the benefit is active, monthly_net_burn = reduced_burn - other_income
    # - benefit. $3,000 burn, $500/mo other income, $200/wk benefit.
    # benefit_monthly = 200 * 52/12 = 866.666...
    # net burn = 3000 - 500 - 866.6667 = 1633.333...
    out = disruption.income_disruption_runway(
        12000, 3000, weekly_benefit=200, max_benefit_weeks=26, other_monthly_income=500
    )
    assert out["benefit_monthly"] == pytest.approx(866.67, abs=0.01)
    assert out["monthly_net_burn"] == pytest.approx(1633.33, abs=0.01)


def test_runway_severance_added_up_front(disruption):
    # Severance lifts the starting balance. $12,000 + $6,000 severance = $18,000,
    # $3,000/mo burn → 6 full months.
    out = disruption.income_disruption_runway(12000, 3000, severance=6000)
    assert out["months_of_runway"] == 6


def test_runway_benefit_extends_runway(disruption):
    # Benefits while active reduce the monthly drain, extending runway versus the
    # plain-drawdown case (4 months).
    out = disruption.income_disruption_runway(12000, 3000, weekly_benefit=300, max_benefit_weeks=26)
    plain = disruption.income_disruption_runway(12000, 3000)
    assert out["months_of_runway"] > plain["months_of_runway"]


def test_runway_benefit_months_conversion(disruption):
    # 26 weeks of benefits → 26 * 12/52 = 6.0 months of benefit eligibility.
    out = disruption.income_disruption_runway(12000, 3000, weekly_benefit=300, max_benefit_weeks=26)
    assert out["benefit_months"] == pytest.approx(6.0, abs=1e-9)


def test_runway_fractional_benefit_month(disruption):
    # 30 weeks → 30 * 12/52 = 6.923076... months: 6 full benefit months plus a
    # partial 7th month paying 0.923076... of the monthly benefit.
    out = disruption.income_disruption_runway(12000, 3000, weekly_benefit=300, max_benefit_weeks=30, detail="full")
    assert out["benefit_months"] == pytest.approx(30 * 12 / 52, abs=1e-9)
    benefit_monthly = 300 * 52 / 12
    rows = out["scenario"]
    # Months 1..6 (index 0..5) get the full benefit; the inflow is the benefit.
    assert rows[0]["inflow"] == pytest.approx(benefit_monthly, abs=0.01)
    assert rows[5]["inflow"] == pytest.approx(benefit_monthly, abs=0.01)
    # Month 7 (index 6) gets the fractional benefit.
    frac = (30 * 12 / 52) - 6
    assert rows[6]["inflow"] == pytest.approx(benefit_monthly * frac, abs=0.01)
    # Month 8 (index 7) gets no benefit.
    assert rows[7]["inflow"] == pytest.approx(0.0, abs=0.01)


def test_runway_discretionary_cut_reduces_burn(disruption):
    # A 20% discretionary cut drops a $3,000 burn to $2,400.
    out = disruption.income_disruption_runway(12000, 3000, discretionary_cut_pct=0.20)
    assert out["reduced_monthly_burn"] == pytest.approx(2400.0, abs=0.01)
    # $12,000 / $2,400 = exactly 5 full months.
    assert out["months_of_runway"] == 5


def test_runway_discretionary_cut_extends_runway(disruption):
    cut = disruption.income_disruption_runway(12000, 3000, discretionary_cut_pct=0.20)
    nocut = disruption.income_disruption_runway(12000, 3000)
    assert cut["months_of_runway"] > nocut["months_of_runway"]


def test_runway_indefinite_when_income_covers_burn(disruption):
    # Other income meets the burn exactly → balance never falls. Indefinite.
    out = disruption.income_disruption_runway(12000, 3000, other_monthly_income=3000)
    assert out["months_of_runway"] is None
    assert "indefinite" in out["summary"].lower()


def test_runway_indefinite_when_income_exceeds_burn(disruption):
    out = disruption.income_disruption_runway(12000, 3000, other_monthly_income=4000)
    assert out["months_of_runway"] is None


def test_runway_full_scenario_table_length(disruption):
    # detail='full' adds one row per simulated month up to the negative crossing.
    out = disruption.income_disruption_runway(12000, 3000, detail="full")
    assert "scenario" in out
    # 4 sustainable months + the month that goes negative = 5 rows.
    assert len(out["scenario"]) == 5
    assert set(out["scenario"][0]) >= {"month", "inflow", "burn", "balance"}
    # Last row is the one that crosses below zero.
    assert out["scenario"][-1]["balance"] < 0
    assert out["scenario"][-2]["balance"] >= 0


def test_runway_summary_omits_scenario(disruption):
    out = disruption.income_disruption_runway(12000, 3000, detail="summary")
    assert "scenario" not in out


def test_runway_has_not_advice_note(disruption):
    out = disruption.income_disruption_runway(12000, 3000)
    assert "not financial advice" in out["summary"].lower()


def test_runway_rejects_bad_inputs(disruption):
    with pytest.raises(ValueError):
        disruption.income_disruption_runway(-1, 3000)
    with pytest.raises(ValueError):
        disruption.income_disruption_runway(12000, -1)
    with pytest.raises(ValueError):
        disruption.income_disruption_runway(12000, 3000, discretionary_cut_pct=1.5)
    with pytest.raises(ValueError):
        disruption.income_disruption_runway(12000, 3000, discretionary_cut_pct=-0.1)
    with pytest.raises(ValueError):
        disruption.income_disruption_runway(12000, 3000, detail="bogus")


# ── income_replacement_need ───────────────────────────────────────────────────
def test_replacement_need_basic_gap_and_fund(disruption):
    # Take-home $5,000; benefits $1,500; other $500 → gap $3,000.
    # 6 months coverage → $18,000 fund needed.
    out = disruption.income_replacement_need(5000, benefits_monthly=1500, other_monthly_income=500)
    assert out["monthly_income_gap"] == pytest.approx(3000.0, abs=0.01)
    assert out["emergency_fund_needed"] == pytest.approx(18000.0, abs=0.01)


def test_replacement_need_replacement_ratio(disruption):
    # ratio = (1500 + 500) / 5000 = 0.40
    out = disruption.income_replacement_need(5000, benefits_monthly=1500, other_monthly_income=500)
    assert out["replacement_ratio"] == pytest.approx(0.40, abs=1e-9)


def test_replacement_need_gap_clamped_to_zero(disruption):
    # Replacement income exceeds take-home → no gap, no fund needed.
    out = disruption.income_replacement_need(5000, benefits_monthly=4000, other_monthly_income=2000)
    assert out["monthly_income_gap"] == 0.0
    assert out["emergency_fund_needed"] == 0.0


def test_replacement_need_custom_coverage_months(disruption):
    out = disruption.income_replacement_need(5000, desired_coverage_months=3)
    assert out["emergency_fund_needed"] == pytest.approx(15000.0, abs=0.01)


def test_replacement_need_has_not_advice_note(disruption):
    out = disruption.income_replacement_need(5000)
    assert "not financial advice" in out["summary"].lower()


def test_replacement_need_rejects_bad_inputs(disruption):
    with pytest.raises(ValueError):
        disruption.income_replacement_need(0)
    with pytest.raises(ValueError):
        disruption.income_replacement_need(5000, desired_coverage_months=0)
