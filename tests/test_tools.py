"""Tests for tools.py — the @tool shims over the finance/ engine, and the
core/advanced split (plan §5)."""

from __future__ import annotations

import importlib

import pytest

CORE_NAMES = {
    "time_value_of_money",
    "loan_payment",
    "loan_amortization",
    "debt_payoff_strategy",
    "compound_growth",
    "savings_goal",
    "retirement_projection",
    "budget_breakdown",
    "emergency_fund_target",
    "net_worth",
    "normalize_transactions",
    "summarize_spending",
    "income_disruption_runway",
    "inflation_adjust",
    # npv + irr are exposed directly by default (plan §10/Q1); only mirr is advanced.
    "npv",
    "irr",
}

ADVANCED_NAMES = {
    "mirr",
    "loan_payoff_extra",
    "debt_to_income",
    "safe_withdrawal",
    "cashflow_forecast",
    "income_replacement_need",
    "detect_recurring",
}


@pytest.fixture
def tools(plugin):
    return importlib.import_module(plugin.__name__ + ".tools")


def _by_name(tool_list):
    return {t.name: t for t in tool_list}


# ── tool sets ─────────────────────────────────────────────────────────────────
def test_core_set(tools):
    core, _ = tools.build_tools({})
    assert _by_name(core).keys() == CORE_NAMES


def test_advanced_set(tools):
    _, advanced = tools.build_tools({})
    assert _by_name(advanced).keys() == ADVANCED_NAMES


def test_core_and_advanced_are_disjoint(tools):
    core, advanced = tools.build_tools({})
    assert not (_by_name(core).keys() & _by_name(advanced).keys())


def test_every_tool_has_a_description(tools):
    core, advanced = tools.build_tools({})
    for t in core + advanced:
        assert t.description and t.description.strip()


# ── time_value_of_money dispatch ──────────────────────────────────────────────
def test_tvm_solves_pmt(tools):
    core = _by_name(tools.build_tools({})[0])
    out = core["time_value_of_money"].invoke({"solve_for": "pmt", "rate": 0.0824 / 12, "nper": 12, "pv": 2500})
    assert out["value"] == pytest.approx(-217.7486, abs=1e-3)


def test_tvm_solves_fv(tools):
    core = _by_name(tools.build_tools({})[0])
    out = core["time_value_of_money"].invoke(
        {"solve_for": "fv", "rate": 0.05 / 12, "nper": 120, "pmt": -100, "pv": -100}
    )
    assert out["value"] == pytest.approx(15692.93, abs=0.01)


def test_tvm_solves_rate(tools):
    core = _by_name(tools.build_tools({})[0])
    out = core["time_value_of_money"].invoke({"solve_for": "rate", "nper": 12, "pmt": -100, "pv": 1000})
    assert out["value"] == pytest.approx(0.029228540769, abs=1e-6)


def test_tvm_rejects_unknown_solve_for(tools):
    core = _by_name(tools.build_tools({})[0])
    with pytest.raises(ValueError):
        core["time_value_of_money"].invoke({"solve_for": "wat", "rate": 0.01, "nper": 12, "pmt": -100})


def test_tvm_requires_inputs_for_target(tools):
    core = _by_name(tools.build_tools({})[0])
    # solving for pmt needs rate, nper, pv — omitting nper must error, not miscompute.
    with pytest.raises(ValueError):
        core["time_value_of_money"].invoke({"solve_for": "pmt", "rate": 0.01, "pv": 2500})


# ── config-default injection ──────────────────────────────────────────────────
def test_compound_growth_uses_config_inflation_when_omitted(tools):
    core = _by_name(tools.build_tools({"default_inflation_rate": 0.03})[0])
    out = core["compound_growth"].invoke({"starting_balance": 10000, "contribution": 500, "rate": 0.07, "years": 30})
    assert "real_fv" in out  # inflation default applied → real value reported
    assert out["inflation_rate"] == 0.03


def test_safe_withdrawal_uses_config_rate_when_omitted(tools):
    advanced = _by_name(tools.build_tools({"default_withdrawal_rate": 0.04})[1])
    out = advanced["safe_withdrawal"].invoke({"nest_egg": 1000000})
    assert out["sustainable_income"] == pytest.approx(40000.0, abs=0.01)


# ── thin-wrapper passthrough ──────────────────────────────────────────────────
def test_loan_payment_wrapper_returns_engine_dict(tools):
    core = _by_name(tools.build_tools({})[0])
    out = core["loan_payment"].invoke({"principal": 200000, "apr": 0.06, "term_years": 30})
    assert out["payment"] == pytest.approx(1199.10, abs=0.01)


def test_normalize_wrapper_returns_canonical(tools):
    core = _by_name(tools.build_tools({})[0])
    out = core["normalize_transactions"].invoke({"raw": [{"date": "2024-01-01", "amount": -20, "merchant": "Uber"}]})
    assert out["transactions"][0]["category"] == "Transport"


def test_npv_and_irr_are_core_and_callable(tools):
    core = _by_name(tools.build_tools({})[0])
    assert core["npv"].invoke({"rate": 0.0, "values": [-100, 50, 80]})["npv"] == pytest.approx(30.0, abs=1e-9)
    assert core["irr"].invoke({"values": [-100, 50, 80]})["irr"] > 0


def test_summarize_spending_window_via_dates(tools):
    core = _by_name(tools.build_tools({})[0])
    txns = [
        {"date": "2024-01-10", "amount": -10.0, "merchant": "A", "category": "Dining"},
        {"date": "2024-02-10", "amount": -20.0, "merchant": "B", "category": "Dining"},
    ]
    out = core["summarize_spending"].invoke(
        {"transactions": txns, "group_by": "category", "start_date": "2024-01-01", "end_date": "2024-01-31"}
    )
    assert out["total_outflow"] == pytest.approx(10.0, abs=0.01)  # only the January txn


def test_summarize_spending_rejects_half_open_window(tools):
    core = _by_name(tools.build_tools({})[0])
    with pytest.raises(ValueError):
        core["summarize_spending"].invoke({"transactions": [], "start_date": "2024-01-01"})
