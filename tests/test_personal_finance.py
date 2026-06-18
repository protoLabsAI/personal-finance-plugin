"""Registration + manifest tests for Personal Finance — host-free (no protoAgent)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plugin_testkit import FakeRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

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
}
ADVANCED_NAMES = {
    "npv",
    "irr",
    "mirr",
    "loan_payoff_extra",
    "debt_to_income",
    "safe_withdrawal",
    "cashflow_forecast",
    "income_replacement_need",
    "detect_recurring",
}


def test_register_runs_host_free(plugin, registry):
    plugin.register(registry)  # must not raise with no host present
    assert registry.tools


def test_register_binds_core_14_by_default(plugin):
    reg = FakeRegistry()  # no config → advanced off
    plugin.register(reg)
    names = {t.name for t in reg.tools}
    assert names == CORE_NAMES
    assert len(reg.tools) == 14


def test_register_registers_skill_dir(plugin):
    reg = FakeRegistry()
    plugin.register(reg)
    assert "skills" in reg.skill_dirs


def test_advanced_tools_gated_off_by_default(plugin):
    reg = FakeRegistry()
    plugin.register(reg)
    names = {t.name for t in reg.tools}
    assert not (names & ADVANCED_NAMES)


def test_advanced_tools_enabled_via_config(plugin):
    reg = FakeRegistry({"advanced_tools": True})
    plugin.register(reg)
    names = {t.name for t in reg.tools}
    assert names == CORE_NAMES | ADVANCED_NAMES
    assert len(reg.tools) == 23


def test_skill_present_with_frontmatter():
    skill = ROOT / "skills" / "personal-finance" / "SKILL.md"
    assert skill.exists()
    text = skill.read_text()
    assert text.startswith("---")
    front = yaml.safe_load(text.split("---", 2)[1])
    assert front["name"] == "personal-finance"
    assert front["description"].strip()


def test_manifest_is_valid_and_zero_deps():
    m = yaml.safe_load((ROOT / "protoagent.plugin.yaml").read_text())
    assert m["id"] == "personal-finance" and m["version"]
    # Lean trust surface (plan §1): no pip deps, no network/filesystem.
    assert m.get("requires_pip", []) == []
    cfg = m["config"]
    assert {"default_inflation_rate", "default_withdrawal_rate", "currency", "advanced_tools"} <= cfg.keys()
    assert cfg["advanced_tools"] is False
