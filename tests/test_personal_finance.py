"""Registration + manifest tests for Personal Finance — host-free (no protoAgent)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plugin_testkit import FakeRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# Tool rosters come from tools.py (single source of truth) — no local copy here.
@pytest.fixture
def CORE_NAMES(plugin):
    return importlib.import_module(plugin.__name__ + ".tools").CORE_TOOL_NAMES


@pytest.fixture
def ADVANCED_NAMES(plugin):
    return importlib.import_module(plugin.__name__ + ".tools").ADVANCED_TOOL_NAMES


def test_register_runs_host_free(plugin, registry):
    plugin.register(registry)  # must not raise with no host present
    assert registry.tools


def test_register_binds_core_by_default(plugin, CORE_NAMES):
    reg = FakeRegistry()  # no config → advanced off
    plugin.register(reg)
    names = {t.name for t in reg.tools}
    assert names == CORE_NAMES
    assert len(reg.tools) == 16  # the 14 domain tools + npv + irr (plan §10/Q1)


def test_register_registers_skill_dir(plugin):
    reg = FakeRegistry()
    plugin.register(reg)
    assert "skills" in reg.skill_dirs


def test_advanced_tools_gated_off_by_default(plugin, ADVANCED_NAMES):
    reg = FakeRegistry()
    plugin.register(reg)
    names = {t.name for t in reg.tools}
    assert not (names & ADVANCED_NAMES)


def test_advanced_tools_enabled_via_config(plugin, CORE_NAMES, ADVANCED_NAMES):
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
