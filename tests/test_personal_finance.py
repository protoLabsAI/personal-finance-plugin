"""Smoke tests for Personal Finance — host-free (no protoAgent running)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_register_runs_host_free(plugin, registry):
    plugin.register(registry)  # must not raise with no host present
    # The scaffold wires its contributions here — replace with your real assertions.
    assert registry.tools or registry.routers or registry.surfaces or registry.subagents


def test_manifest_is_valid():
    m = yaml.safe_load((ROOT / "protoagent.plugin.yaml").read_text())
    assert m["id"] == "personal-finance" and m["version"]
