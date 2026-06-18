"""Personal Finance — a protoAgent plugin (scaffolded by plugin-devkit)."""

from __future__ import annotations

from langchain_core.tools import tool


def register(registry):
    """Wire this plugin's contributions into the agent (ADR 0018)."""

    @tool
    def personal_finance_hello(name: str = "world") -> str:
        """Say hello — replace with your tool's real work."""
        return f"hello, {name}, from personal-finance"
    registry.register_tool(personal_finance_hello)

    # Event bus (ADR 0039) — coordinate without importing other plugins:
    #   registry.emit("did_something", {"id": 1})   # → "personal_finance.did_something" on the bus
    #   registry.on("other-plugin.*", lambda evt: ...) # react to anyone's events
