"""Personal Finance — a protoAgent plugin of deterministic finance tools.

Tools-only: the math lives in ``finance/`` (pure, unit-tested), the @tool shims
in ``tools.py``, and one skill in ``skills/``. No router, surface, or MCP.
"""

from __future__ import annotations


def register(registry):
    """Wire this plugin's contributions into the agent (ADR 0018).

    Binds the core tool set by default; the advanced set is bound only when the
    ``advanced_tools`` config flag is set, to keep the agent's tool surface lean
    (plan §5). Reads the plugin's config section for tool defaults.
    """
    # Imported lazily (not at module top) so bare-pytest collection of this
    # repo-root file as a top-level module doesn't hit a parent-less relative
    # import; under the host loader the package context resolves it.
    from . import tools

    config = getattr(registry, "config", None) or {}
    core, advanced = tools.build_tools(config)
    registry.register_tools(core)
    if config.get("advanced_tools"):
        registry.register_tools(advanced)
    registry.register_skill_dir("skills")
