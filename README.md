# Personal Finance

A protoAgent plugin of **deterministic, source-agnostic** personal & family
finance tools: time-value-of-money math, loan/debt analysis, savings &
retirement projections, budgeting, income-disruption planning, and
transaction-data normalization.

Pure Python — **no network, no secrets, no external dependencies** — so it's
trivially trustable and host-free testable.

## Why

- **Stop the model doing arithmetic.** The LLM picks the tool and arguments; the
  math is exact and reproducible. Same inputs → same outputs, every time.
- **Source-agnostic.** Tools take plain numbers and row-lists, never a connector
  handle. Works with hand-typed figures, a CSV, or a data connector's dump.
- **Lean trust surface.** `requires_pip: []`, `capabilities: {network: [],
  filesystem: none}` — safe to enable anywhere.
- **Honest framing.** Outputs are *projections under stated assumptions*, not
  advice; the modeling tools echo their assumptions and a not-advice caveat.

## Tools

Returns are typed dicts of figures plus a short `summary` string. Large outputs
(amortization rows, month-by-month tables, yearly schedules) gate behind a
`detail="full"` flag to protect context.

**Core (bound by default):**

| Tool | Purpose |
|------|---------|
| `time_value_of_money` | solve for `fv`/`pv`/`pmt`/`nper`/`rate` |
| `npv`, `irr` | cash-flow series valuation |
| `loan_payment`, `loan_amortization` | payment, total interest, schedule |
| `debt_payoff_strategy` | snowball vs. avalanche payoff |
| `compound_growth`, `savings_goal` | investment growth; required contribution |
| `retirement_projection` | accumulation → drawdown, shortfall, depletion age |
| `budget_breakdown`, `emergency_fund_target`, `net_worth` | household budgeting |
| `normalize_transactions`, `summarize_spending` | transaction analysis |
| `income_disruption_runway` | months of runway after a job loss |
| `inflation_adjust` | real ↔ nominal value |

**Advanced (bound when `advanced_tools` is enabled):** `mirr`,
`loan_payoff_extra`, `refinance_breakeven`, `debt_to_income`, `compare_apy`,
`safe_withdrawal`, `rmd`, `social_security_estimate`, `cashflow_forecast`,
`income_replacement_need`, `detect_recurring`.

## Composing with a data source

This plugin **computes**; it does not fetch. Pair it with a connector at the
agent layer (no cross-plugin imports): e.g. call `monarch__get_transactions`,
feed the raw dump to `normalize_transactions`, then run `summarize_spending`,
`net_worth`, or the projection tools on the normalized result.

## The TVM engine

`finance/tvm.py` re-implements **all 10 numpy-financial functions** (`fv`, `pv`,
`pmt`, `nper`, `ipmt`, `ppmt`, `npv`, `irr`, `mirr`, `rate`) from first
principles with **zero dependency on numpy**. The iterative solvers (`rate`,
`irr`) use Newton–Raphson with a bisection fallback and raise a
`ConvergenceError` rather than returning a silent `NaN` when no root exists.
Every function is golden-tested against `numpy_financial` (a dev-only
cross-check, never a runtime dependency).

## Layout

```
personal-finance/
  protoagent.plugin.yaml   # manifest: config + settings (tools-only)
  __init__.py              # register(): binds core tools, advanced when enabled
  tools.py                 # @tool shims over the engine
  finance/                 # the pure-function engine (tvm, loans, growth,
                           #   retirement, budget, normalize, disruption,
                           #   inflation, money helpers)
  skills/personal-finance/ # the agent skill (when + how to use)
  tests/                   # host-free suite (golden values, properties)
```

## Configuration (`protoagent.plugin.yaml`)

| Key | Default | Use |
|-----|---------|-----|
| `default_inflation_rate` | `0.03` | used when a tool's inflation input is omitted |
| `default_withdrawal_rate` | `0.04` | safe-withdrawal default |
| `currency` | `USD` | reserved; surfaced to the agent via the skill |
| `advanced_tools` | `false` | bind the advanced tool set |

The plugin ships `enabled: false` — operators opt in.

## Development

Host-free: the suite loads the plugin as a package via a vendored testkit, with
no running protoAgent.

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check . && ruff format --check tests/
```

CI (`.github/workflows/ci.yml`) runs `ruff` + `pytest` on push and PR (on
`ubuntu-latest`, or the org's Namespace runner when `NSC_RUNNER` is set).

## Not financial advice

The modeling tools produce projections under stated assumptions. They are not
financial advice.
