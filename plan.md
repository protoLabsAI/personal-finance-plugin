# `personal-finance` plugin — build plan

> Status: **PLAN — for review.** No code yet. We review this, then build.
> Target repo: `~/Dev/personal-finance-plugin` · plugin id: `personal-finance`

A protoAgent plugin of **deterministic, source-agnostic** personal/family finance
tools: time-value-of-money math (a hand-rolled re-implementation of *every*
NumPy-Financial function), plus projections, budgeting, retirement, unemployment
planning, and transaction-data normalization. Pure Python — **no network, no
secrets, no external deps** — so it's trivially trustable and host-free testable.

It is **not** a data connector. It computes over numbers the agent already has —
typed by the user, or pulled from a connector like the `monarch-money` plugin. The
two compose **at the agent layer** (no cross-plugin imports — ADR 0039): the agent
calls `monarch__get_transactions`, feeds the dump to `normalize_transactions`, then
runs `summarize_spending` / `retirement_projection` / etc.

---

## 1. Design principles

1. **Deterministic & pure.** Every tool is a pure function: same inputs → same
   outputs, no randomness, no clock, no I/O. The LLM picks the tool + args; the
   math is exact. This is the whole point — stop the model doing arithmetic.
2. **Source-agnostic.** Tools take plain numbers / row-lists, never a Monarch
   handle. Works with Monarch, Plaid, a CSV, or hand-typed figures.
3. **Modular, no cross-dependency.** Does not import `monarch-money` (or anything).
   Composition happens in the agent's turn. Mirrors the connector-vs-computation
   split we agreed on.
4. **Lean trust surface.** `requires_pip: []`, `capabilities: {network: [],
   filesystem: none}`. Safe to enable anywhere, unlike the OAuth/network Monarch
   plugin.
5. **Structured output, not prose.** Tools return typed dicts (figures + a short
   `summary` string), so the agent can reason and a future view can render. Large
   outputs (amortization rows, month-by-month) gate behind a `detail` flag to
   protect context (ADR 0005).
6. **Honest framing.** Outputs are *projections under stated assumptions*, not
   advice. Every modeling tool echoes its assumptions; the skill enforces a
   "not financial advice" framing.

---

## 2. Repository layout

```
personal-finance/
  protoagent.plugin.yaml      # manifest (tools-only; config for default rates/assumptions)
  __init__.py                 # register(): register_tools(...) + register_skill_dir("skills")
  finance/
    tvm.py                    # the hand-rolled NumPy-Financial engine (§3)
    loans.py                  # loan/debt tools (§4B)
    growth.py                 # savings & investment projections (§4C)
    retirement.py             # retirement, SWR, RMD, SS (§4D)
    budget.py                 # budgeting & household (§4E)
    normalize.py              # transaction normalization + analysis (§4F)
    disruption.py             # unemployment / income-disruption (§4G)
    money.py                  # shared: rounding, sign helpers, period/frequency maps
  tools.py                    # the LangChain @tool wrappers (thin; call finance/*)
  skills/
    personal-finance/SKILL.md # when + how to use; Monarch composition; disclaimer
  tests/                      # host-free suite (golden values vs numpy_financial), CI, pyproject
```

The math lives in `finance/` as plain functions (unit-tested directly via the
devkit testkit's package loader); `tools.py` holds only the `@tool` shims (parse,
validate, shape output). Keeps the engine testable without the tool layer.

---

## 3. The hand-rolled NumPy-Financial engine (`finance/tvm.py`)

We re-implement **all 10** NumPy-Financial functions from first principles, with
**zero dependency on numpy / numpy-financial**. They share one master equation
(the TVM identity from the [numpy-financial docs](https://numpy.org/numpy-financial/latest/)):

```
fv + pv·(1+r)^n + pmt·(1 + r·when)/r·((1+r)^n − 1) = 0        (r ≠ 0)
fv + pv + pmt·n = 0                                            (r = 0)
```

`when` = `1` (begin / annuity-due) or `0` (end / ordinary, default). **Cash-flow
sign convention** (matching numpy_financial): money you *pay out* is negative,
money you *receive* is positive. We follow it strictly in the primitives, and the
high-level tools (§4) translate intuitive positive inputs (a loan principal, a
target balance) to/from it internally so the agent isn't burdened with signs.

| Fn | Solves for | Method |
|----|-----------|--------|
| `fv(rate, nper, pmt, pv, when)` | future value | closed form |
| `pv(rate, nper, pmt, fv, when)` | present value | closed form |
| `pmt(rate, nper, pv, fv, when)` | level payment | closed form |
| `nper(rate, pmt, pv, fv, when)` | # periods | closed form (log); r=0 linear |
| `rate(nper, pmt, pv, fv, when, guess)` | periodic rate | **Newton–Raphson**, bisection fallback |
| `ipmt(rate, per, nper, pv, fv, when)` | interest part of period `per` | from remaining balance |
| `ppmt(rate, per, nper, pv, fv, when)` | principal part (`pmt − ipmt`) | derived |
| `npv(rate, values)` | net present value of a cash-flow series | Σ discounting |
| `irr(values, guess)` | internal rate of return | **Newton**, bracketed bisection fallback |
| `mirr(values, finance_rate, reinvest_rate)` | modified IRR | npv of in/out flows |

**Numerical methods.** `rate` and `irr` have no closed form → Newton–Raphson with
an analytic-ish derivative, a bracketing/bisection fallback for non-convergence,
capped iterations, and a clear error (not silent NaN) when no root exists. Golden
tests pin every function to `numpy_financial`'s documented outputs (e.g.
`pmt(0.0824/12, 12, 2500) ≈ −217.40`; `ipmt(0.05,3,10,−1000) ≈ −15.04`).

**Precision.** TVM uses `float` (matches numpy_financial; differences < 1e-6).
Currency *aggregation* in `normalize.py` uses `Decimal` to avoid float drift when
summing thousands of transactions. Money outputs round to cents at the edge only.

**Exposure (anti-pollution, ADR 0005).** All 10 are implemented, but we do **not**
bind all 10 as separate agent tools. Instead:
- one ergonomic **`time_value_of_money(solve_for=…)`** tool covers fv/pv/pmt/nper/rate;
- **`npv`**, **`irr`**, **`mirr`** are exposed directly (genuinely useful standalone);
- `ipmt`/`ppmt` stay internal, powering amortization.
The full engine is still importable + unit-tested; we just keep the *agent* surface lean.

---

## 4. Tool surface (agent-facing)

Grouped by domain. Each is a `@tool` returning a structured dict. **Source** = the
calculator family it implements (calculator.net / FINRED) or the standard it follows.

### A. TVM / cash-flow primitives
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `time_value_of_money` | `solve_for` (rate\|nper\|pmt\|pv\|fv) + the others, `when` → the solved value | TVM engine §3 |
| `npv` | discount rate, cash-flow list → NPV | Present Value / Finance calc |
| `irr` | cash-flow list → IRR (+ converged flag) | IRR calc |
| `mirr` | cash-flows, finance & reinvest rates → MIRR | IRR/ROI calc |

### B. Loans & debt
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `loan_payment` | principal, APR, term, freq → payment, total interest, total paid | Loan/Mortgage/Auto calc |
| `loan_amortization` | + `detail` (summary\|full) → schedule rows (date, principal, interest, balance) | Amortization calc |
| `loan_payoff_extra` | + extra payment → new payoff date, interest saved | Mortgage Payoff calc |
| `refinance_breakeven` | current vs new rate/term/balance, closing costs → break-even month, lifetime savings | Refinance calc |
| `debt_payoff_strategy` | list of debts (balance, APR, min), budget, method (snowball\|avalanche) → payoff order, months, total interest, comparison | Debt Payoff / Debt Destroyer (FINRED) |
| `debt_to_income` | gross income, debt payments → DTI ratio + front/back-end | DTI calc |

### C. Savings & investment projections
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `compound_growth` | starting balance, contribution, freq, rate, years, `inflation_rate?` → nominal & **real** FV, contributions vs growth, optional yearly schedule | Compound Interest / Investment / Savings calc |
| `savings_goal` | target, by-date, starting balance, rate → required periodic contribution | Savings/College Savings calc |
| `compare_apy` | list of (nominal rate, compounding freq) → effective APY ranking | Compare Savings Rates (FINRED) |

### D. Retirement
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `retirement_projection` | current age, retire age, life expectancy, current savings, annual contribution, expected return, **inflation toggle (real/nominal)**, **Social Security** (annual benefit + claim age, includable/excludable), desired income → balance at retirement, sustainable income, **shortfall/surplus**, depletion age | Retirement / 401K calc |
| `safe_withdrawal` | nest egg OR target income, rate (default 4%), horizon → sustainable income / required nest egg | Trinity / 4% rule |
| `rmd` | age, account balance → required minimum distribution | IRS **Uniform Lifetime Table** |
| `social_security_estimate` | **v2** — in v1, SS is an *input* (benefit + claim age) to `retirement_projection`; standalone PIA estimate deferred | Social Security calc |

### E. Budgeting & household
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `budget_breakdown` | income, categorized expenses → surplus/deficit, savings rate, **50/30/20** needs/wants/savings split, flags | Budget calc / Student Budget (FINRED) |
| `emergency_fund_target` | essential monthly expenses, months (default 3–6), current reserve, monthly saving → target, gap, time-to-fund | Emergency Fund (FINRED) |
| `net_worth` | assets, liabilities (or normalized accounts) → net worth, asset/liability breakdown | Net Worth (FINRED) |
| `cashflow_forecast` | recurring income & expenses, horizon, starting cash → month-by-month balance, min balance, shortfall months | Cash-flow projection |

### F. Transaction-data normalization (the connector bridge)
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `normalize_transactions` | raw txn dump (Monarch `get_transactions` rows, or CSV-shaped list) → **canonical dataset**: `{date, amount (signed), merchant, category, account, type, currency}`; de-duped, sign-normalized (outflow −), category-mapped to a canonical taxonomy, currency tagged | — (purpose-built) |
| `summarize_spending` | normalized txns, group-by (category\|merchant\|month), window → totals, averages, top-N, trend | Spending analysis |
| `detect_recurring` | normalized txns → inferred subscriptions/recurring bills (cadence, amount, next-due) | Recurring detection |

`normalize_transactions` is the deterministic glue the user asked for: hand a
Monarch dump in, get an analysis-ready dataset out. Its canonical category taxonomy
is a rule table (merchant/keyword → category), fully deterministic and testable.

### G. Unemployment / income-disruption planning
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `income_disruption_runway` | liquid savings, essential monthly burn, severance, unemployment benefit (weekly × max weeks), partial/spousal income, optional discretionary-cut % → **months of runway**, depletion date, monthly net burn, scenario table | Emergency planning (FINRED Should-My-Spouse-Work? adjacent) |
| `income_replacement_need` | prior take-home, benefits, desired coverage months → income gap, emergency-fund needed | Income replacement |

### H. Inflation / general
| Tool | Inputs → Outputs | Source |
|------|------------------|--------|
| `inflation_adjust` | amount, rate, years, direction (to-future\|to-present) → adjusted (real) value | Inflation calc |

**Total: ~24 tools.** See §5 for keeping that from polluting context.

---

## 5. Tool-pollution mitigation (ADR 0005)

~24 tools exceeds the ~10–15 comfort zone where tool selection stays sharp. Plan:
- **Curate the engine surface** — 10 TVM functions collapse to 4 agent tools (§3).
- **Lean default + opt-in groups.** Bind a **core set** by default (loan_payment,
  loan_amortization, debt_payoff_strategy, compound_growth, savings_goal,
  retirement_projection, budget_breakdown, emergency_fund_target, net_worth,
  normalize_transactions, summarize_spending, income_disruption_runway, inflation_adjust,
  time_value_of_money). Gate the rest (refinance, mirr, rmd, compare_apy, …) behind a
  manifest config flag (`tools.advanced: true`) so a household install isn't paying
  context for niche tools. (Implementation: `register()` reads config and registers
  the advanced set conditionally.)
- Document the split so operators can widen/narrow via the standard MCP-style
  include/exclude if they prefer.

**Resolved (Q4):** core-14 bound by default; advanced set gated by `advanced_tools`.

---

## 6. Skills (`skills/personal-finance/SKILL.md`)

One skill, triggered on planning/modeling/projection/budgeting/"what-if" questions.
It teaches the agent to:
- **Prefer these tools over arithmetic** — never hand-compute a payment or projection.
- **Compose with a data source.** When the `monarch-money` connector is live, pull
  real figures (`monarch__get_accounts`, `monarch__get_transactions`), run them
  through `normalize_transactions`, then compute — instead of asking the user for
  numbers it could fetch. When no connector, ask for the inputs.
- **State assumptions & units.** Echo rate, inflation toggle, SS inclusion, horizon;
  report currency and as-of context.
- **Frame as projection, not advice** — include the not-financial-advice caveat on
  retirement/debt/disruption outputs.
- **Pick the narrowest tool** and use `detail=summary` unless a full schedule is asked.

**Resolved (Q5):** one combined skill in v1.

---

## 7. Sources & references

- **NumPy-Financial** — the function set, signatures, sign convention, and master
  TVM equation the engine re-implements: <https://numpy.org/numpy-financial/latest/>
  (fv, pv, pmt, ppmt, ipmt, nper, rate, npv, irr, mirr). Golden tests pin to its outputs.
- **calculator.net financial calculators** — scenario taxonomy (mortgage, auto,
  loan, investment, retirement, debt, budget, inflation): <https://www.calculator.net/financial-calculator.html>
- **FINRED (DoD Financial Readiness)** — family/household angle (budgeting, emergency
  fund, net worth, debt destroyer, spouse-work, college/savings, compare-savings):
  <https://finred.usalearning.gov/ToolsAndAddRes/Calculators>
- **Standards baked into specific tools** (each documented in-tool):
  - IRS **Uniform Lifetime Table** → `rmd`
  - **50/30/20** budgeting rule → `budget_breakdown`
  - **4% safe-withdrawal** (Trinity study) → `safe_withdrawal`
  - Effective-APY formula → `compare_apy`
- **Composition partner:** the `monarch-money` plugin is the canonical *data* source
  `normalize_transactions` consumes — but only via the agent, never imported.

---

## 8. Plugin mechanics — per Plugin Devkit v0.2.0

Following the devkit's `building-plugins` guidance (manifest + `register()`,
tools via `register_tools`, explicit `register_skill_dir`, standalone repo + tests).

**Scaffold:** `python -m server plugin new "Personal Finance" --skill --tests`
(tools-only; **no `--view`** in v1 — a dashboard is a later add, §10).

**Manifest (`protoagent.plugin.yaml`) sketch:**
```yaml
id: personal-finance
name: Personal Finance
version: 0.1.0
description: Deterministic personal & family finance tools — TVM, projections, budgeting, retirement, normalization.
enabled: false
requires_pip: []                 # hand-rolled; zero deps
capabilities: { network: [], filesystem: none }
config_section: personal_finance
config:
  default_inflation_rate: 0.03   # used when a tool's inflation input is omitted
  default_withdrawal_rate: 0.04  # safe-withdrawal default
  currency: USD
  advanced_tools: false          # §5 opt-in for niche tools
settings:
  - { key: default_inflation_rate, label: "Default inflation rate", type: number }
  - { key: default_withdrawal_rate, label: "Safe withdrawal rate", type: number }
  - { key: currency, label: "Default currency", type: string }
  - { key: advanced_tools, label: "Expose advanced tools", type: bool }
```

**`register(registry)`:** read config, `register_tools(core_tools)`, conditionally
`register_tools(advanced_tools)` when `advanced_tools`, `register_skill_dir("skills")`.
No router, no surface, no MCP — tools + skill only.

---

## 9. Testing strategy

- **Host-free** via the devkit's vendored testkit (`load_plugin` as a package), so
  `finance/*` modules are unit-tested directly, no running protoAgent.
- **Golden numerics:** every TVM function asserted against `numpy_financial`'s
  documented values (and `ipmt + ppmt == pmt`, `Σ npv` identities). We pin numbers,
  not the library — `numpy_financial` is a *dev*-only cross-check, never a runtime dep.
- **Property tests** where cheap (e.g. `pv(fv(x)) == x` round-trips; amortization
  final balance ≈ 0).
- **Normalization fixtures:** a sample Monarch `get_transactions` dump → asserted
  canonical dataset (sign, dedupe, category mapping, recurring detection).
- **Manifest + register** smoke test (tools registered; advanced gating respected).
- CI: `ruff` + `pytest` (the scaffold's GitHub Actions), green from first commit.

---

## 10. Resolved decisions (locked)

- **Q1 — engine exposure: YES.** Collapse fv/pv/pmt/nper/rate into one
  `time_value_of_money(solve_for=…)` tool; expose `npv` + `irr` directly; `mirr` is
  advanced; `ipmt`/`ppmt` stay internal.
- **Q2 — money type: confirmed.** `float` for TVM, `Decimal` for transaction
  aggregation; round to cents only at the edge.
- **Q3 — Social Security: accept-as-input (v1).** SS enters `retirement_projection`
  as `{annual_benefit, claim_age, include}`. A standalone simplified-PIA
  `social_security_estimate` is deferred to v2.
- **Q4 — tool split: confirmed.** Core-14 bound by default; advanced (refinance,
  mirr, rmd, compare_apy, …) gated behind the `advanced_tools` config flag.
- **Q5 — skills: one combined skill** in v1.
- **Q6 — v1 scope: per §11** (engine + groups B, C, E, F, G, H + `retirement_projection`
  & `safe_withdrawal`).

## 11. Phasing

- **v1 (this build):** TVM engine (all 10) + groups B, C, E, F, G, H, and
  `retirement_projection` + `safe_withdrawal` from D. Skill. Standalone repo + tests.
- **v2:** `rmd`, `social_security_estimate`, `compare_apy`, `refinance_breakeven`,
  `mirr`; a console **view** (a scenario dashboard / net-worth + spending panels);
  richer recurring detection; multi-currency normalization.

---

*Prepared for review. On approval: scaffold via the devkit, implement `finance/`
engine + tests first (golden-locked), then the tool shims + skill, then wire the
standalone repo (CI/pyproject) and a first green commit.*
