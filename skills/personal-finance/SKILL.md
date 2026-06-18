---
name: personal-finance
description: >-
  Use when the user asks to plan, model, project, or analyze personal or family
  finances — loan/mortgage/auto payments, debt payoff, savings & investment
  growth, retirement, budgeting, net worth, emergency funds, cash flow,
  inflation, job-loss/income-disruption runway, or analyzing a transaction
  export. Also use whenever a money calculation would otherwise be done by hand.
---

# Personal Finance

This plugin provides **deterministic** finance tools. Same inputs always give the
same exact figures — the model picks the tool and arguments; the math is done by
the tool, not by you.

## Core rules

1. **Never do the arithmetic yourself.** Do not hand-compute a loan payment,
   future value, amortization figure, payoff timeline, or projection. Call the
   tool. If no single tool fits, use `time_value_of_money` (solve for
   `fv`/`pv`/`pmt`/`nper`/`rate`) rather than computing it in your head.

2. **Compose with a data source when one is connected.** When the user wants
   analysis of *their* finances and a connector like the `monarch-money` plugin
   is available, pull the real figures first — e.g. `monarch__get_accounts`,
   `monarch__get_transactions` — then pass the raw transaction dump straight into
   `normalize_transactions`, and run `summarize_spending` / `detect_recurring` /
   `net_worth` / the projection tools on the normalized result. Do not ask the
   user to retype numbers a connector can fetch. When no connector is present,
   ask for the specific inputs the chosen tool needs.

3. **State assumptions and units.** Echo the rate, frequency, inflation
   assumption, Social Security inclusion, and horizon you used, and report the
   currency and any as-of context. The modeling tools return an `assumptions`
   block and a `summary` — surface those.

4. **Frame projections as projections, not advice.** For retirement, debt,
   savings, and income-disruption outputs, present results as *projections under
   stated assumptions* and include a brief "this is not financial advice" note.
   The relevant tools already include this caveat in their `summary`.

5. **Pick the narrowest tool and keep output small.** Prefer the most specific
   tool for the question. Leave `detail="summary"` unless the user explicitly
   wants a full schedule (amortization rows, month-by-month cash flow, yearly
   growth); only then pass `detail="full"`.

## Choosing a tool

Tools marked **(advanced)** are only bound when the operator has enabled
`advanced_tools` in the plugin config. If an advanced tool isn't available, fall
back to the closest default tool or ask the operator to enable the advanced set.

- **Loan / mortgage / auto payment, total interest** → `loan_payment`; full
  schedule → `loan_amortization`; effect of paying extra → `loan_payoff_extra`
  **(advanced)**.
- **Paying off multiple debts** → `debt_payoff_strategy` (snowball vs avalanche).
- **Borrowing capacity / lender ratios** → `debt_to_income` **(advanced)**.
- **Investment / savings growth over time** → `compound_growth`; contribution
  needed to hit a target → `savings_goal`.
- **Retirement readiness** → `retirement_projection`; sustainable spending or
  required nest egg → `safe_withdrawal` **(advanced)**.
- **Monthly budget, 50/30/20, savings rate** → `budget_breakdown`; emergency
  fund sizing → `emergency_fund_target`; net worth → `net_worth`; runway of cash
  over coming months → `cashflow_forecast` **(advanced)**.
- **Job loss / income gap** → `income_disruption_runway`;
  `income_replacement_need` **(advanced)**.
- **Inflation / real vs nominal value** → `inflation_adjust`.
- **Generic time-value-of-money** → `time_value_of_money`.
- **Cash-flow series valuation** → `npv`, `irr`; modified IRR → `mirr`
  **(advanced)**.
- **Analyzing a transaction export** → `normalize_transactions` first, then
  `summarize_spending`, then `detect_recurring` **(advanced)**.

## Sign conventions

The high-level tools take **intuitive positive numbers** (a loan principal, a
balance, a contribution, a target). Only `time_value_of_money`, `npv`, `irr`,
and `mirr` use the cash-flow sign convention directly: money paid out is
negative, money received is positive.
