# Cleanup backlog

Deferred quality items surfaced by code review. None are correctness bugs for
normal inputs (those were fixed inline); these are efficiency, simplification,
altitude, and convention improvements to tackle **after** the integration layer.

## Cross-module

- [ ] **Shared `detail` validation.** The `if detail not in ("summary","full"): raise ValueError` check is copy-pasted across `budget.py` (×3), `disruption.py`, `normalize.py`. Extract a `money._validate_detail(detail)` helper (alongside `periods_per_year`/`round_cents`).
- [ ] **Shared inflation factor.** `(1+rate)**years` is hand-rolled in `inflation.py`, `growth.py` (real-FV discount), and `retirement.py` (real return). Centralize as one helper so the sign/convention lives in one place.

## normalize.py

- [ ] **`detect_recurring` cadence gaps.** Median gaps of ~16–27d and ~32–359d match no bucket (e.g. quarterly ~90d, every-3-weeks ~21d) and the charge is dropped as irregular. Widen/parameterize the cadence buckets.
- [ ] **`detect_recurring` `next_due`.** Weekly/biweekly add a fixed 7/14-day step rather than the observed median gap, so `next_due` isn't truly derived from the data spacing.
- [ ] **`detect_recurring` unused `detail` param.** Validated but has no behavioral effect ("full" returns identical output). Either drop it or make "full" add per-occurrence rows.
- [ ] **Dedupe key edges.** Key rounds amount to cents (sub-cent distinct txns collapse) and a missing `account` ('') makes same day/amount/merchant always dedupe. Consider including a stable index or raw amount.
- [ ] **`_parse_date` format surface.** Only ISO `YYYY-MM-DD` (with optional time) parses; `01/15/2024`, `2024-1-5`, epoch ints raise. Documented as "raises", but the accepted surface is narrower than connectors may emit — broaden parsing.

## retirement.py

- [ ] **`safe_withdrawal` `horizon` is inert.** It doesn't affect `sustainable_income = nest_egg*rate` or `required_nest_egg = target_income/rate`; it only appears in echoed dicts and summary text. Either make the model horizon-sensitive or document it as a pure echo.
- [ ] **`safe_withdrawal` duplicate echo.** `rate`/`horizon` appear at the top level AND inside `assumptions`; the two return branches duplicate the whole dict shape. `retirement_projection` echoes assumptions only in `assumptions` — pick one convention across both tools.
- [ ] **`_depletion_age` uses `round_cents` on an age.** Currency rounding applied to an age yields e.g. "age 68.37". Use a sensible age precision (note: existing tests pin 2-decimal values, so update those tests when changing).

## disruption.py

- [ ] **`income_disruption_runway` builds rows in summary mode.** The full per-month rows list (up to `_MAX_MONTHS`) is built + rounded on every call but only attached when `detail=='full'`. Gate row construction behind `detail=='full'` (mirror `loans._amortization_rows`).

## budget.py

- [ ] **`budget_breakdown` flags aliasing.** The `needs_over_50pct` flag is appended to `out["flags"]` after `out` is built, relying on list identity and on the summary being assembled afterward. Compute all flags before assembling `out`/summary.

## tvm.py / loans.py (earlier deferrals)

- [ ] **`rate`/`irr` step-size-only convergence.** Newton accepts `abs(step) < tol` without checking the residual is ~0, so a stall could return a non-root. (Theoretical; a naive absolute-residual guard risks destabilizing golden tests because residual magnitude scales with cash-flow size — needs a scaled check.)
- [ ] **`fv`/`pv`/`pmt` complex results.** For `rate < -1` and non-integer `nper`, `(1+rate)**nper` returns a complex number silently. Invalid financial domain, but unguarded.
- [ ] **`debt_payoff_strategy` drops balance-0 debts.** A debt input with `balance == 0` is filtered out of `active` and never appears in `payoff_order`.
