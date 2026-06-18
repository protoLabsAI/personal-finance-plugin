# Cleanup backlog

Deferred quality items surfaced by code review. The bulk were resolved in the
cleanup pass (see git history); only the item below remains, intentionally
deferred.

## Open

- [ ] **`rate`/`irr` step-size-only convergence (tvm.py).** Newton accepts
  `abs(step) < tol` without checking the residual is ~0, so a stall could
  theoretically return a non-root. Deferred deliberately: a naive absolute
  residual guard destabilizes the golden tests because residual magnitude
  scales with cash-flow size — this needs a *scaled* residual check, designed
  and verified carefully against the full golden suite.

## Resolved in the cleanup pass

Cross-module
- [x] Shared `detail` validation → `money.validate_detail` (used by budget ×3,
  disruption, normalize, growth, loans).
- [x] Shared inflation factor → `money.inflation_factor` (inflation.py, growth.py).

normalize.py
- [x] `detect_recurring` cadence gaps → nearest-cadence-within-±15% (quarterly etc. now detected).
- [x] `detect_recurring` `next_due` derived from the observed cadence/median gap.
- [x] `detect_recurring` `detail='full'` now adds an `occurrences_list`.
- [x] Dedupe key uses full-precision amount (no sub-cent collapse).
- [x] `_parse_date` accepts ISO, `MM/DD/YYYY`, `YYYY/MM/DD`, unpadded ISO.

retirement.py
- [x] `safe_withdrawal` horizon documented as horizon-independent (not faked).
- [x] `safe_withdrawal` single result dict; rate/horizon echoed only in `assumptions`.
- [x] `_depletion_age` rounds an age to 1 decimal (not `round_cents`).

disruption.py
- [x] `income_disruption_runway` builds per-month rows only when `detail='full'`.

budget.py
- [x] `budget_breakdown` computes all flags before assembling the result.

integration layer
- [x] `retirement_projection`/`compound_growth` wrapper docstrings clarify the
  inflation-default behavior.
- [x] `currency` documented as reserved (plan §8) in the manifest.
- [x] Rate defaults promoted to `DEFAULT_INFLATION_RATE`/`DEFAULT_WITHDRAWAL_RATE`
  constants with a manifest drift-guard test.
- [x] Tool-name rosters exported from `tools.py` as the single source of truth.
- [x] `build_tools` forwarders moved to module level; only the 4 config-consuming
  tools stay in the closure.
- [x] `time_value_of_money` dispatch collapsed into one table.

tvm.py / loans.py
- [x] `fv`/`pv`/`pmt` reject `rate <= -1` (no silent complex result).
- [x] `debt_payoff_strategy` reports already-cleared (balance 0) debts in `payoff_order`.
