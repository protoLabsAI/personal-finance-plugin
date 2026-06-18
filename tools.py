"""Agent-facing @tool shims over the finance/ engine (plan §4). Thin: each tool
parses/forwards to a pure finance function and returns its typed dict. The
engine is fully unit-tested on its own; these wrappers add only argument shape,
config-default injection, and the LLM-facing descriptions.

`build_tools(config)` returns ``(core, advanced)``. register() binds the core-14
by default and the advanced set only when ``advanced_tools`` is enabled (§5).
"""

from __future__ import annotations

from langchain_core.tools import tool

from .finance import budget, disruption, growth, inflation, loans, normalize, retirement, tvm

__all__ = ["build_tools"]


def build_tools(config: dict):
    """Construct the tool objects, closing over plugin config for defaults.

    Returns ``(core_tools, advanced_tools)`` — two lists of LangChain tools.
    """
    inflation_default = config.get("default_inflation_rate", 0.03)
    withdrawal_default = config.get("default_withdrawal_rate", 0.04)

    # ── TVM / cash-flow ──────────────────────────────────────────────────────
    @tool
    def time_value_of_money(
        solve_for: str,
        rate: float | None = None,
        nper: float | None = None,
        pmt: float | None = None,
        pv: float | None = None,
        fv: float = 0.0,
        when: int = 0,
    ) -> dict:
        """Solve a time-value-of-money problem for one unknown.

        solve_for is one of 'fv', 'pv', 'pmt', 'nper', or 'rate'; supply the
        other variables. Cash-flow sign convention: money paid out is negative,
        money received is positive (e.g. a loan principal you receive is
        positive pv, the payment you make is negative pmt). when=0 for
        end-of-period (default) or 1 for begin-of-period.
        """
        required = {
            "fv": ("rate", "nper", "pmt", "pv"),
            "pv": ("rate", "nper", "pmt"),
            "pmt": ("rate", "nper", "pv"),
            "nper": ("rate", "pmt", "pv"),
            "rate": ("nper", "pmt", "pv"),
        }
        if solve_for not in required:
            raise ValueError(f"solve_for must be one of {sorted(required)}; got {solve_for!r}")
        provided = {"rate": rate, "nper": nper, "pmt": pmt, "pv": pv}
        missing = [name for name in required[solve_for] if provided[name] is None]
        if missing:
            raise ValueError(f"solving for {solve_for} requires: {', '.join(missing)}")

        if solve_for == "fv":
            value = tvm.fv(rate, nper, pmt, pv, when)
        elif solve_for == "pv":
            value = tvm.pv(rate, nper, pmt, fv, when)
        elif solve_for == "pmt":
            value = tvm.pmt(rate, nper, pv, fv, when)
        elif solve_for == "nper":
            value = tvm.nper(rate, pmt, pv, fv, when)
        else:  # rate
            value = tvm.rate(nper, pmt, pv, fv, when)
        return {"solve_for": solve_for, "value": value, "when": when}

    @tool
    def npv(rate: float, values: list[float]) -> dict:
        """Net present value of a cash-flow series at the given periodic discount
        rate. The first value is at t=0 (undiscounted). Outflows negative."""
        result = tvm.npv(rate, values)
        return {"npv": result, "rate": rate, "periods": len(values)}

    @tool
    def irr(values: list[float]) -> dict:
        """Internal rate of return: the periodic rate where the cash-flow series'
        NPV is zero. Needs at least one sign change (e.g. an initial outflow then
        inflows)."""
        return {"irr": tvm.irr(values), "periods": len(values)}

    @tool
    def mirr(values: list[float], finance_rate: float, reinvest_rate: float) -> dict:
        """Modified internal rate of return: outflows discounted at finance_rate,
        inflows compounded at reinvest_rate. Needs both an inflow and an outflow."""
        return {"mirr": tvm.mirr(values, finance_rate, reinvest_rate)}

    # ── loans & debt ─────────────────────────────────────────────────────────
    @tool
    def loan_payment(principal: float, apr: float, term_years: float, freq: str = "monthly") -> dict:
        """Level payment for a fully-amortizing loan plus total interest and total
        paid. apr is the nominal annual rate (0.06 == 6%)."""
        return loans.loan_payment(principal, apr, term_years, freq)

    @tool
    def loan_amortization(
        principal: float, apr: float, term_years: float, freq: str = "monthly", detail: str = "summary"
    ) -> dict:
        """Loan amortization. detail='summary' returns totals; detail='full' adds
        the per-period schedule (period, payment, principal, interest, balance)."""
        return loans.loan_amortization(principal, apr, term_years, freq, detail)

    @tool
    def loan_payoff_extra(
        principal: float, apr: float, term_years: float, extra_payment: float, freq: str = "monthly"
    ) -> dict:
        """How paying extra each period shortens a loan's term and cuts total
        interest, versus the baseline schedule."""
        return loans.loan_payoff_extra(principal, apr, term_years, extra_payment, freq)

    @tool
    def debt_payoff_strategy(debts: list[dict], budget: float, method: str = "avalanche") -> dict:
        """Order multiple debts for payoff and project months + total interest.
        Each debt is {name, balance, apr, min_payment}. method 'avalanche'
        (highest APR first) or 'snowball' (smallest balance first); both are
        always compared. budget must cover the sum of minimum payments."""
        return loans.debt_payoff_strategy(debts, budget, method)

    @tool
    def debt_to_income(gross_monthly_income: float, housing_payment: float, other_debt_payments: float = 0.0) -> dict:
        """Debt-to-income ratios: front-end (housing/income) and back-end (all
        debt/income), as decimals (0.36 == 36%)."""
        return loans.debt_to_income(gross_monthly_income, housing_payment, other_debt_payments)

    # ── savings & investment ─────────────────────────────────────────────────
    @tool
    def compound_growth(
        starting_balance: float,
        contribution: float,
        rate: float,
        years: float,
        freq: str = "monthly",
        inflation_rate: float | None = None,
        detail: str = "summary",
    ) -> dict:
        """Project an investment of a starting balance plus regular contributions
        growing at an annual rate. Reports nominal future value, contributions vs
        growth, and the inflation-adjusted (real) value. detail='full' adds a
        yearly schedule."""
        infl = inflation_default if inflation_rate is None else inflation_rate
        return growth.compound_growth(starting_balance, contribution, rate, years, freq, infl, detail)

    @tool
    def savings_goal(
        target: float, years: float, starting_balance: float = 0.0, rate: float = 0.0, freq: str = "monthly"
    ) -> dict:
        """Required periodic contribution to reach a savings target in a given
        number of years, accounting for a starting balance and expected return."""
        return growth.savings_goal(target, years, starting_balance, rate, freq)

    # ── retirement ───────────────────────────────────────────────────────────
    @tool
    def retirement_projection(
        current_age: int,
        retirement_age: int,
        life_expectancy: int,
        current_savings: float,
        annual_contribution: float,
        expected_return: float,
        desired_annual_income: float,
        inflation_rate: float | None = None,
        social_security: dict | None = None,
    ) -> dict:
        """Project retirement: grow savings to retirement, then the sustainable
        annual income vs the desired income, any surplus/shortfall, and the age
        funds run out. social_security is {annual_benefit, claim_age, include}.
        Output is a projection under stated assumptions, not financial advice."""
        infl = inflation_default if inflation_rate is None else inflation_rate
        return retirement.retirement_projection(
            current_age,
            retirement_age,
            life_expectancy,
            current_savings,
            annual_contribution,
            expected_return,
            desired_annual_income,
            infl,
            social_security,
        )

    @tool
    def safe_withdrawal(
        nest_egg: float | None = None, target_income: float | None = None, rate: float | None = None, horizon: int = 30
    ) -> dict:
        """Safe-withdrawal (4%/Trinity) estimate. Give nest_egg to get the
        sustainable annual income, OR target_income to get the required nest egg
        (exactly one). rate defaults to the configured withdrawal rate."""
        r = withdrawal_default if rate is None else rate
        return retirement.safe_withdrawal(nest_egg, target_income, r, horizon)

    # ── budgeting & household ────────────────────────────────────────────────
    @tool
    def budget_breakdown(
        monthly_income: float,
        expenses: dict,
        needs_categories: list[str] | None = None,
        detail: str = "summary",
    ) -> dict:
        """Budget breakdown: surplus/deficit, savings rate, 50/30/20 targets, and
        flags. expenses maps category -> monthly amount; needs_categories (if
        given) classifies which categories are needs vs wants."""
        return budget.budget_breakdown(monthly_income, expenses, needs_categories, detail)

    @tool
    def emergency_fund_target(
        essential_monthly_expenses: float, months: int = 6, current_reserve: float = 0.0, monthly_saving: float = 0.0
    ) -> dict:
        """Emergency-fund target (months of essential expenses), the gap to it,
        and time to fund at a given monthly saving rate."""
        return budget.emergency_fund_target(essential_monthly_expenses, months, current_reserve, monthly_saving)

    @tool
    def net_worth(assets: dict, liabilities: dict, detail: str = "summary") -> dict:
        """Net worth from assets and liabilities (each a name -> amount map).
        detail='full' adds the per-item breakdowns."""
        return budget.net_worth(assets, liabilities, detail)

    @tool
    def cashflow_forecast(
        starting_cash: float, monthly_income: float, monthly_expenses: float, months: int, detail: str = "summary"
    ) -> dict:
        """Month-by-month cash-flow forecast: ending balance, minimum balance, and
        count of shortfall months. detail='full' adds the monthly rows."""
        return budget.cashflow_forecast(starting_cash, monthly_income, monthly_expenses, months, detail)

    # ── transaction normalization & analysis ─────────────────────────────────
    @tool
    def normalize_transactions(raw: list[dict]) -> dict:
        """Normalize a raw transaction dump (e.g. from a Monarch get_transactions
        call, or CSV-shaped rows) into a canonical, de-duplicated, sign-normalized,
        category-mapped dataset ready for analysis."""
        return normalize.normalize_transactions(raw)

    @tool
    def summarize_spending(
        transactions: list[dict],
        group_by: str = "category",
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 5,
    ) -> dict:
        """Summarize normalized transactions grouped by 'category', 'merchant', or
        'month': per-group spend, totals, and the top-N groups. start_date and
        end_date (ISO 'YYYY-MM-DD'), if both given, restrict to that inclusive
        window."""
        if (start_date is None) != (end_date is None):
            raise ValueError("provide both start_date and end_date, or neither")
        window = (start_date, end_date) if start_date is not None else None
        return normalize.summarize_spending(transactions, group_by, window, top_n)

    @tool
    def detect_recurring(transactions: list[dict], detail: str = "summary") -> dict:
        """Infer recurring charges (subscriptions, bills) from normalized
        transactions: merchant, amount, cadence, and next due date."""
        return normalize.detect_recurring(transactions, detail)

    # ── income disruption ────────────────────────────────────────────────────
    @tool
    def income_disruption_runway(
        liquid_savings: float,
        essential_monthly_burn: float,
        severance: float = 0.0,
        weekly_benefit: float = 0.0,
        max_benefit_weeks: int = 0,
        other_monthly_income: float = 0.0,
        discretionary_cut_pct: float = 0.0,
        detail: str = "summary",
    ) -> dict:
        """Months of runway during an income disruption given savings, essential
        burn, severance, unemployment benefits, other income, and an optional
        discretionary spending cut. Projection under stated assumptions, not
        financial advice."""
        return disruption.income_disruption_runway(
            liquid_savings,
            essential_monthly_burn,
            severance,
            weekly_benefit,
            max_benefit_weeks,
            other_monthly_income,
            discretionary_cut_pct,
            detail,
        )

    @tool
    def income_replacement_need(
        prior_monthly_takehome: float,
        benefits_monthly: float = 0.0,
        other_monthly_income: float = 0.0,
        desired_coverage_months: int = 6,
    ) -> dict:
        """The monthly income gap to replace after a job loss and the emergency
        fund needed to cover it for the desired number of months."""
        return disruption.income_replacement_need(
            prior_monthly_takehome, benefits_monthly, other_monthly_income, desired_coverage_months
        )

    # ── inflation ────────────────────────────────────────────────────────────
    @tool
    def inflation_adjust(amount: float, years: float, rate: float | None = None, direction: str = "to_present") -> dict:
        """Adjust an amount for inflation. direction='to_present' gives the real
        (today's-dollars) value of a future amount; 'to_future' gives the nominal
        amount needed later to match today's purchasing power. rate defaults to
        the configured inflation rate."""
        r = inflation_default if rate is None else rate
        return inflation.inflation_adjust(amount, r, years, direction)

    core = [
        time_value_of_money,
        loan_payment,
        loan_amortization,
        debt_payoff_strategy,
        compound_growth,
        savings_goal,
        retirement_projection,
        budget_breakdown,
        emergency_fund_target,
        net_worth,
        normalize_transactions,
        summarize_spending,
        income_disruption_runway,
        inflation_adjust,
        # npv + irr are exposed directly by default (plan §10/Q1); mirr is advanced.
        npv,
        irr,
    ]
    advanced = [
        mirr,
        loan_payoff_extra,
        debt_to_income,
        safe_withdrawal,
        cashflow_forecast,
        income_replacement_need,
        detect_recurring,
    ]
    return core, advanced
