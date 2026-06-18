"""Transaction normalization & analysis (plan §4F). The deterministic bridge
between a data connector (e.g. a Monarch ``get_transactions`` dump) and the
analysis tools. Pure functions over plain dicts; the @tool shims in tools.py
wrap these. Every tool returns a typed dict of figures plus a short ``summary``.

Purity policy: no I/O, no clock, no randomness. ``datetime`` is used only to
PARSE and do arithmetic on the input rows' own dates — any "next due" notion is
derived from the data, never from the real clock. Currency aggregation uses
Decimal (plan §3) to avoid float drift when summing many transactions; outputs
round to cents at the edge via ``money.round_cents``.
"""

from __future__ import annotations

import re
import statistics
from datetime import date, datetime, timedelta

from . import money

# Words in an explicit type/kind field that indicate an outflow vs an inflow.
_DEBIT_WORDS = {"debit", "expense", "withdrawal", "payment", "purchase", "charge", "fee", "pos", "atm"}
_CREDIT_WORDS = {"credit", "income", "deposit", "refund", "interest", "reimbursement", "payroll"}

__all__ = ["normalize_transactions", "summarize_spending", "detect_recurring"]

# Canonical taxonomy (small, fixed). Everything maps into one of these.
CATEGORIES = [
    "Groceries",
    "Dining",
    "Transport",
    "Utilities",
    "Housing",
    "Income",
    "Entertainment",
    "Shopping",
    "Health",
    "Subscriptions",
    "Transfer",
    "Other",
]

# Deterministic rule table: case-insensitive substring → canonical category.
# Order matters only in that the first hit wins; keep each substring distinctive.
_CATEGORY_RULES: list[tuple[str, str]] = [
    # Transport
    ("uber", "Transport"),
    ("lyft", "Transport"),
    ("shell", "Transport"),
    ("exxon", "Transport"),
    ("chevron", "Transport"),
    ("gas", "Transport"),
    ("fuel", "Transport"),
    ("parking", "Transport"),
    ("transit", "Transport"),
    # Groceries
    ("whole foods", "Groceries"),
    ("safeway", "Groceries"),
    ("trader joe", "Groceries"),
    ("kroger", "Groceries"),
    ("grocery", "Groceries"),
    ("groceries", "Groceries"),
    ("supermarket", "Groceries"),
    # Dining
    ("starbucks", "Dining"),
    ("restaurant", "Dining"),
    ("cafe", "Dining"),
    ("coffee", "Dining"),
    ("dining", "Dining"),
    ("mcdonald", "Dining"),
    ("chipotle", "Dining"),
    ("pizza", "Dining"),
    # Subscriptions
    ("netflix", "Subscriptions"),
    ("spotify", "Subscriptions"),
    ("hulu", "Subscriptions"),
    ("disney+", "Subscriptions"),
    ("subscription", "Subscriptions"),
    ("prime video", "Subscriptions"),
    # Utilities
    ("utilities", "Utilities"),
    ("utility", "Utilities"),
    ("electric", "Utilities"),
    ("water", "Utilities"),
    ("internet", "Utilities"),
    ("comcast", "Utilities"),
    ("verizon", "Utilities"),
    ("at&t", "Utilities"),
    # Housing
    ("rent", "Housing"),
    ("mortgage", "Housing"),
    ("landlord", "Housing"),
    ("hoa", "Housing"),
    # Income
    ("payroll", "Income"),
    ("paycheck", "Income"),
    ("salary", "Income"),
    ("direct deposit", "Income"),
    ("deposit", "Income"),
    ("income", "Income"),
    # Health
    ("pharmacy", "Health"),
    ("doctor", "Health"),
    ("clinic", "Health"),
    ("hospital", "Health"),
    ("gym", "Health"),
    ("fitness", "Health"),
    ("health", "Health"),
    # Entertainment
    ("cinema", "Entertainment"),
    ("movie", "Entertainment"),
    ("theater", "Entertainment"),
    ("concert", "Entertainment"),
    ("entertainment", "Entertainment"),
    # Shopping
    ("amazon", "Shopping"),
    ("target", "Shopping"),
    ("walmart", "Shopping"),
    ("shopping", "Shopping"),
    ("store", "Shopping"),
    # Transfer
    ("transfer", "Transfer"),
    ("venmo", "Transfer"),
    ("zelle", "Transfer"),
]

# Keys an input row might carry for each canonical field (checked in order).
_DATE_KEYS = ("date", "posted", "transaction_date", "transactionDate")
_AMOUNT_KEYS = ("amount", "value")
_MERCHANT_KEYS = ("merchant", "name", "description", "payee")
_CATEGORY_KEYS = ("category",)
_ACCOUNT_KEYS = ("account", "account_name", "accountName")
_CURRENCY_KEYS = ("currency", "currency_code")
_TYPE_KEYS = ("type", "kind")


def _first(row: dict, keys: tuple[str, ...]):
    """First present, non-None value among ``keys``."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _parse_date(value) -> str:
    """Parse a date into a canonical 'YYYY-MM-DD' string. Accepts ISO
    'YYYY-MM-DD' (with an optional time suffix), zero-unpadded ISO like
    '2024-1-5', 'YYYY/MM/DD', and US 'MM/DD/YYYY'. Raises ValueError on
    anything unparseable so the caller can flag the offending row."""
    text = str(value).strip()
    # Tolerate a trailing time component ('2024-01-15T00:00:00') / 'T' or space.
    head = re.split(r"[T ]", text, maxsplit=1)[0]
    # Try strict ISO first (fast path, also validates the calendar date).
    try:
        return date.fromisoformat(head).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(head, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {value!r}")


def _map_category(raw_category, merchant: str) -> str:
    """Prefer an explicit raw category if it maps cleanly; else infer from the
    merchant text; else 'Other'."""
    if raw_category is not None:
        mapped = _match_rules(str(raw_category))
        if mapped is not None:
            return mapped
    return _match_rules(merchant) or "Other"


def _match_rules(text: str) -> str | None:
    """Canonical category for ``text`` via case-insensitive substring rules, or
    None if nothing matches. A value already equal to a canonical name wins."""
    low = text.strip().lower()
    if not low:
        return None
    for canonical in CATEGORIES:
        if low == canonical.lower():
            return canonical
    for needle, canonical in _CATEGORY_RULES:
        # Match on word boundaries so 'gas' doesn't fire inside 'Vegas' nor
        # 'store' inside 'Restore'. Skip the trailing boundary for needles that
        # end in punctuation (e.g. 'disney+', 'at&t'), which have no word edge.
        pattern = r"\b" + re.escape(needle)
        if needle[-1].isalnum():
            pattern += r"\b"
        if re.search(pattern, low):
            return canonical
    return None


def _signed_amount(raw_amount: float, row: dict) -> float:
    """Final signed amount. An explicit type/kind drives the sign (debit/expense
    → negative, credit/income → positive); otherwise the numeric sign is kept."""
    kind = _first(row, _TYPE_KEYS)
    magnitude = abs(raw_amount)
    if kind is not None:
        # Match any indicative word in the type string (e.g. "POS purchase",
        # "ACH credit"), not just an exact token, so multi-word types still
        # drive the sign. Debit wins ties ("credit card purchase" is a debit).
        words = set(str(kind).strip().lower().replace("/", " ").split())
        if words & _DEBIT_WORDS:
            return -magnitude
        if words & _CREDIT_WORDS:
            return magnitude
    return float(raw_amount)


def normalize_transactions(raw: list[dict]) -> dict:
    """Map heterogeneous connector rows to a canonical schema, dedupe, and report.

    Canonical row: ``{date, amount, merchant, category, account, type, currency}``
    where ``amount`` is signed (outflow negative, inflow positive) and ``type`` is
    'expense' when amount<0 else 'income'. Rows with no parseable date or amount
    raise ValueError naming the offending row index. Identical rows (same date,
    amount, merchant, account) are collapsed to one.
    """
    if not isinstance(raw, list):
        raise ValueError("raw must be a list of transaction dicts")

    canonical: list[dict] = []
    seen: set[tuple] = set()
    removed = 0

    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} is not a dict")
        raw_date = _first(row, _DATE_KEYS)
        raw_amount = _first(row, _AMOUNT_KEYS)
        if raw_date is None:
            raise ValueError(f"row {i} has no parseable date")
        if raw_amount is None:
            raise ValueError(f"row {i} has no parseable amount")
        try:
            iso_date = _parse_date(raw_date)
        except (ValueError, TypeError):
            raise ValueError(f"row {i} has an unparseable date: {raw_date!r}") from None
        try:
            amount = _signed_amount(float(raw_amount), row)
        except (ValueError, TypeError):
            raise ValueError(f"row {i} has an unparseable amount: {raw_amount!r}") from None

        merchant = str(_first(row, _MERCHANT_KEYS) or "").strip()
        account = str(_first(row, _ACCOUNT_KEYS) or "").strip()
        currency = str(_first(row, _CURRENCY_KEYS) or "USD").strip().upper()
        category = _map_category(_first(row, _CATEGORY_KEYS), merchant)

        # Dedupe on the canonical full-precision amount so sub-cent-distinct
        # transactions (which round to the same cents) are NOT collapsed.
        key = (iso_date, money.money(amount), merchant, account)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        canonical.append(
            {
                "date": iso_date,
                "amount": money.round_cents(amount),
                "merchant": merchant,
                "category": category,
                "account": account,
                "type": "expense" if amount < 0 else "income",
                "currency": currency,
            }
        )

    currencies = sorted({t["currency"] for t in canonical})
    return {
        "transactions": canonical,
        "count": len(canonical),
        "removed_duplicates": removed,
        "currencies": currencies,
        "summary": (
            f"Normalized {len(canonical)} transaction(s) "
            f"({removed} duplicate(s) removed) across {len(currencies)} "
            f"currency/currencies: {', '.join(currencies) or 'none'}."
        ),
    }


def _in_window(iso_date: str, window: tuple[str, str] | None) -> bool:
    if window is None:
        return True
    start, end = window
    return start <= iso_date <= end


def _group_key(txn: dict, group_by: str) -> str:
    if group_by == "category":
        return txn["category"]
    if group_by == "merchant":
        return txn["merchant"]
    return txn["date"][:7]  # month: 'YYYY-MM'


def summarize_spending(
    transactions: list[dict], group_by: str = "category", window: tuple[str, str] | None = None, top_n: int = 5
) -> dict:
    """Aggregate canonical rows into per-group outflow plus headline totals.

    group_by: 'category' | 'merchant' | 'month'. window: optional inclusive
    (start_iso, end_iso) date filter. Per-group spend is the sum of negative
    amounts expressed as a positive number; ``top`` lists the top_n groups by
    spend. Aggregation is done in Decimal and rounded at the edge.
    """
    if group_by not in ("category", "merchant", "month"):
        raise ValueError("group_by must be 'category', 'merchant', or 'month'")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    rows = [t for t in transactions if _in_window(t["date"], window)]
    groups: dict[str, money.Decimal] = {}
    total_outflow = money.money(0)
    total_inflow = money.money(0)
    for t in rows:
        amt = money.money(t["amount"])
        if amt < 0:
            spend = -amt
            total_outflow += spend
            key = _group_key(t, group_by)
            groups[key] = groups.get(key, money.money(0)) + spend
        else:
            total_inflow += amt

    group_out = {k: money.round_cents(v) for k, v in groups.items()}
    ranked = sorted(group_out.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    top = [{"key": k, "amount": v} for k, v in ranked]
    net = money.round_cents(total_inflow - total_outflow)
    avg = money.round_cents(total_outflow / len(groups)) if groups else 0.0

    return {
        "groups": group_out,
        "top": top,
        "total_outflow": money.round_cents(total_outflow),
        "total_inflow": money.round_cents(total_inflow),
        "net": net,
        "average_per_group": avg,
        "group_by": group_by,
        "summary": (
            f"By {group_by}: ${money.round_cents(total_outflow):,.2f} spent across "
            f"{len(group_out)} group(s), ${money.round_cents(total_inflow):,.2f} in, "
            f"net ${net:,.2f}." + (f" Top: {top[0]['key']} (${top[0]['amount']:,.2f})." if top else "")
        ),
    }


# Cadence targets in days. The median gap is classified to the NEAREST target
# (by relative distance) and accepted only if within ±15% of it; otherwise the
# series is treated as irregular (not recurring). Calendar-aware cadences carry
# the number of months/years to advance for next_due; data-driven ones (weekly,
# biweekly) add the observed median gap instead.
_TOLERANCE = 0.15
_CADENCE_TARGETS: list[tuple[str, float]] = [
    ("weekly", 7.0),
    ("biweekly", 14.0),
    ("monthly", 30.44),
    ("quarterly", 91.31),
    ("annual", 365.25),
]
# Calendar-step (in months) for cadences advanced by month/year arithmetic.
_CADENCE_MONTHS: dict[str, int] = {"monthly": 1, "quarterly": 3, "annual": 12}


def _cadence_for_gap(gap_days: float) -> str | None:
    """Nearest cadence label within ±15% of its target, else None (irregular)."""
    if gap_days <= 0:
        return None
    label, target = min(_CADENCE_TARGETS, key=lambda lt: abs(gap_days - lt[1]))
    if abs(gap_days - target) <= _TOLERANCE * target:
        return label
    return None


def _add_months(d: date, months: int) -> str:
    """Advance ``d`` by whole calendar months, clamping the day to month length."""
    month0 = d.month - 1 + months
    year = d.year + month0 // 12
    month = month0 % 12 + 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day).isoformat()


def _add_cadence(last: str, label: str, median_gap: float) -> str:
    """next_due = last date + one cadence interval, derived from data only.

    Weekly/biweekly advance by the observed median gap (in whole days); monthly,
    quarterly, and annual use calendar-aware, day-clamped month/year arithmetic.
    """
    d = date.fromisoformat(last)
    months = _CADENCE_MONTHS.get(label)
    if months is not None:
        return _add_months(d, months)
    return (d + timedelta(days=round(median_gap))).isoformat()


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


def detect_recurring(transactions: list[dict], detail: str = "summary") -> dict:
    """Infer recurring charges from canonical rows.

    Heuristic: group expenses (amount<0) by merchant; a merchant is recurring
    when it has >=3 charges of similar amount (each within ~5% of the median
    magnitude) at a regular cadence. Cadence comes from the median gap between
    consecutive (sorted) dates, classified to the nearest of {weekly, biweekly,
    monthly, quarterly, annual} within ±15% (else irregular → not recurring).
    next_due = last date + one cadence interval, derived purely from the input
    dates (no real clock). detail='full' adds a per-merchant ``occurrences_list``
    of {date, amount}; 'summary' omits it.
    """
    money.validate_detail(detail)

    by_merchant: dict[str, list[dict]] = {}
    for t in transactions:
        if t["amount"] < 0:
            by_merchant.setdefault(t["merchant"], []).append(t)

    recurring: list[dict] = []
    for merchant, charges in by_merchant.items():
        if len(charges) < 3:
            continue
        charges = sorted(charges, key=lambda c: c["date"])
        magnitudes = [abs(c["amount"]) for c in charges]
        median_amt = statistics.median(magnitudes)
        if median_amt <= 0:
            continue
        if any(abs(m - median_amt) > 0.05 * median_amt for m in magnitudes):
            continue  # amounts not similar enough

        dates = [date.fromisoformat(c["date"]) for c in charges]
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        median_gap = statistics.median(gaps)
        label = _cadence_for_gap(median_gap)
        if label is None:
            continue  # irregular spacing
        last = charges[-1]["date"]
        entry = {
            "merchant": merchant,
            "amount": money.round_cents(median_amt),
            "cadence": label,
            "occurrences": len(charges),
            "next_due": _add_cadence(last, label, median_gap),
        }
        if detail == "full":
            entry["occurrences_list"] = [{"date": c["date"], "amount": c["amount"]} for c in charges]
        recurring.append(entry)

    recurring.sort(key=lambda r: r["merchant"])
    out = {
        "recurring": recurring,
        "count": len(recurring),
        "summary": (
            f"Found {len(recurring)} recurring charge(s): "
            + ", ".join(f"{r['merchant']} ${r['amount']:,.2f}/{r['cadence']}" for r in recurring)
            if recurring
            else "No recurring charges detected."
        ),
    }
    return out
