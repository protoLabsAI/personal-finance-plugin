"""Tests for finance/normalize.py — the deterministic bridge between a data
connector dump and the analysis tools (plan §4F).

All fixtures are small, hand-built transaction lists with concrete expected
outputs. Every function is pure & deterministic: no clock, no I/O — any
"next due"/"today" notion is derived only from the input rows' own dates.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def normalize(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.normalize")


# ── normalize_transactions: key-name robustness ───────────────────────────────
def test_normalize_maps_varied_key_names(normalize):
    raw = [
        {
            "posted": "2024-01-15",
            "value": -42.50,
            "name": "Whole Foods Market",
            "category": "groceries",
            "accountName": "Checking",
            "currency_code": "USD",
        }
    ]
    out = normalize.normalize_transactions(raw)
    t = out["transactions"][0]
    assert t["date"] == "2024-01-15"
    assert t["amount"] == -42.50
    assert t["merchant"] == "Whole Foods Market"
    assert t["category"] == "Groceries"
    assert t["account"] == "Checking"
    assert t["currency"] == "USD"
    assert t["type"] == "expense"


def test_normalize_alternate_keys_transactiondate_payee_account(normalize):
    raw = [{"transactionDate": "2024-02-01", "amount": 1000.0, "payee": "ACME Payroll", "account": "Savings"}]
    out = normalize.normalize_transactions(raw)
    t = out["transactions"][0]
    assert t["date"] == "2024-02-01"
    assert t["merchant"] == "ACME Payroll"
    assert t["account"] == "Savings"
    assert t["currency"] == "USD"  # default


def test_normalize_uses_description_as_merchant_fallback(normalize):
    raw = [{"date": "2024-03-03", "amount": -9.99, "description": "Netflix.com"}]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["merchant"] == "Netflix.com"


# ── sign normalization ────────────────────────────────────────────────────────
def test_normalize_preserves_numeric_sign_when_no_type(normalize):
    raw = [
        {"date": "2024-01-01", "amount": -50.0, "merchant": "Shell"},
        {"date": "2024-01-02", "amount": 200.0, "merchant": "Deposit"},
    ]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["amount"] == -50.0
    assert out["transactions"][0]["type"] == "expense"
    assert out["transactions"][1]["amount"] == 200.0
    assert out["transactions"][1]["type"] == "income"


def test_normalize_debit_credit_type_sets_sign(normalize):
    # Magnitudes are positive in the source; type drives the sign.
    raw = [
        {"date": "2024-01-01", "amount": 30.0, "merchant": "Cafe", "type": "debit"},
        {"date": "2024-01-02", "amount": 1500.0, "merchant": "Paycheck", "type": "credit"},
    ]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["amount"] == -30.0
    assert out["transactions"][0]["type"] == "expense"
    assert out["transactions"][1]["amount"] == 1500.0
    assert out["transactions"][1]["type"] == "income"


def test_normalize_expense_income_kind_sets_sign(normalize):
    raw = [
        {"date": "2024-01-01", "amount": 30.0, "merchant": "Cafe", "kind": "expense"},
        {"date": "2024-01-02", "amount": 1500.0, "merchant": "Paycheck", "kind": "income"},
    ]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["amount"] == -30.0
    assert out["transactions"][1]["amount"] == 1500.0


def test_normalize_unrecognized_debit_type_still_outflow(normalize):
    # A positive magnitude carrying a debit-like type word ("POS purchase")
    # must be recorded as an expense, not silently passed through as income.
    raw = [{"date": "2024-01-01", "amount": 50.0, "merchant": "Shop", "type": "POS purchase"}]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["amount"] == -50.0
    assert out["transactions"][0]["type"] == "expense"


# ── category mapping ──────────────────────────────────────────────────────────
def test_normalize_category_no_substring_false_match(normalize):
    # Keywords must match on word boundaries: 'gas' must not fire inside 'Vegas',
    # nor 'store' inside 'Restore'. Both should fall through to 'Other'.
    raw = [
        {"date": "2024-01-01", "amount": -200.0, "merchant": "Las Vegas Hotel"},
        {"date": "2024-01-02", "amount": -90.0, "merchant": "Restore Spa"},
    ]
    out = normalize.normalize_transactions(raw)
    assert [t["category"] for t in out["transactions"]] == ["Other", "Other"]


def test_normalize_category_real_keyword_still_matches(normalize):
    # The boundary fix must not break legitimate whole-word matches.
    raw = [
        {"date": "2024-01-01", "amount": -40.0, "merchant": "Shell Gas Station"},
        {"date": "2024-01-02", "amount": -12.0, "merchant": "Whole Foods Market"},
    ]
    out = normalize.normalize_transactions(raw)
    assert [t["category"] for t in out["transactions"]] == ["Transport", "Groceries"]


def test_normalize_category_from_merchant_keywords(normalize):
    raw = [
        {"date": "2024-01-01", "amount": -20.0, "merchant": "Uber Trip"},
        {"date": "2024-01-02", "amount": -15.0, "merchant": "Spotify"},
        {"date": "2024-01-03", "amount": -60.0, "merchant": "Safeway #123"},
        {"date": "2024-01-04", "amount": -5.0, "merchant": "Starbucks"},
        {"date": "2024-01-05", "amount": -1500.0, "merchant": "Monthly Rent"},
        {"date": "2024-01-06", "amount": 3000.0, "merchant": "Payroll ACME"},
    ]
    out = normalize.normalize_transactions(raw)
    cats = [t["category"] for t in out["transactions"]]
    assert cats == ["Transport", "Subscriptions", "Groceries", "Dining", "Housing", "Income"]


def test_normalize_explicit_category_preferred_over_merchant(normalize):
    # Raw category 'utilities' maps cleanly even though merchant has no keyword.
    raw = [{"date": "2024-01-01", "amount": -120.0, "merchant": "PG&E", "category": "utilities"}]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["category"] == "Utilities"


def test_normalize_unmappable_falls_back_to_other(normalize):
    raw = [{"date": "2024-01-01", "amount": -10.0, "merchant": "Mystery LLC"}]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["category"] == "Other"


def test_normalize_unmappable_raw_category_inferred_from_merchant(normalize):
    # 'misc' doesn't map; fall through to merchant keyword 'shell' -> Transport.
    raw = [{"date": "2024-01-01", "amount": -40.0, "merchant": "Shell Gas", "category": "misc"}]
    out = normalize.normalize_transactions(raw)
    assert out["transactions"][0]["category"] == "Transport"


# ── dedupe ────────────────────────────────────────────────────────────────────
def test_normalize_dedupes_identical_rows(normalize):
    raw = [
        {"date": "2024-01-01", "amount": -10.0, "merchant": "Cafe", "account": "Checking"},
        {"date": "2024-01-01", "amount": -10.0, "merchant": "Cafe", "account": "Checking"},
        {"date": "2024-01-01", "amount": -10.0, "merchant": "Cafe", "account": "Savings"},
    ]
    out = normalize.normalize_transactions(raw)
    assert out["count"] == 2
    assert out["removed_duplicates"] == 1


# ── multi-currency ────────────────────────────────────────────────────────────
def test_normalize_collects_sorted_unique_currencies(normalize):
    raw = [
        {"date": "2024-01-01", "amount": -10.0, "merchant": "A", "currency": "USD"},
        {"date": "2024-01-02", "amount": -10.0, "merchant": "B", "currency": "EUR"},
        {"date": "2024-01-03", "amount": -10.0, "merchant": "C", "currency_code": "GBP"},
        {"date": "2024-01-04", "amount": -10.0, "merchant": "D"},  # default USD
    ]
    out = normalize.normalize_transactions(raw)
    assert out["currencies"] == ["EUR", "GBP", "USD"]


# ── validation ────────────────────────────────────────────────────────────────
def test_normalize_raises_on_missing_date(normalize):
    raw = [{"amount": -10.0, "merchant": "A"}]
    with pytest.raises(ValueError):
        normalize.normalize_transactions(raw)


def test_normalize_raises_on_missing_amount(normalize):
    raw = [{"date": "2024-01-01", "merchant": "A"}]
    with pytest.raises(ValueError):
        normalize.normalize_transactions(raw)


def test_normalize_raises_lists_bad_row_index(normalize):
    raw = [
        {"date": "2024-01-01", "amount": -10.0, "merchant": "A"},
        {"date": "not-a-date", "amount": -10.0, "merchant": "B"},
    ]
    with pytest.raises(ValueError, match="1"):
        normalize.normalize_transactions(raw)


def test_normalize_has_summary_and_count(normalize):
    raw = [{"date": "2024-01-01", "amount": -10.0, "merchant": "A"}]
    out = normalize.normalize_transactions(raw)
    assert out["count"] == 1
    assert isinstance(out["summary"], str) and out["summary"]


# ── summarize_spending ────────────────────────────────────────────────────────
def _canon(date, amount, merchant, category, currency="USD"):
    return {
        "date": date,
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "account": "Checking",
        "type": "expense" if amount < 0 else "income",
        "currency": currency,
    }


@pytest.fixture
def sample_canon():
    return [
        _canon("2024-01-05", -100.0, "Safeway", "Groceries"),
        _canon("2024-01-20", -50.0, "Whole Foods", "Groceries"),
        _canon("2024-02-10", -40.0, "Uber", "Transport"),
        _canon("2024-02-15", -60.0, "Lyft", "Transport"),
        _canon("2024-02-28", 3000.0, "Payroll", "Income"),
        _canon("2024-03-01", -200.0, "Rent", "Housing"),
    ]


def test_summarize_by_category(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon, group_by="category")
    assert out["groups"]["Groceries"] == 150.0
    assert out["groups"]["Transport"] == 100.0
    assert out["groups"]["Housing"] == 200.0
    assert "Income" not in out["groups"]  # inflow, not spend
    assert out["total_outflow"] == 450.0
    assert out["total_inflow"] == 3000.0
    assert out["net"] == 2550.0


def test_summarize_by_merchant(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon, group_by="merchant")
    assert out["groups"]["Safeway"] == 100.0
    assert out["groups"]["Lyft"] == 60.0


def test_summarize_by_month(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon, group_by="month")
    assert out["groups"]["2024-01"] == 150.0
    assert out["groups"]["2024-02"] == 100.0
    assert out["groups"]["2024-03"] == 200.0


def test_summarize_top_n(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon, group_by="category", top_n=2)
    assert [g["key"] for g in out["top"]] == ["Housing", "Groceries"]
    assert out["top"][0]["amount"] == 200.0
    assert len(out["top"]) == 2


def test_summarize_window_filter_inclusive(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon, group_by="category", window=("2024-02-01", "2024-02-28"))
    assert out["groups"] == {"Transport": 100.0}
    assert out["total_inflow"] == 3000.0


def test_summarize_rejects_bad_group_by(normalize, sample_canon):
    with pytest.raises(ValueError):
        normalize.summarize_spending(sample_canon, group_by="weekday")


def test_summarize_has_summary(normalize, sample_canon):
    out = normalize.summarize_spending(sample_canon)
    assert isinstance(out["summary"], str) and out["summary"]


# ── detect_recurring ──────────────────────────────────────────────────────────
def test_detect_recurring_monthly_with_next_due(normalize):
    txns = [
        _canon("2024-01-05", -15.99, "Netflix", "Subscriptions"),
        _canon("2024-02-05", -15.99, "Netflix", "Subscriptions"),
        _canon("2024-03-05", -15.99, "Netflix", "Subscriptions"),
    ]
    out = normalize.detect_recurring(txns)
    assert out["count"] == 1
    r = out["recurring"][0]
    assert r["merchant"] == "Netflix"
    assert r["cadence"] == "monthly"
    assert r["occurrences"] == 3
    assert r["amount"] == 15.99
    assert r["next_due"] == "2024-04-05"  # last date + ~1 month, from data only


def test_detect_recurring_weekly(normalize):
    txns = [
        _canon("2024-01-01", -12.0, "Gym", "Health"),
        _canon("2024-01-08", -12.0, "Gym", "Health"),
        _canon("2024-01-15", -12.0, "Gym", "Health"),
        _canon("2024-01-22", -12.0, "Gym", "Health"),
    ]
    out = normalize.detect_recurring(txns)
    r = next(x for x in out["recurring"] if x["merchant"] == "Gym")
    assert r["cadence"] == "weekly"
    assert r["next_due"] == "2024-01-29"


def test_detect_recurring_ignores_too_few_or_irregular(normalize):
    txns = [
        _canon("2024-01-01", -5.0, "OneOff", "Other"),
        _canon("2024-01-02", -5.0, "OneOff", "Other"),  # only 2 charges
        _canon("2024-01-01", -10.0, "Sporadic", "Other"),
        _canon("2024-02-15", -90.0, "Sporadic", "Other"),  # amounts not similar
        _canon("2024-04-30", -10.0, "Sporadic", "Other"),
    ]
    out = normalize.detect_recurring(txns)
    assert out["count"] == 0


def test_detect_recurring_detail_summary_hides_list(normalize):
    txns = [
        _canon("2024-01-05", -15.99, "Netflix", "Subscriptions"),
        _canon("2024-02-05", -15.99, "Netflix", "Subscriptions"),
        _canon("2024-03-05", -15.99, "Netflix", "Subscriptions"),
    ]
    out = normalize.detect_recurring(txns, detail="summary")
    assert out["count"] == 1
    assert isinstance(out["summary"], str) and out["summary"]


def test_detect_recurring_rejects_bad_detail(normalize):
    with pytest.raises(ValueError):
        normalize.detect_recurring([], detail="loud")
