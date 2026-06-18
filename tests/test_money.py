"""Tests for finance/money.py — shared rounding / frequency / rate helpers."""

from __future__ import annotations

import importlib
from decimal import Decimal

import pytest


@pytest.fixture
def money(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.money")


# ── periods_per_year ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "freq,expected",
    [
        ("annual", 1),
        ("semiannual", 2),
        ("quarterly", 4),
        ("monthly", 12),
        ("biweekly", 26),
        ("weekly", 52),
        ("daily", 365),
    ],
)
def test_periods_per_year_known(money, freq, expected):
    assert money.periods_per_year(freq) == expected


def test_periods_per_year_is_case_insensitive(money):
    assert money.periods_per_year("Monthly") == 12


def test_periods_per_year_rejects_unknown(money):
    with pytest.raises(ValueError):
        money.periods_per_year("fortnightly")


# ── periodic_rate ─────────────────────────────────────────────────────────────
def test_periodic_rate_monthly(money):
    assert money.periodic_rate(0.06, "monthly") == pytest.approx(0.005)


def test_periodic_rate_annual_is_unchanged(money):
    assert money.periodic_rate(0.06, "annual") == pytest.approx(0.06)


# ── round_cents ───────────────────────────────────────────────────────────────
def test_round_cents_half_up(money):
    assert money.round_cents(2.675) == 2.68


def test_round_cents_rounds_up_at_exact_half(money):
    assert money.round_cents(1.005) == 1.01


def test_round_cents_returns_float(money):
    assert isinstance(money.round_cents(3.14159), float)
    assert money.round_cents(3.14159) == 3.14


# ── money (Decimal aggregation) ───────────────────────────────────────────────
def test_money_returns_decimal(money):
    assert money.money(10.1) == Decimal("10.1")


def test_money_sum_avoids_float_drift(money):
    total = sum((money.money(0.1) for _ in range(3)), money.money(0))
    assert total == Decimal("0.3")  # 0.1*3 != 0.3 in float, but exact in Decimal
