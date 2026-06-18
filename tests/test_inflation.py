"""Tests for finance/inflation.py — inflation adjustment (plan §4H)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def inflation(plugin):
    return importlib.import_module(plugin.__name__ + ".finance.inflation")


# ── inflation_adjust ───────────────────────────────────────────────────────────
def test_to_present_discounts_future_dollars(inflation):
    out = inflation.inflation_adjust(1000, 0.03, 10, "to_present")
    assert out["adjusted"] == pytest.approx(1000 / 1.03**10, abs=0.01)
    assert out["adjusted"] == pytest.approx(744.09, abs=0.01)


def test_to_future_inflates_todays_dollars(inflation):
    out = inflation.inflation_adjust(1000, 0.03, 10, "to_future")
    assert out["adjusted"] == pytest.approx(1000 * 1.03**10, abs=0.01)
    assert out["adjusted"] == pytest.approx(1343.92, abs=0.01)


def test_defaults_to_present(inflation):
    out = inflation.inflation_adjust(1000, 0.03, 10)
    assert out["direction"] == "to_present"
    assert out["adjusted"] == pytest.approx(744.09, abs=0.01)


def test_years_zero_returns_amount_unchanged(inflation):
    present = inflation.inflation_adjust(1000, 0.03, 0, "to_present")
    future = inflation.inflation_adjust(1000, 0.03, 0, "to_future")
    assert present["adjusted"] == pytest.approx(1000.0, abs=0.01)
    assert future["adjusted"] == pytest.approx(1000.0, abs=0.01)


def test_zero_rate_returns_amount_unchanged(inflation):
    out = inflation.inflation_adjust(1000, 0.0, 10, "to_future")
    assert out["adjusted"] == pytest.approx(1000.0, abs=0.01)


def test_round_trip_present_then_future(inflation):
    present = inflation.inflation_adjust(1000, 0.03, 10, "to_present")
    future = inflation.inflation_adjust(present["adjusted"], 0.03, 10, "to_future")
    assert future["adjusted"] == pytest.approx(1000.0, abs=0.01)


def test_returns_full_shape(inflation):
    out = inflation.inflation_adjust(1000, 0.03, 10, "to_present")
    assert set(out) >= {"amount", "rate", "years", "direction", "adjusted", "summary"}
    assert out["amount"] == 1000
    assert out["rate"] == 0.03
    assert out["years"] == 10
    assert isinstance(out["summary"], str) and out["summary"]


def test_rejects_unknown_direction(inflation):
    with pytest.raises(ValueError):
        inflation.inflation_adjust(1000, 0.03, 10, "sideways")


def test_rejects_negative_years(inflation):
    with pytest.raises(ValueError):
        inflation.inflation_adjust(1000, 0.03, -1, "to_present")


def test_rejects_rate_at_or_below_minus_one(inflation):
    # rate <= -100% makes (1+rate) non-positive: division by zero / complex.
    # Must raise a clean ValueError rather than crash.
    with pytest.raises(ValueError):
        inflation.inflation_adjust(1000, -1.0, 10, "to_present")
    with pytest.raises(ValueError):
        inflation.inflation_adjust(1000, -2.0, 2.5, "to_future")
