"""Shared test fixtures for marginbt test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marginbt.execution.rule_based.types import FillPolicyConfig, MarginConfig, MetricsConfig


@pytest.fixture
def sample_ohlc() -> dict[str, pd.Series]:
    """Generate deterministic 360-bar OHLC data for testing.

    Returns a dict with keys: close, open, high, low, entries, short_entries, sl_stop.
    This is the same data used by the regression snapshot.
    """
    n = 360
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    x = np.linspace(0, 12 * np.pi, n)
    close = pd.Series(100 + 2.0 * np.sin(x) + 0.4 * np.sin(3 * x), index=idx, dtype="float64")
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.25
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.25
    long_entries = (close < close.rolling(20).mean().bfill()).shift(1, fill_value=False)
    short_entries = (close > close.rolling(20).mean().bfill()).shift(1, fill_value=False)
    sl_stop = pd.Series(0.02, index=idx, dtype="float64")
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "entries": long_entries.astype(bool),
        "short_entries": short_entries.astype(bool),
        "sl_stop": sl_stop,
    }


@pytest.fixture
def default_margin_cfg() -> MarginConfig:
    """Default MarginConfig for tests."""
    return MarginConfig(leverage=1.0, maintenance_margin_rate=0.005, liquidation_fee_rate=0.0)


@pytest.fixture
def default_metrics_cfg() -> MetricsConfig:
    """Default MetricsConfig for tests."""
    return MetricsConfig(risk_free_annual=0.0, year_days=365)


@pytest.fixture
def default_fill_cfg() -> FillPolicyConfig:
    """Default FillPolicyConfig for tests."""
    return FillPolicyConfig(gap_sl_policy="bar_open", same_bar_conflict_policy="risk_first")
