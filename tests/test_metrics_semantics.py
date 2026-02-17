"""Semantic tests for metric calculations."""

from __future__ import annotations

import math

import pandas as pd

from marginbt.execution.rule_based.math_utils import _calc_sharpe


class TestSharpeSemantics:
    """Verify Sharpe ratio edge cases with constant return streams."""

    def test_constant_positive_returns_gives_positive_inf(self) -> None:
        sharpe = _calc_sharpe(pd.Series([0.01] * 50), rf_annual=0.0, bars_per_year=365.0 * 24.0)
        assert math.isinf(sharpe) and sharpe > 0, f"Expected +inf, got {sharpe}"

    def test_constant_negative_returns_gives_negative_inf(self) -> None:
        sharpe = _calc_sharpe(pd.Series([-0.01] * 50), rf_annual=0.0, bars_per_year=365.0 * 24.0)
        assert math.isinf(sharpe) and sharpe < 0, f"Expected -inf, got {sharpe}"

    def test_constant_zero_returns_gives_zero(self) -> None:
        sharpe = _calc_sharpe(pd.Series([0.0] * 50), rf_annual=0.0, bars_per_year=365.0 * 24.0)
        assert sharpe == 0.0, f"Expected 0.0, got {sharpe}"
