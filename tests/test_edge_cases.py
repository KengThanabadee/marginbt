"""Edge case tests for the rule-based execution engine."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from marginbt.execution.rule_based.engine import run_rule_based_execution
from marginbt.execution.rule_based.types import (
    BacktestResult,
    FillPolicyConfig,
    MarginConfig,
    MetricsConfig,
)


def _run_case(
    *,
    open_: Iterable[float],
    high: Iterable[float],
    low: Iterable[float],
    close: Iterable[float],
    bb_middle: Iterable[float],
    long_entries: Iterable[bool],
    sl_pct: Iterable[float],
    init_cash: float = 100.0,
    leverage: float = 1.0,
    risk_per_trade: float = 0.25,
    metrics_cfg: MetricsConfig | None = None,
    fill_cfg: FillPolicyConfig | None = None,
    min_qty: float | None = None,
    enforce_min_constraints: bool = False,
    instrument: str = "PAXG_USDT_Perp",
) -> BacktestResult:
    open_values = list(open_)
    idx = pd.date_range("2025-01-01", periods=len(open_values), freq="1h")
    return run_rule_based_execution(
        index=idx,
        open_=pd.Series(open_values, index=idx),
        high=pd.Series(list(high), index=idx),
        low=pd.Series(list(low), index=idx),
        close=pd.Series(list(close), index=idx),
        bb_middle=pd.Series(list(bb_middle), index=idx),
        long_entries=pd.Series(list(long_entries), index=idx, dtype=bool),
        short_entries=pd.Series(False, index=idx, dtype=bool),
        sl_pct=pd.Series(list(sl_pct), index=idx),
        allow_short=False,
        init_cash=init_cash,
        risk_per_trade=risk_per_trade,
        fee_per_side=0.0,
        slippage=0.0,
        min_qty=min_qty,
        enforce_min_constraints=enforce_min_constraints,
        skip_if_below_min=True,
        instrument=instrument,
        margin_cfg=MarginConfig(leverage=leverage, maintenance_margin_rate=0.005, liquidation_fee_rate=0.0),
        metrics_cfg=metrics_cfg or MetricsConfig(risk_free_annual=0.0, year_days=365),
        fill_cfg=fill_cfg or FillPolicyConfig(gap_sl_policy="bar_open", same_bar_conflict_policy="risk_first"),
    )


class TestGapSL:
    """Tests for gap-through stop-loss behavior."""

    def test_gap_sl_fills_at_bar_open(self) -> None:
        result = _run_case(
            open_=[100.0, 100.0, 90.0],
            high=[101.0, 101.0, 101.0],
            low=[99.0, 99.0, 89.0],
            close=[100.0, 100.0, 100.0],
            bb_middle=[120.0, 120.0, 120.0],
            long_entries=[True, False, False],
            sl_pct=[0.05, 0.05, 0.05],
            leverage=1.0,
            risk_per_trade=0.25,
        )
        trades = result.trades
        assert not trades.empty, "Expected at least one trade"
        assert str(trades.iloc[0]["reason"]).startswith("stop_open"), (
            f"Expected stop_open exit reason, got {trades.iloc[0]['reason']}"
        )


class TestSameBarConflict:
    """Tests for same-bar SL/TP conflict resolution."""

    def test_risk_first_policy_prefers_sl(self) -> None:
        result = _run_case(
            open_=[100.0, 100.0],
            high=[100.5, 106.0],
            low=[99.5, 94.0],
            close=[100.0, 100.0],
            bb_middle=[105.0, 200.0],
            long_entries=[True, False],
            sl_pct=[0.05, 0.05],
            leverage=1.0,
            risk_per_trade=0.25,
        )
        trades = result.trades
        assert not trades.empty, "Expected at least one trade"
        assert trades.iloc[0]["reason"] == "stop_intrabar_conflict", (
            f"Expected stop_intrabar_conflict, got {trades.iloc[0]['reason']}"
        )


class TestLiquidation:
    """Tests for liquidation mechanics."""

    def test_liquidation_triggers_on_large_drop(self) -> None:
        result = _run_case(
            open_=[100.0, 100.0],
            high=[101.0, 101.0],
            low=[99.0, 80.0],
            close=[100.0, 85.0],
            bb_middle=[200.0, 200.0],
            long_entries=[True, False],
            sl_pct=[0.02, 0.02],
            leverage=10.0,
            risk_per_trade=0.5,
        )
        trades = result.trades
        assert not trades.empty, "Expected at least one trade"
        assert str(trades.iloc[0]["reason"]).startswith("liquidation"), (
            f"Expected liquidation exit reason, got {trades.iloc[0]['reason']}"
        )


class TestEmptyInput:
    """Tests for empty input handling."""

    def test_empty_index_returns_zero_period(self) -> None:
        result = run_rule_based_execution(
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
            open_=pd.Series(dtype="float64"),
            high=pd.Series(dtype="float64"),
            low=pd.Series(dtype="float64"),
            close=pd.Series(dtype="float64"),
            bb_middle=pd.Series(dtype="float64"),
            long_entries=pd.Series(dtype=bool),
            short_entries=pd.Series(dtype=bool),
            sl_pct=pd.Series(dtype="float64"),
            allow_short=False,
            init_cash=100.0,
            risk_per_trade=0.25,
            fee_per_side=0.0,
            slippage=0.0,
            min_qty=None,
            enforce_min_constraints=False,
            skip_if_below_min=True,
            instrument="PAXG_USDT_Perp",
        )
        assert int(result.stats().get("Period", -1)) == 0, "Empty input should return Period=0"
