from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

GapSLPolicy = Literal["bar_open"]
SameBarConflictPolicy = Literal["risk_first"]
SizeType = Literal["risk", "amount", "value", "percent"]
StopEntryPrice = Literal["fillprice", "close", "open"]


@dataclass(frozen=True)
class MarginConfig:
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.0


@dataclass(frozen=True)
class MetricsConfig:
    risk_free_annual: float = 0.03
    year_days: int = 365
    daily_loss_limit_pct: float = 1.0
    kill_switch_drawdown_pct: float = 1.0


@dataclass(frozen=True)
class FillPolicyConfig:
    gap_sl_policy: GapSLPolicy = "bar_open"
    same_bar_conflict_policy: SameBarConflictPolicy = "risk_first"


@dataclass
class BacktestResult:
    """Result of a backtest run.

    Attributes
    ----------
    stats_map : dict
        Key performance metrics (Total Return, Sharpe, Max Drawdown, etc.).
    equity_curve : pd.Series
        Per-bar equity value.
    returns_series : pd.Series
        Per-bar simple returns.
    trades : pd.DataFrame
        Trade log with entry/exit timestamps, prices, PnL, and exit reason.
    exec_meta : dict
        Execution metadata (skip reasons, risk halt counts, etc.).
    used_margin_curve : pd.Series or None
        Per-bar used margin (rule-based mode only).
    free_margin_curve : pd.Series or None
        Per-bar free margin (rule-based mode only).
    """

    stats_map: dict[str, Any]
    equity_curve: pd.Series
    returns_series: pd.Series
    trades: pd.DataFrame
    exec_meta: dict[str, Any]
    used_margin_curve: pd.Series | None = None
    free_margin_curve: pd.Series | None = None

    def stats(self) -> pd.Series:
        return pd.Series(self.stats_map, dtype="object")



@dataclass
class _PendingSignal:
    side: int


@dataclass
class _Position:
    side: int
    qty: float
    entry_price: float
    stop_price: float
    entry_margin: float
    entry_idx: int
    entry_ts: pd.Timestamp
    trailing_high: float = 0.0
    trailing_low: float = float("inf")
    tp_price_level: float = float("nan")


__all__ = [
    "GapSLPolicy",
    "SameBarConflictPolicy",
    "MarginConfig",
    "MetricsConfig",
    "FillPolicyConfig",
    "BacktestResult",
    "_PendingSignal",
    "_Position",
]
