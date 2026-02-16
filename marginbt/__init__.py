"""marginbt - margin-aware execution backtest engine with VBT-like ergonomics.

Quick start::

    import marginbt as mbt

    engine = mbt.BacktestEngine(init_cash=100, fees=0.00045, leverage=10)
    result = engine.run(close=close, entries=entries, sl_stop=sl_pct)

    # VBT-like sugar API
    result2 = mbt.Portfolio.from_signals(close, entries=entries, exits=exits)
"""

from marginbt.engine import BacktestEngine, EngineConfig
from marginbt.execution import (
    BacktestResult,
    FillPolicyConfig,
    GapSLPolicy,
    MarginConfig,
    MetricsConfig,
    SameBarConflictPolicy,
    SizeType,
    StopEntryPrice,
)
from marginbt.portfolio import Portfolio

__all__ = [
    "BacktestEngine",
    "EngineConfig",
    "Portfolio",
    "BacktestResult",
    "GapSLPolicy",
    "SameBarConflictPolicy",
    "SizeType",
    "StopEntryPrice",
    "MarginConfig",
    "MetricsConfig",
    "FillPolicyConfig",
]
