from __future__ import annotations

from typing import Any

import pandas as pd

from marginbt.engine import BacktestEngine
from marginbt.types import BacktestResult


class Portfolio:
    """VBT-like facade that delegates signal backtests to BacktestEngine."""

    @classmethod
    def from_signals(
        cls,
        close: pd.Series,
        entries: pd.Series | None = None,
        exits: pd.Series | None = None,
        *,
        engine: BacktestEngine | None = None,
        **run_kwargs: Any,
    ) -> BacktestResult:
        bt_engine = engine or BacktestEngine()
        return bt_engine.run(close=close, entries=entries, exits=exits, **run_kwargs)


__all__ = ["Portfolio"]
