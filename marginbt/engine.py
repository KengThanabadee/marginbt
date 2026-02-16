"""BacktestEngine - margin-aware backtest engine with VBT-like API.

Like ``vbt.Portfolio.from_signals()`` but with leverage, margin, liquidation,
daily-loss halt, and global kill-switch support.

Quick start::

    from marginbt import BacktestEngine

    engine = BacktestEngine(init_cash=100, fees=0.00045, leverage=10)
    result = engine.run(
        close=close, open=open_1h, high=high_1h, low=low_1h,
        entries=my_entries, sl_stop=my_sl_pct,
    )
    result.stats()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from marginbt.execution.rule_based.engine import run_rule_based_execution
from marginbt.execution.rule_based.types import (
    BacktestResult,
    FillPolicyConfig,
    GapSLPolicy,
    MarginConfig,
    MetricsConfig,
    SameBarConflictPolicy,
    SizeType,
    StopEntryPrice,
)


@dataclass(frozen=True)
class EngineConfig:
    """All engine-level settings grouped for serialization/logging."""

    init_cash: float = 100.0
    fees: float = 0.00045
    slippage: float = 0.0002
    freq: str = "1h"
    leverage: float = 10.0
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.0
    daily_loss_limit_pct: float = 0.015
    kill_switch_drawdown_pct: float = 0.015
    risk_free_annual: float = 0.03
    year_days: int = 365
    gap_sl_policy: GapSLPolicy = "bar_open"
    same_bar_conflict_policy: SameBarConflictPolicy = "risk_first"


class BacktestEngine:
    """Margin-aware backtest engine.

    Parameters
    ----------
    init_cash : float
        Starting capital (default 100 USDT).
    fees : float
        Fee rate per side, e.g. 0.00045 = 0.045%.
    slippage : float
        Adverse slippage factor applied on entry and exit.
    freq : str
        Bar frequency label (for display only, e.g. ``"1h"``).
    leverage : float
        Maximum leverage.  Used for position-size cap and margin accounting.
    maintenance_margin_rate : float
        Exchange maintenance-margin rate for liquidation check.
    liquidation_fee_rate : float
        Extra fee charged on liquidation events.
    daily_loss_limit_pct : float
        Intraday loss threshold that halts new entries for the rest of the day.
    kill_switch_drawdown_pct : float
        Peak-to-trough drawdown threshold that halts the strategy globally.
    risk_free_annual : float
        Annual risk-free rate used in Sharpe calculation.
    year_days : int
        Calendar days per year (365 for crypto).
    gap_sl_policy : str
        How to fill a stop that gaps past the stop level (``"bar_open"``).
    same_bar_conflict_policy : str
        Priority when SL and TP trigger on the same bar (``"risk_first"``).
    """

    def __init__(
        self,
        init_cash: float = 100.0,
        fees: float = 0.00045,
        slippage: float = 0.0002,
        freq: str = "1h",
        leverage: float = 10.0,
        maintenance_margin_rate: float = 0.005,
        liquidation_fee_rate: float = 0.0,
        daily_loss_limit_pct: float = 0.015,
        kill_switch_drawdown_pct: float = 0.015,
        risk_free_annual: float = 0.03,
        year_days: int = 365,
        gap_sl_policy: GapSLPolicy = "bar_open",
        same_bar_conflict_policy: SameBarConflictPolicy = "risk_first",
    ) -> None:
        self.config = EngineConfig(
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
            leverage=leverage,
            maintenance_margin_rate=maintenance_margin_rate,
            liquidation_fee_rate=liquidation_fee_rate,
            daily_loss_limit_pct=daily_loss_limit_pct,
            kill_switch_drawdown_pct=kill_switch_drawdown_pct,
            risk_free_annual=risk_free_annual,
            year_days=year_days,
            gap_sl_policy=gap_sl_policy,
            same_bar_conflict_policy=same_bar_conflict_policy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        close: pd.Series,
        *,
        open: pd.Series | None = None,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
        entries: pd.Series | None = None,
        exits: pd.Series | None = None,
        short_entries: pd.Series | None = None,
        short_exits: pd.Series | None = None,
        sl_stop: pd.Series | float | None = None,
        sl_trail: bool = False,
        tp_stop: float | None = None,
        tp_price: pd.Series | float | None = None,
        price: pd.Series | float | None = None,
        stop_entry_price: StopEntryPrice = "fillprice",
        size: float | None = None,
        size_type: SizeType = "risk",
        risk_per_trade: float = 0.0025,
        direction: str = "longonly",
        min_qty: float | None = None,
        enforce_min_constraints: bool = True,
        skip_if_below_min: bool = True,
        instrument: str | None = None,
    ) -> BacktestResult:
        """Run a backtest with the given signals and price data.

        Parameters
        ----------
        close : pd.Series
            Close prices (required).  Index must be a sorted DatetimeIndex.
        open, high, low : pd.Series, optional
            OHLC price data.  If omitted, ``close`` is used as fallback.
        entries : pd.Series[bool], optional
            Long entry signals.
        exits : pd.Series[bool], optional
            Long exit signals (boolean TP).
        short_entries : pd.Series[bool], optional
            Short entry signals.
        short_exits : pd.Series[bool], optional
            Short exit signals (boolean TP).
        sl_stop : pd.Series[float] or float, optional
            Stop-loss as a fraction of entry price (e.g. 0.02 = 2%).
            Can be a constant or a per-bar Series.
        sl_trail : bool
            If True, the stop-loss trails the highest high (long) or
            lowest low (short) since entry, ratcheting up/down.
        tp_stop : float, optional
            Take-profit as a fraction of entry price (e.g. 0.2 = 20%).
            VBT-compatible.  Overrides ``tp_price`` and ``exits``.
        tp_price : pd.Series[float] or float, optional
            Take-profit price level.  When price crosses this level,
            the engine triggers a TP exit.  Overrides ``exits`` if both given.
        price : pd.Series[float] or float, optional
            Custom fill price for entries.  Default is ``close`` (VBT default).
        stop_entry_price : str
            Which price SL/TP percentages are calculated from:
            ``"fillprice"`` (default, after slippage), ``"close"``, ``"open"``.
        size : float, optional
            Position size value.  Interpretation depends on ``size_type``.
        size_type : str
            How ``size`` is interpreted:
            ``"risk"`` (default) = risk-based with ``risk_per_trade``,
            ``"amount"`` = fixed qty, ``"value"`` = fixed notional,
            ``"percent"`` = fraction of equity as notional.
        risk_per_trade : float
            Fraction of equity risked per trade (default 0.25%).
            Only used when ``size_type="risk"``.
        direction : str
            ``"longonly"``, ``"shortonly"``, or ``"both"``.
        min_qty : float, optional
            Minimum order quantity (exchange constraint).
        enforce_min_constraints : bool
            Whether to enforce ``min_qty`` checks.
        skip_if_below_min : bool
            Skip entry if computed qty is below ``min_qty``.
        instrument : str, optional
            Instrument label for metadata.

        Returns
        -------
        BacktestResult
            Result object with ``.stats()``, ``.trades``,
            ``.equity_curve``, and ``.exec_meta``.
        """
        cfg = self.config
        idx = close.index

        # --- Defaults ------------------------------------------------
        open_s = open if open is not None else close.copy()
        high_s = high if high is not None else close.copy()
        low_s = low if low is not None else close.copy()

        allow_short = direction in ("shortonly", "both")

        if entries is None:
            entries = pd.Series(False, index=idx, dtype=bool)
        if short_entries is None:
            short_entries = pd.Series(False, index=idx, dtype=bool)

        # --- SL: scalar -> Series ------------------------------------
        if sl_stop is None:
            sl_pct = pd.Series(0.0, index=idx, dtype="float64")
        elif isinstance(sl_stop, (int, float)):
            sl_pct = pd.Series(float(sl_stop), index=idx, dtype="float64")
        else:
            sl_pct = sl_stop

        # --- TP: resolve to bb_middle-style price level or boolean ---
        if tp_price is not None:
            if isinstance(tp_price, (int, float)):
                bb_middle = pd.Series(float(tp_price), index=idx, dtype="float64")
            else:
                bb_middle = tp_price
        elif exits is not None:
            # Derive a synthetic TP price from boolean exit signals.
            # When exit is True, set TP price to current close so
            # the engine triggers a TP on that bar.
            bb_middle = close.where(exits, float("nan"))
        else:
            bb_middle = pd.Series(float("nan"), index=idx, dtype="float64")

        # --- Price override: scalar -> Series -------------------------
        # VBT default fill price is close.  Internal engine default is open.
        # So we pass close as override to match VBT convention.
        if price is not None:
            if isinstance(price, (int, float)):
                price_override = pd.Series(float(price), index=idx, dtype="float64")
            else:
                price_override = price
        else:
            price_override = close  # VBT default: fill at close

        # --- Delegate to internal engine -----------------------------
        margin_cfg = MarginConfig(
            leverage=cfg.leverage,
            maintenance_margin_rate=cfg.maintenance_margin_rate,
            liquidation_fee_rate=cfg.liquidation_fee_rate,
        )
        metrics_cfg = MetricsConfig(
            risk_free_annual=cfg.risk_free_annual,
            year_days=cfg.year_days,
            daily_loss_limit_pct=cfg.daily_loss_limit_pct,
            kill_switch_drawdown_pct=cfg.kill_switch_drawdown_pct,
        )
        fill_cfg = FillPolicyConfig(
            gap_sl_policy=cfg.gap_sl_policy,
            same_bar_conflict_policy=cfg.same_bar_conflict_policy,
        )

        base_meta: dict[str, Any] = {
            "engine": "backtest_engine",
            "instrument": instrument,
            "freq": cfg.freq,
            "direction": direction,
        }

        return run_rule_based_execution(
            index=idx,
            open_=open_s,
            high=high_s,
            low=low_s,
            close=close,
            bb_middle=bb_middle,
            long_entries=entries,
            short_entries=short_entries,
            sl_pct=sl_pct,
            allow_short=allow_short,
            init_cash=cfg.init_cash,
            risk_per_trade=risk_per_trade,
            fee_per_side=cfg.fees,
            slippage=cfg.slippage,
            min_qty=min_qty,
            enforce_min_constraints=enforce_min_constraints,
            skip_if_below_min=skip_if_below_min,
            instrument=instrument,
            margin_cfg=margin_cfg,
            metrics_cfg=metrics_cfg,
            fill_cfg=fill_cfg,
            base_meta=base_meta,
            sl_trail=sl_trail,
            tp_stop_pct=tp_stop,
            entry_price_override=price_override,
            stop_entry_price=stop_entry_price,
            size_override=size,
            size_type=size_type,
        )

    def describe(self) -> dict[str, Any]:
        """Return a human-readable dict of all engine settings."""
        cfg = self.config
        return {
            "init_cash": cfg.init_cash,
            "fees": cfg.fees,
            "slippage": cfg.slippage,
            "freq": cfg.freq,
            "leverage": cfg.leverage,
            "maintenance_margin_rate": cfg.maintenance_margin_rate,
            "liquidation_fee_rate": cfg.liquidation_fee_rate,
            "daily_loss_limit_pct": cfg.daily_loss_limit_pct,
            "kill_switch_drawdown_pct": cfg.kill_switch_drawdown_pct,
            "risk_free_annual": cfg.risk_free_annual,
            "year_days": cfg.year_days,
            "gap_sl_policy": cfg.gap_sl_policy,
            "same_bar_conflict_policy": cfg.same_bar_conflict_policy,
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"BacktestEngine(init_cash={cfg.init_cash}, fees={cfg.fees}, "
            f"slippage={cfg.slippage}, leverage={cfg.leverage}, "
            f"freq={cfg.freq!r})"
        )


__all__ = ["BacktestEngine", "EngineConfig"]


