from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from marginbt.execution.rule_based.math_utils import (
    _adverse_entry_price,
    _adverse_exit_price,
    _bars_per_year,
    _calc_return,
    _calc_sharpe,
    _calc_trade_pnl,
    _maintenance_margin,
    _position_equity,
    _select_signal_side,
    _to_bool_series,
    _to_float_series,
)
from marginbt.execution.rule_based.types import (
    BacktestResult,
    FillPolicyConfig,
    MarginConfig,
    MetricsConfig,
    SizeType,
    StopEntryPrice,
    _PendingSignal,
    _Position,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutable state container — passed between helpers to avoid deep nesting
# ---------------------------------------------------------------------------

@dataclass
class _EngineState:
    """Mutable state tracked across the bar-by-bar loop."""

    balance: float
    position: _Position | None = None
    pending: _PendingSignal | None = None

    # Counters
    attempted_signals: int = 0
    executed_signals: int = 0
    skipped_invalid_sl: int = 0
    skipped_min_qty: int = 0
    skipped_missing_min_table: int = 0
    skipped_position_busy: int = 0
    skipped_end_of_data: int = 0
    skipped_risk_halt: int = 0

    total_fees_paid: float = 0.0
    liquidation_count: int = 0
    risk_halt_daily_count: int = 0
    risk_halt_global_triggered: bool = False
    risk_halt_global_count: int = 0
    forced_close_by_risk_control: int = 0
    forced_close_reason_counts: dict[str, int] | None = None

    # Day tracking
    current_day: pd.Timestamp | None = None
    day_start_equity: float = 0.0
    daily_halted: bool = False
    global_halted: bool = False
    peak_equity: float = 0.0
    last_equity_close: float = 0.0

    # Trade records
    trades: list[dict[str, Any]] | None = None
    tx_days: set[pd.Timestamp] | None = None

    # Curves
    equity_values: list[float] | None = None
    used_margin_values: list[float] | None = None
    free_margin_values: list[float] | None = None

    def __post_init__(self) -> None:
        if self.forced_close_reason_counts is None:
            self.forced_close_reason_counts = {}
        if self.trades is None:
            self.trades = []
        if self.tx_days is None:
            self.tx_days = set()
        if self.equity_values is None:
            self.equity_values = []
        if self.used_margin_values is None:
            self.used_margin_values = []
        if self.free_margin_values is None:
            self.free_margin_values = []


# ---------------------------------------------------------------------------
# Helper: close position
# ---------------------------------------------------------------------------

def _close_position(
    state: _EngineState,
    reason: str,
    ts: pd.Timestamp,
    raw_exit_price: float,
    liq_exit: bool,
    slippage: float,
    fee_per_side: float,
    m_cfg: MarginConfig,
) -> None:
    """Close the current position and record the trade."""
    if state.position is None:
        return
    side = state.position.side
    qty = state.position.qty
    exit_price = _adverse_exit_price(float(raw_exit_price), side, slippage)
    exit_notional = abs(qty) * exit_price
    pnl = _calc_trade_pnl(side, qty, state.position.entry_price, exit_price)
    exit_fee = exit_notional * fee_per_side
    liq_fee = exit_notional * m_cfg.liquidation_fee_rate if liq_exit else 0.0
    state.balance += pnl - exit_fee - liq_fee
    if liq_exit:
        state.balance = max(0.0, state.balance)
    state.total_fees_paid += exit_fee + liq_fee
    if liq_exit:
        state.liquidation_count += 1

    assert state.tx_days is not None
    state.tx_days.add(pd.Timestamp(ts).normalize())

    assert state.trades is not None
    state.trades.append(
        {
            "entry_ts": state.position.entry_ts,
            "exit_ts": pd.Timestamp(ts),
            "side": "long" if side > 0 else "short",
            "qty": qty,
            "entry_price": state.position.entry_price,
            "exit_price": exit_price,
            "gross_pnl": pnl,
            "fees_paid": exit_fee + liq_fee,
            "net_pnl": pnl - exit_fee - liq_fee,
            "reason": reason,
        }
    )

    logger.debug(
        "Closed %s position: qty=%.6f entry=%.4f exit=%.4f pnl=%.4f reason=%s",
        "LONG" if side > 0 else "SHORT",
        qty,
        state.position.entry_price,
        exit_price,
        pnl - exit_fee - liq_fee,
        reason,
    )
    state.position = None


# ---------------------------------------------------------------------------
# Helper: fill pending signal → open position
# ---------------------------------------------------------------------------

def _fill_pending_signal(
    state: _EngineState,
    i: int,
    ts: pd.Timestamp,
    sl_now: float,
    c: float,
    o: float,
    *,
    entry_price_override: pd.Series | None,
    stop_entry_price: StopEntryPrice,
    size_type: SizeType,
    size_override: float | None,
    risk_per_trade: float,
    slippage: float,
    fee_per_side: float,
    m_cfg: MarginConfig,
    tp_stop_pct: float | None,
    min_qty: float | None,
    enforce_min_constraints: bool,
    skip_if_below_min: bool,
) -> None:
    """Attempt to fill a pending signal. Updates state in-place."""
    if state.pending is None or state.position is not None:
        return
    if state.daily_halted or state.global_halted:
        return

    # --- Determine raw fill price ---
    raw_fill = float(entry_price_override.iloc[i]) if entry_price_override is not None else o

    valid_sl = bool(np.isfinite(sl_now) and sl_now > 0 and raw_fill > 0)
    if not valid_sl:
        state.skipped_invalid_sl += 1
        state.pending = None
        return

    side = state.pending.side
    entry_price = _adverse_entry_price(raw_fill, side, slippage)

    # --- Stop reference price ---
    if stop_entry_price == "fillprice":
        stop_ref = entry_price
    elif stop_entry_price == "close":
        stop_ref = c
    elif stop_entry_price == "open":
        stop_ref = o
    else:
        stop_ref = entry_price

    # --- Sizing ---
    balance = state.balance
    if size_type == "risk":
        stop_distance = sl_now * raw_fill
        estimated_notional = (balance * risk_per_trade / sl_now) if sl_now > 0 else 0.0
        estimated_fee = estimated_notional * fee_per_side
        effective_balance = max(0.0, balance - estimated_fee)
        risk_cash = effective_balance * risk_per_trade
        qty_risk = risk_cash / stop_distance if stop_distance > 0 else 0.0
        qty_lev = (effective_balance * m_cfg.leverage) / entry_price if entry_price > 0 else 0.0
        qty_base = max(0.0, min(qty_risk, qty_lev))
    elif size_type == "amount":
        qty_base = float(size_override) if size_override is not None else 0.0
        qty_lev = (max(0.0, balance) * m_cfg.leverage) / entry_price if entry_price > 0 else 0.0
        qty_base = max(0.0, min(qty_base, qty_lev))
    elif size_type == "value":
        notional_target = float(size_override) if size_override is not None else 0.0
        qty_base = notional_target / entry_price if entry_price > 0 else 0.0
        qty_lev = (max(0.0, balance) * m_cfg.leverage) / entry_price if entry_price > 0 else 0.0
        qty_base = max(0.0, min(qty_base, qty_lev))
    elif size_type == "percent":
        pct = float(size_override) if size_override is not None else 0.0
        notional_target = max(0.0, balance) * pct
        qty_base = notional_target / entry_price if entry_price > 0 else 0.0
        qty_lev = (max(0.0, balance) * m_cfg.leverage) / entry_price if entry_price > 0 else 0.0
        qty_base = max(0.0, min(qty_base, qty_lev))
    else:
        qty_base = 0.0

    order_notional = abs(qty_base) * entry_price

    missing_min_table = bool(enforce_min_constraints and min_qty is None)
    if enforce_min_constraints and min_qty is not None:
        below_qty = qty_base < float(min_qty)
        dynamic_min_notional = float(min_qty) * entry_price
        below_dynamic_notional = order_notional < dynamic_min_notional
    else:
        below_qty = False
        below_dynamic_notional = False

    rejected = False
    if missing_min_table:
        rejected = True
    if skip_if_below_min and enforce_min_constraints and min_qty is not None and (
        below_qty or below_dynamic_notional
    ):
        rejected = True
    if qty_base <= 0:
        rejected = True

    if rejected:
        if missing_min_table:
            state.skipped_missing_min_table += 1
        if below_qty:
            state.skipped_min_qty += 1
        state.pending = None
        return

    stop_price = stop_ref * (1.0 - sl_now) if side > 0 else stop_ref * (1.0 + sl_now)

    # --- Per-trade TP from tp_stop_pct ---
    if tp_stop_pct is not None:
        trade_tp_level = stop_ref * (1.0 + tp_stop_pct) if side > 0 else stop_ref * (1.0 - tp_stop_pct)
    else:
        trade_tp_level = float("nan")

    notional = order_notional
    entry_margin = notional / m_cfg.leverage
    entry_fee = notional * fee_per_side
    state.balance -= entry_fee
    state.total_fees_paid += entry_fee

    assert state.tx_days is not None
    state.tx_days.add(ts.normalize())

    state.position = _Position(
        side=side,
        qty=qty_base,
        entry_price=entry_price,
        stop_price=stop_price,
        entry_margin=entry_margin,
        entry_idx=i,
        entry_ts=ts,
        trailing_high=entry_price,
        trailing_low=entry_price,
        tp_price_level=trade_tp_level,
    )
    state.executed_signals += 1
    state.pending = None

    logger.debug(
        "Opened %s position: qty=%.6f price=%.4f sl=%.4f tp=%.4f",
        "LONG" if side > 0 else "SHORT",
        qty_base,
        entry_price,
        stop_price,
        trade_tp_level,
    )


# ---------------------------------------------------------------------------
# Helper: check exit conditions (SL, TP, liquidation)
# ---------------------------------------------------------------------------

def _check_exit_conditions(
    state: _EngineState,
    ts: pd.Timestamp,
    o: float,
    h: float,
    low_price: float,
    c: float,
    tp_level: float,
    sl_now: float,
    *,
    sl_trail: bool,
    slippage: float,
    fee_per_side: float,
    m_cfg: MarginConfig,
    f_cfg: FillPolicyConfig,
) -> None:
    """Check and execute SL/TP/liquidation exits on the current position."""
    if state.position is None:
        return

    side = state.position.side
    qty = state.position.qty
    long_pos = side > 0

    # --- Trailing stop: update stop_price ---
    if sl_trail and np.isfinite(sl_now) and sl_now > 0:
        if long_pos:
            state.position.trailing_high = max(state.position.trailing_high, h)
            new_stop = state.position.trailing_high * (1.0 - sl_now)
            if new_stop > state.position.stop_price:
                state.position.stop_price = new_stop
        else:
            state.position.trailing_low = min(state.position.trailing_low, low_price)
            new_stop = state.position.trailing_low * (1.0 + sl_now)
            if new_stop < state.position.stop_price:
                state.position.stop_price = new_stop

    # --- Resolve effective TP level ---
    eff_tp = state.position.tp_price_level if np.isfinite(state.position.tp_price_level) else tp_level

    # --- Check at open price ---
    open_liq = _position_equity(side, qty, state.position.entry_price, state.position.entry_margin, o) <= _maintenance_margin(
        qty, o, m_cfg.maintenance_margin_rate
    )
    gap_stop = (o <= state.position.stop_price) if long_pos else (o >= state.position.stop_price)
    gap_tp = (o >= eff_tp) if long_pos and np.isfinite(eff_tp) else False
    if not long_pos and np.isfinite(eff_tp):
        gap_tp = o <= eff_tp

    normal_exit_reason: str | None = None
    normal_exit_price: float | None = None

    if open_liq:
        _close_position(state, reason="liquidation_open", ts=ts, raw_exit_price=float(o), liq_exit=True,
                        slippage=slippage, fee_per_side=fee_per_side, m_cfg=m_cfg)
        return
    elif gap_stop or gap_tp:
        if gap_stop and gap_tp:
            if f_cfg.same_bar_conflict_policy == "risk_first":
                normal_exit_reason = "stop_open_conflict"
                normal_exit_price = o if f_cfg.gap_sl_policy == "bar_open" else state.position.stop_price
            else:
                normal_exit_reason = "tp_open_conflict"
                normal_exit_price = o
        elif gap_stop:
            normal_exit_reason = "stop_open"
            normal_exit_price = o if f_cfg.gap_sl_policy == "bar_open" else state.position.stop_price
        else:
            normal_exit_reason = "tp_open"
            normal_exit_price = o
    else:
        # --- Check intrabar ---
        stop_hit = (low_price <= state.position.stop_price) if long_pos else (h >= state.position.stop_price)
        tp_hit = (h >= eff_tp) if long_pos and np.isfinite(eff_tp) else False
        if not long_pos and np.isfinite(eff_tp):
            tp_hit = low_price <= eff_tp

        worst_mark = low_price if long_pos else h
        intrabar_liq = _position_equity(
            side, qty, state.position.entry_price, state.position.entry_margin, worst_mark
        ) <= _maintenance_margin(qty, worst_mark, m_cfg.maintenance_margin_rate)

        if intrabar_liq:
            _close_position(state, reason="liquidation_intrabar", ts=ts, raw_exit_price=float(worst_mark), liq_exit=True,
                            slippage=slippage, fee_per_side=fee_per_side, m_cfg=m_cfg)
            return
        elif stop_hit and tp_hit:
            if f_cfg.same_bar_conflict_policy == "risk_first":
                normal_exit_reason = "stop_intrabar_conflict"
                normal_exit_price = state.position.stop_price
            else:
                normal_exit_reason = "tp_intrabar_conflict"
                normal_exit_price = eff_tp
        elif stop_hit:
            normal_exit_reason = "stop_intrabar"
            normal_exit_price = state.position.stop_price
        elif tp_hit:
            normal_exit_reason = "tp_intrabar"
            normal_exit_price = eff_tp

    if normal_exit_reason is not None and normal_exit_price is not None and state.position is not None:
        _close_position(state, reason=normal_exit_reason, ts=ts, raw_exit_price=float(normal_exit_price), liq_exit=False,
                        slippage=slippage, fee_per_side=fee_per_side, m_cfg=m_cfg)


# ---------------------------------------------------------------------------
# Helper: check risk controls (daily loss + global kill switch)
# ---------------------------------------------------------------------------

def _check_risk_controls(
    state: _EngineState,
    ts: pd.Timestamp,
    c: float,
    *,
    slippage: float,
    fee_per_side: float,
    m_cfg: MarginConfig,
    metric_cfg: MetricsConfig,
) -> None:
    """Check and enforce daily loss limit and global kill switch."""
    if state.position is not None:
        unreal_close_for_risk = _calc_trade_pnl(state.position.side, state.position.qty, state.position.entry_price, c)
        equity_for_risk = state.balance + unreal_close_for_risk
    else:
        equity_for_risk = state.balance

    dd_now = (equity_for_risk / state.peak_equity - 1.0) if state.peak_equity > 0 else 0.0
    global_trigger = (not state.global_halted) and (dd_now <= -metric_cfg.kill_switch_drawdown_pct)
    day_ret_now = (equity_for_risk - state.day_start_equity) / state.day_start_equity if state.day_start_equity > 0 else 0.0
    daily_trigger = (not state.global_halted and not state.daily_halted) and (day_ret_now <= -metric_cfg.daily_loss_limit_pct)

    risk_reason: str | None = None
    if global_trigger:
        state.global_halted = True
        state.risk_halt_global_triggered = True
        state.risk_halt_global_count += 1
        risk_reason = "risk_kill_switch"
        logger.debug("Global kill switch triggered at %s (dd=%.4f%%)", ts, dd_now * 100)
    elif daily_trigger:
        state.daily_halted = True
        state.risk_halt_daily_count += 1
        risk_reason = "risk_daily_loss"
        logger.debug("Daily loss limit triggered at %s (day_ret=%.4f%%)", ts, day_ret_now * 100)

    if risk_reason is not None and state.position is not None:
        _close_position(state, reason=risk_reason, ts=ts, raw_exit_price=c, liq_exit=False,
                        slippage=slippage, fee_per_side=fee_per_side, m_cfg=m_cfg)
        state.forced_close_by_risk_control += 1
        assert state.forced_close_reason_counts is not None
        state.forced_close_reason_counts[risk_reason] = state.forced_close_reason_counts.get(risk_reason, 0) + 1


# ---------------------------------------------------------------------------
# Helper: build stats and metadata
# ---------------------------------------------------------------------------

def _build_stats_and_meta(
    state: _EngineState,
    dt_index: pd.DatetimeIndex,
    init_cash: float,
    allow_short: bool,
    instrument: str | None,
    min_qty: float | None,
    enforce_min_constraints: bool,
    m_cfg: MarginConfig,
    metric_cfg: MetricsConfig,
    f_cfg: FillPolicyConfig,
    base_meta: dict[str, Any] | None,
) -> BacktestResult:
    """Assemble stats, metadata, and return the BacktestResult."""
    assert state.equity_values is not None
    assert state.used_margin_values is not None
    assert state.free_margin_values is not None
    assert state.trades is not None
    assert state.tx_days is not None
    assert state.forced_close_reason_counts is not None

    equity_curve = pd.Series(state.equity_values, index=dt_index, name="equity")
    used_margin_curve = pd.Series(state.used_margin_values, index=dt_index, name="used_margin")
    free_margin_curve = pd.Series(state.free_margin_values, index=dt_index, name="free_margin")

    returns_values: list[float] = []
    prev_val = float(init_cash)
    for curr in equity_curve.to_list():
        returns_values.append(_calc_return(prev_val, float(curr)))
        prev_val = float(curr)
    returns_series = pd.Series(returns_values, index=dt_index, name="returns")

    total_return = (float(equity_curve.iloc[-1]) - float(init_cash)) / float(init_cash) if init_cash != 0 else float("nan")
    roll_peak = equity_curve.cummax()
    drawdown = (equity_curve / roll_peak) - 1.0
    max_dd = float(-drawdown.min()) if len(drawdown) > 0 else float("nan")
    bpy = _bars_per_year(dt_index, metric_cfg.year_days)
    sharpe = _calc_sharpe(returns_series, metric_cfg.risk_free_annual, bpy)

    trades_df = pd.DataFrame(state.trades)
    total_trades = int(len(trades_df))
    win_rate = float("nan")
    if total_trades > 0:
        wins = int((trades_df["net_pnl"] > 0).sum())
        win_rate = float(wins / total_trades * 100.0)

    date_norm = pd.Series(dt_index).dt.normalize()
    market_open_days = int(date_norm.nunique())
    period_days = int((date_norm.iloc[-1] - date_norm.iloc[0]).days + 1) if len(date_norm) > 0 else 0
    active_transaction_days = int(len(state.tx_days))

    daily_loss_hit_count = int(state.risk_halt_daily_count)
    kill_switch_hit_count = int(state.risk_halt_global_count)
    skipped_total = max(0, state.attempted_signals - state.executed_signals)
    missing_min_table = bool(enforce_min_constraints and min_qty is None)

    stats_map: dict[str, Any] = {
        "Start": dt_index[0],
        "End": dt_index[-1],
        "Period": len(dt_index),
        "Period Days": period_days,
        "Start Value": float(init_cash),
        "End Value": float(equity_curve.iloc[-1]),
        "Total Return [%]": float(total_return * 100.0),
        "Max Drawdown [%]": float(max_dd * 100.0),
        "Sharpe Ratio": float(sharpe),
        "Total Trades": total_trades,
        "Win Rate [%]": win_rate,
        "Total Fees Paid": float(state.total_fees_paid),
        "Market Open Days": market_open_days,
        "Active Transaction Days": active_transaction_days,
        "Daily Loss Hit Count": daily_loss_hit_count,
        "Kill Switch Hit Count": kill_switch_hit_count,
        "Risk Halt Daily Count": int(state.risk_halt_daily_count),
        "Risk Halt Global Triggered": bool(state.risk_halt_global_triggered),
        "Forced Close by Risk Control": int(state.forced_close_by_risk_control),
    }

    meta = {
        **(base_meta or {}),
        "engine": "rule_based_execution",
        "execution_mode": "rule_based_execution",
        "allow_short": allow_short,
        "instrument": instrument,
        "requested_leverage": float(m_cfg.leverage),
        "applied_leverage": float(m_cfg.leverage),
        "maintenance_margin_rate": float(m_cfg.maintenance_margin_rate),
        "liquidation_fee_rate": float(m_cfg.liquidation_fee_rate),
        "risk_free_annual": float(metric_cfg.risk_free_annual),
        "year_days": int(metric_cfg.year_days),
        "bars_per_year": float(bpy),
        "gap_sl_policy": f_cfg.gap_sl_policy,
        "same_bar_conflict_policy": f_cfg.same_bar_conflict_policy,
        "day_policy": "market_open_days_for_annualization",
        "market_open_days": market_open_days,
        "active_transaction_days": active_transaction_days,
        "attempted_signals": int(state.attempted_signals),
        "executed_signals": int(state.executed_signals),
        "skipped_invalid_sl": int(state.skipped_invalid_sl),
        "skipped_min_qty": int(state.skipped_min_qty),
        "skipped_missing_min_table": int(state.skipped_missing_min_table),
        "skipped_position_busy": int(state.skipped_position_busy),
        "skipped_end_of_data": int(state.skipped_end_of_data),
        "skipped_risk_halt": int(state.skipped_risk_halt),
        "skipped_total": int(skipped_total),
        "min_qty": min_qty,
        "dynamic_min_notional_enabled": bool(enforce_min_constraints),
        "missing_min_table": missing_min_table,
        "liquidation_count": int(state.liquidation_count),
        "risk_halt_daily_count": int(state.risk_halt_daily_count),
        "risk_halt_global_triggered": bool(state.risk_halt_global_triggered),
        "forced_close_reason_counts": state.forced_close_reason_counts,
        "forced_close_by_risk_control": int(state.forced_close_by_risk_control),
    }

    return BacktestResult(
        stats_map=stats_map,
        equity_curve=equity_curve,
        returns_series=returns_series,
        trades=trades_df,
        exec_meta=meta,
        used_margin_curve=used_margin_curve,
        free_margin_curve=free_margin_curve,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rule_based_execution(
    *,
    index: pd.Index,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_middle: pd.Series,
    long_entries: pd.Series,
    short_entries: pd.Series,
    sl_pct: pd.Series,
    allow_short: bool,
    init_cash: float,
    risk_per_trade: float,
    fee_per_side: float,
    slippage: float,
    min_qty: float | None,
    enforce_min_constraints: bool,
    skip_if_below_min: bool,
    instrument: str | None,
    margin_cfg: MarginConfig | None = None,
    metrics_cfg: MetricsConfig | None = None,
    fill_cfg: FillPolicyConfig | None = None,
    base_meta: dict[str, Any] | None = None,
    # ---- VBT-compatible params ----
    sl_trail: bool = False,
    tp_stop_pct: float | None = None,
    entry_price_override: pd.Series | None = None,
    stop_entry_price: StopEntryPrice = "fillprice",
    size_override: float | None = None,
    size_type: SizeType = "risk",
) -> BacktestResult:
    m_cfg = margin_cfg or MarginConfig()
    metric_cfg = metrics_cfg or MetricsConfig()
    f_cfg = fill_cfg or FillPolicyConfig()
    dt_index = pd.DatetimeIndex(index)

    if m_cfg.leverage <= 0:
        raise ValueError("MarginConfig.leverage must be > 0.")
    if m_cfg.maintenance_margin_rate < 0:
        raise ValueError("MarginConfig.maintenance_margin_rate must be >= 0.")
    if m_cfg.liquidation_fee_rate < 0:
        raise ValueError("MarginConfig.liquidation_fee_rate must be >= 0.")
    if metric_cfg.year_days <= 0:
        raise ValueError("MetricsConfig.year_days must be > 0.")
    if risk_per_trade < 0:
        raise ValueError("risk_per_trade must be >= 0.")
    if fee_per_side < 0:
        raise ValueError("fee_per_side must be >= 0.")
    if slippage < 0:
        raise ValueError("slippage must be >= 0.")
    if min_qty is not None and min_qty <= 0:
        raise ValueError("min_qty must be > 0 when provided.")
    if not dt_index.is_monotonic_increasing:
        raise ValueError("Index must be monotonic increasing.")
    if dt_index.has_duplicates:
        raise ValueError("Index must not contain duplicate timestamps.")

    def _empty_result(reason: str) -> BacktestResult:
        empty_equity = pd.Series(dtype="float64", index=dt_index, name="equity")
        empty_returns = pd.Series(dtype="float64", index=dt_index, name="returns")
        empty_margin = pd.Series(dtype="float64", index=dt_index, name="used_margin")
        empty_free = pd.Series(dtype="float64", index=dt_index, name="free_margin")
        stats_map: dict[str, Any] = {
            "Start": pd.NaT,
            "End": pd.NaT,
            "Period": 0,
            "Period Days": 0,
            "Start Value": float(init_cash),
            "End Value": float(init_cash),
            "Total Return [%]": 0.0,
            "Max Drawdown [%]": 0.0,
            "Sharpe Ratio": float("nan"),
            "Total Trades": 0,
            "Win Rate [%]": float("nan"),
            "Total Fees Paid": 0.0,
            "Market Open Days": 0,
            "Active Transaction Days": 0,
            "Daily Loss Hit Count": 0,
            "Kill Switch Hit Count": 0,
            "Risk Halt Daily Count": 0,
            "Risk Halt Global Triggered": False,
            "Forced Close by Risk Control": 0,
        }
        meta = {
            **(base_meta or {}),
            "engine": "rule_based_execution",
            "execution_mode": "rule_based_execution",
            "allow_short": allow_short,
            "instrument": instrument,
            "requested_leverage": float(m_cfg.leverage),
            "applied_leverage": float(m_cfg.leverage),
            "maintenance_margin_rate": float(m_cfg.maintenance_margin_rate),
            "liquidation_fee_rate": float(m_cfg.liquidation_fee_rate),
            "risk_free_annual": float(metric_cfg.risk_free_annual),
            "year_days": int(metric_cfg.year_days),
            "bars_per_year": float("nan"),
            "gap_sl_policy": f_cfg.gap_sl_policy,
            "same_bar_conflict_policy": f_cfg.same_bar_conflict_policy,
            "day_policy": "market_open_days_for_annualization",
            "market_open_days": 0,
            "active_transaction_days": 0,
            "attempted_signals": 0,
            "executed_signals": 0,
            "skipped_invalid_sl": 0,
            "skipped_min_qty": 0,
            "skipped_missing_min_table": 0,
            "skipped_position_busy": 0,
            "skipped_end_of_data": 0,
            "skipped_risk_halt": 0,
            "skipped_total": 0,
            "min_qty": min_qty,
            "dynamic_min_notional_enabled": bool(enforce_min_constraints),
            "missing_min_table": bool(enforce_min_constraints and min_qty is None),
            "liquidation_count": 0,
            "risk_halt_daily_count": 0,
            "risk_halt_global_triggered": False,
            "forced_close_reason_counts": {},
            "forced_close_by_risk_control": 0,
            "empty_run": True,
            "empty_run_reason": reason,
        }
        return BacktestResult(
            stats_map=stats_map,
            equity_curve=empty_equity,
            returns_series=empty_returns,
            trades=pd.DataFrame(),
            exec_meta=meta,
            used_margin_curve=empty_margin,
            free_margin_curve=empty_free,
        )

    if len(dt_index) == 0:
        return _empty_result("empty_index")

    open_s = _to_float_series(open_, dt_index)
    high_s = _to_float_series(high, dt_index)
    low_s = _to_float_series(low, dt_index)
    close_s = _to_float_series(close, dt_index)
    middle_s = _to_float_series(bb_middle, dt_index)
    middle_known_s = middle_s.shift(1)
    sl_s = _to_float_series(sl_pct, dt_index)
    long_s = _to_bool_series(long_entries, dt_index)
    short_s = _to_bool_series(short_entries, dt_index)

    ohlc = pd.DataFrame({"open": open_s, "high": high_s, "low": low_s, "close": close_s}, index=dt_index)
    if not np.isfinite(ohlc.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLC contains NaN or infinite values.")
    if bool((high_s < low_s).any()):
        raise ValueError("Found bars with high < low.")

    # --- Initialize state ---
    state = _EngineState(
        balance=float(init_cash),
        day_start_equity=float(init_cash),
        peak_equity=float(init_cash),
        last_equity_close=float(init_cash),
    )

    # --- Main bar-by-bar loop ---
    for i, ts_raw in enumerate(dt_index):
        ts = pd.Timestamp(ts_raw)
        bar_day = ts.normalize()

        if state.current_day is None or bar_day != state.current_day:
            state.current_day = bar_day
            state.day_start_equity = state.last_equity_close
            state.daily_halted = False

        if (state.daily_halted or state.global_halted) and state.pending is not None:
            state.pending = None
            state.skipped_risk_halt += 1

        o = float(open_s.iloc[i])
        h = float(high_s.iloc[i])
        low_price = float(low_s.iloc[i])
        c = float(close_s.iloc[i])
        tp_level = float(middle_known_s.iloc[i]) if pd.notna(middle_known_s.iloc[i]) else float("nan")
        sl_now = float(sl_s.iloc[i]) if pd.notna(sl_s.iloc[i]) else float("nan")

        # 1. Fill pending signal
        _fill_pending_signal(
            state, i, ts, sl_now, c, o,
            entry_price_override=entry_price_override,
            stop_entry_price=stop_entry_price,
            size_type=size_type,
            size_override=size_override,
            risk_per_trade=risk_per_trade,
            slippage=slippage,
            fee_per_side=fee_per_side,
            m_cfg=m_cfg,
            tp_stop_pct=tp_stop_pct,
            min_qty=min_qty,
            enforce_min_constraints=enforce_min_constraints,
            skip_if_below_min=skip_if_below_min,
        )

        # 2. Check exit conditions
        _check_exit_conditions(
            state, ts, o, h, low_price, c, tp_level, sl_now,
            sl_trail=sl_trail,
            slippage=slippage,
            fee_per_side=fee_per_side,
            m_cfg=m_cfg,
            f_cfg=f_cfg,
        )

        # 3. Check risk controls
        _check_risk_controls(
            state, ts, c,
            slippage=slippage,
            fee_per_side=fee_per_side,
            m_cfg=m_cfg,
            metric_cfg=metric_cfg,
        )

        # 4. Record equity and margin
        if state.position is not None:
            unreal_close = _calc_trade_pnl(state.position.side, state.position.qty, state.position.entry_price, c)
            equity_close = state.balance + unreal_close
            used_margin = abs(state.position.qty) * c / m_cfg.leverage
            free_margin = equity_close - used_margin
        else:
            equity_close = state.balance
            used_margin = 0.0
            free_margin = state.balance

        assert state.equity_values is not None
        assert state.used_margin_values is not None
        assert state.free_margin_values is not None

        state.equity_values.append(float(equity_close))
        state.used_margin_values.append(float(used_margin))
        state.free_margin_values.append(float(free_margin))
        if np.isfinite(equity_close):
            state.peak_equity = max(state.peak_equity, float(equity_close))
        state.last_equity_close = float(equity_close)

        # 5. Scan for new signals
        long_sig = bool(long_s.iloc[i])
        short_sig = bool(short_s.iloc[i])
        signal_side = _select_signal_side(long_sig, short_sig, allow_short=allow_short)
        if signal_side is not None:
            state.attempted_signals += 1
            if i >= len(dt_index) - 1:
                state.skipped_end_of_data += 1
            elif state.global_halted or state.daily_halted:
                state.skipped_risk_halt += 1
            elif state.position is None and state.pending is None:
                state.pending = _PendingSignal(side=signal_side)
            else:
                state.skipped_position_busy += 1

    # --- Force close at end of data ---
    if state.position is not None:
        ts = pd.Timestamp(dt_index[-1])
        _close_position(state, reason="forced_close_end", ts=ts, raw_exit_price=float(close_s.iloc[-1]), liq_exit=False,
                        slippage=slippage, fee_per_side=fee_per_side, m_cfg=m_cfg)

        assert state.equity_values is not None
        assert state.used_margin_values is not None
        assert state.free_margin_values is not None

        if len(state.equity_values) > 0:
            state.equity_values[-1] = float(state.balance)
            state.used_margin_values[-1] = 0.0
            state.free_margin_values[-1] = float(state.balance)

    return _build_stats_and_meta(
        state, dt_index, init_cash, allow_short, instrument,
        min_qty, enforce_min_constraints,
        m_cfg, metric_cfg, f_cfg, base_meta,
    )


__all__ = ["run_rule_based_execution"]
