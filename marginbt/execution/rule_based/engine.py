from __future__ import annotations

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

    balance = float(init_cash)
    equity_values: list[float] = []
    used_margin_values: list[float] = []
    free_margin_values: list[float] = []

    trades: list[dict[str, Any]] = []
    tx_days: set[pd.Timestamp] = set()

    position: _Position | None = None
    pending: _PendingSignal | None = None

    attempted_signals = 0
    executed_signals = 0
    skipped_invalid_sl = 0
    skipped_min_qty = 0
    skipped_missing_min_table = 0
    skipped_position_busy = 0
    skipped_end_of_data = 0
    skipped_risk_halt = 0

    total_fees_paid = 0.0
    liquidation_count = 0
    risk_halt_daily_count = 0
    risk_halt_global_triggered = False
    risk_halt_global_count = 0
    forced_close_by_risk_control = 0
    forced_close_reason_counts: dict[str, int] = {}

    current_day: pd.Timestamp | None = None
    day_start_equity = float(init_cash)
    daily_halted = False
    global_halted = False
    peak_equity = float(init_cash)
    last_equity_close = float(init_cash)

    def _close_position(reason: str, ts: pd.Timestamp, raw_exit_price: float, liq_exit: bool) -> None:
        nonlocal position, balance, total_fees_paid, liquidation_count
        if position is None:
            return
        side = position.side
        qty = position.qty
        exit_price = _adverse_exit_price(float(raw_exit_price), side, slippage)
        exit_notional = abs(qty) * exit_price
        pnl = _calc_trade_pnl(side, qty, position.entry_price, exit_price)
        exit_fee = exit_notional * fee_per_side
        liq_fee = exit_notional * m_cfg.liquidation_fee_rate if liq_exit else 0.0
        balance += pnl - exit_fee - liq_fee
        if liq_exit:
            balance = max(0.0, balance)
        total_fees_paid += exit_fee + liq_fee
        if liq_exit:
            liquidation_count += 1
        tx_days.add(pd.Timestamp(ts).normalize())
        trades.append(
            {
                "entry_ts": position.entry_ts,
                "exit_ts": pd.Timestamp(ts),
                "side": "long" if side > 0 else "short",
                "qty": qty,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "gross_pnl": pnl,
                "fees_paid": exit_fee + liq_fee,
                "net_pnl": pnl - exit_fee - liq_fee,
                "reason": reason,
            }
        )
        position = None

    for i, ts_raw in enumerate(dt_index):
        ts = pd.Timestamp(ts_raw)
        bar_day = ts.normalize()

        if current_day is None or bar_day != current_day:
            current_day = bar_day
            day_start_equity = last_equity_close
            daily_halted = False

        if (daily_halted or global_halted) and pending is not None:
            pending = None
            skipped_risk_halt += 1

        o = float(open_s.iloc[i])
        h = float(high_s.iloc[i])
        l = float(low_s.iloc[i])
        c = float(close_s.iloc[i])
        tp_level = float(middle_known_s.iloc[i]) if pd.notna(middle_known_s.iloc[i]) else float("nan")
        sl_now = float(sl_s.iloc[i]) if pd.notna(sl_s.iloc[i]) else float("nan")

        if pending is not None and position is None and not (daily_halted or global_halted):
            # --- Determine raw fill price ---
            if entry_price_override is not None:
                raw_fill = float(entry_price_override.iloc[i])
            else:
                raw_fill = o  # internal default = open (backward compat)

            valid_sl = bool(np.isfinite(sl_now) and sl_now > 0 and raw_fill > 0)
            if not valid_sl:
                skipped_invalid_sl += 1
                pending = None
            else:
                side = pending.side
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
                        skipped_missing_min_table += 1
                    if below_qty:
                        skipped_min_qty += 1
                    pending = None
                else:
                    stop_price = stop_ref * (1.0 - sl_now) if side > 0 else stop_ref * (1.0 + sl_now)

                    # --- Per-trade TP from tp_stop_pct ---
                    if tp_stop_pct is not None:
                        trade_tp_level = stop_ref * (1.0 + tp_stop_pct) if side > 0 else stop_ref * (1.0 - tp_stop_pct)
                    else:
                        trade_tp_level = float("nan")

                    notional = order_notional
                    entry_margin = notional / m_cfg.leverage
                    entry_fee = notional * fee_per_side
                    balance -= entry_fee
                    total_fees_paid += entry_fee
                    tx_days.add(ts.normalize())
                    position = _Position(
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
                    executed_signals += 1
                    pending = None

        normal_exit_reason: str | None = None
        normal_exit_price: float | None = None

        if position is not None:
            side = position.side
            qty = position.qty
            long_pos = side > 0

            # --- Trailing stop: update stop_price ---
            if sl_trail and np.isfinite(sl_now) and sl_now > 0:
                if long_pos:
                    position.trailing_high = max(position.trailing_high, h)
                    new_stop = position.trailing_high * (1.0 - sl_now)
                    if new_stop > position.stop_price:
                        position.stop_price = new_stop
                else:
                    position.trailing_low = min(position.trailing_low, l)
                    new_stop = position.trailing_low * (1.0 + sl_now)
                    if new_stop < position.stop_price:
                        position.stop_price = new_stop

            # --- Resolve effective TP level ---
            # Priority: per-trade tp_stop_pct > per-bar tp_price (bb_middle) > exits
            if np.isfinite(position.tp_price_level):
                eff_tp = position.tp_price_level
            else:
                eff_tp = tp_level

            open_liq = _position_equity(side, qty, position.entry_price, position.entry_margin, o) <= _maintenance_margin(
                qty, o, m_cfg.maintenance_margin_rate
            )
            gap_stop = (o <= position.stop_price) if long_pos else (o >= position.stop_price)
            gap_tp = (o >= eff_tp) if long_pos and np.isfinite(eff_tp) else False
            if not long_pos and np.isfinite(eff_tp):
                gap_tp = o <= eff_tp

            if open_liq:
                _close_position(reason="liquidation_open", ts=ts, raw_exit_price=float(o), liq_exit=True)
            elif gap_stop or gap_tp:
                if gap_stop and gap_tp:
                    if f_cfg.same_bar_conflict_policy == "risk_first":
                        normal_exit_reason = "stop_open_conflict"
                        normal_exit_price = o if f_cfg.gap_sl_policy == "bar_open" else position.stop_price
                    else:
                        normal_exit_reason = "tp_open_conflict"
                        normal_exit_price = o
                elif gap_stop:
                    normal_exit_reason = "stop_open"
                    normal_exit_price = o if f_cfg.gap_sl_policy == "bar_open" else position.stop_price
                else:
                    normal_exit_reason = "tp_open"
                    normal_exit_price = o
            else:
                stop_hit = (l <= position.stop_price) if long_pos else (h >= position.stop_price)
                tp_hit = (h >= eff_tp) if long_pos and np.isfinite(eff_tp) else False
                if not long_pos and np.isfinite(eff_tp):
                    tp_hit = l <= eff_tp

                worst_mark = l if long_pos else h
                intrabar_liq = _position_equity(
                    side, qty, position.entry_price, position.entry_margin, worst_mark
                ) <= _maintenance_margin(qty, worst_mark, m_cfg.maintenance_margin_rate)

                if intrabar_liq:
                    _close_position(reason="liquidation_intrabar", ts=ts, raw_exit_price=float(worst_mark), liq_exit=True)
                elif stop_hit and tp_hit:
                    if f_cfg.same_bar_conflict_policy == "risk_first":
                        normal_exit_reason = "stop_intrabar_conflict"
                        normal_exit_price = position.stop_price
                    else:
                        normal_exit_reason = "tp_intrabar_conflict"
                        normal_exit_price = eff_tp
                elif stop_hit:
                    normal_exit_reason = "stop_intrabar"
                    normal_exit_price = position.stop_price
                elif tp_hit:
                    normal_exit_reason = "tp_intrabar"
                    normal_exit_price = eff_tp

        if normal_exit_reason is not None and normal_exit_price is not None and position is not None:
            _close_position(reason=normal_exit_reason, ts=ts, raw_exit_price=float(normal_exit_price), liq_exit=False)

        if position is not None:
            unreal_close_for_risk = _calc_trade_pnl(position.side, position.qty, position.entry_price, c)
            equity_for_risk = balance + unreal_close_for_risk
        else:
            equity_for_risk = balance

        dd_now = (equity_for_risk / peak_equity - 1.0) if peak_equity > 0 else 0.0
        global_trigger = (not global_halted) and (dd_now <= -metric_cfg.kill_switch_drawdown_pct)
        day_ret_now = (equity_for_risk - day_start_equity) / day_start_equity if day_start_equity > 0 else 0.0
        daily_trigger = (not global_halted and not daily_halted) and (day_ret_now <= -metric_cfg.daily_loss_limit_pct)

        risk_reason: str | None = None
        if global_trigger:
            global_halted = True
            risk_halt_global_triggered = True
            risk_halt_global_count += 1
            risk_reason = "risk_kill_switch"
        elif daily_trigger:
            daily_halted = True
            risk_halt_daily_count += 1
            risk_reason = "risk_daily_loss"

        if risk_reason is not None and position is not None:
            _close_position(reason=risk_reason, ts=ts, raw_exit_price=c, liq_exit=False)
            forced_close_by_risk_control += 1
            forced_close_reason_counts[risk_reason] = forced_close_reason_counts.get(risk_reason, 0) + 1

        if position is not None:
            unreal_close = _calc_trade_pnl(position.side, position.qty, position.entry_price, c)
            equity_close = balance + unreal_close
            used_margin = abs(position.qty) * c / m_cfg.leverage
            free_margin = equity_close - used_margin
        else:
            equity_close = balance
            used_margin = 0.0
            free_margin = balance

        equity_values.append(float(equity_close))
        used_margin_values.append(float(used_margin))
        free_margin_values.append(float(free_margin))
        if np.isfinite(equity_close):
            peak_equity = max(peak_equity, float(equity_close))
        last_equity_close = float(equity_close)

        long_sig = bool(long_s.iloc[i])
        short_sig = bool(short_s.iloc[i])
        signal_side = _select_signal_side(long_sig, short_sig, allow_short=allow_short)
        if signal_side is not None:
            attempted_signals += 1
            if i >= len(dt_index) - 1:
                skipped_end_of_data += 1
            elif global_halted or daily_halted:
                skipped_risk_halt += 1
            elif position is None and pending is None:
                pending = _PendingSignal(side=signal_side)
            else:
                skipped_position_busy += 1

    if position is not None:
        ts = pd.Timestamp(dt_index[-1])
        _close_position(reason="forced_close_end", ts=ts, raw_exit_price=float(close_s.iloc[-1]), liq_exit=False)
        if len(equity_values) > 0:
            equity_values[-1] = float(balance)
            used_margin_values[-1] = 0.0
            free_margin_values[-1] = float(balance)

    equity_curve = pd.Series(equity_values, index=dt_index, name="equity")
    used_margin_curve = pd.Series(used_margin_values, index=dt_index, name="used_margin")
    free_margin_curve = pd.Series(free_margin_values, index=dt_index, name="free_margin")
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

    trades_df = pd.DataFrame(trades)
    total_trades = int(len(trades_df))
    win_rate = float("nan")
    if total_trades > 0:
        wins = int((trades_df["net_pnl"] > 0).sum())
        win_rate = float(wins / total_trades * 100.0)

    date_norm = pd.Series(dt_index).dt.normalize()
    market_open_days = int(date_norm.nunique())
    period_days = int((date_norm.iloc[-1] - date_norm.iloc[0]).days + 1) if len(date_norm) > 0 else 0
    active_transaction_days = int(len(tx_days))

    daily_loss_hit_count = int(risk_halt_daily_count)
    kill_switch_hit_count = int(risk_halt_global_count)
    skipped_total = max(0, attempted_signals - executed_signals)
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
        "Total Fees Paid": float(total_fees_paid),
        "Market Open Days": market_open_days,
        "Active Transaction Days": active_transaction_days,
        "Daily Loss Hit Count": daily_loss_hit_count,
        "Kill Switch Hit Count": kill_switch_hit_count,
        "Risk Halt Daily Count": int(risk_halt_daily_count),
        "Risk Halt Global Triggered": bool(risk_halt_global_triggered),
        "Forced Close by Risk Control": int(forced_close_by_risk_control),
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
        "attempted_signals": int(attempted_signals),
        "executed_signals": int(executed_signals),
        "skipped_invalid_sl": int(skipped_invalid_sl),
        "skipped_min_qty": int(skipped_min_qty),
        "skipped_missing_min_table": int(skipped_missing_min_table),
        "skipped_position_busy": int(skipped_position_busy),
        "skipped_end_of_data": int(skipped_end_of_data),
        "skipped_risk_halt": int(skipped_risk_halt),
        "skipped_total": int(skipped_total),
        "min_qty": min_qty,
        "dynamic_min_notional_enabled": bool(enforce_min_constraints),
        "missing_min_table": missing_min_table,
        "liquidation_count": int(liquidation_count),
        "risk_halt_daily_count": int(risk_halt_daily_count),
        "risk_halt_global_triggered": bool(risk_halt_global_triggered),
        "forced_close_reason_counts": forced_close_reason_counts,
        "forced_close_by_risk_control": int(forced_close_by_risk_control),
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

__all__ = ["run_rule_based_execution"]

