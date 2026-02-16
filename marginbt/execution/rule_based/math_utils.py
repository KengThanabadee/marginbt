from __future__ import annotations

import numpy as np
import pandas as pd


def _to_float_series(series: pd.Series, index: pd.Index) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if not s.index.equals(index):
        s = s.reindex(index)
    return s


def _to_bool_series(series: pd.Series, index: pd.Index) -> pd.Series:
    s = series.astype("boolean")
    if not s.index.equals(index):
        s = s.reindex(index)
    return s.fillna(False).astype(bool)


def _adverse_entry_price(raw_price: float, side: int, slippage: float) -> float:
    if side > 0:
        return raw_price * (1.0 + slippage)
    return raw_price * (1.0 - slippage)


def _adverse_exit_price(raw_price: float, side: int, slippage: float) -> float:
    if side > 0:
        return raw_price * (1.0 - slippage)
    return raw_price * (1.0 + slippage)


def _calc_return(prev_value: float, curr_value: float) -> float:
    if prev_value == 0:
        if curr_value == 0:
            return 0.0
        return float(np.inf * np.sign(curr_value))
    return float((curr_value - prev_value) / prev_value)


def _calc_sharpe(returns: pd.Series, rf_annual: float, bars_per_year: float) -> float:
    if len(returns) < 2 or not np.isfinite(bars_per_year) or bars_per_year <= 0:
        return float("nan")
    rf_bar = (1.0 + rf_annual) ** (1.0 / bars_per_year) - 1.0
    adj = returns.to_numpy(dtype=np.float64) - rf_bar
    mean = np.nanmean(adj)
    std = np.nanstd(adj, ddof=1)
    if np.isnan(std):
        return float("nan")
    if std == 0.0:
        if mean > 0.0:
            return float(np.inf)
        if mean < 0.0:
            return float(-np.inf)
        return 0.0
    return float(mean / std * np.sqrt(bars_per_year))


def _bars_per_year(index: pd.Index, year_days: int) -> float:
    if len(index) < 2:
        return float(year_days * 24)
    freq_delta = pd.Series(index[1:] - index[:-1]).median()
    if pd.isna(freq_delta) or freq_delta.total_seconds() <= 0:
        return float(year_days * 24)
    hours_per_bar = freq_delta.total_seconds() / 3600.0
    return float((year_days * 24.0) / hours_per_bar)


def _calc_trade_pnl(side: int, qty: float, entry_price: float, exit_price: float) -> float:
    return float(side * qty * (exit_price - entry_price))


def _position_equity(side: int, qty: float, entry_price: float, entry_margin: float, mark_price: float) -> float:
    return float(entry_margin + side * qty * (mark_price - entry_price))


def _maintenance_margin(qty: float, mark_price: float, maintenance_margin_rate: float) -> float:
    return float(abs(qty) * mark_price * maintenance_margin_rate)


def _select_signal_side(long_sig: bool, short_sig: bool, allow_short: bool) -> int | None:
    if long_sig and not short_sig:
        return 1
    if allow_short and short_sig and not long_sig:
        return -1
    if allow_short and long_sig and short_sig:
        return 1
    return None


__all__ = [
    "_to_float_series",
    "_to_bool_series",
    "_adverse_entry_price",
    "_adverse_exit_price",
    "_calc_return",
    "_calc_sharpe",
    "_bars_per_year",
    "_calc_trade_pnl",
    "_position_equity",
    "_maintenance_margin",
    "_select_signal_side",
]

