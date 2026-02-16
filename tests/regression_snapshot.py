from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginbt import BacktestEngine

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "regression_snapshot.json"

STATS_KEYS = [
    "Total Return [%]",
    "Max Drawdown [%]",
    "Sharpe Ratio",
    "Total Trades",
    "Win Rate [%]",
    "Market Open Days",
    "Active Transaction Days",
]

META_KEYS = [
    "engine",
    "allow_short",
    "instrument",
    "attempted_signals",
    "executed_signals",
    "skipped_invalid_sl",
    "skipped_min_qty",
    "skipped_missing_min_table",
    "skipped_total",
    "dynamic_min_notional_enabled",
    "liquidation_count",
    "risk_halt_daily_count",
    "risk_halt_global_triggered",
    "forced_close_by_risk_control",
]


def _make_series(n: int = 360) -> dict[str, pd.Series]:
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


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        return round(x, 12)
    if pd.isna(value):
        return "NaN"
    return value


def _collect_result_payload() -> dict[str, Any]:
    s = _make_series()
    engine = BacktestEngine(init_cash=100.0, fees=0.00045, slippage=0.0002, leverage=10.0)
    result = engine.run(
        close=s["close"],
        open=s["open"],
        high=s["high"],
        low=s["low"],
        entries=s["entries"],
        short_entries=s["short_entries"],
        sl_stop=s["sl_stop"],
        risk_per_trade=0.0025,
        direction="both",
        instrument="PAXG_USDT_Perp",
    )
    stats = result.stats()
    meta = result.exec_meta
    return {
        "backtest_engine": {
            "stats": {k: _normalize_scalar(stats.get(k)) for k in STATS_KEYS},
            "meta": {k: _normalize_scalar(meta.get(k)) for k in META_KEYS},
        }
    }


def _is_close_num(left: Any, right: Any, atol: float = 1e-9) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= atol
    return left == right


def _verify(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for mode, expected_mode in expected.items():
        if mode not in actual:
            errors.append(f"Missing mode: {mode}")
            continue
        actual_mode = actual[mode]
        for group in ("stats", "meta"):
            exp_group = expected_mode.get(group, {})
            act_group = actual_mode.get(group, {})
            for key, exp_val in exp_group.items():
                if key not in act_group:
                    errors.append(f"{mode}.{group}.{key}: missing actual key")
                    continue
                act_val = act_group[key]
                if not _is_close_num(act_val, exp_val):
                    errors.append(f"{mode}.{group}.{key}: expected={exp_val} actual={act_val}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture/verify deterministic regression snapshot for marginbt core.")
    parser.add_argument("--mode", choices=["capture", "verify"], required=True)
    parser.add_argument("--fixture", default=str(FIXTURE_PATH))
    args = parser.parse_args()

    fixture = Path(args.fixture)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    payload = _collect_result_payload()

    if args.mode == "capture":
        fixture.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"SNAPSHOT: CAPTURED -> {fixture}")
        return 0

    if not fixture.exists():
        print(f"SNAPSHOT: VERIFY FAIL -> fixture not found: {fixture}")
        return 1
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    errors = _verify(payload, expected)
    if errors:
        print("SNAPSHOT: VERIFY FAIL")
        for i, err in enumerate(errors, start=1):
            print(f"{i}. {err}")
        return 1

    print("SNAPSHOT: VERIFY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
