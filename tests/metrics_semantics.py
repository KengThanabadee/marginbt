from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginbt.execution.rule_based.math_utils import _calc_sharpe


def main() -> int:
    errors: list[str] = []
    bars_per_year = 365.0 * 24.0

    sharpe_pos = _calc_sharpe(pd.Series([0.01] * 50), rf_annual=0.0, bars_per_year=bars_per_year)
    sharpe_neg = _calc_sharpe(pd.Series([-0.01] * 50), rf_annual=0.0, bars_per_year=bars_per_year)
    sharpe_zero = _calc_sharpe(pd.Series([0.0] * 50), rf_annual=0.0, bars_per_year=bars_per_year)

    if not (math.isinf(sharpe_pos) and sharpe_pos > 0):
        errors.append(f"Expected +inf for constant positive returns, got {sharpe_pos}.")
    if not (math.isinf(sharpe_neg) and sharpe_neg < 0):
        errors.append(f"Expected -inf for constant negative returns, got {sharpe_neg}.")
    if sharpe_zero != 0.0:
        errors.append(f"Expected 0.0 for constant zero returns, got {sharpe_zero}.")

    if errors:
        print("METRICS SEMANTICS CHECK: FAIL")
        for i, err in enumerate(errors, start=1):
            print(f"{i}. {err}")
        return 1

    print("METRICS SEMANTICS CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
