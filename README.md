# marginbt

A powerful Python backtesting engine designed for margin accounts. I built this to bridge the gap between VectorBT's ease of use and the complex requirements of margin-aware execution.

## Overview

`marginbt` is a professional-grade backtesting engine designed for high-precision execution simulation. While many libraries focus on signal generation, I built `marginbt` to solve the complex challenges of **risk-controlled execution**—handling leveraged margin, automated circuit breakers, and deterministic conflict resolution that standard backtesters often oversimplify.

## Key Features

- **Advanced Risk Controls**: Built-in "Daily Loss Limit" (circuit breaker) and "Global Kill-Switch" (drawdown-based halt) to protect capital.
- **Precision Margin Engine**: Real-time tracking of initial and maintenance margin with realistic liquidation simulation and fees.
- **Robust Conflict Resolution**: Configurable "Same-bar Conflict Policy" (e.g., risk-first) to handle scenarios where SL and TP are both triggered within the same bar.
- **Dynamic Risk Sizing**: Automatically calculate position sizes based on equity percentage at risk, taking the Stop-Loss distance into account.
- **Gap-Aware Execution**: Handle price gaps with configurable policies (e.g., filling at bar open if price gaps past your SL).
- **Deterministic Verification**: Integration with a regression snapshot system to ensure your backtest results remain consistent as you develop.
- **VBT-like Ergonomics**: A familiar and simple API for researchers coming from VectorBT, but with much deeper execution control.

## Inspiration

This project is deeply inspired by the excellent [vectorbt](https://github.com/polakowo/vectorbt) library by Oleg Polakow. I aim to provide similar ergonomics (`from_signals`, `entries`, `exits`) while implementing a specialized margin-aware execution engine.

## Installation

```bash
# Clone the repository
git clone https://github.com/KengThanabadee/marginbt.git
cd marginbt

# Install the package and dev dependencies
pip install -e .[dev]
```

## Quick Start

```python
import marginbt as mbt
import pandas as pd
import numpy as np

# Sample data
close = pd.Series([100, 101, 102, 101, 100, 99, 98], name='Close')
entries = pd.Series([True, False, False, False, False, False, False])
exits = pd.Series([False, False, False, False, True, False, False])

# Initialize engine (defaults: 10k USDT, no leverage, 1% risk)
engine = mbt.BacktestEngine()

# Run backtest
result = engine.run(close=close, entries=entries, exits=exits)

# View results
print(result.summary())
```

## Running Tests

Run unit tests:

```bash
pytest -q
```

Verify regression snapshots:

```bash
python tests/regression_snapshot.py --mode verify
```

Run the full local quality gate (same command used in CI):

```bash
python scripts/verify.py
```

Enable repository-managed pre-push hook:

```bash
python scripts/install_git_hooks.py
```

## Documentation

- `docs/README.md` documentation index
- `docs/PRACTICAL_GATE.md` practical merge and quality-gate policy
- `docs/USAGE_GUIDE.md` full API and behavior guide

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
