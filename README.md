# marginbt

Margin-aware execution backtest engine with VBT-like ergonomics.

## Overview

`marginbt` is a Python library for backtesting trading strategies with a focus on margin requirements, liquidations, and flexible position sizing. I aim to provide the power of low-level execution logic while maintaining the ease of use of VectorBT.

## Key Features

- **Margin Management**: Track maintenance margin, initial margin, and leverage.
- **Liquidation Logic**: Simulate liquidations based on margin ratios.
- **Flexible Sizing**: Support for percent of equity, absolute units, and more.
- **VBT-like Ergonomics**: Simple API for running backtests from signals or rules.
- **Deterministic Verification**: Regression snapshots to ensure consistency over time.

## Inspiration

This project is deeply inspired by the excellent [vectorbt](https://github.com/polakowo/vectorbt) library by Oleg Polakow. I aim to provide similar ergonomics (`from_signals`, `entries`, `exits`) while implementing a specialized margin-aware execution engine.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marginbt.git
cd marginbt

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
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

# Initialize engine
engine = mbt.BacktestEngine(init_cash=1000, leverage=10)

# Run backtest
result = engine.run(close=close, entries=entries, exits=exits)

# View results
print(result.summary())
```

## Running Tests

To run the test suite:

```bash
pytest -q
```

To verify regression snapshots:

```bash
python tests/regression_snapshot.py --mode verify
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
