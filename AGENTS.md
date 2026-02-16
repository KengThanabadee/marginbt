# Repository Guidelines

## Project Structure & Module Organization
- Core package: `marginbt/`.
- Public API and orchestration: `marginbt/__init__.py`, `marginbt/engine.py`.
- Domain models and validation: `marginbt/types.py`, `marginbt/portfolio.py`, `marginbt/config_validate.py`.
- Rule-based execution engine: `marginbt/execution/rule_based/` (`engine.py`, `types.py`, `math_utils.py`).
- Tests and fixtures: `tests/` with deterministic fixture data in `tests/fixtures/regression_snapshot.json`.

## Build, Test, and Development Commands
- Install runtime deps: `pip install -r requirements.txt`
- Install package + dev deps: `pip install -e .[dev]`
- Run all pytest-style tests: `pytest -q`
- Run script-based edge checks: `python tests/edge_cases.py`
- Run metric semantics check: `python tests/metrics_semantics.py`
- Verify regression snapshot: `python tests/regression_snapshot.py --mode verify`
- Refresh regression snapshot intentionally: `python tests/regression_snapshot.py --mode capture`

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indentation and type hints.
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep modules focused and small; place new execution logic under `marginbt/execution/`.
- Prefer explicit, deterministic numeric handling (see snapshot and metric tests).

## Testing Guidelines
- Add tests in `tests/` as either:
  - `pytest`-discoverable files named `test_*.py`, or
  - executable verification scripts with a `main()` returning non-zero on failure.
- Use fixed inputs and seeded/deterministic series for backtest assertions.
- For behavior changes, update `tests/fixtures/regression_snapshot.json` only with clear rationale in the PR.

## Commit & Pull Request Guidelines
- Local Git history is not available in this workspace snapshot; use Conventional Commit style going forward (for example, `feat(engine): add liquidation guard`).
- Keep commits small and single-purpose.
- PRs should include:
  - what changed and why,
  - linked issue/ticket,
  - exact verification commands run,
  - fixture diffs or sample output when behavior/metrics change.

## Security & Configuration Tips
- Do not commit secrets, API keys, or environment-specific credentials.
- Validate config assumptions early (for example via `marginbt/config_validate.py`) and fail fast on invalid inputs.
