# Contributing to marginbt

Thank you for your interest in contributing!  `marginbt` is a precision margin execution engine, so correctness and testability are paramount.

## Development Setup

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/marginbt.git
   cd marginbt
   ```

2. **Install Dependencies**
   Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e .[dev]
   ```
   This installs pinned versions of `pytest`, `ruff`, and `mypy`.

3. **Enable local pre-push gate (recommended)**
   Configure repository-managed hooks:
   ```bash
   python scripts/install_git_hooks.py
   ```
   This sets `core.hooksPath=.githooks` for this local clone and enables the `pre-push` check.

## Running Tests

Use `pytest` for all tests.

```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/test_edge_cases.py
```

### Regression Snapshots
To ensure deterministic execution, verify results against a "gold standard" snapshot.

```bash
# Verify behavior matches snapshot (Run this before PR!)
python tests/regression_snapshot.py --mode verify
```

### One-command Verify Check

Run the exact local quality gate used by CI:

```bash
python scripts/verify.py
```

This command runs:
- `ruff check marginbt/ tests/`
- `pytest -q`
- `python tests/regression_snapshot.py --mode verify`

## Code Style & Linting

Use strict typing and linting.

```bash
# Lint code
ruff check marginbt/ tests/

# Type check
mypy marginbt/
```

- Follow PEP 8.
- Use type hints for **all** public functions.
- Add docstrings (Google style preferred).

## Pull Request Guidelines

1. **Create a Branch**: `git checkout -b feat/my-new-feature`
2. **Add Tests**: If fixing a bug, add a regression test. If adding a feature, add unit tests.
3. **Update Docs**: If changing API, update `docs/USAGE_GUIDE.md`.
4. **Pass CI**: Ensure `pytest`, `ruff`, and `mypy` all pass locally.
5. **Submit PR**: Describe *what* you changed and *why*. Link to issues if applicable.

## Branch Protection (Maintainer Setup)

Use `docs/PRACTICAL_GATE.md` as the source of truth for branch protection and quality gate setup.

## Release Process (Maintainers Only)

1. Bump version in `marginbt/__init__.py` and `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Create a git tag: `git tag v0.1.x`.
4. Push tag: `git push origin --tags`.
