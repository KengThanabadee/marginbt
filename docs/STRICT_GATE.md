# Strict Gate Setup

This repository uses a strict quality gate to prevent broken pushes from reaching `main`.

## 1) Local gate (pre-push)

Install dev dependencies:

```bash
pip install -e .[dev]
```

Enable repository-managed hooks in your local clone:

```bash
python scripts/install_git_hooks.py
```

This activates `.githooks/pre-push`, which runs:

```bash
python scripts/verify.py
```

If any check fails, push is blocked.

## 2) Single standard verify command

Run this before pushing:

```bash
python scripts/verify.py
```

The command runs:

1. `ruff check marginbt/ tests/`
2. `pytest -q`
3. `python tests/regression_snapshot.py --mode verify`

## 3) GitHub branch protection (maintainer action)

Apply these settings to branch `main`:

1. Require a pull request before merging.
2. Require status checks to pass before merging.
3. Required checks:
   - `CI / test (3.10)`
   - `CI / test (3.11)`
   - `CI / test (3.12)`
4. Require branches to be up to date before merging.
5. Restrict who can push to matching branches (block direct pushes).
