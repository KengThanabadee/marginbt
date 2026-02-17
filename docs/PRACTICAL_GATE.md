# Practical Gate Setup

This guide defines the minimum workflow to keep `main` stable without unnecessary process overhead.

## Local Workflow (Default)

Install dev dependencies:

```bash
pip install -e .[dev]
```

Enable repository-managed git hooks:

```bash
python scripts/install_git_hooks.py
```

The pre-push hook runs:

```bash
python scripts/verify.py
```

If any check fails, push is blocked.

## Merge Policy (Default)

For branch `main`, enable:

1. Require a pull request before merging.
2. Require status checks to pass before merging.
3. Require branches to be up to date before merging.

## Optional Hardening (Enable Later)

If needed, add:

1. Required status checks:
   - `CI / test (3.10)`
   - `CI / test (3.11)`
   - `CI / test (3.12)`
2. "Do not allow bypassing the above settings".
3. "Restrict who can push to matching branches" (if available in your UI).

## UI Notes

1. Required checks may not appear until the repository has recent check history.
2. If required checks are not selectable yet, keep status checks enabled and run `python scripts/verify.py` before merge.
