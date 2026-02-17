# Practical Gate Setup

This guide defines the default workflow to keep `main` stable with a simple, repeatable process.

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
4. Set required approvals to `0` for solo-maintainer workflow.
5. Add required status checks when they appear in the selector:
   - `CI / test (3.10)`
   - `CI / test (3.11)`
   - `CI / test (3.12)`
