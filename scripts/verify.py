from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, cmd: list[str]) -> int:
    print(f"\n==> {name}")
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    checks: list[tuple[str, list[str]]] = [
        ("Lint with ruff", [sys.executable, "-m", "ruff", "check", "marginbt/", "tests/"]),
        ("Run tests", [sys.executable, "-m", "pytest", "-q"]),
        ("Verify regression snapshot", [sys.executable, "tests/regression_snapshot.py", "--mode", "verify"]),
    ]

    for name, cmd in checks:
        code = _run(name, cmd)
        if code != 0:
            print(f"\nFAIL: {name} (exit {code})")
            return code

    print("\nPASS: All strict-gate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
