from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS_PATH = ".githooks"
PRE_PUSH_HOOK = ROOT / ".githooks" / "pre-push"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def main() -> int:
    set_result = _run(["git", "config", "--local", "core.hooksPath", HOOKS_PATH])
    if set_result.returncode != 0:
        if set_result.stderr:
            print(set_result.stderr.strip())
        print("FAIL: could not set git core.hooksPath.")
        return int(set_result.returncode)

    get_result = _run(["git", "config", "--get", "core.hooksPath"])
    configured = (get_result.stdout or "").strip()
    if configured != HOOKS_PATH:
        print(f"FAIL: expected core.hooksPath={HOOKS_PATH}, got {configured!r}")
        return 1

    if PRE_PUSH_HOOK.exists():
        mode = PRE_PUSH_HOOK.stat().st_mode
        PRE_PUSH_HOOK.chmod(mode | 0o111)

    print(f"PASS: configured git core.hooksPath={HOOKS_PATH}")
    print("Pre-push hook is now active for this local clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
