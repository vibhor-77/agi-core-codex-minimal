"""Small CLI for the readable 4-pillar scaffold."""

from __future__ import annotations

import sys

from domains import run_arc, run_inspect, run_synthetic


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "inspect":
        if len(sys.argv) < 3:
            raise SystemExit("usage: python minimal.py inspect TASK_ID")
        run_inspect(sys.argv[2])
        return
    if mode in {"synthetic", "both"}:
        run_synthetic()
    if mode in {"arc", "both"}:
        run_arc()


if __name__ == "__main__":
    main()
