"""Command line: python -m src.runs {register,status,phase1}"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ..config import (
    LOCK_PATH,
    REPO_ROOT,
    PreregistrationError,
    register_thresholds,
    registration_is_committed,
    verify_registration,
)
from .phase1 import run_phase1
from .spec import RunSpec


def _cmd_register(args: argparse.Namespace) -> int:
    lock = register_thresholds(force=args.force)
    print(f"Registered thresholds sha256={lock['sha256'][:16]}... at {lock['registered_utc']}")
    print(f"Lock file: {LOCK_PATH.relative_to(REPO_ROOT)}")
    print("Commit configs/thresholds.yaml and the lock BEFORE scoring the test split (AGENTS.md section 3).")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    try:
        info = verify_registration()
    except PreregistrationError as e:
        print(f"NOT VALID: {e}")
        return 1
    print(f"Registered: sha256={info['sha256'][:16]}... at {info['registered_utc']}")
    print(f"Committed in git: {registration_is_committed()}")
    return 0


def _cmd_phase1(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec)
    spec = RunSpec.from_yaml(spec_path if spec_path.is_absolute() else REPO_ROOT / spec_path)
    if args.mode:
        spec.mode = args.mode
    if args.include_test:
        spec.include_test = True
    if args.budget_usd is not None:
        spec.budget_usd = args.budget_usd
    try:
        res = asyncio.run(run_phase1(spec))
    except PreregistrationError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"\nArtifacts: {res['output_dir']}")
    print((Path(res["output_dir"]) / "report.md").read_text(encoding="utf-8"))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m src.runs")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="Pre-register configs/thresholds.yaml (writes the lock)")
    r.add_argument("--force", action="store_true", help="Overwrite an existing registration (only if no test result was seen)")
    r.set_defaults(fn=_cmd_register)

    s = sub.add_parser("status", help="Check whether the current thresholds match the registration")
    s.set_defaults(fn=_cmd_status)

    ph = sub.add_parser("phase1", help="Run the Phase 1 Jev calibration probe")
    ph.add_argument("--spec", default="configs/runs/phase1.yaml")
    ph.add_argument("--mode", choices=["live", "mock"], help="Override the spec's mode")
    ph.add_argument("--include-test", action="store_true",
                    help="Also score the TEST split (requires a valid pre-registration)")
    ph.add_argument("--budget-usd", type=float, help="Hard spend cap for live calls (overrides the spec)")
    ph.set_defaults(fn=_cmd_phase1)

    args = p.parse_args()
    sys.exit(args.fn(args))
