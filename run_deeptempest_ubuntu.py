#!/usr/bin/env python3
"""Ubuntu-friendly launcher and health check utility for Deep-Tempest."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent

REQUIRED_MODULES = ["numpy", "torch", "cv2", "PIL", "matplotlib", "scipy", "skimage"]
OPTIONAL_MODULES = ["gnuradio", "torchvision"]


class CheckError(RuntimeError):
    pass


def _run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True, check=False)


def _print_result(name: str, ok: bool, detail: str) -> None:
    icon = "[OK]" if ok else "[FAIL]"
    print(f"{icon:<6} {name}: {detail}")


def _check_python() -> None:
    ok = sys.version_info >= (3, 10)
    _print_result("python", ok, sys.version.split()[0])
    if not ok:
        raise CheckError("Python >= 3.10 is required.")


def _check_modules(modules: Iterable[str], *, required: bool) -> list[str]:
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
            _print_result(f"import {module}", True, "available")
        except Exception as exc:  # noqa: BLE001
            missing.append(module)
            _print_result(f"import {module}", False, str(exc))

    if required and missing:
        raise CheckError(f"Missing required modules: {', '.join(missing)}")
    return missing


def _check_paths() -> None:
    for rel in ["README.md", "gr-tempest", "end-to-end", "text_generation"]:
        path = ROOT / rel
        ok = path.exists()
        _print_result("path", ok, rel)
        if not ok:
            raise CheckError(f"Required project path not found: {path}")


def cmd_check(_: argparse.Namespace) -> int:
    _check_python()
    _check_paths()
    _check_modules(REQUIRED_MODULES, required=True)
    missing_optional = _check_modules(OPTIONAL_MODULES, required=False)
    if missing_optional:
        print("\nOptional dependencies missing:")
        for module in missing_optional:
            print(f"  - {module}")
    print("\nEnvironment check completed.")
    return 0


def cmd_compile(_: argparse.Namespace) -> int:
    proc = _run_cmd([sys.executable, "-m", "compileall", "-q", "."], cwd=ROOT)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise CheckError("compileall failed")
    _print_result("compileall", True, "All Python files compiled")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ubuntu launcher and diagnostics for Deep-Tempest")
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="Run environment checks")
    check_parser.set_defaults(func=cmd_check)

    compile_parser = sub.add_parser("compile", help="Compile all Python files to detect syntax issues")
    compile_parser.set_defaults(func=cmd_compile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CheckError as exc:
        _print_result("summary", False, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
