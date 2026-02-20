from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str


@dataclass
class CheckReport:
    python_version: str
    passed: bool
    items: list[CheckItem]

    def to_json(self) -> str:
        payload = {
            "python_version": self.python_version,
            "passed": self.passed,
            "items": [asdict(i) for i in self.items],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def run_import_checks(required: Iterable[str], optional: Iterable[str]) -> CheckReport:
    items: list[CheckItem] = []
    py_ok = sys.version_info >= (3, 10)
    items.append(CheckItem("python", py_ok, sys.version.split()[0]))

    def check_mod(module: str, required_mod: bool) -> None:
        try:
            importlib.import_module(module)
            items.append(CheckItem(f"import:{module}", True, "available"))
        except Exception as exc:  # noqa: BLE001 - diagnostic context
            prefix = "required" if required_mod else "optional"
            items.append(CheckItem(f"import:{module}", not required_mod, f"{prefix} missing: {exc}"))

    for mod in required:
        check_mod(mod, True)
    for mod in optional:
        check_mod(mod, False)

    must_pass = all(i.ok for i in items if i.name == "python" or i.name.startswith("import:"))
    return CheckReport(sys.version.split()[0], must_pass, items)


def check_project_layout(required_paths: Iterable[str]) -> CheckReport:
    items: list[CheckItem] = []
    for rel in required_paths:
        path = ROOT / rel
        items.append(CheckItem(f"path:{rel}", path.exists(), str(path)))
    passed = all(i.ok for i in items)
    return CheckReport(sys.version.split()[0], passed, items)


def run_compileall() -> CheckItem:
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "."],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return CheckItem("compileall", True, "success")
    detail = (proc.stderr or proc.stdout or "compileall failed").strip()
    return CheckItem("compileall", False, detail)


def run_text_generation(output_dir: str | None = None) -> CheckItem:
    script = ROOT / "text_generation" / "generate_random_text.py"
    if not script.exists():
        return CheckItem("generate_text", False, f"missing: {script}")

    workdir = script.parent
    env = dict(os.environ)
    if output_dir:
        env["DEEPTEMPEST_OUTPUT_DIR"] = output_dir

    proc = subprocess.run(
        [sys.executable, script.name],
        cwd=str(workdir),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode == 0:
        return CheckItem("generate_text", True, "completed")
    detail = (proc.stderr or proc.stdout or "generation failed").strip()
    return CheckItem("generate_text", False, detail)
