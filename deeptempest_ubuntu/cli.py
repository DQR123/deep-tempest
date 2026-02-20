from __future__ import annotations

import argparse

from .core import check_project_layout, run_compileall, run_import_checks, run_text_generation

REQUIRED_MODULES = ["numpy", "torch", "cv2", "PIL", "matplotlib", "scipy", "skimage"]
OPTIONAL_MODULES = ["gnuradio", "torchvision"]
REQUIRED_PATHS = ["README.md", "gr-tempest", "end-to-end", "text_generation"]


def _print_item(name: str, ok: bool, detail: str) -> None:
    icon = "[OK]" if ok else "[FAIL]"
    print(f"{icon:<6} {name}: {detail}")


def cmd_doctor(args: argparse.Namespace) -> int:
    layout = check_project_layout(REQUIRED_PATHS)
    imports = run_import_checks(REQUIRED_MODULES, OPTIONAL_MODULES)

    for item in layout.items + imports.items:
        _print_item(item.name, item.ok, item.detail)

    passed = layout.passed and imports.passed
    if args.json:
        print(imports.to_json())
    return 0 if passed else 1


def cmd_compile(_: argparse.Namespace) -> int:
    item = run_compileall()
    _print_item(item.name, item.ok, item.detail)
    return 0 if item.ok else 1


def cmd_generate(args: argparse.Namespace) -> int:
    item = run_text_generation(args.output_dir)
    _print_item(item.name, item.ok, item.detail)
    return 0 if item.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep-Tempest Ubuntu rewrite CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("doctor", help="Run Ubuntu environment diagnostics")
    p1.add_argument("--json", action="store_true", help="Print import report as JSON")
    p1.set_defaults(func=cmd_doctor)

    p2 = sub.add_parser("compile", help="Compile all Python files to find syntax errors")
    p2.set_defaults(func=cmd_compile)

    p3 = sub.add_parser("generate", help="Run text generation smoke test")
    p3.add_argument("--output-dir", default=None, help="Custom output directory for generated images")
    p3.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
