"""
The ``machina`` command line.

    machina params sync  --csv params/params.csv --modules pkg.power,pkg.timing [--prune]
    machina params check --csv params/params.csv --modules pkg.power,pkg.timing [--allow-unfilled]

``sync`` writes; ``check`` never does, and is what CI runs. ``--modules``
names the modules that call :func:`machina.params.param` -- machina is a
library, so the consumer owns that list (icarus hardcodes its own). The
working directory is put on ``sys.path`` first, as ``python -m`` would, so a
consumer's own packages import from its repository root.

``--contract module:Name`` activates a different params format for the run,
which is how icarus keeps its MAVLink ten-column table:
``--contract icarus_dynamics.contract:MachpilotContract``.

``machina report`` arrives in Phase 4.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

__all__ = ["main"]


def _load_contract(spec: str):
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise SystemExit(f"--contract must look like module.path:Name, got {spec!r}")
    target = getattr(importlib.import_module(module_name), attr)
    return target() if isinstance(target, type) else target


def _prepare(args):
    from machina.params import modules

    return modules.load_all(modules.parse_module_list(args.modules))


def _sync(args) -> int:
    from machina.params import sync

    decls = _prepare(args)
    diff = sync.sync(args.csv, decls, prune=args.prune)
    lines = diff.render(pruned=args.prune)
    for line in lines:
        print(line)
    from machina.params import table
    blank = [r["name"] for r in table.read(args.csv) if not r["value"]]
    print(f"{args.csv}: {len(decls)} declared, {len(lines)} change(s).")
    if blank:
        print(f"{len(blank)} param(s) awaiting a value: {', '.join(blank)}. "
              f"The tool does not guess -- fill in value, provenance and source by hand.")
    return 0


def _check(args) -> int:
    from machina.params import check

    decls = _prepare(args)
    errors, warnings = check.check(args.csv, decls, allow_unfilled=args.allow_unfilled)
    for line in warnings:
        print(f"warning:{line}")
    for line in errors:
        print(f"error:{line}")
    if errors:
        print(f"{args.csv}: {len(errors)} error(s). Not fit to publish.")
        return 1
    print(f"{args.csv}: OK ({len(decls)} params, {len(warnings)} warning(s)).")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="machina", description="machina command line: the params pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    params = commands.add_parser("params", help="declare -> sync -> fill -> check")
    actions = params.add_subparsers(dest="action", required=True)

    def common(sub):
        sub.add_argument("--csv", required=True, type=Path, help="the params table")
        sub.add_argument("--modules", required=True,
                         help="comma-separated modules that call param()")
        sub.add_argument("--contract", default=None,
                         help="module.path:Name of a params contract (default: machina's)")

    sync = actions.add_parser("sync", help="add rows for new declarations (writes the CSV)")
    common(sync)
    sync.add_argument("--prune", action="store_true",
                      help="remove rows that no longer have a declaration")
    sync.set_defaults(handler=_sync)

    check = actions.add_parser("check", help="verify the table against the code (never writes)")
    common(check)
    check.add_argument("--allow-unfilled", action="store_true",
                       help="accept blank values (a work-in-progress branch, never CI)")
    check.set_defaults(handler=_check)
    return parser


def main(argv=None) -> int:
    """Run one command. The active params contract is restored afterwards, so calling
    ``main()`` twice in one process (a test, a notebook) does not leak state."""
    from machina.params import contract

    # Redirected output on Windows is cp1252; a stray non-ASCII doc must not crash the gate.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = _parser().parse_args(argv)
    # Declaring modules and contracts import from the consumer's repository root, as
    # `python -m` would allow; a console script does not put the working directory on the path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        if args.contract:
            with contract.using(_load_contract(args.contract)):
                return args.handler(args)
        return args.handler(args)
    except ValueError as exc:
        print(f"error: {args.csv}: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
