"""
Verify the params CSV without writing to it. This is the CI gate.

Adopted from icarus-dynamics ``params/check.py``. Two independent questions,
and both have to hold:

1. **Does the table agree with the code?** Nothing declared is missing,
   nothing tool-owned has drifted, nothing is orphaned, and the file is in its
   canonical encoding. Computed by re-running ``sync``'s plan against the file.
2. **Is the table legal?** The active contract's linter, strict by default --
   so an empty ``value`` fails, and under machina's default contract so does a
   filled value with no provenance code.

The file is read as bytes, so a table that picked up CRLF line ends from a
Windows checkout is reported as non-canonical rather than silently accepted.
"""

from pathlib import Path

from machina.params import contract, sync, table

__all__ = ["check"]


def check(csv_path: Path, decls: dict, *, allow_unfilled: bool = False) -> tuple:
    """``(errors, warnings)``. Only errors mean the table is unfit to publish.

    Float32 quantization is why this is a pair rather than one list: almost
    every ordinary decimal is inexact on the wire, so those are reported with
    their loss and fail nothing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return ([f"  {csv_path} does not exist -- run `machina params sync` to create it"], [])

    text = csv_path.read_bytes().decode("utf-8")
    try:
        rows = table.loads(text.replace("\r\n", "\n"))
    except ValueError as exc:
        return ([f"  {exc}"], [])

    errors = sync.plan(decls, rows).render()

    # Canonical form, checked by re-emitting. Catches what the row-level diff cannot see:
    # quoting style, a stray trailing newline, CRLF line ends, float spelling.
    if not errors and table.dumps(rows) != text:
        errors.append(
            "  file is not in canonical form (quoting, line endings or number spelling); "
            "run `machina params sync` to re-emit it")

    issues = contract.get().lint_text(text.replace("\r\n", "\n"), allow_unfilled=allow_unfilled)
    errors.extend(f"  {issue}" for issue in issues if not issue.warning)
    warnings = [f"  {issue}" for issue in issues if issue.warning]
    return (errors, warnings)
