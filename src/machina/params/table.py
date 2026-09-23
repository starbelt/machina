"""
Reading and writing the params CSV in exactly one canonical form.

Adopted from icarus-dynamics ``params/table.py``. There is one encoding of any
given table. That is not tidiness: the determinism double-build byte-diffs two
independent runs, so an emitter with any freedom in it -- quoting style, float
spelling, line endings -- would produce a build that fails intermittently.

Column ownership, which ``sync`` and ``check`` both depend on:

``TOOL_OWNED``
    derived from the declaration in Python; ``sync`` overwrites these.
human-owned
    every other column except ``name``: ``value``, ``source`` and (under the
    default contract) ``provenance``. No tool writes them, ever. The set comes
    from the active contract, so icarus's ten-column format keeps its two.
"""

import csv
import io
from pathlib import Path

from machina.params import contract
from machina.params.registry import ParamDecl

__all__ = ["TOOL_OWNED", "human_owned", "columns", "Row", "format_number", "row_from_decl",
           "dumps", "loads", "parse_rows", "read", "write"]

TOOL_OWNED = ("type", "unit", "min", "max", "default", "mutability", "comment")

# A bare CR or LF inside a field is refused on both sides of the pipe. Nothing in a params
# table legitimately spans two lines, and the interpreters disagree about the bytes: 3.12's
# csv.writer quotes such a field, 3.10's does not, and 3.10's csv.reader then refuses its own
# output. Refusing it is what leaves dumps() with no freedom in it.
_NEWLINE_ADVICE = "a params table field is one line -- remove it and re-run `machina params sync`"
# Stands in for a bare CR while parsing, so 3.10 and 3.12 read the same fields and name the
# same row. Private-use, so it cannot collide with anything a params table would carry.
_CR_MARK = ""

Row = dict


def columns() -> tuple:
    """The exact header of the active contract."""
    return tuple(contract.get().COLUMNS)


def human_owned() -> tuple:
    """Columns no tool writes, in header order."""
    return tuple(c for c in columns() if c != "name" and c not in TOOL_OWNED)


def format_number(value, is_integer: bool) -> str:
    """One spelling per number.

    Integer types print without a decimal point; float types always carry one,
    so ``lo=0`` on an f32 param is ``0.0`` rather than sometimes ``0``.
    ``repr()`` of a float is the shortest string that round-trips exactly, which
    keeps the CSV both readable and lossless.
    """
    if value is None:
        return ""
    return str(int(value)) if is_integer else repr(float(value))


def row_from_decl(decl: ParamDecl, **human) -> Row:
    """The row a declaration implies, carrying over any human-owned values given.

    Unknown keyword names are refused rather than dropped, so a misspelt
    ``provenence=`` cannot silently lose a provenance code.
    """
    owned = human_owned()
    unknown = sorted(k for k in human if k not in owned)
    if unknown:
        raise ValueError(f"{unknown} are not human-owned columns of this format; "
                         f"those are {list(owned)}")
    derived = {
        "name": decl.name,
        "type": decl.type,
        "unit": decl.unit,
        "min": format_number(decl.lo, decl.is_integer),
        "max": format_number(decl.hi, decl.is_integer),
        "default": format_number(decl.default, decl.is_integer),
        "mutability": decl.mutability,
        "comment": decl.doc,
    }
    return {c: derived[c] if c in derived else human.get(c, "") for c in columns()}


def _refuse_newline(where) -> None:
    raise ValueError(f"row {where} contains a bare carriage return or newline inside a field; "
                     f"{_NEWLINE_ADVICE}")


def parse_rows(text: str) -> list:
    """The raw fields of every line, refusing a bare CR or LF inside any field.

    One parser for both readers -- :func:`loads` and the default contract's linter -- so the
    two agree about what a table is, and the same row gets named on either interpreter.

    3.10's ``csv.reader`` raises on a bare CR in an unquoted field where 3.12's accepts it
    silently, so the CR is swapped for a marker the reader has no opinion about *before*
    parsing. Both then see the same fields, and the scan below reports the same row. A CR
    that is half of a CRLF line end is a separate complaint, made by ``check`` when the file
    fails to re-emit; it is collapsed here so it cannot be mistaken for this one.
    """
    scan = text.replace("\r\n", "\n")
    marked = "\r" in scan
    if marked:
        scan = scan.replace("\r", _CR_MARK)
    try:
        rows = list(csv.reader(io.StringIO(scan)))
    except csv.Error as exc:
        raise ValueError(f"a field contains a bare carriage return or newline "
                         f"({exc}); {_NEWLINE_ADVICE}") from None
    if not rows:
        return rows
    # Name the offending row the way a human would find it: by its name column when the header
    # has one and the cell is filled, else by its 1-based position in the file.
    name_at = rows[0].index("name") if "name" in rows[0] else None
    for number, fields in enumerate(rows, start=1):
        for field in fields:
            if (marked and _CR_MARK in field) or "\n" in field:
                named = name_at is not None and number > 1 and name_at < len(fields)
                _refuse_newline(repr(fields[name_at]) if named and fields[name_at] else number)
    return rows


def dumps(rows: list) -> str:
    """Canonical text: the contract's header, rows sorted by name, LF line ends.

    A field holding a bare CR or LF is refused before anything is written: the two
    interpreters' writers spell it differently, and this emitter has one spelling per table.
    """
    header = columns()
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for row in sorted(rows, key=lambda r: r["name"]):
        fields = [row[c] for c in header]
        for field in fields:
            if "\r" in str(field) or "\n" in str(field):
                _refuse_newline(repr(row["name"]) if row["name"] else "(unnamed)")
        writer.writerow(fields)
    return out.getvalue()


def loads(text: str) -> list:
    header = columns()
    if text.startswith("\ufeff"):
        raise ValueError(
            "the file starts with a UTF-8 byte-order mark, which Excel's 'CSV UTF-8' save "
            "adds. Save it as plain CSV (or run `machina params sync` to re-emit it)."
        )
    rows = parse_rows(text)
    if not rows:
        return []
    if tuple(rows[0]) != header:
        raise ValueError(f"header must be exactly {','.join(header)}")
    out = []
    for lineno, fields in enumerate(rows[1:], start=2):
        if not fields:
            continue
        if len(fields) != len(header):
            raise ValueError(f"line {lineno} has {len(fields)} fields; the header has "
                             f"{len(header)}")
        out.append(dict(zip(header, fields)))
    return out


def read(path: Path) -> list:
    """The rows of a table. Bytes are decoded as ``check`` decodes them: CRLF line ends
    are read as LF, and a lone CR inside a field is refused rather than translated."""
    path = Path(path)
    if not path.exists():
        return []
    return loads(path.read_bytes().decode("utf-8").replace("\r\n", "\n"))


def write(path: Path, rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so Windows does not turn the canonical LF into CRLF on the way out.
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(dumps(rows))
