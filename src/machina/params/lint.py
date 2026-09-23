"""
``DefaultContract`` -- machina's own params CSV format and its linter.

A reimplementation, not a port. icarus-dynamics re-exports its format rules
from ``lint_params`` in the private ``machpilot-rt`` repository, which is not
in its tree and not distributable. What icarus's code and tests pin about that
linter -- the ten-column header, ``"value is empty"`` for a blank value, and
``"on the f32 wire"`` for a float32 quantization warning -- is kept, so the two
read the same way.

What differs, on purpose (Decision Log #47 and #50):

=====================  =====================================================
column set             icarus's ten plus ``provenance``, next to ``value``
names                  ``[A-Za-z][A-Za-z0-9_]*``, up to 64 characters; the
                       16-character uppercase rule is MAVLink's, not ours
types                  ``f32 f64 i8 i16 i32 u8 u16 u32``
mutabilities           ``fixed design flight disarmed boot``
provenance             a filled ``value`` needs a code: ``D`` documented,
                       ``P`` physics, ``E`` estimate, ``A`` assumption,
                       ``M`` measured
=====================  =====================================================

The tool never guesses a value. ``sync`` leaves ``value`` blank and this
linter refuses the blank unless told the table is a work in progress.
"""

import math
import re
import struct

from machina.params import table
from machina.params.contract import LintIssue
from machina.units import NON_SI_UNITS, UNIT_RE, is_si, is_well_formed

__all__ = ["DefaultContract", "INT_RANGES", "PROVENANCE_MEANINGS"]

# What any reader of a CSV -- strtod, a spreadsheet, another language -- parses the same way.
# Python's float() also accepts "1_0", surrounding spaces and non-ASCII digits; nothing else does.
_DECIMAL = re.compile(r"[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?")
_INFINITE = re.compile(r"[-+]?inf")

INT_RANGES = {
    "i8": (-2 ** 7, 2 ** 7 - 1), "i16": (-2 ** 15, 2 ** 15 - 1), "i32": (-2 ** 31, 2 ** 31 - 1),
    "u8": (0, 2 ** 8 - 1), "u16": (0, 2 ** 16 - 1), "u32": (0, 2 ** 32 - 1),
}

PROVENANCE_MEANINGS = {
    "D": "documented (a datasheet, a standard, a cited paper)",
    "P": "physics (derived from first principles)",
    "E": "estimate (an engineering judgement with a basis)",
    "A": "assumption (chosen, and could be otherwise)",
    "M": "measured (a benchmark, a test, a log)",
}


class DefaultContract:
    """machina's params format. See the module docstring for the rules."""

    COLUMNS = ("name", "value", "type", "unit", "min", "max", "default", "mutability",
               "provenance", "source", "comment")
    NAME_RULE = "[A-Za-z][A-Za-z0-9_]*"
    NAME_RE = re.compile(NAME_RULE)
    MAX_NAME_LEN = 64
    NAME_LENGTH_REASON = "a name that long stops being readable in a report table"
    NUMERIC_TYPES = ("f32", "f64", "i8", "i16", "i32", "u8", "u16", "u32")
    INT_TYPES = ("i8", "i16", "i32", "u8", "u16", "u32")
    MUTABILITIES = ("fixed", "design", "flight", "disarmed", "boot")
    PROVENANCE_CODES = ("D", "P", "E", "A", "M")
    NON_SI_UNITS = NON_SI_UNITS
    UNIT_RE = UNIT_RE
    is_si = staticmethod(is_si)      # the allow-list in machina.units; icarus supplies its own

    def lint_text(self, text: str, *, allow_unfilled: bool = False) -> list:
        """Every problem with the table, in file order. Warnings do not fail it."""
        # A table the parser cannot read at all is one issue, not a traceback -- the same
        # shape as an Excel BOM, which falls out below as a header mismatch.
        try:
            rows = table.parse_rows(text)
        except ValueError as exc:
            return [LintIssue(str(exc))]
        if not rows:
            return [LintIssue(f"the table is empty; it needs at least the header "
                              f"{','.join(self.COLUMNS)}")]
        if tuple(rows[0]) != self.COLUMNS:
            return [LintIssue(f"header must be exactly {','.join(self.COLUMNS)}", line=1)]

        issues, names = [], []
        for lineno, fields in enumerate(rows[1:], start=2):
            if not fields:
                continue
            if len(fields) != len(self.COLUMNS):
                issues.append(LintIssue(
                    f"has {len(fields)} fields; the header has {len(self.COLUMNS)}",
                    line=lineno))
                continue
            row = dict(zip(self.COLUMNS, fields))
            names.append(row["name"])
            issues.extend(self._lint_row(row, lineno, allow_unfilled))

        reported = []
        for name in names:
            count = names.count(name)
            if count > 1 and name not in reported:
                issues.append(LintIssue(f"appears {count} times; names are unique", name=name))
                reported.append(name)
        if names != sorted(names):
            issues.append(LintIssue("rows are not sorted by name"))
        return issues

    # --- one row ----------------------------------------------------------------------------

    def _lint_row(self, row: dict, lineno: int, allow_unfilled: bool) -> list:
        name = row["name"]
        issues = []

        def fail(message, warning=False):
            issues.append(LintIssue(message, warning=warning, line=lineno, name=name or None))

        if not self.NAME_RE.fullmatch(name):
            fail(f"name must match {self.NAME_RULE}")
        elif len(name) > self.MAX_NAME_LEN:
            fail(f"name is {len(name)} characters; the limit is {self.MAX_NAME_LEN} "
                 f"({self.NAME_LENGTH_REASON})")

        kind = row["type"]
        if kind not in self.NUMERIC_TYPES:
            fail(f"type {kind!r} is not one of {' '.join(self.NUMERIC_TYPES)}")
            kind = None

        unit = row["unit"]
        if not unit:
            fail('unit is mandatory (use "1" for dimensionless)')
        elif not is_well_formed(unit):
            fail(f"unit {unit!r} has unexpected characters")
        elif not is_si(unit):
            fail(f"unit {unit!r} is not SI; convert at the boundary that produced the number")

        if row["mutability"] not in self.MUTABILITIES:
            fail(f"mutability {row['mutability']!r} is not one of {' | '.join(self.MUTABILITIES)}")

        if not row["comment"].strip():
            fail("comment is empty; a parameter nobody can explain is one nobody can review")

        lo = self._number(row["min"], "min", kind, fail, allow_infinite=True)
        hi = self._number(row["max"], "max", kind, fail, allow_infinite=True)
        if lo is not None and hi is not None and lo > hi:
            fail(f"min {row['min']} is greater than max {row['max']}")

        if row["default"]:
            default = self._number(row["default"], "default", kind, fail)
            self._in_range(default, "default", lo, hi, fail)

        provenance = row["provenance"]
        if provenance and provenance not in self.PROVENANCE_CODES:
            fail(f"provenance {provenance!r} is not one of {' '.join(self.PROVENANCE_CODES)}")

        if not row["value"]:
            if not allow_unfilled:
                fail("value is empty. The tool does not guess a number: fill it in by hand, "
                     "with a provenance code and a source")
            return issues

        value = self._number(row["value"], "value", kind, fail)
        self._in_range(value, "value", lo, hi, fail)
        if not provenance:
            fail(f"a filled value needs a provenance code "
                 f"({', '.join(self.PROVENANCE_CODES)}): where did the number come from?")
        if value is not None and kind == "f32":
            self._f32_loss(value, fail)
        return issues

    # --- helpers ----------------------------------------------------------------------------

    def _number(self, text, column, kind, fail, *, allow_infinite=False):
        if text.strip().lower().lstrip("+-") == "nan":
            fail(f"{column} is NaN")
            return None
        if _INFINITE.fullmatch(text) and not allow_infinite:
            fail(f"{column} {text!r} is infinite")
            return None
        if not (_DECIMAL.fullmatch(text) or (allow_infinite and _INFINITE.fullmatch(text))):
            fail(f"{column} {text!r} is not a number in plain decimal form (digits, one '.', "
                 f"optional exponent; no spaces, underscores or non-ASCII digits)")
            return None
        try:
            value = float(text)
        except ValueError:
            fail(f"{column} {text!r} is not a number")
            return None
        if math.isnan(value):
            fail(f"{column} is NaN")
            return None
        if math.isinf(value):
            if allow_infinite and kind not in self.INT_TYPES:
                return value
            fail(f"{column} {text!r} is infinite")
            return None
        if kind in self.INT_TYPES:
            if not value.is_integer():
                fail(f"{column} {text!r} is not an integer, but the type is {kind}")
                return None
            low, high = INT_RANGES[kind]
            if not low <= value <= high:
                fail(f"{column} {text!r} does not fit in {kind} [{low}, {high}]")
                return None
        return value

    @staticmethod
    def _in_range(value, column, lo, hi, fail):
        if value is None or lo is None or hi is None:
            return
        if not lo <= value <= hi:
            fail(f"{column} {value!r} is outside [{lo!r}, {hi!r}]")

    @staticmethod
    def _f32_loss(value, fail):
        try:
            wire = struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            fail(f"value {value!r} exceeds the f32 range")
            return
        if wire != value:
            fail(f"value {value!r} is {wire!r} on the f32 wire (loss {abs(wire - value):.3g})",
                 warning=True)
