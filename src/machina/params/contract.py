"""
The params CSV format, as an injectable contract.

icarus-dynamics' params pipeline imports its format rules -- the column set,
the name pattern, the type and mutability vocabularies, and the linter itself
-- from ``machpilot-rt``, a private repository that arrives as a Nix flake
input. That module is not distributable, and it encodes a MAVLink wire format
machina has no reason to impose on a thesis spreadsheet. So the format is a
*contract*: an object carrying those rules, chosen at run time.

:class:`machina.params.lint.DefaultContract` is machina's own (Decision Log
#50): eleven columns, relaxed names, and a mandatory provenance code on every
filled value (#47). icarus injects a contract wrapping its pinned linter with
``use(MachpilotContract())`` and keeps its ten-column file and strict
uppercase names unchanged.

Nothing here imports the default contract at module load, and nothing in
:mod:`machina.params` reads the contract until it is asked to validate
something. That is deliberate: icarus's eager ``from .declare import param``
is what made its whole test suite fail at collection outside ``nix develop``.
"""

import re
from dataclasses import dataclass
from typing import Protocol

__all__ = ["Contract", "LintIssue", "use", "get", "reset", "using"]


@dataclass(frozen=True)
class LintIssue:
    """One problem with a params table. Only non-warnings make a table unfit to publish.

    Named ``LintIssue`` rather than icarus's ``Problem`` because ``Problem`` is
    the Phase 3 compiler class.
    """

    message: str
    warning: bool = False
    line: int = None
    name: str = None

    def __str__(self) -> str:
        where = []
        if self.line is not None:
            where.append(f"line {self.line}")
        if self.name:
            where.append(self.name)
        prefix = f"{', '.join(where)}: " if where else ""
        return f"{prefix}{self.message}"


class Contract(Protocol):
    """What a params format must supply.

    ``COLUMNS`` is the exact header. Every column that is not ``name`` and not
    one of :data:`machina.params.table.TOOL_OWNED` is human-owned: ``sync``
    never writes it.
    """

    COLUMNS: tuple
    NAME_RE: re.Pattern
    MAX_NAME_LEN: int
    NAME_RULE: str               # human-readable NAME_RE, for error messages
    NAME_LENGTH_REASON: str      # why the limit exists, for error messages
    NUMERIC_TYPES: tuple
    INT_TYPES: tuple
    MUTABILITIES: tuple
    PROVENANCE_CODES: tuple      # empty when the format has no provenance column
    NON_SI_UNITS: frozenset
    UNIT_RE: re.Pattern
    # Optional: is_si(unit) -> bool. When present it decides; otherwise a unit is refused
    # when it is in NON_SI_UNITS, which is icarus's rule.

    def lint_text(self, text: str, *, allow_unfilled: bool = False) -> list: ...


_active = None


def use(contract) -> None:
    """Make ``contract`` the format every params tool validates against."""
    global _active
    missing = [attr for attr in ("COLUMNS", "NAME_RE", "MAX_NAME_LEN", "NUMERIC_TYPES",
                                 "MUTABILITIES", "UNIT_RE", "lint_text")
               if not hasattr(contract, attr)]
    if missing:
        raise TypeError(
            f"{type(contract).__name__} is not a params contract: it is missing {missing}. "
            f"See machina/params/contract.py for the protocol."
        )
    if "name" not in contract.COLUMNS or "value" not in contract.COLUMNS:
        raise TypeError(
            f"{type(contract).__name__}.COLUMNS must include 'name' and 'value'; "
            f"got {contract.COLUMNS}."
        )
    _active = contract


def get():
    """The active contract, defaulting to :class:`DefaultContract` on first use."""
    global _active
    if _active is None:
        from machina.params.lint import DefaultContract
        _active = DefaultContract()
    return _active


def reset() -> None:
    """Return to the default contract. Tests only."""
    global _active
    _active = None


class using:
    """``with using(contract): ...`` -- activate a contract for a block. Tests only."""

    def __init__(self, contract):
        self._contract = contract
        self._previous = None

    def __enter__(self):
        global _active
        self._previous = _active
        use(self._contract)
        return self._contract

    def __exit__(self, *exc):
        global _active
        _active = self._previous
        return False
