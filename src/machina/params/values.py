"""
Numbers out of the params CSV, with their provenance, for the compiler.

This is the seam Phase 3's ``Problem`` reads through: ``Quantity(param="X")``
links a component's leaf to a declaration, and ``values_from(csv)`` supplies
the number together with where it came from. A ``PARAMETER`` or ``FIXED`` role
takes the value; a ``VARIABLE`` or ``DISCRETE`` role takes it as the initial
guess and the declaration's ``min``/``max`` as bounds. Provenance and source
travel onto the solver backend's records so a report can print them.

Values are read from the CSV's own ``type`` column, so this needs no
declarations loaded -- but it does refuse a blank value, because a compile
that silently skipped one would fall back to a default nobody chose.
"""

from dataclasses import dataclass
from pathlib import Path

from machina.params import contract, table

__all__ = ["ParamValue", "values_from"]


@dataclass(frozen=True)
class ParamValue:
    """One filled row of the params table."""

    name: str
    value: float
    unit: str
    lo: float
    hi: float
    provenance: str
    source: str
    doc: str

    def __float__(self) -> float:
        return float(self.value)


def _number(text: str, integer: bool):
    return int(float(text)) if integer else float(text)


def values_from(csv_path: Path, *, allow_unfilled: bool = False) -> dict:
    """``{name: ParamValue}`` in table order.

    A blank value is an error naming every blank row, unless
    ``allow_unfilled`` -- then those rows are left out, and the compiler's own
    default handling decides what happens to the quantities that wanted them.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist -- run `machina params sync`.")
    text = csv_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    problems = [str(i) for i in contract.get().lint_text(text, allow_unfilled=True)
                if not i.warning]
    if problems:
        raise ValueError(
            f"{csv_path} is not a valid params table; run `machina params check` for the full "
            f"report. First problems: {problems[:3]}"
        )
    rows = table.read(csv_path)
    blank = [r["name"] for r in rows if not r["value"]]
    if blank and not allow_unfilled:
        raise ValueError(
            f"{csv_path}: {blank} have no value. The tool does not guess a number; fill them "
            f"in by hand with a provenance code, or pass allow_unfilled=True to leave them out."
        )
    out = {}
    for row in rows:
        if not row["value"]:
            continue
        integer = not row["type"].startswith("f")
        try:
            out[row["name"]] = ParamValue(
                name=row["name"],
                value=_number(row["value"], integer),
                unit=row["unit"],
                lo=float(row["min"]) if row["min"] else float("-inf"),
                hi=float(row["max"]) if row["max"] else float("inf"),
                provenance=row.get("provenance", "") or None,
                source=row.get("source", "") or None,
                doc=row.get("comment", ""),
            )
        except ValueError:
            raise ValueError(
                f"{csv_path}: {row['name']} has a value {row['value']!r} that is not a number; "
                f"run `machina params check` for the full list."
            ) from None
    return out
