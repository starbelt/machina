"""
The unit rule, in one place.

Signals (:mod:`machina.model.signals`) and parameter declarations
(:mod:`machina.params`) both require an SI unit string, and they must agree.
icarus-dynamics deliberately restated the rule in two files because its
parameter contract arrived as a Nix flake input and importing it from the
physics model would have pointed a dependency the wrong way. machina has no
such constraint, so the rule is defined once here and imported by both.

Units are plain strings: ``"m"``, ``"m/s^2"``, ``"kg*m^2"``, ``"1"`` for
dimensionless. Non-SI units are rejected outright rather than converted --
the conversion belongs at the boundary that produced the number, and display
units (degrees, RPM, feet) belong in plot labels.
"""

import re

UNIT_RE = re.compile(r"^[A-Za-z0-9_^/*.\-]+$")

NON_SI_UNITS = frozenset({
    "deg", "degree", "degrees", "rpm", "psi", "bar", "ft", "feet", "foot",
    "knot", "knots", "kt", "mph", "C", "F", "hr", "hour", "min", "day",
    "nmi", "mile", "miles", "lb", "lbf", "lbm", "slug", "in", "inch",
})


def is_well_formed(unit: str) -> bool:
    """True when ``unit`` is a non-empty string of the allowed characters."""
    return bool(unit) and bool(UNIT_RE.match(unit))


def is_si(unit: str) -> bool:
    """True when ``unit`` is well formed and not on the non-SI list."""
    return is_well_formed(unit) and unit not in NON_SI_UNITS
