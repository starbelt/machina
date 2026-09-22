"""
The unit rule, in one place.

Signals (:mod:`machina.model.signals`), quantities
(:class:`machina.model.Quantity`) and parameter declarations
(:mod:`machina.params`) all require an SI unit string, and they must agree.
icarus-dynamics deliberately restated the rule in two files because its
parameter contract arrived as a Nix flake input and importing it from the
physics model would have pointed a dependency the wrong way. machina has no
such constraint, so the rule is defined once here and imported by all three.

Units are plain strings: ``"m"``, ``"m/s^2"``, ``"kg*m^2"``, ``"1"`` for
dimensionless. A unit is split into symbols on ``* / ^ . -`` and numeric
exponents are dropped; **any** symbol on the non-SI list refuses the whole
unit, so ``"deg/s"`` and ``"ft/s"`` fail as surely as ``"deg"`` does. The
conversion belongs at the boundary that produced the number, and display
units (degrees, RPM, feet) belong in plot labels.

SI prefixes are SI: ``"km"``, ``"ms"`` and ``"kW"`` are accepted. Whether a
pack should work in metres rather than kilometres is a numerics decision for
that pack (the April orbital code is in km), not something this module can
settle by refusing the prefix.

``"C"`` and ``"F"`` are refused as Celsius and Fahrenheit, as icarus refuses
them, even though coulomb and farad share the symbols: write ``"A*s"`` and
``"A*s/V"`` for those, which nothing in either project has needed yet.
"""

import re

__all__ = ["UNIT_RE", "NON_SI_UNITS", "is_well_formed", "is_si", "symbols"]

UNIT_RE = re.compile(r"^[A-Za-z0-9_^/*.\-]+$")

NON_SI_UNITS = frozenset({
    # angle
    "deg", "degree", "degrees", "arcmin", "arcsec", "grad",
    # rotation rate
    "rpm", "rps",
    # temperature scales (K is SI)
    "C", "F", "degC", "degF", "degR",
    # length
    "ft", "feet", "foot", "in", "inch", "inches", "yd", "mi", "mile", "miles", "nmi", "NM",
    # speed
    "knot", "knots", "kt", "kts", "mph", "kph", "fps",
    # time beyond the second
    "min", "hr", "h", "hour", "hours", "day", "days", "yr", "year", "years",
    # mass and force
    "lb", "lbs", "lbf", "lbm", "slug", "oz", "ton", "tons",
    # pressure
    "psi", "psf", "bar", "mbar", "atm", "torr", "inHg", "mmHg",
    # energy and power
    "hp", "cal", "kcal", "BTU", "Btu",
})

_SPLIT = re.compile(r"[*/^.\-]")


def is_well_formed(unit: str) -> bool:
    """True when ``unit`` is a non-empty string of the allowed characters."""
    return isinstance(unit, str) and bool(unit) and bool(UNIT_RE.match(unit))


def symbols(unit: str) -> list:
    """The unit symbols in ``unit``, numeric exponents dropped: ``"kg*m^2/s"`` -> kg, m, s."""
    return [token for token in _SPLIT.split(unit) if token and not token.isdigit()]


def is_si(unit: str) -> bool:
    """True when ``unit`` is well formed and none of its symbols is non-SI."""
    return is_well_formed(unit) and not any(token in NON_SI_UNITS for token in symbols(unit))
