"""
The unit rule, in one place.

Signals (:mod:`machina.model.signals`), quantities
(:class:`machina.model.Quantity`) and parameter declarations
(:mod:`machina.params`) all require an SI unit string, and they must agree.
icarus-dynamics deliberately restated the rule in two files because its
parameter contract arrived as a Nix flake input and importing it from the
physics model would have pointed a dependency the wrong way. machina has no
such constraint, so the rule is defined once here and imported by all three.

Units are plain strings: ``"m"``, ``"m/s^2"``, ``"kg*m^2"``, ``"W/(m^2*K)"``,
``"1"`` for dimensionless. A unit is split into symbols on ``* / ^ . - ( )``
and numeric exponents are dropped; **every** symbol must be an SI base or
derived unit, optionally with an SI prefix. It is an allow-list because a
deny-list cannot keep up with spellings -- ``RPM``, ``hrs``, ``kWh``, ``AU``
would all slip through one. The conversion belongs at the boundary that
produced the number, and display units (degrees, RPM, feet) belong in plot
labels.

SI prefixes are SI: ``"km"``, ``"ms"`` and ``"kW"`` are accepted. Whether a
pack should work in metres rather than kilometres is a numerics decision for
that pack (the April orbital code is in km), not something this module can
settle by refusing the prefix.

Coulomb and farad are left out on purpose: icarus refuses ``"C"`` and ``"F"``
as Celsius and Fahrenheit, and machina keeps the same answer. Write ``"A*s"``
and ``"A*s/V"`` if either is ever needed.
"""

import re

__all__ = ["UNIT_RE", "NON_SI_UNITS", "SI_SYMBOLS", "SI_PREFIXES", "is_well_formed", "is_si",
           "symbols"]

UNIT_RE = re.compile(r"^[A-Za-z0-9_^/*.()\-]+$")

SI_SYMBOLS = frozenset({
    # base
    "m", "g", "s", "A", "K", "mol", "cd",
    # derived, with special names
    "rad", "sr", "Hz", "N", "Pa", "J", "W", "V", "ohm", "S", "Wb", "T", "H", "lm", "lx",
    "Bq", "Gy", "Sv", "kat",
})

SI_PREFIXES = ("da", "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "d", "c", "m", "u", "n",
               "p", "f", "a", "z", "y")

# Not used by is_si (the allow-list decides); kept as the list of common mistakes that
# params contracts and error messages refer to, and for icarus compatibility.
NON_SI_UNITS = frozenset({
    "deg", "degree", "degrees", "rpm", "RPM", "psi", "bar", "ft", "feet", "foot", "in", "inch",
    "knot", "knots", "kt", "mph", "C", "F", "degC", "degF", "hr", "h", "hour", "min", "day",
    "nmi", "mile", "miles", "lb", "lbf", "lbm", "slug", "Wh", "kWh", "Ah", "mAh", "AU", "gal",
})

_SPLIT = re.compile(r"[*/^.()\-]")


def is_well_formed(unit: str) -> bool:
    """True when ``unit`` is a non-empty string of the allowed characters."""
    return isinstance(unit, str) and bool(unit) and bool(UNIT_RE.match(unit))


def symbols(unit: str) -> list:
    """The unit symbols in ``unit``, numeric exponents dropped: ``"kg*m^2/s"`` -> kg, m, s."""
    return [token for token in _SPLIT.split(unit) if token and not token.isdigit()]


def _is_si_symbol(token: str) -> bool:
    if token in SI_SYMBOLS:
        return True
    return any(token.startswith(prefix) and token[len(prefix):] in SI_SYMBOLS
               for prefix in SI_PREFIXES)


def is_si(unit: str) -> bool:
    """True when ``unit`` is well formed and every symbol in it is SI (prefixes allowed)."""
    return is_well_formed(unit) and all(_is_si_symbol(token) for token in symbols(unit))
