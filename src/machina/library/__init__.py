"""
machina.library -- generic, domain-free building blocks.

The numeric guards (:mod:`machina.library.numerics`), the factory registry
(:mod:`machina.library.registry`) and the generic factories: ``cost.quadratic``,
``cost.rosenbrock``, ``cost.least_squares`` (:mod:`~machina.library.cost`),
``constraint.linear`` (:mod:`~machina.library.constraint`), ``util.sum`` and
``util.rotate_x/y/z`` (:mod:`~machina.library.util`).

Importing this package registers the generic factories, and ``import machina``
imports it. Domain factories live in the packs and register when their pack is
imported.
"""

from machina.library import constraint, cost, registry, util  # noqa: F401
from machina.library.numerics import EPS, TINY, safe_divide, safe_norm, safe_sqrt

__all__ = ["TINY", "EPS", "safe_sqrt", "safe_norm", "safe_divide"]
