"""
Read-only records describing what has been registered with a ``SolverBackend``.

These are the public introspection surface of the solver backend. The viz tool, the
compiler and (later) the report and study packages read these instead of the
backend's internal state. Records are snapshots: they are rebuilt on every
accessor call and reflect the *current* bounds, guesses and parameter values
(after ``set_bounds``, ``fix``, ``set_parameter`` and friends).

``VariableRecord``, ``ParameterRecord``, ``ConstraintRecord`` and
``CostRecord`` hold CasADi MX objects and therefore cannot be pickled.
``LayoutEntry`` is numeric-only; ``SolutionResult`` carries tuples of these so
that a result stays self-describing and picklable.

Array conventions match ``SolutionResult.x_opt``: vector quantities
(``shape[1] == 1``) are 1-D arrays, matrix quantities are ``(rows, cols)``
arrays.
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class LayoutEntry:
    """Position of one named block in a flat vector (``x``, ``p`` or ``g``)."""

    name: str
    start: int
    stop: int
    shape: tuple[int, int]

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def numel(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, eq=False)
class VariableRecord:
    name: str
    shape: tuple[int, int]
    slice: slice            # position in the flat decision vector
    lb: np.ndarray          # current lower bound, physical units
    ub: np.ndarray          # current upper bound, physical units
    x0: np.ndarray          # current initial guess, physical units
    discrete: np.ndarray    # bool per element
    scale: np.ndarray       # decision vector holds x / scale
    symbol: ca.MX           # the handle returned by add_variable()
    unit: str = ""
    doc: str = ""
    provenance: str | None = None
    source: str | None = None
    component: str | None = None
    frame: str | None = None

    @property
    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def is_discrete(self) -> bool:
        return bool(np.any(self.discrete))

    @property
    def is_fixed(self) -> bool:
        return bool(np.all(self.lb == self.ub))


@dataclass(frozen=True, eq=False)
class ParameterRecord:
    name: str
    shape: tuple[int, int]
    slice: slice                # position in the flat parameter vector
    value: np.ndarray | None    # stored value, None until one is supplied
    symbol: ca.MX
    unit: str = ""
    doc: str = ""
    provenance: str | None = None
    source: str | None = None
    component: str | None = None
    frame: str | None = None

    @property
    def numel(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True, eq=False)
class ConstraintRecord:
    name: str
    index: int              # registration index
    slice: slice            # rows in the flat constraint vector
    lb: np.ndarray
    ub: np.ndarray
    scale: np.ndarray       # solver sees expr / scale
    expr: ca.MX             # physical (unscaled) expression
    auto_named: bool = False
    doc: str = ""
    component: str | None = None

    @property
    def n_rows(self) -> int:
        return self.slice.stop - self.slice.start

    @property
    def is_equality(self) -> bool:
        return bool(np.all(self.lb == self.ub))


@dataclass(frozen=True, eq=False)
class CostRecord:
    name: str
    index: int
    expr: ca.MX             # unweighted term
    weight: object = 1.0    # float, or an MX parameter expression
    auto_named: bool = False
    doc: str = ""
    component: str | None = None

    @property
    def weighted(self) -> ca.MX:
        """The term as it enters the objective."""
        if isinstance(self.weight, (int, float)) and self.weight == 1.0:
            return self.expr
        return self.weight * self.expr
