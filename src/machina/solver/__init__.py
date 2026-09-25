"""Solver backend (CasADi ``nlpsol`` wrapper), records and result packaging."""

from machina.solver.backend import SolverBackend
from machina.solver.errors import SolverError
from machina.solver.records import (
    ConstraintRecord,
    CostRecord,
    LayoutEntry,
    ParameterRecord,
    VariableRecord,
)
from machina.solver.result import ConstraintValue, SolutionResult

__all__ = [
    "SolverBackend",
    "SolutionResult",
    "SolverError",
    "ConstraintValue",
    "VariableRecord",
    "ParameterRecord",
    "ConstraintRecord",
    "CostRecord",
    "LayoutEntry",
]
