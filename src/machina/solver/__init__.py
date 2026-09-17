"""Layer 1: solver backend (CasADi ``nlpsol`` wrapper) and result packaging."""

from machina.solver.backend import SolverBackend
from machina.solver.result import SolutionResult

__all__ = ["SolverBackend", "SolutionResult"]
