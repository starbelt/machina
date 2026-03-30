from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class SolutionResult:
    """
    Structured output from a single ``SolverBackend.solve()`` call.

    All numeric fields are plain numpy arrays (converted from CasADi DM via
    ``.full().flatten()``), so callers never need to interact with CasADi types
    after the solve returns.

    Named variable values are the primary access point. Use dict-style indexing
    (``result['sma']``) as a shorthand for ``result.x_opt['sma']``.

    Lagrange multipliers
    --------------------
    ``lam_x`` and ``lam_g`` are the dual variables for variable bounds and
    constraints respectively. A nonzero value at index i indicates that bound
    or constraint is active at the solution. These are the natural warm-start
    seeds for a subsequent solve on a nearby problem.

    ``lam_p`` contains the sensitivity of the optimal objective with respect to
    the parameter vector (d f* / d p). It requires ``{'calc_lam_p': True}`` in
    solver options to be populated with nonzero values; it is set to ``None``
    when no parameters were registered.

    Field ordering note
    -------------------
    ``lam_p`` is declared last because it is the only field with a default value
    (``None``). Python dataclasses require all default fields to follow all
    non-default fields.
    """

    # Whether the solver converged. True when IPOPT return_status is
    # 'Solve_Succeeded' or 'Solved_To_Acceptable_Level'; False for all others
    # (infeasible, max iterations, etc.). Check this before using x_opt.
    success: bool

    # Optimal values for each registered decision variable, keyed by the name
    # supplied to add_variable(). Each value is a 1-D numpy array of length n.
    x_opt: Dict[str, np.ndarray]

    # Optimal objective value, as a Python float.
    f_opt: float

    # Constraint vector values at the optimum. Length equals the total number
    # of constraint rows across all add_constraint() calls. Values near zero
    # indicate active equality constraints; check against lbg/ubg for inequality
    # activity (not stored here — see raw_sol if needed).
    g_opt: np.ndarray

    # Lagrange multipliers for variable bounds (lbx/ubx). Length equals the
    # total number of decision variable elements across all add_variable() calls.
    lam_x: np.ndarray

    # Lagrange multipliers for constraints (lbg/ubg). Length equals len(g_opt).
    lam_g: np.ndarray

    # Raw solver statistics dict from CasADi (solver.stats()). Always contains
    # 'return_status', 'iter_count', and 't_wall_total' for IPOPT.
    stats: dict

    # Raw CasADi solution dict (keys: 'x', 'f', 'g', 'lam_x', 'lam_g', 'lam_p').
    # Retained for advanced use: warm-start seeding, custom post-processing,
    # or passing to SolverBackend.extract() directly.
    raw_sol: dict

    # Parameter sensitivities (d f* / d p). None when no parameters were
    # registered. Requires {'calc_lam_p': True} in solver options for nonzero
    # values. Must be last field because it carries a default value.
    lam_p: Optional[np.ndarray] = None

    def __getitem__(self, name: str) -> np.ndarray:
        """Shorthand for ``result.x_opt[name]``."""
        return self.x_opt[name]
