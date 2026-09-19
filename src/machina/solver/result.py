from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class ConstraintValue:
    """One named constraint at the solution, in physical units."""

    name: str
    value: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    # CasADi convention: positive when the upper bound is active, negative
    # when the lower bound is active. For a budget ``g <= limit`` the
    # multiplier is the marginal change of the optimal objective per unit of
    # extra limit, with the sign flipped.
    multiplier: np.ndarray
    active: np.ndarray      # bool per row: value sits on a finite bound

    @property
    def violation(self) -> np.ndarray:
        return np.maximum(0.0, np.maximum(self.lb - self.value, self.value - self.ub))


@dataclass
class SolutionResult:
    """
    Structured output from a single ``SolverBackend.solve()`` call.

    All numeric fields are plain numpy arrays in **physical units**: when the
    backend scales variables or constraints, values and multipliers are
    converted back before they land here. Callers never need to interact with
    CasADi types after the solve returns, and a result can be pickled.

    Named variable values are the primary access point. Use dict-style indexing
    (``result['sma']``) as a shorthand for ``result.x_opt['sma']``.

    Lagrange multipliers
    --------------------
    ``lam_x`` and ``lam_g`` are the dual variables for variable bounds and
    constraints respectively. A nonzero value at index i indicates that bound
    or constraint is active at the solution. These are the warm-start seeds
    for a subsequent solve on a nearby problem (``solve(warm_start=result)``).
    Use ``constraint(name)`` and ``bound_multiplier(name)`` for named access.
    Mixed-integer plugins (bonmin) return NaN multipliers: duals are not defined
    for an integer solve. ``g_opt`` is still populated.

    ``lam_p`` is CasADi's multiplier on the parameter vector. With CasADi's
    Lagrangian convention it equals **minus** the sensitivity of the optimal
    objective: ``d f* / d p = -lam_p`` (checked against finite differences in
    the test suite). CasADi computes it by default (``calc_lam_p``); it is
    ``None`` when no parameters were registered. Use ``sensitivity(name)``,
    which applies the sign, for named access.

    Field ordering note
    -------------------
    Every field from ``lam_p`` onward carries a default, so the positional
    construction ``SolutionResult(success, x_opt, f_opt, g_opt, lam_x, lam_g,
    stats, raw_sol)`` keeps working.
    """

    # Whether the solver converged: ``bool(stats['success'])``, which every
    # nlpsol plugin provides. For IPOPT this is True for 'Solve_Succeeded' and
    # 'Solved_To_Acceptable_Level'. Check this before using x_opt.
    success: bool

    # Optimal values for each registered decision variable, keyed by the name
    # supplied to add_variable(). Vector variables are 1-D arrays of length n;
    # matrix variables are (rows, cols) arrays.
    x_opt: dict[str, np.ndarray]

    # Optimal objective value, as a Python float.
    f_opt: float

    # Constraint vector values at the optimum. Length equals the total number
    # of constraint rows across all add_constraint() calls.
    g_opt: np.ndarray

    # Lagrange multipliers for variable bounds (lbx/ubx). Length equals the
    # total number of decision variable elements across all add_variable() calls.
    lam_x: np.ndarray

    # Lagrange multipliers for constraints (lbg/ubg). Length equals len(g_opt).
    lam_g: np.ndarray

    # Raw solver statistics dict from CasADi (solver.stats()). Contents depend
    # on the plugin; 'return_status', 'success' and 'iter_count' are common.
    stats: dict

    # CasADi-style solution dict (keys: 'x', 'f', 'g', 'lam_x', 'lam_g',
    # 'lam_p') in physical units, usable with SolverBackend.extract().
    raw_sol: dict

    # CasADi's parameter multipliers: lam_p = -(d f* / d p). None when no
    # parameters were registered. See sensitivity().
    lam_p: np.ndarray | None = None

    # The plugin's own return status, as a string (fatrop reports an int).
    status: str = ""

    # CasADi's plugin-independent status: 'SOLVER_RET_SUCCESS',
    # 'SOLVER_RET_LIMITED' (iteration or time limit) or 'SOLVER_RET_UNKNOWN'.
    unified_status: str = ""

    # nlpsol plugin that produced this result.
    plugin: str = ""

    # Parameter values used for this solve, by name.
    p_opt: dict[str, np.ndarray] = field(default_factory=dict)

    # Names of discrete variables that were solved as continuous because the
    # backend was built with discrete_mode='relax'. Empty otherwise.
    relaxed_discrete: tuple[str, ...] = ()

    iterations: int | None = None

    # Wall-clock seconds spent inside the solver call.
    t_wall: float | None = None

    # True when this solve ran on the warm-start solver (duals were supplied).
    warm_started: bool = False

    # Value of each cost term as it enters the objective (weight applied), by
    # name. Sums to f_opt.
    cost_terms: dict[str, float] = field(default_factory=dict)

    # Constraint bounds used for this solve, flat, parallel to g_opt.
    lbg: np.ndarray | None = None
    ubg: np.ndarray | None = None

    # Flat-vector layouts (tuples of LayoutEntry) for x, p and g.
    x_layout: tuple = ()
    p_layout: tuple = ()
    g_layout: tuple = ()

    def __getitem__(self, name: str) -> np.ndarray:
        """Shorthand for ``result.x_opt[name]``."""
        return self.x_opt[name]

    # ------------------------------------------------------------------
    # Named access
    # ------------------------------------------------------------------

    @staticmethod
    def _find(layout: tuple, name: str, kind: str):
        for entry in layout:
            if entry.name == name:
                return entry
        raise KeyError(
            f"No {kind} named '{name}' in this result. "
            f"Available: {[e.name for e in layout]}"
        )

    @staticmethod
    def _natural(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        return flat.reshape(shape, order='F') if shape[1] > 1 else flat

    def constraint(self, name: str, tol: float = 1e-6) -> ConstraintValue:
        """
        Value, bounds, multiplier and activity of one named constraint.

        A row is active when its value lies within ``tol * max(1, |bound|)``
        of a finite bound.
        """
        entry = self._find(self.g_layout, name, 'constraint')
        sl = entry.slice
        value, lb, ub = self.g_opt[sl], self.lbg[sl], self.ubg[sl]
        with np.errstate(invalid='ignore'):
            at_lb = np.isfinite(lb) & (value <= lb + tol * np.maximum(1.0, np.abs(lb)))
            at_ub = np.isfinite(ub) & (value >= ub - tol * np.maximum(1.0, np.abs(ub)))
        return ConstraintValue(name=name, value=value, lb=lb, ub=ub,
                               multiplier=self.lam_g[sl], active=at_lb | at_ub)

    def bound_multiplier(self, name: str) -> np.ndarray:
        """Multipliers on a variable's bounds (slice of ``lam_x``), in its shape."""
        entry = self._find(self.x_layout, name, 'variable')
        return self._natural(self.lam_x[entry.slice], entry.shape)

    def sensitivity(self, name: str) -> np.ndarray:
        """
        d f* / d p for one named parameter, in the parameter's shape.

        This is ``-lam_p``: the first-order change of the optimal objective per
        unit change of the parameter, valid while the active set holds.
        """
        entry = self._find(self.p_layout, name, 'parameter')
        return -self._natural(self.lam_p[entry.slice], entry.shape)

    @property
    def max_constraint_violation(self) -> float:
        """Largest violation of ``lbg <= g <= ubg``; 0.0 without constraints."""
        if self.lbg is None or self.g_opt.size == 0:
            return 0.0
        return float(np.max(np.maximum(0.0, np.maximum(self.lbg - self.g_opt,
                                                       self.g_opt - self.ubg))))
