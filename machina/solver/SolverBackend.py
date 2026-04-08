import warnings
import casadi as ca
import numpy as np
from .SolutionResult import SolutionResult


class SolverBackend:
    """
    Thin, domain-agnostic wrapper around CasADi's ``nlpsol`` interface.

    This class owns all CasADi symbolic state and manages the mapping between
    named variables and their positions in the flat global decision vector.
    It knows nothing about orbital mechanics, mission design, or problem
    structure — that knowledge lives in the blocks and agents above it.

    NLP form
    --------
    The solver finds x* that solves:

        minimize    f(x, p)
        subject to  lbg <= g(x, p) <= ubg
                    lbx <= x       <= ubx

    where x is the decision vector (assembled from all registered variables),
    p is the fixed parameter vector (assembled from all registered parameters),
    f is the scalar objective (sum of all cost terms), and g is the constraint
    vector (stack of all registered constraint expressions).

    Lifecycle
    ---------
    1. Register variables, parameters, constraints, and cost terms.
    2. Call ``build()`` once to assemble the NLP and create the solver.
    3. Call ``solve()`` one or more times (with different ``p_val`` to re-solve
       parametrically without rebuilding).

    Design note — ``nlpsol`` vs ``Opti``
    -------------------------------------
    CasADi's ``Opti`` interface is convenient but owns variable indexing
    internally, making it hard to inspect or control decision vector layout.
    ``nlpsol`` gives this framework explicit control over the global vector
    structure, which is essential for interleaved state/control ordering in
    trajectory problems and for clean result extraction by name.
    """

    def __init__(self, solver: str = 'ipopt', solver_opts: dict = None):
        """
        Initialize the solver backend.

        Args:
            solver:       CasADi solver plugin name. Default is ``'ipopt'``.
                          Other options include ``'sqpmethod'``, ``'fatrop'``.
            solver_opts:  Solver options passed through to CasADi as-is.
                          Accepts flat dot-notation (``{'ipopt.tol': 1e-8}``)
                          or nested dicts (``{'ipopt': {'tol': 1e-8}}``).
                          These are merged on top of the hardcoded defaults
                          set in ``build()``, so only specify what you want
                          to override.
        """
        self.solver_name: str = solver
        self.solver_opts: dict = solver_opts if solver_opts is not None else {}

        # Decision vector accumulators. Each add_variable() call appends one
        # MX symbol to _w and extends the three bound/init lists by n elements.
        # They stay as lists until build() concatenates them with ca.vertcat.
        self._w:   list[ca.MX] = []
        self._w0:  list[float] = []
        self._lbw: list[float] = []
        self._ubw: list[float] = []

        # Constraint accumulators. Parallel structure to the variable lists.
        # IPOPT treats a constraint as equality when lbg[i] == ubg[i].
        self._g:    list[ca.MX] = []
        self._lbg:  list[float] = []
        self._ubg:  list[float] = []

        # Parameter accumulators. Parameters appear in f and g but are not
        # optimized; their values are supplied at solve() time.
        self._p: list[ca.MX] = []

        # Scalar objective. Initialized as a CasADi zero (not Python 0) so
        # that the first add_cost() call produces an MX expression, not a
        # mixed Python-int/MX sum that CasADi may handle inconsistently.
        self._J: ca.MX = ca.MX.zeros(1, 1)

        # Running byte-offsets into the flat decision and parameter vectors.
        # Used to record slice indices in _var_map and _param_map as variables
        # and parameters are registered.
        self._offset:   int = 0
        self._p_offset: int = 0

        # Index maps for result extraction. Maps name → (start, end, shape) in
        # the flat vector. shape is (n, 1) for vectors, (rows, cols) for
        # matrices. Used to reshape extracted slices back to their original form.
        self._var_map:   dict[str, tuple[int, int, tuple[int, int]]] = {}
        self._param_map: dict[str, tuple[int, int, tuple[int, int]]] = {}

        # Shared namespace for collision detection across variables AND
        # parameters. Constraints and cost terms intentionally do not go here —
        # they are identified by position in their respective lists, not by name.
        self._names: set[str] = set()

        # Metadata lists for reporting and debugging. These do not affect the
        # NLP formulation; they exist so callers can inspect what was registered.
        self._cost_terms:       list[tuple[str, ca.MX]] = []
        self._constraint_names: list[tuple[str, int]]   = []

        self._solver: ca.Function = None
        self._built:  bool = False
        self._solved: bool = False  # True after the first solve() call completes

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def add_variable(self, name: str, n: int | tuple[int, int],
                     lb=-np.inf, ub=np.inf,
                     initial_guess=0.0) -> ca.MX:
        """
        Create a decision variable and register it in the global decision vector.

        The returned MX symbol is the handle used to build constraint and cost
        expressions. Internally, the variable occupies a contiguous slice
        [offset, offset+numel) in the flat vector; this slice is recorded in
        ``_var_map`` so results can be extracted by name after solving.

        For matrix variables, the decision vector stores elements column-major
        (via ``ca.vec``). Extraction automatically reshapes the result back to
        ``(rows, cols)`` using Fortran order, matching CasADi's column-major
        convention.

        Args:
            name:          Unique name for this variable. Must not already exist
                           in the variable/parameter namespace.
            n:             Size of the variable. Pass an ``int`` for an n×1
                           column vector, or a ``(rows, cols)`` tuple for a
                           matrix. The returned MX symbol has the requested
                           shape; bounds and initial guess are broadcast to
                           ``rows * cols`` elements.
            lb:            Lower bound. Scalar is broadcast to all elements.
                           Can also be a list or array of length ``rows*cols``
                           (column-major order for matrices).
            ub:            Upper bound. Same broadcast rules as lb.
            initial_guess: Starting point for the solver. Scalar or array.
                           A good initial guess is important for convergence on
                           nonlinear problems; zero is used when nothing better
                           is known.

        Returns:
            MX symbolic variable with the requested shape — use this to form
            objective and constraint expressions passed to ``add_cost`` and
            ``add_constraint``.

        Raises:
            ValueError: If ``name`` already exists in the variable/parameter
                        namespace.
        """
        if name in self._names:
            raise ValueError(f"Variable name '{name}' already exists.")
        self._names.add(name)

        if isinstance(n, tuple):
            rows, cols = n
            numel = rows * cols
            var = ca.MX.sym(name, rows, cols)
            # ca.vec flattens column-major so the matrix occupies a contiguous
            # column vector block in the global decision vector, as nlpsol requires.
            self._w.append(ca.vec(var))
        else:
            rows, cols = n, 1
            numel = n
            var = ca.MX.sym(name, n)
            self._w.append(var)

        # Scalars are broadcast; arrays are flattened to handle matrix-shaped
        # inputs consistently. Both paths produce a flat list of length numel.
        lb = np.full(numel, lb).tolist() if np.isscalar(lb) else list(np.asarray(lb).flatten())
        ub = np.full(numel, ub).tolist() if np.isscalar(ub) else list(np.asarray(ub).flatten())
        initial_guess = np.full(numel, initial_guess).tolist() if np.isscalar(initial_guess) else list(np.asarray(initial_guess).flatten())

        self._w0.extend(initial_guess)
        self._lbw.extend(lb)
        self._ubw.extend(ub)

        self._var_map[name] = (self._offset, self._offset + numel, (rows, cols))
        self._offset += numel

        return var

    def add_parameter(self, name: str, n: int | tuple[int, int]) -> ca.MX:
        """
        Create a fixed parameter and register it in the global parameter vector.

        Parameters appear in the objective and constraints but are not optimized.
        Their numeric values are supplied at ``solve()`` time via ``p_val``,
        enabling the same built solver to be re-solved with different problem
        data without rebuilding — critical for parametric sweeps or MPC-style
        re-planning.

        Args:
            name: Unique name. Must not already exist in the variable/parameter
                  namespace (parameters and variables share one namespace to
                  prevent ambiguous expression graphs).
            n:    Size of the parameter. Pass an ``int`` for an n×1 column
                  vector, or a ``(rows, cols)`` tuple for a matrix. The
                  returned MX symbol has the requested shape; the corresponding
                  ``p_val`` slice must supply ``rows*cols`` values in
                  column-major order.

        Returns:
            MX symbolic parameter — use this in objective and constraint
            expressions exactly like a variable symbol.

        Raises:
            ValueError: If ``name`` already exists in the variable/parameter
                        namespace.
        """
        if name in self._names:
            raise ValueError(f"Parameter name '{name}' already exists.")
        self._names.add(name)

        if isinstance(n, tuple):
            rows, cols = n
            numel = rows * cols
            param = ca.MX.sym(name, rows, cols)
            self._p.append(ca.vec(param))
        else:
            rows, cols = n, 1
            numel = n
            param = ca.MX.sym(name, n)
            self._p.append(param)

        self._param_map[name] = (self._p_offset, self._p_offset + numel, (rows, cols))
        self._p_offset += numel

        return param

    def add_constraint(self, expr: ca.MX,
                       lb=-np.inf, ub=np.inf,
                       name: str = None):
        """
        Add a constraint expression to the problem.

        The constraint is stored as-is (an MX expression). Bounds are stored
        as flat numeric lists. CasADi/IPOPT interprets:
          - lb == ub    → equality constraint (IPOPT detects this automatically)
          - finite lb, infinite ub  → one-sided lower bound (g(x) >= lb)
          - infinite lb, finite ub  → one-sided upper bound (g(x) <= ub)
          - both finite → double-sided inequality

        Args:
            expr: MX expression. Can be vector-valued; its number of rows
                  (``expr.shape[0]``) determines how many constraint rows are
                  added. The expression should be written so that the desired
                  bound applies to the expression value directly:
                  e.g., to enforce x >= 2, pass ``expr=x`` with ``lb=2``.
            lb:   Lower bound. Scalar is broadcast to all rows.
            ub:   Upper bound. Scalar is broadcast to all rows.
            name: Optional label for debugging and reporting. Does NOT enter
                  the variable/parameter namespace — constraint names need not
                  be globally unique.
        """
        n = expr.shape[0]
        lb = np.full(n, lb).tolist() if np.isscalar(lb) else list(lb)
        ub = np.full(n, ub).tolist() if np.isscalar(ub) else list(ub)

        self._g.append(expr)
        self._lbg.extend(lb)
        self._ubg.extend(ub)

        # name and n stored for post-solve reporting; not used by the solver.
        self._constraint_names.append((name, n))

    def add_equality(self, expr: ca.MX, name: str = None):
        """
        Convenience wrapper: add an equality constraint ``expr == 0``.

        Equivalent to ``add_constraint(expr, lb=0, ub=0, name=name)``.
        Write the expression so it equals zero at the desired solution:
        e.g., to enforce x + y == 1, pass ``expr = x + y - 1``.
        """
        self.add_constraint(expr, lb=0.0, ub=0.0, name=name)

    def add_cost(self, expr: ca.MX, name: str = None):
        """
        Add a scalar cost term to the objective.

        All terms are accumulated into a single objective ``f = sum(terms)``.
        CasADi builds this as one symbolic expression, so the solver sees the
        full combined objective and can compute exact gradients across all terms.

        Args:
            expr: Scalar MX expression (shape must be (1, 1)).
            name: Optional label for objective breakdown reporting.

        Raises:
            ValueError: If ``expr`` is not shape (1, 1). Vector-valued
                        expressions must be reduced to a scalar first, e.g.
                        via ``ca.sumsqr(v)`` or ``v.T @ v``.
        """
        if expr.shape != (1, 1):
            raise ValueError(
                f"Cost term must be scalar (1, 1), got {expr.shape}."
                + (f" Name: '{name}'." if name else "")
            )
        self._J += expr
        self._cost_terms.append((name, expr))

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self, opts: dict = None) -> ca.Function:
        """
        Assemble the NLP and create the ``nlpsol`` solver object.

        This method concatenates all accumulated variable and constraint lists
        into the flat vectors CasADi expects, assembles the NLP dict, merges
        solver options, and calls ``ca.nlpsol``. After ``build()`` returns,
        the backend is locked — no further variables, parameters, or constraints
        can be registered.

        Option precedence (later layers override earlier ones):
            1. Hardcoded defaults (tol, max_iter, linear_solver, etc.)
            2. ``solver_opts`` supplied at ``__init__`` time
            3. ``opts`` supplied to this ``build()`` call

        Args:
            opts: Additional or override solver options for this build only.
                  Same format as ``solver_opts`` in ``__init__``.

        Returns:
            The ``ca.nlpsol`` Function object (also stored as ``_solver``).

        Raises:
            RuntimeError: If no variables have been registered, or if
                          ``build()`` has already been called on this instance.

        Note on ``expand``:
            Setting ``expand=True`` (via opts) converts the MX graph to SX at
            build time for faster evaluation. It is incompatible with
            ``ca.Callback`` objects. Omitted from the defaults so that
            iteration callbacks remain usable.
        """
        if self._built:
            raise RuntimeError("Solver has already been built.")
        if not self._w:
            raise RuntimeError(
                "No decision variables registered. Call add_variable() before build()."
            )
        if not self._cost_terms:
            warnings.warn(
                "No cost terms registered. The objective will be zero (feasibility problem).",
                UserWarning,
                stacklevel=2,
            )

        w = ca.vertcat(*self._w)
        # ca.MX(0, 1) is the correct empty column vector for a problem with no
        # constraints; ca.MX.zeros(0) would be a scalar and confuse IPOPT.
        g = ca.vertcat(*self._g) if self._g else ca.MX(0, 1)

        nlp = {'x': w, 'f': self._J, 'g': g}
        # Only add 'p' when parameters are registered. If 'p' were always
        # present (even as an empty vector), every solver() call would require
        # a p= keyword argument, breaking the no-parameter usage.
        if self._p:
            nlp['p'] = ca.vertcat(*self._p)

        defaults = {
            'print_time':           True,
            'ipopt.tol':            1e-8,
            'ipopt.max_iter':       2000,
            'ipopt.linear_solver':  'mumps',
            'ipopt.mu_strategy':    'adaptive',
            'ipopt.print_level':    5,
        }
        merged_opts = {**defaults, **self.solver_opts}
        if opts is not None:
            merged_opts.update(opts)

        self._solver = ca.nlpsol('solver', self.solver_name, nlp, merged_opts)
        self._built = True
        return self._solver

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------

    def solve(self, p_val=None) -> SolutionResult:
        """
        Call the solver and return a structured result.

        Uses the initial guesses and bounds registered during the build phase.
        For parametric problems, ``p_val`` provides the numeric parameter values
        in the same order they were registered via ``add_parameter()``.

        Args:
            p_val: Numeric values for all registered parameters, as a flat list
                   or numpy array, in registration order. Must be provided if
                   any parameters were registered; must be ``None`` (or omitted)
                   if none were. Passing wrong-length ``p_val`` raises early
                   rather than letting CasADi produce a cryptic dimension error.

        Returns:
            ``SolutionResult`` with all solution fields populated and all
            variable values accessible by name via ``result['name']``.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
            ValueError:   If parameters are registered but ``p_val`` is None,
                          or if ``p_val`` has the wrong number of elements.
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")

        p_arr = None
        if self._p:
            if p_val is None:
                raise ValueError(
                    "Parameters were registered but p_val is None. Provide parameter values."
                )
            p_arr = np.asarray(p_val).flatten()
            if len(p_arr) != self._p_offset:
                raise ValueError(
                    f"p_val has length {len(p_arr)}, but {self._p_offset} "
                    f"parameter value(s) are required."
                )

        kwargs = dict(
            x0=self._w0,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg,
            ubg=self._ubg,
        )
        if self._p:
            # Pass the normalized flat array, not the original p_val. This
            # ensures CasADi always receives a consistent 1-D input regardless
            # of whether the caller passed a list, 2-D array, or column vector.
            kwargs['p'] = p_arr

        sol = self._solver(**kwargs)
        self._solved = True
        solver_stats = self._solver.stats()

        return_status = solver_stats.get('return_status', '')
        success = return_status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')

        # Slice each named variable out of the flat solution vector using the
        # index ranges recorded at registration time. Matrix variables are
        # reshaped back to (rows, cols) using Fortran (column-major) order,
        # matching the ca.vec() column-major flattening used at registration.
        x_opt = {}
        for name, (i, j, shape) in self._var_map.items():
            vals = sol['x'][i:j].full().flatten()
            x_opt[name] = vals.reshape(shape, order='F') if shape[1] > 1 else vals

        # Convert all CasADi DM outputs to numpy. lam_p (parameter sensitivity)
        # is only meaningful when parameters are registered; set to None otherwise
        # so callers don't have to special-case an empty array.
        f_opt = float(sol['f'])
        g_opt = sol['g'].full().flatten()
        lam_x = sol['lam_x'].full().flatten()
        lam_g = sol['lam_g'].full().flatten()
        lam_p = sol['lam_p'].full().flatten() if self._p else None

        return SolutionResult(
            success=success,
            x_opt=x_opt,
            f_opt=f_opt,
            g_opt=g_opt,
            lam_x=lam_x,
            lam_g=lam_g,
            stats=solver_stats,
            raw_sol=dict(sol),
            lam_p=lam_p,
        )

    # -------------------------------------------------------------------------
    # Post-solve utilities
    # -------------------------------------------------------------------------

    def extract(self, sol: dict, name: str) -> np.ndarray:
        """
        Slice a named variable's values out of a raw CasADi solution dict.

        This is a lower-level alternative to ``result.x_opt[name]``; it works
        directly on the CasADi solution dict (e.g., ``result.raw_sol``) and is
        useful when the caller wants to avoid the overhead of building the full
        ``SolutionResult``, or when post-processing the raw dict directly.

        Args:
            sol:  CasADi solution dict (must contain the ``'x'`` key).
            name: Variable name as registered with ``add_variable``.

        Returns:
            1-D numpy array of optimal values for this variable.
        """
        i, j, shape = self._var_map[name]
        vals = sol['x'][i:j].full().flatten()
        return vals.reshape(shape, order='F') if shape[1] > 1 else vals

    def stats(self) -> dict:
        """
        Return solver statistics from the most recent solve call.

        Delegates to ``self._solver.stats()``. Key fields include:
          - ``return_status``: IPOPT exit status string
          - ``iter_count``:    number of IPOPT iterations
          - ``t_wall_total``:  wall-clock time in seconds

        Returns:
            Dict of solver statistics. Contents depend on the solver plugin.

        Raises:
            RuntimeError: If ``solve()`` has not been called yet. CasADi's
                          ``stats()`` returns stale or empty data before any
                          solve has run; this guard surfaces that as an explicit
                          error rather than silent incorrect output.
        """
        if not self._solved:
            raise RuntimeError("No solve has been run yet. Call solve() before stats().")
        return self._solver.stats()
