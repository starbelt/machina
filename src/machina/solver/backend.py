import time
import warnings
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from . import plugins
from .errors import SolverError
from .records import (
    ConstraintRecord,
    CostRecord,
    LayoutEntry,
    ParameterRecord,
    VariableRecord,
)
from .result import SolutionResult

_DISCRETE_MODES = ('native', 'relax')


# -----------------------------------------------------------------------------
# Shape and value helpers
# -----------------------------------------------------------------------------

def _shape_of(n, name: str) -> tuple[int, int]:
    """Normalise the ``n`` argument of add_variable/add_parameter to (rows, cols)."""
    if isinstance(n, tuple):
        if len(n) != 2:
            raise ValueError(f"'{name}': shape must be an int or (rows, cols), got {n!r}.")
        rows, cols = int(n[0]), int(n[1])
    elif isinstance(n, (int, np.integer)):
        rows, cols = int(n), 1
    else:
        raise TypeError(f"'{name}': shape must be an int or (rows, cols), got {n!r}.")
    if rows < 1 or cols < 1:
        raise ValueError(f"'{name}': shape must be positive, got {(rows, cols)}.")
    return rows, cols


def _flatten(value, shape: tuple[int, int], *, what: str, name: str,
             dtype=float) -> np.ndarray:
    """
    Return ``value`` as a flat array of length ``rows*cols`` in **column-major**
    order, matching the ``ca.vec`` layout of the symbol it belongs to.

    Scalars broadcast. A 2-D array of the declared shape is raveled with
    ``order='F'``. A 1-D array of the right length is taken as already
    column-major. Anything else is an error that names the quantity.
    """
    numel = shape[0] * shape[1]
    if isinstance(value, ca.DM):
        value = value.full()
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} of '{name}' is not numeric: {value!r}.") from exc

    if arr.ndim == 0:
        flat = np.full(numel, arr.item(), dtype=dtype)
    elif arr.ndim == 1 and arr.size == numel:
        flat = arr.copy()
    elif arr.ndim == 2 and arr.shape == shape:
        flat = arr.ravel(order='F').copy()
    elif arr.ndim == 2 and 1 in shape and 1 in arr.shape and arr.size == numel:
        flat = arr.ravel().copy()      # row given for a column vector, or vice versa
    else:
        raise ValueError(
            f"{what} of '{name}' has shape {arr.shape}; expected a scalar, "
            f"an array of shape {shape}, or a flat column-major array of "
            f"length {numel}."
        )
    if dtype is float and np.isnan(flat).any():
        raise ValueError(f"{what} of '{name}' contains NaN.")
    return flat


def _natural(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Flat column-major -> 1-D for vectors, (rows, cols) for matrices."""
    return flat.reshape(shape, order='F') if shape[1] > 1 else flat.copy()


def _as_mx(expr, what: str, name) -> ca.MX:
    """Accept MX, convert numeric constants, refuse SX with the fix spelled out."""
    if isinstance(expr, ca.MX):
        return expr
    label = f" '{name}'" if name else ""
    if isinstance(expr, ca.SX):
        raise TypeError(
            f"{what}{label} is an SX expression. The NLP is assembled in MX: wrap "
            "the SX graph in a ca.Function and call it with the MX symbols "
            "returned by add_variable()/add_parameter()."
        )
    try:
        return ca.MX(ca.DM(expr))
    except Exception as exc:
        raise TypeError(
            f"{what}{label} must be a CasADi MX expression or a numeric constant, "
            f"got {type(expr).__name__}."
        ) from exc


# -----------------------------------------------------------------------------
# Internal registration entries (mutable; public view is records.py)
# -----------------------------------------------------------------------------

@dataclass
class _Var:
    name: str
    shape: tuple[int, int]
    start: int
    stop: int
    symbol: ca.MX
    flat: ca.MX                 # what enters the decision vector (ca.vec for matrices)
    lb: np.ndarray              # current values, flat column-major, physical units
    ub: np.ndarray
    x0: np.ndarray
    declared: tuple             # (lb, ub, x0) as registered, for reset()/unfix()
    discrete: np.ndarray
    scale: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class _Param:
    name: str
    shape: tuple[int, int]
    start: int
    stop: int
    symbol: ca.MX
    flat: ca.MX
    value: np.ndarray | None
    meta: dict = field(default_factory=dict)


@dataclass
class _Con:
    name: str
    index: int
    start: int
    stop: int
    expr: ca.MX
    lb: np.ndarray
    ub: np.ndarray
    scale: np.ndarray
    auto_named: bool
    doc: str
    component: str | None


@dataclass
class _Cost:
    name: str
    index: int
    expr: ca.MX
    weight: object
    auto_named: bool
    doc: str
    component: str | None


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
       Registration is locked from here on; bounds, guesses and parameter
       values stay editable (``set_bounds``, ``fix``, ``set_parameter``, ...).
    3. Call ``solve()`` one or more times. ``rebuild()`` re-targets the same
       registered problem at another plugin, option set or discrete mode.

    MX only
    -------
    The NLP is assembled in MX. Library and component functions are SX inside
    a ``ca.Function`` and are *called* with the MX symbols this class hands
    out. SX expressions are refused; numeric constants are accepted.

    Units and scaling
    -----------------
    Everything a caller passes in or reads back is in physical units. When a
    variable or constraint is registered with ``scale != 1`` the solver works
    on ``x / scale`` and ``g / scale`` internally; bounds, guesses, results and
    multipliers are converted at the boundary. With every scale equal to 1 the
    assembled NLP is exactly the unscaled one.

    Design note — ``nlpsol`` vs ``Opti``
    -------------------------------------
    CasADi's ``Opti`` interface is convenient but owns variable indexing
    internally, making it hard to inspect or control decision vector layout.
    ``nlpsol`` gives this framework explicit control over the global vector
    structure, which is essential for interleaved state/control ordering in
    trajectory problems and for clean result extraction by name.
    """

    def __init__(self, solver: str = 'ipopt', solver_opts: dict = None, *,
                 verbose: bool = True):
        """
        Initialize the solver backend.

        Args:
            solver:       CasADi solver plugin name. Default is ``'ipopt'``.
                          Others: ``'bonmin'`` (mixed-integer), ``'sqpmethod'``,
                          ``'fatrop'``. Per-plugin defaults live in
                          ``machina.solver.plugins``.
            solver_opts:  Solver options passed through to CasADi. Accepts flat
                          dot-notation (``{'ipopt.tol': 1e-8}``) or nested
                          dicts (``{'ipopt': {'tol': 1e-8}}``). Merged on top
                          of the plugin defaults, so only specify what you
                          want to override. An option carrying another
                          plugin's prefix raises ``SolverError`` at build.
            verbose:      ``False`` applies the plugin's quiet options (for
                          IPOPT: no banner, ``print_level`` 0, no timing
                          table). Explicit ``solver_opts`` still win.
        """
        self.solver_name: str = solver
        self.solver_opts: dict = solver_opts if solver_opts is not None else {}
        self.verbose: bool = verbose

        # Registration state. Insertion-ordered dicts; registration order is
        # the layout of the flat vectors and the order of every report.
        self._vars:   dict[str, _Var] = {}
        self._params: dict[str, _Param] = {}
        self._cons:   dict[str, _Con] = {}
        self._costs:  dict[str, _Cost] = {}

        # Scalar objective. Initialized as a CasADi zero (not Python 0) so
        # that the first add_cost() call produces an MX expression, not a
        # mixed Python-int/MX sum that CasADi may handle inconsistently.
        self._J: ca.MX = ca.MX.zeros(1, 1)

        # Running offsets into the flat decision, parameter and constraint vectors.
        self._n_x: int = 0
        self._n_p: int = 0
        self._n_g: int = 0

        self._solver: ca.Function = None
        self._warm_solver: ca.Function = None   # built lazily, see solve()
        self._last_solver: ca.Function = None
        self._eval_fn: ca.Function = None       # (x, p) -> (g, cost terms), built lazily
        self._nlp: dict = None
        self._build_opts: dict = {}
        self._extra_opts: dict = {}     # options the backend adds itself ('discrete')
        self._discrete_mode: str = 'native'
        self._relaxed: tuple[str, ...] = ()
        self._scaled: bool = False
        self._built:  bool = False
        self._solved: bool = False  # True after the first solve() call completes

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _check_unlocked(self, method: str) -> None:
        if self._built:
            raise RuntimeError(
                f"{method}() called after build(): registration is locked. "
                "Register everything before build(); use set_bounds(), "
                "set_initial_guess(), set_parameter() or fix() to change data, "
                "and rebuild() only to change the solver or its options."
            )

    def _check_new_name(self, name: str, kind: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{kind} name must be a non-empty string, got {name!r}.")
        if name in self._vars or name in self._params:
            raise ValueError(f"{kind} name '{name}' already exists.")

    def add_variable(self, name: str, n: int | tuple[int, int],
                     lb=-np.inf, ub=np.inf,
                     initial_guess=0.0, *,
                     discrete=False, scale=1.0,
                     unit: str = "", doc: str = "",
                     provenance: str = None, source: str = None,
                     component: str = None, frame: str = None) -> ca.MX:
        """
        Create a decision variable and register it in the global decision vector.

        The returned MX symbol is the handle used to build constraint and cost
        expressions. Internally, the variable occupies a contiguous slice
        [offset, offset+numel) in the flat vector, so results can be extracted
        by name after solving.

        For matrix variables, the decision vector stores elements column-major
        (via ``ca.vec``). Array-valued bounds and guesses given in the natural
        ``(rows, cols)`` shape are flattened the same way, and extraction
        reshapes back using Fortran order.

        Args:
            name:          Unique name for this variable. Must not already exist
                           in the variable/parameter namespace.
            n:             Size of the variable. Pass an ``int`` for an n×1
                           column vector, or a ``(rows, cols)`` tuple for a
                           matrix. The returned MX symbol has the requested
                           shape.
            lb:            Lower bound. A scalar broadcasts; an array may be
                           ``(rows, cols)`` or a flat column-major array of
                           length ``rows*cols``.
            ub:            Upper bound. Same rules as lb.
            initial_guess: Starting point for the solver. Same rules as lb.
                           A good initial guess is important for convergence on
                           nonlinear problems; zero is used when nothing better
                           is known.
            discrete:      ``True`` (or a per-element bool array) marks integer
                           elements. See ``build(discrete_mode=...)``.
            scale:         Nominal magnitude, > 0. The solver works on
                           ``x / scale``; everything the caller sees stays in
                           physical units. Must be 1 for discrete elements.
            unit, doc, provenance, source, component, frame:
                           Metadata carried on the public record for reports
                           and debugging. Not used by the solver.

        Returns:
            MX symbolic variable with the requested shape, in physical units.

        Raises:
            ValueError:   duplicate name, malformed shape, wrong-sized or NaN
                          bounds, ``lb > ub``, non-positive scale, or a scaled
                          discrete element.
            RuntimeError: if called after ``build()``.
        """
        self._check_unlocked('add_variable')
        self._check_new_name(name, 'Variable')
        rows, cols = _shape_of(n, name)
        shape = (rows, cols)
        numel = rows * cols

        lb_f = _flatten(lb, shape, what='lb', name=name)
        ub_f = _flatten(ub, shape, what='ub', name=name)
        x0_f = _flatten(initial_guess, shape, what='initial_guess', name=name)
        if (lb_f > ub_f).any():
            raise ValueError(f"Variable '{name}': lb > ub for at least one element.")
        disc = _flatten(discrete, shape, what='discrete', name=name, dtype=bool)
        sc = _flatten(scale, shape, what='scale', name=name)
        if not (np.isfinite(sc).all() and (sc > 0).all()):
            raise ValueError(f"Variable '{name}': scale must be finite and > 0.")
        if (disc & (sc != 1.0)).any():
            raise ValueError(
                f"Variable '{name}': discrete elements must have scale 1 "
                "(the integrality would apply to x/scale)."
            )

        if isinstance(n, tuple):
            var = ca.MX.sym(name, rows, cols)
            # ca.vec flattens column-major so the matrix occupies a contiguous
            # column vector block in the global decision vector, as nlpsol requires.
            flat = ca.vec(var)
        else:
            var = ca.MX.sym(name, rows)
            flat = var

        self._vars[name] = _Var(
            name=name, shape=shape, start=self._n_x, stop=self._n_x + numel,
            symbol=var, flat=flat, lb=lb_f, ub=ub_f, x0=x0_f,
            declared=(lb_f.copy(), ub_f.copy(), x0_f.copy()),
            discrete=disc, scale=sc,
            meta=dict(unit=unit, doc=doc, provenance=provenance, source=source,
                      component=component, frame=frame),
        )
        self._n_x += numel
        return var

    def add_parameter(self, name: str, n: int | tuple[int, int], *,
                      value=None,
                      unit: str = "", doc: str = "",
                      provenance: str = None, source: str = None,
                      component: str = None, frame: str = None) -> ca.MX:
        """
        Create a fixed parameter and register it in the global parameter vector.

        Parameters appear in the objective and constraints but are not optimized.
        Their values can be stored here (``value=``), changed later with
        ``set_parameter()``, or supplied per call through ``solve(p_val=...)``,
        so the same built solver can be re-solved with different problem data
        without rebuilding — critical for parametric sweeps or MPC-style
        re-planning.

        Args:
            name:  Unique name. Must not already exist in the variable/parameter
                   namespace (parameters and variables share one namespace to
                   prevent ambiguous expression graphs).
            n:     Size of the parameter. Pass an ``int`` for an n×1 column
                   vector, or a ``(rows, cols)`` tuple for a matrix.
            value: Optional stored value: scalar, ``(rows, cols)`` array, or a
                   flat column-major array.
            unit, doc, provenance, source, component, frame:
                   Metadata carried on the public record.

        Returns:
            MX symbolic parameter — use this in objective and constraint
            expressions exactly like a variable symbol.
        """
        self._check_unlocked('add_parameter')
        self._check_new_name(name, 'Parameter')
        rows, cols = _shape_of(n, name)
        shape = (rows, cols)
        numel = rows * cols

        if isinstance(n, tuple):
            param = ca.MX.sym(name, rows, cols)
            flat = ca.vec(param)
        else:
            param = ca.MX.sym(name, rows)
            flat = param

        stored = None if value is None else _flatten(value, shape, what='value', name=name)
        self._params[name] = _Param(
            name=name, shape=shape, start=self._n_p, stop=self._n_p + numel,
            symbol=param, flat=flat, value=stored,
            meta=dict(unit=unit, doc=doc, provenance=provenance, source=source,
                      component=component, frame=frame),
        )
        self._n_p += numel
        return param

    def add_constraint(self, expr: ca.MX,
                       lb=-np.inf, ub=np.inf,
                       name: str = None, *,
                       scale=1.0, doc: str = "", component: str = None):
        """
        Add a constraint expression to the problem.

        CasADi/IPOPT interprets the bounds as:
          - lb == ub    → equality constraint (IPOPT detects this automatically)
          - finite lb, infinite ub  → one-sided lower bound (g(x) >= lb)
          - infinite lb, finite ub  → one-sided upper bound (g(x) <= ub)
          - both finite → double-sided inequality

        Args:
            expr:  MX column expression; its number of rows determines how many
                   constraint rows are added. Write it so that the bound
                   applies to the expression value directly: to enforce
                   x >= 2, pass ``expr=x`` with ``lb=2``. A numeric constant is
                   accepted (a fixed-role substitution can make a constraint
                   constant); an SX expression is refused.
            lb:    Lower bound. Scalar is broadcast to all rows.
            ub:    Upper bound. Scalar is broadcast to all rows.
            name:  Label for reporting and for named access on the result
                   (``result.constraint(name)``). Unique among constraints when
                   given; ``None`` gets ``g<index>``. Constraint names do NOT
                   enter the variable/parameter namespace.
            scale: Nominal magnitude, > 0 (scalar or per row). The solver sees
                   ``expr / scale``; values and multipliers are reported in
                   physical units.
            doc, component: Metadata carried on the public record.
        """
        self._check_unlocked('add_constraint')
        expr = _as_mx(expr, 'Constraint', name)
        if expr.shape[1] != 1:
            raise ValueError(
                f"Constraint{f' {name!r}' if name else ''} has shape {expr.shape}; "
                "constraints must be column vectors. Use ca.vec(expr)."
            )
        n = expr.shape[0]
        index = len(self._cons)
        auto = name is None
        cname = f'g{index}' if auto else name
        if cname in self._cons:
            raise ValueError(
                f"Constraint name '{cname}' already exists. Constraint names "
                "must be unique so results can be read by name."
            )
        lb_f = _flatten(lb, (n, 1), what='lb', name=cname)
        ub_f = _flatten(ub, (n, 1), what='ub', name=cname)
        if (lb_f > ub_f).any():
            raise ValueError(f"Constraint '{cname}': lb > ub for at least one row.")
        sc = _flatten(scale, (n, 1), what='scale', name=cname)
        if not (np.isfinite(sc).all() and (sc > 0).all()):
            raise ValueError(f"Constraint '{cname}': scale must be finite and > 0.")

        self._cons[cname] = _Con(
            name=cname, index=index, start=self._n_g, stop=self._n_g + n,
            expr=expr, lb=lb_f, ub=ub_f, scale=sc, auto_named=auto,
            doc=doc, component=component,
        )
        self._n_g += n

    def add_equality(self, expr: ca.MX, name: str = None, *,
                     scale=1.0, doc: str = "", component: str = None):
        """
        Convenience wrapper: add an equality constraint ``expr == 0``.

        Equivalent to ``add_constraint(expr, lb=0, ub=0, name=name)``.
        Write the expression so it equals zero at the desired solution:
        e.g., to enforce x + y == 1, pass ``expr = x + y - 1``.
        """
        self.add_constraint(expr, lb=0.0, ub=0.0, name=name,
                            scale=scale, doc=doc, component=component)

    def add_cost(self, expr: ca.MX, name: str = None, *,
                 weight=1.0, doc: str = "", component: str = None):
        """
        Add a scalar cost term to the objective.

        All terms are accumulated into a single objective
        ``f = sum(weight_i * term_i)``. CasADi builds this as one symbolic
        expression, so the solver sees the full combined objective and can
        compute exact gradients across all terms.

        Args:
            expr:   Scalar MX expression (shape must be (1, 1)).
            name:   Label for the objective breakdown (``result.cost_terms``).
                    Unique among cost terms when given; ``None`` gets
                    ``J<index>``.
            weight: Float, or a scalar MX (typically a parameter, so weights
                    can be swept without a rebuild). The record keeps the
                    unweighted term and the weight separately.
            doc, component: Metadata carried on the public record.

        Raises:
            ValueError: If ``expr`` is not shape (1, 1). Vector-valued
                        expressions must be reduced to a scalar first, e.g.
                        via ``ca.sumsqr(v)`` or ``v.T @ v``.
        """
        self._check_unlocked('add_cost')
        expr = _as_mx(expr, 'Cost term', name)
        if expr.shape != (1, 1):
            raise ValueError(
                f"Cost term must be scalar (1, 1), got {expr.shape}."
                + (f" Name: '{name}'." if name else "")
            )
        index = len(self._costs)
        auto = name is None
        cname = f'J{index}' if auto else name
        if cname in self._costs:
            raise ValueError(
                f"Cost term name '{cname}' already exists. Cost names must be "
                "unique so the objective breakdown can be read by name."
            )
        if not isinstance(weight, (int, float)):
            weight = _as_mx(weight, 'Cost weight', cname)
            if weight.shape != (1, 1):
                raise ValueError(f"Cost '{cname}': weight must be scalar, got {weight.shape}.")

        entry = _Cost(name=cname, index=index, expr=expr, weight=weight,
                      auto_named=auto, doc=doc, component=component)
        self._costs[cname] = entry
        self._J += self._weighted(entry)

    @staticmethod
    def _weighted(cost: _Cost) -> ca.MX:
        if isinstance(cost.weight, (int, float)) and cost.weight == 1.0:
            return cost.expr
        return cost.weight * cost.expr

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self, opts: dict = None, *, discrete_mode: str = 'native') -> ca.Function:
        """
        Assemble the NLP and create the ``nlpsol`` solver object.

        After ``build()`` returns, registration is locked — no further
        variables, parameters, constraints or costs can be added. Bounds,
        initial guesses and parameter values remain editable.

        Option precedence (later layers override earlier ones):
            1. Plugin defaults (``machina.solver.plugins``)
            2. Plugin quiet options, when ``verbose=False``
            3. ``solver_opts`` supplied at ``__init__`` time
            4. ``opts`` supplied to this ``build()`` call

        Args:
            opts:          Additional or override solver options for this build.
            discrete_mode: What to do with variables registered
                           ``discrete=True``. ``'native'`` hands the per-element
                           integrality flags to the plugin, which must have
                           integer support (bonmin); any other plugin raises
                           ``SolverError``. ``'relax'`` solves them as
                           continuous and lists their names in
                           ``result.relaxed_discrete``.

        Returns:
            The ``ca.nlpsol`` Function object.

        Raises:
            RuntimeError: If no variables have been registered, or if
                          ``build()`` has already been called on this instance
                          (use ``rebuild()``).
            SolverError:  Plugin unavailable, option with a foreign plugin
                          prefix, or discrete variables without integer support.

        Note on ``expand``:
            Setting ``expand=True`` (via opts) converts the MX graph to SX at
            build time for faster evaluation. It is incompatible with
            ``ca.Callback`` objects. Omitted from the defaults so that
            iteration callbacks remain usable.
        """
        if self._built:
            raise RuntimeError(
                "Solver has already been built. Use rebuild() to change the "
                "solver plugin or its options."
            )
        if not self._vars:
            raise RuntimeError(
                "No decision variables registered. Call add_variable() before build()."
            )
        if not self._costs:
            warnings.warn(
                "No cost terms registered. The objective will be zero (feasibility problem).",
                UserWarning,
                stacklevel=2,
            )
        self._build_opts = dict(opts) if opts else {}
        self._construct(discrete_mode)
        self._built = True
        return self._solver

    def rebuild(self, opts: dict = None, *, solver: str = None,
                solver_opts: dict = None, discrete_mode: str = None) -> ca.Function:
        """
        Create a fresh ``nlpsol`` object from the same registered problem.

        Use it to switch plugin (``solver='bonmin'`` after a relaxed IPOPT
        solve), change options (``opts={'expand': True}``) or change
        ``discrete_mode`` without re-running whatever registered the problem.
        Arguments left as ``None`` keep their previous value. When the plugin
        changes, stored options carrying the old plugin's prefix are dropped
        because they cannot apply.
        """
        if not self._built:
            raise RuntimeError("Call build() before rebuild().")
        if solver is not None and solver != self.solver_name:
            old_prefix = plugins.spec_for(self.solver_name).prefix
            self.solver_opts = plugins.strip_prefix(self.solver_opts, old_prefix)
            self._build_opts = plugins.strip_prefix(self._build_opts, old_prefix)
            self.solver_name = solver
        if solver_opts is not None:
            self.solver_opts = dict(solver_opts)
        if opts is not None:
            self._build_opts = dict(opts)
        self._construct(self._discrete_mode if discrete_mode is None else discrete_mode)
        self._solved = False
        return self._solver

    def _physical_nlp(self) -> tuple:
        """(w, p or None, g) MX vectors of the registered problem, physical units."""
        w = ca.vertcat(*[v.flat for v in self._vars.values()])
        # ca.MX(0, 1) is the correct empty column vector for a problem with no
        # constraints; ca.MX.zeros(0) would be a scalar and confuse IPOPT.
        g = (ca.vertcat(*[c.expr for c in self._cons.values()])
             if self._cons else ca.MX(0, 1))
        p = (ca.vertcat(*[q.flat for q in self._params.values()])
             if self._params else None)
        return w, p, g

    def _concat(self, entries, attr: str, dtype=float) -> np.ndarray:
        parts = [getattr(e, attr) for e in entries]
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    def _construct(self, discrete_mode: str) -> None:
        if discrete_mode not in _DISCRETE_MODES:
            raise ValueError(
                f"discrete_mode must be one of {_DISCRETE_MODES}, got {discrete_mode!r}."
            )
        spec = plugins.spec_for(self.solver_name)
        plugins.require_available(self.solver_name)

        w, p, g = self._physical_nlp()
        sx = self._concat(self._vars.values(), 'scale')
        sg = self._concat(self._cons.values(), 'scale')
        self._scaled = bool((sx != 1.0).any() or (sg != 1.0).any())

        if not self._scaled:
            nlp = {'x': w, 'f': self._J, 'g': g}
            # Only add 'p' when parameters are registered. If 'p' were always
            # present (even as an empty vector), every solver() call would
            # require a p= keyword argument, breaking the no-parameter usage.
            if p is not None:
                nlp['p'] = p
        else:
            # Scaled problem: keep the physical graph intact inside a Function
            # and evaluate it at scale * z. Callers' symbols stay pure symbols.
            z = ca.MX.sym('z', self._n_x)
            if p is not None:
                phys = ca.Function('machina_nlp', [w, p], [self._J, g])
                pz = ca.MX.sym('p', self._n_p)
                f_z, g_z = phys(ca.DM(sx) * z, pz)
            else:
                phys = ca.Function('machina_nlp', [w], [self._J, g])
                pz = None
                f_z, g_z = phys(ca.DM(sx) * z)
            if self._n_g:
                g_z = g_z / ca.DM(sg)
            nlp = {'x': z, 'f': f_z, 'g': g_z}
            if pz is not None:
                nlp['p'] = pz

        extra = {}
        flags = self._concat(self._vars.values(), 'discrete', dtype=bool)
        discrete_names = tuple(v.name for v in self._vars.values() if v.discrete.any())
        self._relaxed = ()
        if discrete_names:
            if discrete_mode == 'relax':
                self._relaxed = discrete_names
            elif not spec.supports_discrete:
                raise SolverError(
                    f"Variables {list(discrete_names)} are discrete but the "
                    f"'{spec.name}' plugin has no integer support. Use "
                    "SolverBackend(solver='bonmin') (or rebuild(solver='bonmin')), "
                    "or build(discrete_mode='relax') to solve them as continuous."
                )
            else:
                extra['discrete'] = [bool(b) for b in flags]

        merged = plugins.merge_options(
            spec, verbose=self.verbose,
            solver_opts=self.solver_opts, build_opts=self._build_opts,
        )
        merged.update(extra)

        self._solver = ca.nlpsol('solver', self.solver_name, nlp, merged)
        self._warm_solver = None
        self._last_solver = self._solver
        self._nlp = nlp
        self._extra_opts = extra
        self._discrete_mode = discrete_mode

    def _get_warm_solver(self) -> ca.Function:
        """The nlpsol object used when duals are supplied; built on first use."""
        spec = plugins.spec_for(self.solver_name)
        if not spec.warm_start:
            return self._solver
        if self._warm_solver is None:
            merged = plugins.merge_options(
                spec, verbose=self.verbose, solver_opts=self.solver_opts,
                build_opts=self._build_opts, warm=True,
            )
            merged.update(self._extra_opts)
            self._warm_solver = ca.nlpsol('solver_warm', self.solver_name,
                                          self._nlp, merged)
        return self._warm_solver

    # -------------------------------------------------------------------------
    # Editable data: bounds, guesses, parameter values
    # -------------------------------------------------------------------------

    def _var(self, name: str) -> _Var:
        if name not in self._vars:
            raise KeyError(f"No variable named '{name}'. Registered: {list(self._vars)}")
        return self._vars[name]

    def _param(self, name: str) -> _Param:
        if name not in self._params:
            raise KeyError(f"No parameter named '{name}'. Registered: {list(self._params)}")
        return self._params[name]

    def set_bounds(self, name: str, lb=None, ub=None) -> None:
        """Change a variable's bounds (before or after build; no rebuild needed)."""
        v = self._var(name)
        new_lb = v.lb if lb is None else _flatten(lb, v.shape, what='lb', name=name)
        new_ub = v.ub if ub is None else _flatten(ub, v.shape, what='ub', name=name)
        if (new_lb > new_ub).any():
            raise ValueError(f"Variable '{name}': lb > ub for at least one element.")
        v.lb, v.ub = new_lb, new_ub

    def set_initial_guess(self, name: str, x0) -> None:
        """Change a variable's stored initial guess."""
        v = self._var(name)
        v.x0 = _flatten(x0, v.shape, what='initial_guess', name=name)

    def set_parameter(self, name: str, value) -> None:
        """Store a parameter value; used by every later solve() unless overridden."""
        q = self._param(name)
        q.value = _flatten(value, q.shape, what='value', name=name)

    def fix(self, name: str, value) -> None:
        """Pin a variable: ``lb = ub = initial guess = value``. Undo with ``unfix``."""
        v = self._var(name)
        pinned = _flatten(value, v.shape, what='value', name=name)
        v.lb, v.ub, v.x0 = pinned, pinned.copy(), pinned.copy()

    def unfix(self, name: str) -> None:
        """Restore the bounds a variable was registered with."""
        v = self._var(name)
        v.lb, v.ub = v.declared[0].copy(), v.declared[1].copy()

    def reset(self, name: str = None) -> None:
        """Restore registered bounds and initial guess for one variable, or all."""
        for v in ([self._var(name)] if name is not None else self._vars.values()):
            v.lb, v.ub, v.x0 = (a.copy() for a in v.declared)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------

    def _parameter_vector(self, p_val) -> tuple[np.ndarray, dict]:
        """Flat parameter vector for this call, plus the by-name values used."""
        values = {name: q.value for name, q in self._params.items()}

        if isinstance(p_val, dict):
            for name, value in p_val.items():
                q = self._param(name)
                values[name] = _flatten(value, q.shape, what='p_val', name=name)
        elif p_val is not None:
            flat = np.asarray(p_val, dtype=float).flatten()
            if len(flat) != self._n_p:
                raise ValueError(
                    f"p_val has length {len(flat)}, but {self._n_p} "
                    f"parameter value(s) are required."
                )
            for name, q in self._params.items():
                values[name] = flat[q.start:q.stop]

        missing = [name for name, value in values.items() if value is None]
        if missing:
            lead = "p_val is None and no" if p_val is None else "No"
            raise ValueError(
                f"{lead} value is stored for parameter(s) {missing}. Pass them "
                "in solve(p_val=...), or store them with "
                "add_parameter(value=...) / set_parameter()."
            )
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Parameter '{name}' has a non-finite value: {value}.")
        flat = (np.concatenate([values[n] for n in self._params])
                if self._params else np.zeros(0))
        by_name = {n: _natural(values[n], self._params[n].shape) for n in self._params}
        return flat, by_name

    def _seed(self, given, entries: dict, total: int, base: np.ndarray, what: str) -> np.ndarray:
        """Overlay a flat array or a by-name dict onto ``base``."""
        if given is None:
            return base
        out = base.copy()
        if isinstance(given, dict):
            for name, value in given.items():
                if name not in entries:
                    raise KeyError(f"{what}: no entry named '{name}'.")
                e = entries[name]
                shape = getattr(e, 'shape', (e.stop - e.start, 1))
                out[e.start:e.stop] = _flatten(value, shape, what=what, name=name)
            return out
        flat = np.asarray(given, dtype=float).flatten()
        if len(flat) != total:
            raise ValueError(f"{what} has length {len(flat)}, expected {total}.")
        return flat

    def solve(self, p_val=None, *, x0=None, lam_x0=None, lam_g0=None,
              warm_start: SolutionResult = None) -> SolutionResult:
        """
        Call the solver and return a structured result.

        Args:
            p_val:      Parameter values for this call. ``None`` uses the
                        stored values. A flat list/array supplies the whole
                        vector in registration order (column-major per matrix
                        parameter). A ``{name: value}`` dict overrides just
                        those parameters for this call.
            x0:         Primal starting point for this call, as a flat array
                        or a (partial) ``{name: value}`` dict laid over the
                        stored initial guesses.
            lam_x0:     Bound multipliers to start from (flat, or by variable).
            lam_g0:     Constraint multipliers to start from (flat, or by
                        constraint name).
            warm_start: A previous ``SolutionResult``. Primal values are taken
                        by name wherever name and shape match, so a result
                        from a differently shaped problem still helps; duals
                        are taken only when the layout is identical. Explicit
                        ``x0`` / ``lam_*`` arguments win over it.

        Supplying duals switches the call to a second ``nlpsol`` object that
        carries the plugin's warm-start options (for IPOPT,
        ``warm_start_init_point`` and friends). It is created on first use;
        cold solves never touch it. Everything is in physical units.

        Returns:
            ``SolutionResult`` with all solution fields populated and all
            variable values accessible by name via ``result['name']``.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
            ValueError:   If a parameter has no value, or an array has the
                          wrong length.
        """
        if not self._built:
            raise RuntimeError("Call build() before solve().")
        if not self._params and p_val is not None:
            if (len(p_val) if isinstance(p_val, dict) else np.size(p_val)) > 0:
                raise ValueError("p_val was given but no parameters are registered.")
            p_val = None

        p_flat, p_by_name = self._parameter_vector(p_val)

        lbx = self._concat(self._vars.values(), 'lb')
        ubx = self._concat(self._vars.values(), 'ub')
        lbg = self._concat(self._cons.values(), 'lb')
        ubg = self._concat(self._cons.values(), 'ub')
        x_start = self._concat(self._vars.values(), 'x0')

        if warm_start is not None:
            for name, v in self._vars.items():
                prev = warm_start.x_opt.get(name)
                if prev is not None and np.size(prev) == v.stop - v.start:
                    x_start[v.start:v.stop] = _flatten(prev, v.shape, what='warm_start', name=name)
            same_layout = (warm_start.x_layout == self._layout(self._vars)
                           and warm_start.g_layout == self._layout(self._cons))
            # Mixed-integer plugins (bonmin) return NaN multipliers.
            finite_duals = (np.isfinite(warm_start.lam_x).all()
                            and np.isfinite(warm_start.lam_g).all())
            if same_layout and finite_duals:
                lam_x0 = warm_start.lam_x if lam_x0 is None else lam_x0
                lam_g0 = warm_start.lam_g if lam_g0 is None else lam_g0

        x_start = self._seed(x0, self._vars, self._n_x, x_start, 'x0')
        if not np.isfinite(x_start).all():
            raise ValueError("The initial guess contains a non-finite value.")
        use_duals = lam_x0 is not None or lam_g0 is not None
        lam_x_start = self._seed(lam_x0, self._vars, self._n_x, np.zeros(self._n_x), 'lam_x0')
        lam_g_start = self._seed(lam_g0, self._cons, self._n_g, np.zeros(self._n_g), 'lam_g0')

        sx = self._concat(self._vars.values(), 'scale')
        sg = self._concat(self._cons.values(), 'scale')

        kwargs = dict(x0=x_start / sx, lbx=lbx / sx, ubx=ubx / sx,
                      lbg=lbg / sg, ubg=ubg / sg)
        if self._params:
            kwargs['p'] = p_flat
        if use_duals:
            kwargs['lam_x0'] = lam_x_start * sx
            kwargs['lam_g0'] = lam_g_start * sg
        solver = self._get_warm_solver() if use_duals else self._solver

        t0 = time.perf_counter()
        sol = solver(**kwargs)
        t_wall = time.perf_counter() - t0
        self._solved = True
        self._last_solver = solver
        solver_stats = solver.stats()

        # Back to physical units. With every scale equal to 1 these are no-ops
        # and raw_sol is CasADi's own dict.
        x_flat = sol['x'].full().flatten() * sx
        g_flat = sol['g'].full().flatten() * sg
        lam_x = sol['lam_x'].full().flatten() / sx
        lam_g = sol['lam_g'].full().flatten() / sg
        # lam_p (CasADi's parameter multiplier, -df*/dp) is unaffected by
        # variable/constraint scaling; None without parameters so callers need
        # no special case.
        lam_p = sol['lam_p'].full().flatten() if self._params else None
        f_opt = float(sol['f'])

        # CasADi's bonmin interface returns NaN for g (and for the multipliers,
        # which are not defined for a mixed-integer solve). g is recoverable
        # from the registered graph; the multipliers stay NaN.
        g_eval, cost_values = self._evaluate(x_flat, p_flat)
        if self._n_g and not np.isfinite(g_flat).all():
            g_flat = g_eval

        if self._scaled:
            raw_sol = dict(sol)
            raw_sol.update(x=ca.DM(x_flat), g=ca.DM(g_flat),
                           lam_x=ca.DM(lam_x), lam_g=ca.DM(lam_g))
        else:
            raw_sol = dict(sol)

        # Matrix variables are reshaped back to (rows, cols) using Fortran
        # (column-major) order, matching the ca.vec() flattening at registration.
        x_opt = {name: _natural(x_flat[v.start:v.stop], v.shape)
                 for name, v in self._vars.items()}

        iterations = solver_stats.get('iter_count')
        return SolutionResult(
            success=bool(solver_stats.get('success', False)),
            x_opt=x_opt,
            f_opt=f_opt,
            g_opt=g_flat,
            lam_x=lam_x,
            lam_g=lam_g,
            stats=solver_stats,
            raw_sol=raw_sol,
            lam_p=lam_p,
            status=str(solver_stats.get('return_status', '')),
            unified_status=str(solver_stats.get('unified_return_status', '')),
            plugin=self.solver_name,
            p_opt=p_by_name,
            relaxed_discrete=self._relaxed,
            iterations=None if iterations is None else int(iterations),
            t_wall=t_wall,
            warm_started=solver is not self._solver,
            cost_terms=cost_values,
            lbg=lbg,
            ubg=ubg,
            x_layout=self._layout(self._vars),
            p_layout=self._layout(self._params),
            g_layout=self._layout(self._cons),
        )

    def _evaluate(self, x_flat: np.ndarray, p_flat: np.ndarray) -> tuple[np.ndarray, dict]:
        """Constraint vector and per-term cost values at a point, physical units."""
        if self._eval_fn is None:
            w, p, g = self._physical_nlp()
            terms = (ca.vertcat(*[self._weighted(c) for c in self._costs.values()])
                     if self._costs else ca.MX(0, 1))
            args = [w] if p is None else [w, p]
            self._eval_fn = ca.Function('machina_eval', args, [g, terms])
        args = [x_flat] if not self._params else [x_flat, p_flat]
        g_val, term_val = self._eval_fn(*args)
        values = np.asarray(term_val).flatten()
        costs = {name: float(v) for name, v in zip(self._costs, values)}
        return np.asarray(g_val).flatten(), costs

    @staticmethod
    def _layout(entries: dict) -> tuple:
        return tuple(
            LayoutEntry(e.name, e.start, e.stop,
                        getattr(e, 'shape', (e.stop - e.start, 1)))
            for e in entries.values()
        )

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def variables(self) -> list[VariableRecord]:
        """Registered variables in registration order, with current bounds/guess."""
        return [
            VariableRecord(
                name=v.name, shape=v.shape, slice=slice(v.start, v.stop),
                lb=_natural(v.lb, v.shape), ub=_natural(v.ub, v.shape),
                x0=_natural(v.x0, v.shape), discrete=_natural(v.discrete, v.shape),
                scale=_natural(v.scale, v.shape), symbol=v.symbol, **v.meta,
            )
            for v in self._vars.values()
        ]

    def parameters(self) -> list[ParameterRecord]:
        """Registered parameters in registration order, with stored values."""
        return [
            ParameterRecord(
                name=q.name, shape=q.shape, slice=slice(q.start, q.stop),
                value=None if q.value is None else _natural(q.value, q.shape),
                symbol=q.symbol, **q.meta,
            )
            for q in self._params.values()
        ]

    def constraints(self) -> list[ConstraintRecord]:
        """Registered constraints in registration order."""
        return [
            ConstraintRecord(
                name=c.name, index=c.index, slice=slice(c.start, c.stop),
                lb=c.lb.copy(), ub=c.ub.copy(), scale=c.scale.copy(), expr=c.expr,
                auto_named=c.auto_named, doc=c.doc, component=c.component,
            )
            for c in self._cons.values()
        ]

    def cost_terms(self) -> list[CostRecord]:
        """Registered cost terms in registration order."""
        return [
            CostRecord(name=c.name, index=c.index, expr=c.expr, weight=c.weight,
                       auto_named=c.auto_named, doc=c.doc, component=c.component)
            for c in self._costs.values()
        ]

    def variable_order(self) -> list[str]:
        return list(self._vars)

    def parameter_order(self) -> list[str]:
        return list(self._params)

    def has(self, name: str) -> bool:
        """True if ``name`` is a registered variable or parameter."""
        return name in self._vars or name in self._params

    def _entry(self, name: str):
        if name in self._vars:
            return self._vars[name]
        if name in self._params:
            return self._params[name]
        raise KeyError(
            f"No variable or parameter named '{name}'. "
            f"Variables: {list(self._vars)}; parameters: {list(self._params)}"
        )

    def slice_of(self, name: str) -> slice:
        """Position of a variable in ``x`` or of a parameter in ``p``."""
        e = self._entry(name)
        return slice(e.start, e.stop)

    def shape_of(self, name: str) -> tuple[int, int]:
        return self._entry(name).shape

    def symbol_of(self, name: str) -> ca.MX:
        return self._entry(name).symbol

    def bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Current ``(lb, ub)`` of a variable, in its natural shape."""
        v = self._var(name)
        return _natural(v.lb, v.shape), _natural(v.ub, v.shape)

    def initial_guess(self, name: str) -> np.ndarray:
        v = self._var(name)
        return _natural(v.x0, v.shape)

    def parameter_value(self, name: str) -> np.ndarray | None:
        q = self._param(name)
        return None if q.value is None else _natural(q.value, q.shape)

    @property
    def n_x(self) -> int:
        return self._n_x

    @property
    def n_p(self) -> int:
        return self._n_p

    @property
    def n_g(self) -> int:
        return self._n_g

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def is_solved(self) -> bool:
        return self._solved

    @property
    def plugin(self) -> str:
        return self.solver_name

    @property
    def is_scaled(self) -> bool:
        """True when the built NLP works on scaled variables or constraints."""
        return self._scaled

    @property
    def has_warm_solver(self) -> bool:
        """True once a warm-started solve has constructed the second nlpsol object."""
        return self._warm_solver is not None

    def nlp_function(self) -> ca.Function:
        """The built ``nlpsol`` Function (the cold solver)."""
        if not self._built:
            raise RuntimeError("Call build() before nlp_function().")
        return self._solver

    def nlp_expressions(self) -> dict:
        """
        The registered problem as MX, in physical units and unscaled:
        ``{'x', 'p', 'f', 'g'}``. Available before ``build()``. ``p`` is an
        empty column when no parameters are registered.
        """
        w, p, g = self._physical_nlp()
        return {'x': w, 'p': ca.MX(0, 1) if p is None else p, 'f': self._J, 'g': g}

    # -------------------------------------------------------------------------
    # Post-solve utilities
    # -------------------------------------------------------------------------

    def extract(self, sol: dict, name: str) -> np.ndarray:
        """
        Slice a named variable's values out of a CasADi-style solution dict.

        This is a lower-level alternative to ``result.x_opt[name]``; it works
        on ``result.raw_sol`` (physical units).

        Args:
            sol:  Solution dict (must contain the ``'x'`` key).
            name: Variable name as registered with ``add_variable``.

        Returns:
            1-D numpy array for vector variables; ``(rows, cols)`` array for
            matrix variables.
        """
        v = self._var(name)
        vals = sol['x'][v.start:v.stop].full().flatten()
        return _natural(vals, v.shape)

    def stats(self) -> dict:
        """
        Return solver statistics from the most recent solve call.

        Delegates to the ``nlpsol`` object that ran last (the warm-start
        solver if the last call supplied duals). Common fields:
          - ``return_status``: the plugin's exit status
          - ``success``:       plugin-independent convergence flag
          - ``iter_count``:    number of iterations

        Raises:
            RuntimeError: If ``solve()`` has not been called yet. CasADi's
                          ``stats()`` returns stale or empty data before any
                          solve has run; this guard surfaces that as an explicit
                          error rather than silent incorrect output.
        """
        if not self._solved:
            raise RuntimeError("No solve has been run yet. Call solve() before stats().")
        return self._last_solver.stats()
