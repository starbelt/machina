"""
``Problem`` -- the compiler: declared components in, a solvable NLP out.

``machina.model`` deliberately leaves three decisions open. A component says
what it owns, not what the leaf *is*: a duty cycle is a decision variable in
one study, a swept parameter in the next and a pinned constant in the third.
It says what a leaf's nominal value would be, not where the number came from.
And it declares constraints and cost terms without deciding which of them the
problem pays for. ``Problem`` makes those three decisions, in one place, with
a documented precedence, so that a report can say where every number came
from and a rerun with one override changes exactly one thing.

It is also the only thing that calls ``SolverBackend.add_*``. The builder
never sees the backend; the backend never sees a component. ``compile()``
creates one leaf per declared quantity, hands them to :meth:`Builder.wire`,
and registers every wired constraint and cost under its instance path.

Precedence, highest first
-------------------------
role
    ``overrides[path]["role"]``, ``roles[path]``, ``Quantity.role`` unless it
    is ``FLEXIBLE``, ``Quantity.default_role``. The result is never
    ``FLEXIBLE`` -- that is the component saying it has no opinion, not a
    role a solver can register.
value
    ``overrides[path]["value"]``, ``values[path]``, ``values[Quantity.param]``
    (the params-table link), ``Quantity.default``. A ``None`` from any of them
    is no value at all and falls through to the next source. No value from any
    source is an error naming the quantity -- for a variable too, because a
    silent ``x0 = 0`` is how a nonlinear solve ends up at a stationary point
    nobody chose. A default with no provenance code is a warning listing every
    such quantity at once, and ``strict=True`` makes it an error.
bounds, scale
    ``overrides[path][...]``, then the ``Quantity``, then the ``ParamValue``
    the params table supplied. A ``ParamValue``'s ``lo``/``hi`` become bounds
    only where the ``Quantity`` left that one at +-inf, so a component that
    knows its own physical limits keeps them, bound by bound.
provenance, source
    ``overrides[path][...]``, then the ``ParamValue`` -- a number that came
    from the params table is sourced by that table, not by whatever the
    declaration guessed -- then the ``Quantity``.

Which override keys mean anything depends on the role the quantity resolved
to: ``VARIABLE``/``DISCRETE`` take all of them, ``PARAMETER`` takes
``role value provenance source``, ``FIXED`` takes ``role value``. An override
that the role has no use for is an error rather than a key silently dropped.

Phase 3 compiles *static* problems: quantities, algebraic signals, costs and
constraints. A model that declares states or inputs is refused by
``compile()`` naming them, because nothing here integrates them yet.

Lifecycle::

    Problem(components).compile(roles=..., values=...).build().solve()

``compile()`` twice, ``expr()``/``add_cost()``/``build()`` before
``compile()``, or ``solve()`` before ``build()`` each raise ``RuntimeError``
naming the call to make.
"""

import math
import warnings

import casadi as ca
import numpy as np

from machina.model import signals as sig
from machina.model.builder import Builder
from machina.model.component import PROVENANCE_CODES, Role, numeric
from machina.model.descriptor import SymbolDescriptor
from machina.params.values import ParamValue
from machina.solver import SolverBackend

__all__ = ["Problem"]

_QUANTITY_KEYS = ("role", "value", "x0", "lb", "ub", "scale", "provenance", "source")
_CONSTRAINT_KEYS = ("scale",)
_REGISTERED = (Role.VARIABLE, Role.DISCRETE)
_KEYS_FOR_ROLE = {
    Role.VARIABLE: _QUANTITY_KEYS,
    Role.DISCRETE: _QUANTITY_KEYS,
    Role.PARAMETER: ("role", "value", "provenance", "source"),
    Role.FIXED: ("role", "value"),
}


class Problem:
    """A model, compiled onto one :class:`~machina.solver.SolverBackend`.

    ``components`` is what :class:`~machina.model.Builder` takes: a list of
    components, optionally nested in ``Scope(name, [...])``. ``solver``,
    ``solver_opts`` and ``verbose`` are handed to the backend unchanged.
    """

    def __init__(self, components, *, helpers=None, registry=sig.DEFAULT,
                 solver="ipopt", solver_opts=None, verbose=True):
        self._components = components
        self._helpers = dict(helpers or {})
        self._registry = registry
        self._solver = solver
        self._solver_opts = solver_opts
        self._verbose = verbose

        self._backend = None
        self._builder = None
        self._wired = None
        self._quantity_of = {}
        self._roles = {}
        self._fixed = {}
        self._discrete_mode = "native"
        self._compiled = False
        self._eval_fn = None
        self._eval_order = None

    # --- properties -----------------------------------------------------------------------

    @property
    def builder(self) -> Builder:
        """The declared :class:`~machina.model.Builder`, once ``compile()`` has run."""
        self._require_compiled("builder")
        return self._builder

    @property
    def backend(self) -> SolverBackend:
        """The solver backend. Registered by ``compile()``, solvable after ``build()``.

        ``compile()`` registers into a backend of its own and hands it over
        only once every leaf, constraint and cost is in, so a compile that
        raised leaves no half-registered backend behind to trip over.
        """
        self._require_compiled("backend")
        return self._backend

    @property
    def roles(self) -> dict:
        """``{instance path: Role}`` as resolved by ``compile()``, in declaration order."""
        self._require_compiled("roles")
        return dict(self._roles)

    @property
    def fixed(self) -> dict:
        """``{instance path: value}`` for every ``FIXED`` quantity, in declaration order.

        These are the quantities substituted as constants: they are not
        variables, not parameters, and nothing in the backend knows about
        them. :meth:`evaluate` still reports them.
        """
        self._require_compiled("fixed")
        return {path: value.copy() for path, value in self._fixed.items()}

    # --- compile --------------------------------------------------------------------------

    def compile(self, *, roles=None, values=None, overrides=None,
                discrete_mode="native", strict=False) -> "Problem":
        """Declare, resolve every leaf, wire, and register the result.

        Args:
            roles:      ``{quantity path: Role or role name}``.
            values:     ``{quantity path or params name: number, array or ParamValue}``.
                        ``machina.params.values_from(csv)`` can be passed whole:
                        a key naming nothing in this model is ignored when its
                        value is a ``ParamValue`` (a table row no quantity links
                        to), and an error for anything else.
            overrides:  ``{quantity path: {key: value}}`` with keys drawn from
                        ``role value x0 lb ub scale provenance source`` -- and
                        from the subset the resolved role uses -- or
                        ``{constraint path: {"scale": ...}}``.
            discrete_mode: kept for :meth:`build`; ``'native'`` needs an integer
                        plugin, ``'relax'`` solves discrete leaves as continuous.
            strict:     turn the unsourced-default warning into an error.

        Returns:
            ``self``, so the call chains into ``build()``.
        """
        if self._compiled:
            raise RuntimeError(
                "compile() has already run on this Problem. A second compile would register "
                "every leaf twice; build a new Problem, or use set_value()/set_bounds()/fix() "
                "to change data on this one."
            )
        roles = dict(roles or {})
        values = dict(values or {})
        overrides = dict(overrides or {})

        builder = Builder(self._components, self._helpers, registry=self._registry).declare()
        self._builder = builder
        self._quantity_of = {path: (quantity, owner)
                             for path, quantity, owner in builder.quantities()}
        constraint_of = {path: (constraint, owner)
                         for path, constraint, owner in builder.constraints()}
        _check_static(builder)
        self._check_keys(roles, values, overrides, constraint_of)

        backend = SolverBackend(self._solver, self._solver_opts, verbose=self._verbose)
        resolved, fixed, leaves, unsourced = {}, {}, {}, []
        for path, quantity, owner in builder.quantities():
            over = overrides.get(path, {})
            role = _role_for(path, quantity, over, roles)
            _check_override(path, role, over)
            value, param_value, bare_default = _value_for(path, quantity, over, values)
            if bare_default:
                unsourced.append(path)
            resolved[path] = role
            leaves[path] = _register(backend, fixed, path, quantity, owner, role, over,
                                     value, param_value)

        self._report_defaults(unsourced, strict)

        wired = builder.wire(leaves)
        for constraint in wired.constraints:
            over = overrides.get(constraint.path, {})
            declared = constraint_of[constraint.path][0]
            backend.add_constraint(
                constraint.expr, lb=constraint.lb, ub=constraint.ub, name=constraint.path,
                scale=over.get("scale", declared.scale), doc=constraint.doc,
                component=constraint.component)
        for cost in wired.costs:
            backend.add_cost(cost.expr, name=cost.path, weight=cost.weight,
                             doc=cost.doc, component=cost.component)

        self._backend = backend
        self._wired = wired
        self._roles = resolved
        self._fixed = fixed
        self._discrete_mode = discrete_mode
        self._compiled = True
        return self

    def _check_keys(self, roles, values, overrides, constraint_of) -> None:
        """Every key names something this model declared, with the allowed sub-keys."""
        known = self._builder.quantity_order
        param_names = []
        for path in known:
            name = self._quantity_of[path][0].param
            if name is not None and name not in param_names:
                param_names.append(name)

        for key in roles:
            if key not in self._quantity_of:
                raise ValueError(
                    f"compile(roles=...) names {key!r}, which is not a quantity of this model. "
                    f"Quantities: {known}."
                )
        for key, value in values.items():
            if key in self._quantity_of or key in param_names:
                continue
            if isinstance(value, ParamValue):
                continue      # a whole values_from(csv) table: a row nothing links to
            raise ValueError(
                f"compile(values=...) names {key!r}, which is neither a quantity of this "
                f"model nor a params name any Quantity(param=...) links to. Quantities: "
                f"{known}. Params names: {param_names or '(none)'}. Only a ParamValue row "
                f"is ignored when nothing links to it; {value!r} is not one."
            )
        for key, entries in overrides.items():
            is_quantity = key in self._quantity_of
            if not is_quantity and key not in constraint_of:
                raise ValueError(
                    f"compile(overrides=...) names {key!r}, which is neither a quantity nor a "
                    f"constraint of this model. Quantities: {known}. Constraints: "
                    f"{list(constraint_of) or '(none)'}."
                )
            if not isinstance(entries, dict):
                raise ValueError(
                    f"compile(overrides={{{key!r}: ...}}) must be a dict of override keys, got "
                    f"{entries!r}. Write overrides={{{key!r}: {{'value': ...}}}}."
                )
            allowed = _QUANTITY_KEYS if is_quantity else _CONSTRAINT_KEYS
            for entry in entries:
                if entry not in allowed:
                    what = "quantity" if is_quantity else "constraint"
                    raise ValueError(
                        f"compile(overrides={{{key!r}: ...}}) has key {entry!r}, which is not an "
                        f"override of a {what}. Allowed for a {what}: {list(allowed)}."
                    )

    @staticmethod
    def _report_defaults(unsourced, strict) -> None:
        """One message for every quantity that fell back to an unsourced default."""
        if not unsourced:
            return
        message = (
            f"{len(unsourced)} quantity/quantities took a Quantity(default=...) with no "
            f"provenance code: {unsourced}. Nobody chose those numbers for this problem -- pass "
            f"compile(values={{path: ...}}), or give the declaration provenance= and source= so "
            f"a report can say where each came from."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message} compile(strict=True) makes this an error.",
                      UserWarning, stacklevel=3)

    # --- adding to the compiled problem ---------------------------------------------------

    def add_cost(self, expr, name, *, weight=1.0, doc="") -> None:
        """Add a problem-level cost term, on top of the components' declared ones."""
        self._require_compiled("add_cost")
        self._backend.add_cost(_symbol(expr), name=name, weight=weight, doc=doc,
                               component=None)

    def add_constraint(self, expr, lb=-math.inf, ub=math.inf, name=None, *,
                       scale=1.0, doc="") -> None:
        """Add a problem-level constraint, on top of the components' declared ones."""
        self._require_compiled("add_constraint")
        self._backend.add_constraint(_symbol(expr), lb=lb, ub=ub, name=name, scale=scale,
                                     doc=doc, component=None)

    def expr(self, path: str) -> SymbolDescriptor:
        """The compiled expression for any signal or quantity, by instance path.

        The descriptor carries the declared shape, unit and frame, so a
        problem-level cost or constraint can be written against a signal the
        components computed without reaching into the wiring.

        ``symbol`` is an ``MX`` for anything the solver still decides. A
        ``FIXED`` quantity -- and any signal that folds to a constant because
        everything upstream of it is fixed -- comes back as a ``ca.DM``
        instead; :func:`add_cost` and :func:`add_constraint` take either.
        """
        self._require_compiled("expr")
        if path not in self._wired.values:
            raise ValueError(
                f"{path!r} is not a signal or quantity of this model. Known paths: "
                f"{self._value_order()}."
            )
        shape = self._builder.shape_of(path)
        if path in self._quantity_of:
            quantity = self._quantity_of[path][0]
            unit, frame = quantity.unit, quantity.frame
        else:
            signal = self._builder.registry.get(path.rpartition("/")[2])
            unit, frame = signal.unit, signal.frame
        return SymbolDescriptor(
            symbol=self._wired.values[path], name=path, shape=shape,
            semantic_type=_semantic_type(shape),
            frame=None if frame in (None, "none") else frame, units=unit)

    # --- build and solve ------------------------------------------------------------------

    def build(self, opts=None) -> "Problem":
        """Assemble the NLP. Registration is locked from here on."""
        self._require_compiled("build")
        self._backend.build(opts, discrete_mode=self._discrete_mode)
        return self

    def solve(self, *, values=None, warm_start=None):
        """Solve, optionally overriding parameter values for this call only.

        ``values`` is ``{parameter path: value}``; it reaches the backend as
        ``solve(p_val=...)``, so a sweep costs no rebuild. ``warm_start`` is a
        previous :class:`~machina.solver.SolutionResult`.
        """
        self._require_built("solve")
        p_val = None
        if values:
            p_val = {}
            for path, value in values.items():
                self._check_role(path, Role.PARAMETER, "solve(values=...)")
                p_val[path] = value.value if isinstance(value, ParamValue) else value
        return self._backend.solve(p_val=p_val, warm_start=warm_start)

    def evaluate(self, result) -> dict:
        """Every signal and quantity at the solution, by instance path.

        One ``ca.Function`` over the whole graph, built on first use from the
        backend's own ``(x, p)`` vectors, so the values are the ones the
        solver actually saw. ``FIXED`` quantities come back as the constants
        they are.
        """
        self._require_compiled("evaluate")
        order, function = self._evaluator()
        args = [_flat(result.x_opt, self._backend.variable_order())]
        if self._backend.parameter_order():
            args.append(_flat(result.p_opt, self._backend.parameter_order()))
        out = function.call(args)
        return {path: _natural(value, self._builder.shape_of(path))
                for path, value in zip(order, out)}

    def _evaluator(self) -> tuple:
        if self._eval_fn is None:
            expressions = self._backend.nlp_expressions()
            order = self._value_order()
            args = [expressions["x"]]
            if self._backend.parameter_order():
                args.append(expressions["p"])
            self._eval_fn = ca.Function("evaluate", args,
                                        [_symbol(self._wired.values[p]) for p in order])
            self._eval_order = order
        return self._eval_order, self._eval_fn

    def _value_order(self) -> list:
        """Every wired path, once, in a fixed order: quantities, then algebraic signals.

        ``compile()`` refuses a model with states or inputs, so those two
        orders are empty here by construction.
        """
        order = []
        for group in (self._builder.quantity_order, self._builder.algebraic_order):
            for path in group:
                if path not in order and path in self._wired.values:
                    order.append(path)
        return order

    # --- editable data --------------------------------------------------------------------

    def set_value(self, path: str, value) -> None:
        """Store a new value for a ``PARAMETER`` quantity; every later solve uses it."""
        self._require_compiled("set_value")
        self._check_role(path, Role.PARAMETER, "set_value")
        self._backend.set_parameter(path, value.value if isinstance(value, ParamValue) else value)

    def set_bounds(self, path: str, lb=None, ub=None) -> None:
        """Change a variable's bounds. No rebuild is needed, before or after ``build()``."""
        self._require_compiled("set_bounds")
        self._check_role(path, _REGISTERED, "set_bounds")
        self._backend.set_bounds(path, lb=lb, ub=ub)

    def fix(self, path: str, value) -> None:
        """Pin a variable at ``value`` by its bounds. Undo with :meth:`unfix`."""
        self._require_compiled("fix")
        self._check_role(path, _REGISTERED, "fix")
        self._backend.fix(path, value.value if isinstance(value, ParamValue) else value)

    def unfix(self, path: str) -> None:
        """Restore the bounds a variable was compiled with."""
        self._require_compiled("unfix")
        self._check_role(path, _REGISTERED, "unfix")
        self._backend.unfix(path)

    # --- internals ------------------------------------------------------------------------

    def _check_role(self, path: str, wanted, what: str) -> None:
        wanted = wanted if isinstance(wanted, tuple) else (wanted,)
        names = [role.value for role in wanted]
        eligible = [p for p in self._builder.quantity_order if self._roles[p] in wanted]
        if path not in self._roles:
            raise ValueError(
                f"{what}: {path!r} is not a quantity of this model. Paths compiled as "
                f"{names}: {eligible or '(none)'}."
            )
        if self._roles[path] not in wanted:
            raise ValueError(
                f"{what}: {path!r} compiled as {self._roles[path].value!r}, and {what} is for "
                f"{names}. Recompile with roles={{{path!r}: Role.{wanted[0].name}}}, or use a "
                f"path from {eligible or '(none)'}."
            )

    def _require_compiled(self, what: str) -> None:
        if not self._compiled:
            raise RuntimeError(f"call compile() before {what}().")

    def _require_built(self, what: str) -> None:
        self._require_compiled(what)
        if not self._backend.is_built:
            raise RuntimeError(f"call build() before {what}().")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if not self._compiled:
            return "<Problem not compiled>"
        state = "built" if self._backend.is_built else "compiled"
        return (f"<Problem {state}, {self._backend.n_x} variable element(s), "
                f"{self._backend.n_p} parameter element(s)>")


# --- module helpers ---------------------------------------------------------------------------


def _check_static(builder) -> None:
    """Phase 3 compiles quantities; a state or an input needs a transcription first."""
    if not builder.state_order and not builder.input_order:
        return
    raise ValueError(
        f"Problem compiles static problems (quantities only) in Phase 3; this model "
        f"declares states {builder.state_order} and inputs {builder.input_order}. "
        f"Simulate it with machina.sim, or fix the states."
    )


def _check_override(path, role, over) -> None:
    """The override keys the resolved role can use, and the two metadata values."""
    allowed = _KEYS_FOR_ROLE[role]
    for key in over:
        if key not in allowed:
            raise ValueError(
                f"compile(overrides={{{path!r}: ...}}) has key {key!r}, which means nothing "
                f"for a quantity that compiled as {role.value!r}. Allowed for "
                f"{role.value!r}: {list(allowed)}. Drop the key, or compile the quantity as "
                f"a role that uses it with roles={{{path!r}: ...}}."
            )
    if over.get("provenance") is not None and over["provenance"] not in PROVENANCE_CODES:
        raise ValueError(
            f"compile(overrides={{{path!r}: {{'provenance': {over['provenance']!r}}}}}) is "
            f"not a provenance code. Pass None or one of {list(PROVENANCE_CODES)} "
            f"(documented, physics, estimate, assumption, measured)."
        )
    if over.get("source") is not None and not isinstance(over["source"], str):
        raise ValueError(
            f"compile(overrides={{{path!r}: {{'source': {over['source']!r}}}}}) must be None "
            f"or a string saying where the number came from."
        )


def _register(backend, fixed, path, quantity, owner, role, over, value, param_value):
    """One leaf: a backend variable, a backend parameter, or a constant."""
    shape = quantity.shape
    provenance = _pick(over, "provenance", param_value, "provenance", quantity.provenance)
    source = _pick(over, "source", param_value, "source", quantity.source)

    if role is Role.FIXED:
        flat = numeric(value, shape, f"fixed quantity {path!r}", finite=True)
        fixed[path] = flat.reshape(shape, order="F") if shape[1] > 1 else flat
        return ca.reshape(ca.DM(flat), shape[0], shape[1])

    if role is Role.PARAMETER:
        return backend.add_parameter(
            path, shape, value=value, unit=quantity.unit, doc=quantity.doc,
            provenance=provenance, source=source, component=owner, frame=quantity.frame)

    lb = _bound(path, "lb", over, quantity.lb, shape, param_value, "lo")
    ub = _bound(path, "ub", over, quantity.ub, shape, param_value, "hi")
    return backend.add_variable(
        path, shape, lb=lb, ub=ub, initial_guess=over.get("x0", value),
        discrete=role is Role.DISCRETE, scale=over.get("scale", quantity.scale),
        unit=quantity.unit, doc=quantity.doc, provenance=provenance, source=source,
        component=owner, frame=quantity.frame)


def _role_for(path, quantity, over, roles) -> Role:
    """The role precedence, with the one rule that FLEXIBLE never survives it."""
    if "role" in over:
        role = _as_role(over["role"], f"compile(overrides={{{path!r}: {{'role': ...}}}})")
    elif path in roles:
        role = _as_role(roles[path], f"compile(roles={{{path!r}: ...}})")
    else:
        role = quantity.role if quantity.role is not Role.FLEXIBLE else quantity.default_role
    if role is Role.FLEXIBLE:
        raise ValueError(
            f"{path!r} was given the role FLEXIBLE, which means 'the component has no opinion' "
            f"and is not something the solver can register. Pass one of "
            f"{[r.value for r in Role if r is not Role.FLEXIBLE]}."
        )
    return role


def _as_role(value, what: str) -> Role:
    if isinstance(value, Role):
        return value
    try:
        return Role(value)
    except ValueError:
        raise ValueError(
            f"{what}: {value!r} is not a role. Pass a Role member or one of "
            f"{[r.value for r in Role]}."
        ) from None


def _value_for(path, quantity, over, values) -> tuple:
    """``(value, ParamValue or None, took a bare default)``, highest source first.

    A ``None`` is no value: an override, a ``values`` entry or a ``ParamValue``
    carrying one falls through to the next source, and a chain of them reaches
    the same error as a key nobody wrote.
    """
    offered = []
    if "value" in over:
        offered.append(over["value"])
    if path in values:
        offered.append(values[path])
    if quantity.param is not None and quantity.param in values:
        offered.append(values[quantity.param])
    for candidate in offered:
        value, param_value, bare_default = _unwrap(candidate)
        if value is not None:
            return value, param_value, bare_default
    if quantity.default is not None:
        return quantity.default, None, quantity.provenance is None
    raise ValueError(
        f"no value for quantity {path!r} -- a None from any source is no value at all. Pass "
        f"compile(values={{{path!r}: ...}}), or compile(overrides={{{path!r}: "
        f"{{'value': ...}}}}), or give its Quantity a default=. A variable needs one too: "
        f"an unchosen initial guess is a silent zero."
    )


def _unwrap(value) -> tuple:
    if isinstance(value, ParamValue):
        return value.value, value, False
    return value, None, False


def _pick(over, key, param_value, attribute, fallback):
    """Override, then what the params table said, then what the declaration said."""
    if key in over:
        return over[key]
    if param_value is not None and getattr(param_value, attribute) is not None:
        return getattr(param_value, attribute)
    return fallback


def _bound(path, key, over, declared, shape, param_value, attribute):
    """A ``ParamValue``'s limit applies only where the ``Quantity`` left the bound open."""
    what = f"{path!r} {key} bound"
    if key in over:
        numeric(over[key], shape, what)
        return over[key]
    if param_value is None:
        return declared
    flat = numeric(declared, shape, what)
    open_bound = np.isneginf(flat) if attribute == "lo" else np.isposinf(flat)
    return getattr(param_value, attribute) if open_bound.all() else declared


def _semantic_type(shape) -> str:
    if shape == (1, 1):
        return "scalar"
    return "vector" if shape[1] == 1 else "matrix"


def _symbol(expr):
    """MX passes through; a constant (a FIXED leaf, a folded expression) becomes one."""
    return expr if isinstance(expr, ca.MX) else ca.MX(ca.DM(expr))


def _flat(by_name: dict, order: list) -> np.ndarray:
    """A backend vector, column-major, from a result's by-name values."""
    if not order:
        return np.zeros(0)
    return np.concatenate([np.asarray(by_name[name], dtype=float).ravel(order="F")
                           for name in order])


def _natural(value, shape) -> np.ndarray:
    """A CasADi DM back to a 1-D array for a column, ``(rows, cols)`` for a matrix."""
    array = np.asarray(ca.DM(value).full(), dtype=float)
    return array if shape[1] > 1 else array.ravel(order="F")
