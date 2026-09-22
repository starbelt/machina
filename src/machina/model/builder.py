"""
``Builder`` -- assemble declared components into one symbolic graph.

Adopted from icarus-dynamics ``model/builder.py``, which kept the good ideas
of its own predecessor and closed the holes that let three silent bugs live
there. What is kept:

1. **Shapes come from the signal registry, and every built output is checked
   against them.** A ``(1, 1)`` standing in for a ``(3, 3)`` is only an error
   if something knows a ``(3, 3)`` belonged. machina checks the other end
   too: every argument a component's Function takes must match the shape of
   the signal or quantity it is bound to, so a scalar cannot be broadcast
   silently into a vector argument.
2. **Aggregation is declared per signal, not implied.** ``SUM`` for forces
   and budgets, ``UNIQUE`` for states and owned outputs.
3. **Algebraic signals are substituted, not carried.** The topological sort
   proves there is no loop, so every algebraic signal can be written in terms
   of the leaves and folded in. The result is a plain ODE, which is why
   :mod:`machina.sim` can be a textbook RK4 with nothing to know about the
   model.
4. **Nothing iterates a set.** Every ordering comes from the registry's
   declaration order or from the component list the caller wrote. Sets appear
   only as membership tests or inside ``sorted()``. Python randomises string
   hashing per process, so iterating one would make the expression graph
   run-dependent and fail the two-process determinism gate.

**Every instance path has exactly one role.** A path is a *state* (some
component integrates it), an *algebraic* signal (some component computes it,
or it is read and unproduced), an *input* (read, produced by nobody) or a
*quantity* (owned by one component). A consumed name listed under the wrong
group -- a computed signal under ``states``, a state under ``algebraic``, a
produced signal under ``inputs`` -- is refused, naming the group it belongs
in. Letting it through gave wrong numbers: a missing topological edge, a
reader seeing a partial ``SUM``, an input replaced by zeros.

Three things are added for machina:

``Scope``
    Components nest under named prefixes, so two satellites are ordinary
    instances rather than a special case. Consumed names resolve lexically:
    the component's own scope, then each parent, then the root; a leading
    ``/`` is absolute. Producers always produce into their own scope.

``quantities``
    Leaves a component owns -- decision variables and solver parameters.
    They become a third vector, and the built ``f_system`` keeps the
    two-input ``(x, u)`` signature when there are none, so existing icarus
    components and :func:`machina.sim.build_step_function` are unaffected.

``wire()``
    The build pass, exposed. It is symbol-type agnostic: the NLP compiler
    seeds it with MX leaves from the solver backend, the simulation path with
    SX leaves, and both run the same checks over the same graph structure.
    ``h`` outputs come back as constraints and ``J`` outputs as cost terms,
    in declaration order; the compiler, not the component, registers them.
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from machina.model import signals as sig
from machina.model.component import Component, Scope, _sequence
from machina.model.errors import ModelError, SignalError

__all__ = [
    "Builder", "Wired", "WiredConstraint", "WiredCost", "Placed", "ModelError",
]

_KEY_FOR = (("derivatives", "f"), ("produces", "g"), ("constraints", "h"), ("costs", "J"))


@dataclass(frozen=True)
class Placed:
    """One component instance, with the scope it was flattened into."""

    component: Component
    scope: str
    declaration: object

    @property
    def path(self) -> str:
        """Instance path of the component itself, e.g. ``sat_a/kinematics``."""
        return _join(self.scope, self.component.name)


@dataclass(frozen=True)
class WiredConstraint:
    """One ``h`` output, ready for ``SolverBackend.add_constraint``."""

    path: str
    expr: object
    lb: object
    ub: object
    shape: tuple
    doc: str
    component: str


@dataclass(frozen=True)
class WiredCost:
    """One ``J`` output, ready for ``SolverBackend.add_cost``."""

    path: str
    expr: object
    weight: object
    doc: str
    component: str


@dataclass(frozen=True)
class Wired:
    """The graph, once leaves have been supplied.

    ``values`` holds every signal and quantity by instance path; ``xdot``
    holds the summed derivative of every state by instance path.
    ``constraints`` and ``costs`` are in declaration order -- the order of
    :meth:`Builder.constraints` and :meth:`Builder.costs` -- whatever order
    the components ran in.
    """

    values: dict
    xdot: dict
    constraints: tuple
    costs: tuple


class Builder:
    """Declare, then wire or build.

    ``declare()`` validates every interface and fixes the vector layouts
    without constructing a single symbol. ``wire()`` and ``build()`` then run
    the components in topological order.
    """

    def __init__(self, components, helpers=None, *, registry=None):
        self.registry = sig.DEFAULT if registry is None else registry
        self.helpers = dict(helpers or {})
        self._tree = _sequence(components, "components")

        self.placed: list = []
        self.state_order: list = []
        self.algebraic_order: list = []
        self.input_order: list = []
        self.quantity_order: list = []
        self.unproduced_sums: list = []

        self._declared = False
        self._order: list = []
        self._resolved: dict = {}
        self._quantity_of: dict = {}
        self._integrated_by: dict = {}
        self._computed_by: dict = {}
        self._scopes: dict = {}

        self._flatten_tree()

    # --- flattening -----------------------------------------------------------------------

    def _flatten_tree(self) -> None:
        """Depth-first in list order. The walk order is the scope ABI."""
        self._scopes = {"": 0}

        def walk(members, scope):
            for member in members:
                if isinstance(member, Scope):
                    inner = _join(scope, member.name)
                    self._scopes.setdefault(inner, len(self._scopes))
                    walk(member.components, inner)
                elif isinstance(member, Component):
                    self.placed.append(Placed(member, scope, member.declare()))
                else:
                    raise ModelError(
                        f"{member!r} is neither a Component nor a Scope. A model is a list of "
                        f"components, optionally nested in Scope(name, [...])."
                    )

        walk(self._tree, "")
        if not self.placed:
            raise ModelError("a model needs at least one component.")

        seen: dict = {}
        for item in self.placed:
            if item.path in seen:
                raise ModelError(
                    f"two components share the instance path {item.path!r}. Names reach "
                    f"generated code and instance paths must be unique -- pass a "
                    f"distinguishing suffix, e.g. Aero('left') and Aero('right'), or put them "
                    f"in separate scopes."
                )
            seen[item.path] = item

    # --- declaration pass -----------------------------------------------------------------

    def declare(self) -> "Builder":
        """Validate every interface and lay out the vectors. Constructs no symbols.

        Idempotent: a second call returns the builder unchanged.
        """
        if self._declared:
            return self
        self._check_interfaces()
        self._index_producers()
        self._resolve_references()
        self._check_paths()
        self._lay_out()
        self._check_unproduced()
        self._order = self._topological_order()
        self._declared = True
        return self

    def _check_interfaces(self) -> None:
        """Every name exists, every helper exists, every quantity frame exists."""
        for item in self.placed:
            dec = item.declaration
            for group in ("states", "algebraic", "inputs"):
                for local, reference in getattr(dec, group):
                    self._signal(_leaf(reference),
                                 f"{item.path}: {group} entry {local!r} -> {reference!r}")
            for group in ("derivatives", "produces"):
                for name in getattr(dec, group):
                    self._signal(name, f"{item.path}: {group} entry {name!r}")
            for quantity in dec.quantities:
                try:
                    self.registry.frame(quantity.frame)
                except SignalError as exc:
                    raise SignalError(f"{item.path}: quantity {quantity.name!r}: {exc}") from None
            missing = [h for h in dec.helpers if h not in self.helpers]
            if missing:
                raise ModelError(
                    f"{item.path}: undeclared helper(s) {missing}. Available: "
                    f"{sorted(self.helpers)}."
                )

    def _signal(self, name: str, where: str):
        try:
            return self.registry.get(name)
        except SignalError as exc:
            raise SignalError(f"{where}: {exc}") from None

    def _index_producers(self) -> None:
        """Who integrates and who computes each path, and the conflicts between them."""
        for item in self.placed:
            for name in item.declaration.derivatives:
                self._integrated_by.setdefault(_join(item.scope, name), []).append(item.path)
            for name in item.declaration.produces:
                self._computed_by.setdefault(_join(item.scope, name), []).append(item.path)

        for path in self._integrated_by:
            if path in self._computed_by:
                raise ModelError(
                    f"{path!r} is integrated by {self._integrated_by[path]} and computed by "
                    f"{self._computed_by[path]}. A signal is either a state (a derivative) or an "
                    f"algebraic output (produces), never both."
                )
        for label, table in (("integrate", self._integrated_by), ("produce", self._computed_by)):
            for path in sorted(table):
                owners = table[path]
                if len(owners) > 1 and self.registry.get(_leaf(path)).aggregation \
                        is sig.Aggregation.UNIQUE:
                    raise ModelError(
                        f"{len(owners)} components {label} {path!r}: {sorted(owners)}. "
                        f"{_leaf(path)!r} is declared UNIQUE -- either one of them is wrong, "
                        f"or the signal should be SUM."
                    )

    def _resolve_references(self) -> None:
        """Map every (component, local name) to an instance path, checking its role."""
        for item in self.placed:
            dec = item.declaration
            resolved: dict = {}
            for local, reference in dec.states:
                resolved[local] = self._resolve_state(item, local, reference)
            for local, reference in dec.algebraic:
                resolved[local] = self._resolve_algebraic(item, local, reference)
            for local, reference in dec.inputs:
                resolved[local] = self._resolve_input(item, local, reference)
            for quantity in dec.quantities:
                path = _join(item.scope, quantity.name)
                owner = self._quantity_of.get(path)
                if owner is not None:
                    raise ModelError(
                        f"{item.path} and {owner[0]} both own a quantity at {path!r}. "
                        f"Quantities are named in their component's scope -- rename one, or "
                        f"put the components in separate scopes."
                    )
                producer = self._producer_of(path)
                if producer:
                    raise ModelError(
                        f"{item.path}: quantity {quantity.name!r} collides with the signal "
                        f"{path!r} produced by {producer[0]}. A quantity is owned and "
                        f"private; rename it or produce it as a signal instead."
                    )
                self._quantity_of[path] = (item.path, quantity)
                resolved[quantity.name] = path
            self._resolved[item.path] = resolved

    def _candidates(self, item: Placed, local: str, group: str, reference: str) -> list:
        """Lexical candidates, innermost first; an absolute reference has exactly one."""
        if reference.startswith("/"):
            candidates = [reference[1:]]
        else:
            candidates, walk = [], item.scope
            while True:
                candidates.append(_join(walk, reference))
                if not walk:
                    break
                walk = walk.rpartition("/")[0]
        if "/" in reference.lstrip("/") or reference.startswith("/"):
            scopes = [c for c in candidates if _scope_of(c) in self._scopes]
            if not scopes:
                raise ModelError(
                    f"{item.path}: {group} entry {local!r} -> {reference!r} names a scope that "
                    f"does not exist. Scopes in this model: {list(self._scopes)[1:] or '(none)'}."
                )
        return candidates

    def _producer_of(self, path: str) -> list:
        return self._integrated_by.get(path) or self._computed_by.get(path) or []

    def _resolve_state(self, item: Placed, local: str, reference: str) -> str:
        for path in self._candidates(item, local, "states", reference):
            if path in self._integrated_by:
                return path
            if path in self._computed_by:
                raise ModelError(
                    f"{item.path} lists {reference!r} under states, but {path!r} is computed by "
                    f"{self._computed_by[path]} (a 'produces' output), not integrated. List it "
                    f"under algebraic."
                )
        raise ModelError(
            f"{item.path} reads state {reference!r}, which nothing integrates. A state with no "
            f"derivative is frozen at its initial value -- if that is intended, it is an "
            f"input, not a state."
        )

    def _resolve_algebraic(self, item: Placed, local: str, reference: str) -> str:
        candidates = self._candidates(item, local, "algebraic", reference)
        for path in candidates:
            if path in self._computed_by:
                if item.path in self._computed_by[path]:
                    raise ModelError(
                        f"{item.path} reads {reference!r} and also produces {path!r}: an "
                        f"algebraic loop through one component. Compute the value inside "
                        f"build() instead of reading it back."
                    )
                return path
            if path in self._integrated_by:
                raise ModelError(
                    f"{item.path} lists {reference!r} under algebraic, but {path!r} is a state "
                    f"integrated by {self._integrated_by[path]}. List it under states."
                )
        if "/" in reference:
            raise ModelError(
                f"{item.path}: algebraic entry {local!r} -> {reference!r} names a path nothing "
                f"produces. An explicit path names one instance, so there is no zero to fall "
                f"back on -- check the spelling, or read the signal by its bare name."
            )
        return candidates[0]

    def _resolve_input(self, item: Placed, local: str, reference: str) -> str:
        candidates = self._candidates(item, local, "inputs", reference)
        for path in candidates:
            producer = self._producer_of(path)
            if producer:
                group = "states" if path in self._integrated_by else "algebraic"
                raise ModelError(
                    f"{item.path} lists {reference!r} under inputs, but {path!r} is produced by "
                    f"{producer}. An input is exogenous -- nothing in the model produces it. "
                    f"List it under {group}, or give the input its own signal name."
                )
        return candidates[0]

    def _check_paths(self) -> None:
        """Cross-component path rules: one role per path, unique declared outputs."""
        consumed_as: dict = {}
        for item in self.placed:
            resolved = self._resolved[item.path]
            for group in ("states", "algebraic", "inputs"):
                for local, _ in getattr(item.declaration, group):
                    consumed_as.setdefault(resolved[local], []).append((group, item.path))

        for path, uses in consumed_as.items():
            if path in self._quantity_of:
                owner = self._quantity_of[path][0]
                raise ModelError(
                    f"{uses[0][1]} reads {path!r} as a signal, but it is a quantity owned by "
                    f"{owner}. A quantity is private to its owner; produce a signal from it if "
                    f"another component needs the value."
                )
            groups = []
            for group, _ in uses:
                if group not in groups:
                    groups.append(group)
            if "inputs" in groups and len(groups) > 1:
                readers = [f"{reader} ({group})" for group, reader in uses]
                raise ModelError(
                    f"{path!r} is read as an input by some components and as a signal by "
                    f"others: {readers}. A path has one role; pick one."
                )

        for label in ("constraint", "cost"):
            seen: dict = {}
            for item in self.placed:
                for entry in getattr(item.declaration, f"{label}s"):
                    path = _join(item.scope, entry.name)
                    if path in seen:
                        raise ModelError(
                            f"{item.path} and {seen[path]} both declare the {label} {path!r}. "
                            f"{label.capitalize()} names are unique per scope -- rename one."
                        )
                    seen[path] = item.path

    def _lay_out(self) -> None:
        """Order by ``(registry index, scope index)``, per the vault design.

        The registry walk comes first, so reordering the component list cannot
        silently re-lay-out a vector; the scope index breaks ties between two
        instances of one signal, and it is the order the caller wrote.
        """
        registry_index = {name: i for i, name in enumerate(self.registry.all())}

        states = list(self._integrated_by)
        algebraic = list(self._computed_by)
        inputs = []
        for item in self.placed:
            resolved = self._resolved[item.path]
            for local, _ in item.declaration.algebraic:
                if resolved[local] not in algebraic:
                    algebraic.append(resolved[local])
            for local, _ in item.declaration.inputs:
                if resolved[local] not in inputs:
                    inputs.append(resolved[local])

        def order(paths):
            keyed = [(registry_index[_leaf(p)], self._scopes.get(_scope_of(p), len(self._scopes)),
                      p) for p in paths]
            keyed.sort()
            return [p for _, _, p in keyed]

        self.state_order = order(states)
        self.algebraic_order = order(algebraic)
        self.input_order = order(inputs)
        self.quantity_order = [path for item in self.placed
                               for path in (_join(item.scope, q.name)
                                            for q in item.declaration.quantities)]

    def _check_unproduced(self) -> None:
        # A SUM signal with no producer is ZERO: that is the identity of addition, and it is
        # what lets a reduced model be expressed by dropping a component. A UNIQUE signal with
        # no producer stays an error, because there is no identity to fall back on -- an absent
        # mass is zero, and a zero mass is a division by zero inside the integrator.
        self.unproduced_sums = [
            p for p in self.algebraic_order
            if p not in self._computed_by
            and self.registry.get(_leaf(p)).aggregation is sig.Aggregation.SUM
        ]
        unmet = [p for p in self.algebraic_order
                 if p not in self._computed_by and p not in self.unproduced_sums]
        if unmet:
            readers = [item.path for item in self.placed
                       if any(self._resolved[item.path][local] in unmet
                              for local, _ in item.declaration.algebraic)]
            raise ModelError(
                f"no component produces {unmet} (read by {readers}), and they are declared "
                f"UNIQUE so there is no zero to fall back on. Add a producer, or declare the "
                f"signal SUM."
            )

    def _topological_order(self) -> list:
        """Producers of an algebraic signal before its consumers. Kahn's, kept stable.

        Ties break by the order the components were written, so the build
        order -- and therefore the expression graph -- is a function of the
        model definition alone. Only computed signals create edges: states and
        inputs are leaves, known before anything runs.
        """
        edges = {item.path: [] for item in self.placed}
        indegree = {item.path: 0 for item in self.placed}

        for item in self.placed:
            resolved = self._resolved[item.path]
            for local, _ in item.declaration.algebraic:
                for producer in self._computed_by.get(resolved[local], []):
                    if item.path not in edges[producer]:
                        edges[producer].append(item.path)
                        indegree[item.path] += 1

        queue = [item.path for item in self.placed if indegree[item.path] == 0]
        ordered = []
        while queue:
            path = queue.pop(0)
            ordered.append(path)
            for downstream in edges[path]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)

        if len(ordered) != len(self.placed):
            stuck = sorted(p for p in indegree if p not in ordered)
            raise ModelError(
                f"algebraic dependency cycle among {stuck}. Two components each waiting on the "
                f"other's output is an implicit system -- it needs a solver, not a sort."
            )
        by_path = {item.path: item for item in self.placed}
        return [by_path[p] for p in ordered]

    # --- wiring ---------------------------------------------------------------------------

    def wire(self, leaves: dict) -> Wired:
        """Run every component over the supplied leaves, in topological order.

        ``leaves`` is keyed by instance path and must cover every state, input
        and quantity, each in its declared shape. The values may be SX, MX or
        DM; nothing here depends on the type, which is what lets one component
        serve both back ends.
        """
        self._require_declared("wire")
        expected = list(self.state_order) + list(self.input_order) + list(self.quantity_order)
        missing = [p for p in expected if p not in leaves]
        if missing:
            raise ModelError(
                f"wire() was not given leaves for {missing}. Every state, input and quantity "
                f"needs one; algebraic signals are computed, not supplied."
            )
        expected_set = set(expected)
        extra = sorted(p for p in leaves if p not in expected_set)
        if extra:
            raise ModelError(
                f"wire() was given leaves for {extra}, which are not states, inputs or "
                f"quantities of this model. Check for a typo in the instance path."
            )
        for path in expected:
            got, want = _shape_of(leaves[path]), self._shape(path)
            if got != want:
                raise ModelError(
                    f"wire(): the leaf for {path!r} has shape {got}, but {path!r} is declared "
                    f"{want}. Pass a value of the declared shape; nothing is broadcast."
                )

        values = {p: leaves[p] for p in expected}
        for path in self.unproduced_sums:
            values[path] = ca.DM.zeros(*self.registry.get(_leaf(path)).shape)

        xdot_terms = {p: [] for p in self.state_order}
        constraints_of, costs_of = {}, {}

        for item in self._order:
            dec = item.declaration
            resolved = self._resolved[item.path]
            built = item.component.build({h: self.helpers[h] for h in dec.helpers})
            _check_keys(item.path, built, dec)

            # Only what it declared. A component reading an undeclared signal gets a missing
            # argument rather than a working model with an invisible dependency.
            visible = {local: values[resolved[local]] for local in dec.consumed()}
            shapes = {local: self._shape(resolved[local]) for local in dec.consumed()}

            if "g" in built:
                expected_g = tuple((n, self.registry.get(n).shape) for n in dec.produces)
                for name, expr in zip(dec.produces,
                                      self._call(item, built["g"], expected_g, visible, shapes)):
                    path = _join(item.scope, name)
                    values[path] = expr if path not in values else values[path] + expr
            if "f" in built:
                expected_f = tuple((n, self.registry.get(n).shape) for n in dec.derivatives)
                for name, expr in zip(dec.derivatives,
                                      self._call(item, built["f"], expected_f, visible, shapes)):
                    xdot_terms[_join(item.scope, name)].append(expr)
            if "h" in built:
                expected_h = tuple((c.name, c.shape) for c in dec.constraints)
                constraints_of[item.path] = [
                    WiredConstraint(_join(item.scope, con.name), expr, con.lb, con.ub, con.shape,
                                    con.doc, item.path)
                    for con, expr in zip(dec.constraints,
                                         self._call(item, built["h"], expected_h, visible,
                                                    shapes))]
            if "J" in built:
                expected_j = tuple((c.name, (1, 1)) for c in dec.costs)
                costs_of[item.path] = [
                    WiredCost(_join(item.scope, cost.name), expr, cost.weight, cost.doc,
                              item.path)
                    for cost, expr in zip(dec.costs,
                                          self._call(item, built["J"], expected_j, visible,
                                                     shapes))]

        xdot = {path: _column(sum(xdot_terms[path][1:], xdot_terms[path][0]))
                for path in self.state_order}
        # Declaration order, not execution order, so h and J line up with constraints()/costs().
        constraints = tuple(c for item in self.placed for c in constraints_of.get(item.path, ()))
        costs = tuple(c for item in self.placed for c in costs_of.get(item.path, ()))
        return Wired(values, xdot, constraints, costs)

    def _call(self, item: Placed, fn, expected: tuple, values: dict, shapes: dict) -> list:
        """Apply one component's Function, checking both ends against the declaration.

        Inputs are matched by name against what the component said it reads,
        and each argument's shape against the signal or quantity it is bound
        to; every output's shape is checked against what it said it produces.
        """
        if fn.n_out() != len(expected):
            raise ModelError(
                f"{item.path}: {fn.name()} returns {fn.n_out()} output(s) but declares "
                f"{len(expected)} ({[n for n, _ in expected]}). They are matched positionally."
            )

        args = []
        for i in range(fn.n_in()):
            key = fn.name_in(i)
            if key not in values:
                raise ModelError(
                    f"{item.path}: {fn.name()} takes an argument named {key!r}, which the "
                    f"component does not declare that it reads. Add it to the declaration, or "
                    f"rename the argument -- they are matched by name, not position."
                )
            takes = (fn.size1_in(i), fn.size2_in(i))
            if takes != tuple(shapes[key]):
                raise ModelError(
                    f"{item.path}: {fn.name()} takes {key!r} as {takes}, but it is declared "
                    f"{tuple(shapes[key])}. Fix the ca.SX.sym shape or the declaration -- "
                    f"CasADi would otherwise broadcast or reshape it silently."
                )
            args.append(values[key])

        # Function.call() always returns a list, including for a zero-input or single-output
        # Function -- calling one positionally returns a dict of outputs instead.
        out = list(fn.call(args))

        for (name, want), expr in zip(expected, out):
            got = (expr.size1(), expr.size2())
            if got != tuple(want):
                raise ModelError(
                    f"{item.path}: {name} is declared {tuple(want)} but {fn.name()} produced "
                    f"{got}. A shape that silently collapses to (1, 1) is usually a Function "
                    f"result indexed with [0] -- that takes element [0, 0], not output 0."
                )
        return out

    # --- the ODE build --------------------------------------------------------------------

    def build(self, *, fixed: dict = None) -> dict:
        """Compose the components into ``f_system`` and ``g_system`` in SX.

        The signature is ``f_system(x, u)`` when the model has no free
        quantities and ``f_system(x, u, q)`` when it has some, so a model made
        only of icarus-style components keeps the two-input ABI that
        :func:`machina.sim.build_step_function` and the exporters expect.
        ``fixed`` substitutes numeric values for named quantities; each value
        must have the quantity's size (a matrix is read column-major).
        ``h_system`` and ``J_system`` are added when the model declares
        constraints or costs, with rows in declaration order.
        """
        self._require_declared("build")
        fixed = dict(fixed or {})
        quantity_set = set(self.quantity_order)
        unknown = sorted(p for p in fixed if p not in quantity_set)
        if unknown:
            raise ModelError(
                f"fixed={unknown} are not quantities of this model. Known quantities: "
                f"{self.quantity_order}."
            )
        free = [p for p in self.quantity_order if p not in fixed]

        x = ca.SX.sym("x", self._width(self.state_order))
        u = ca.SX.sym("u", self._width(self.input_order))
        q = ca.SX.sym("q", self._width(free))

        leaves = {}
        leaves.update(self._unpack(x, self.state_order))
        leaves.update(self._unpack(u, self.input_order))
        leaves.update(self._unpack(q, free))
        for path in self.quantity_order:
            if path in fixed:
                leaves[path] = self._fixed_value(path, fixed[path])

        wired = self.wire(leaves)

        args = [x, u] if not free else [x, u, q]
        names = ["x", "u"] if not free else ["x", "u", "q"]
        xdot = ca.vertcat(*[wired.xdot[p] for p in self.state_order])
        algebraic = [_column(wired.values[p]) for p in self.algebraic_order]

        out = {
            "f": ca.Function("f_system", args, [xdot], names, ["xdot"]),
            "g": ca.Function("g_system", args,
                             [ca.vertcat(*algebraic)] if algebraic else [],
                             names, ["z"] if algebraic else []),
        }
        if wired.constraints:
            out["h"] = ca.Function("h_system", args,
                                   [ca.vertcat(*[_column(c.expr) for c in wired.constraints])],
                                   names, ["h"])
        if wired.costs:
            out["J"] = ca.Function("J_system", args,
                                   [ca.vertcat(*[c.expr for c in wired.costs])], names, ["J"])
        return out

    def _fixed_value(self, path: str, value):
        shape = self._shape(path)
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ModelError(f"fixed[{path!r}] must be numeric, got {value!r}.") from None
        if array.size != shape[0] * shape[1] or (array.ndim == 2 and array.shape != shape):
            raise ModelError(
                f"fixed[{path!r}] has shape {array.shape}, but {path!r} is declared {shape}. "
                f"Pass exactly that many values; nothing is broadcast."
            )
        return ca.reshape(ca.DM(array.ravel(order="F")), shape[0], shape[1])

    # --- introspection --------------------------------------------------------------------

    def layout(self, order) -> dict:
        """Where each instance path sits in its vector.

        The one place that knows, so callers pack and unpack by name rather
        than by counting.
        """
        out, at = {}, 0
        for path in order:
            size = self._size(path)
            out[path] = slice(at, at + size)
            at += size
        return out

    @property
    def order(self) -> list:
        """Component instance paths in the order ``wire()`` runs them."""
        self._require_declared("order")
        return [item.path for item in self._order]

    def shape_of(self, path: str) -> tuple:
        """The declared shape of a state, input, algebraic signal or quantity path."""
        return self._shape(path)

    def quantities(self) -> tuple:
        """``(instance path, Quantity, owning component path)``, in declaration order."""
        self._require_declared("quantities")
        return tuple((p, self._quantity_of[p][1], self._quantity_of[p][0])
                     for p in self.quantity_order)

    def constraints(self) -> tuple:
        """``(instance path, Constraint, owning component path)``, in declaration order."""
        return tuple((_join(i.scope, c.name), c, i.path)
                     for i in self.placed for c in i.declaration.constraints)

    def costs(self) -> tuple:
        """``(instance path, Cost, owning component path)``, in declaration order."""
        return tuple((_join(i.scope, c.name), c, i.path)
                     for i in self.placed for c in i.declaration.costs)

    def producers(self) -> dict:
        """Instance path to the component paths that integrate or compute it."""
        self._require_declared("producers")
        out = {}
        for item in self.placed:
            for name in item.declaration.produced():
                out.setdefault(_join(item.scope, name), []).append(item.path)
        return {p: tuple(v) for p, v in out.items()}

    def resolved(self, component_path: str) -> dict:
        """Local name to instance path, for one component instance."""
        self._require_declared("resolved")
        return dict(self._resolved[component_path])

    @property
    def nx(self) -> int:
        return self._width(self.state_order)

    @property
    def nu(self) -> int:
        return self._width(self.input_order)

    @property
    def nq(self) -> int:
        return self._width(self.quantity_order)

    # --- internals ------------------------------------------------------------------------

    def _require_declared(self, what: str) -> None:
        if not self._declared:
            raise ModelError(f"call declare() before {what}().")

    def _size(self, path: str) -> int:
        shape = self._shape(path)
        return shape[0] * shape[1]

    def _shape(self, path: str) -> tuple:
        if path in self._quantity_of:
            return self._quantity_of[path][1].shape
        return self.registry.get(_leaf(path)).shape

    def _width(self, order) -> int:
        return sum(self._size(p) for p in order)

    def _unpack(self, vec, order) -> dict:
        out, at = {}, 0
        for path in order:
            shape = self._shape(path)
            size = shape[0] * shape[1]
            flat = vec[at:at + size]
            out[path] = ca.reshape(flat, shape[0], shape[1]) if shape[1] > 1 else flat
            at += size
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "declared" if self._declared else "not declared"
        return f"<Builder {len(self.placed)} component(s), {state}>"


# --- module helpers ---------------------------------------------------------------------------


def _join(scope: str, name: str) -> str:
    return f"{scope}/{name}" if scope else name


def _leaf(path: str) -> str:
    return path.rpartition("/")[2]


def _scope_of(path: str) -> str:
    return path.rpartition("/")[0]


def _column(expr):
    return expr if expr.is_column() else ca.reshape(expr, expr.size1() * expr.size2(), 1)


def _shape_of(value) -> tuple:
    """(rows, cols) of a leaf: CasADi types as they are, numbers as (1, 1), 1-D arrays as columns."""
    if hasattr(value, "size1") and hasattr(value, "size2"):
        return (value.size1(), value.size2())
    array = np.asarray(value)
    if array.ndim == 0:
        return (1, 1)
    if array.ndim == 1:
        return (array.shape[0], 1)
    return tuple(array.shape)


def _check_keys(component_path: str, built, dec) -> None:
    if not isinstance(built, dict):
        raise ModelError(
            f"{component_path}: build() must return a dict of ca.Function keyed 'f', 'g', "
            f"'h' or 'J', got {type(built).__name__}."
        )
    unknown = sorted(k for k in built if k not in ("f", "g", "h", "J"))
    if unknown:
        raise ModelError(
            f"{component_path}: build() returned unknown key(s) {unknown}. Valid keys are "
            f"'f' (derivatives), 'g' (produced signals), 'h' (constraints) and 'J' (costs)."
        )
    for group, key in _KEY_FOR:
        if getattr(dec, group) and key not in built:
            raise ModelError(
                f"{component_path} declares {group} {list(getattr(dec, group))} but build() "
                f"returned no {key!r} Function. A declared output that is never built would "
                f"silently vanish from the model."
            )
