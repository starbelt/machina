"""
``Builder`` -- assemble declared components into one symbolic graph.

Adopted from icarus-dynamics ``model/builder.py``, which kept the good ideas
of its own predecessor and closed the holes that let three silent bugs live
there. What is kept verbatim in spirit:

1. **Shapes come from the signal registry, and every built output is checked
   against them.** A ``(1, 1)`` standing in for a ``(3, 3)`` is only an error
   if something knows a ``(3, 3)`` belonged.
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
    ``h`` outputs come back as constraints and ``J`` outputs as cost terms;
    the compiler, not the component, registers them.
"""

from dataclasses import dataclass

import casadi as ca

from machina.model import signals as sig
from machina.model.component import Component, Scope
from machina.model.errors import ModelError

__all__ = [
    "Builder", "Wired", "WiredConstraint", "WiredCost", "Placed", "ModelError",
]


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
        self._tree = list(components)

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
        self._signal_of: dict = {}
        self._producers: dict = {}

        self._flatten_tree()

    # --- flattening -----------------------------------------------------------------------

    def _flatten_tree(self) -> None:
        """Depth-first in list order. The walk order is the scope ABI."""
        def walk(members, scope):
            for member in members:
                if isinstance(member, Scope):
                    walk(member.components, _join(scope, member.name))
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
        """Validate every interface and lay out the vectors. Constructs no symbols."""
        self._check_names_and_helpers()
        self._resolve_references()
        self._collect_producers()
        self._lay_out()
        self._check_aggregation()
        self._check_states()
        self._order = self._topological_order()
        self._declared = True
        return self

    def _check_names_and_helpers(self) -> None:
        for item in self.placed:
            dec = item.declaration
            for _, reference in dec.consumed_refs():
                self.registry.get(_leaf(reference))
            for name in dec.produced():
                self.registry.get(name)
            missing = [h for h in dec.helpers if h not in self.helpers]
            if missing:
                raise ModelError(
                    f"{item.path}: undeclared helper(s) {missing}. Available: "
                    f"{sorted(self.helpers)}."
                )
            both = [n for n in dec.derivatives if n in dec.produces]
            if both:
                raise ModelError(
                    f"{item.path}: {both} appear in both derivatives and produces. A signal is "
                    f"either integrated or computed, not both."
                )

    def _resolve_references(self) -> None:
        """Map every (component, local name) to an instance path.

        Produced paths are known first, because a consumed name resolves to
        the innermost enclosing scope that produces it. Inputs are produced by
        nobody, so they resolve to the component's own scope; share one with
        an absolute reference.
        """
        produced: dict = {}
        for item in self.placed:
            for name in item.declaration.produced():
                produced.setdefault(_join(item.scope, name), []).append(item.path)

        for item in self.placed:
            dec = item.declaration
            resolved: dict = {}
            for group in ("states", "algebraic"):
                for local, reference in getattr(dec, group):
                    resolved[local] = self._lexical(item.scope, reference, produced)
            for local, reference in dec.inputs:
                resolved[local] = (reference[1:] if reference.startswith("/")
                                   else _join(item.scope, reference))
            for quantity in dec.quantities:
                path = _join(item.scope, quantity.name)
                owner = self._quantity_of.get(path)
                if owner is not None:
                    raise ModelError(
                        f"{item.path} and {owner[0]} both own a quantity at {path!r}. "
                        f"Quantities are named in their component's scope -- rename one, or "
                        f"put the components in separate scopes."
                    )
                if path in produced:
                    raise ModelError(
                        f"{item.path}: quantity {quantity.name!r} collides with the signal "
                        f"{path!r} produced by {produced[path][0]}. A quantity is owned and "
                        f"private; rename it or produce it as a signal instead."
                    )
                self._quantity_of[path] = (item.path, quantity)
                resolved[quantity.name] = path
            self._resolved[item.path] = resolved

        for path in produced:
            self._signal_of[path] = _leaf(path)

    def _lexical(self, scope: str, reference: str, produced: dict) -> str:
        """``scope/name``, then each parent, then the root; ``/a/b`` is absolute."""
        if reference.startswith("/"):
            return reference[1:]
        candidates = []
        walk = scope
        while True:
            candidates.append(_join(walk, reference))
            if not walk:
                break
            walk = walk.rpartition("/")[0]
        for candidate in candidates:
            if candidate in produced:
                return candidate
        return candidates[0]

    def _collect_producers(self) -> None:
        producers: dict = {}
        for item in self.placed:
            for name in item.declaration.produced():
                producers.setdefault(_join(item.scope, name), []).append(item.path)
        self._producers = producers

    def _lay_out(self) -> None:
        """Order by ``(registry index, scope index)``, per the vault design.

        The registry walk comes first, so reordering the component list cannot
        silently re-lay-out a vector; the scope index breaks ties between two
        instances of one signal, and it is the order the caller wrote.
        """
        # Every scope a path could resolve into: the root, each component scope in flattening
        # order, and each of their ancestors (a consumed name may resolve to one).
        scope_index = {"": 0}
        for item in self.placed:
            parts = item.scope.split("/") if item.scope else []
            for depth in range(1, len(parts) + 1):
                scope_index.setdefault("/".join(parts[:depth]), len(scope_index))
        registry_index = {name: i for i, name in enumerate(self.registry.all())}

        derivative_paths, algebraic_paths, input_paths = {}, {}, {}
        for item in self.placed:
            dec = item.declaration
            resolved = self._resolved[item.path]
            for name in dec.derivatives:
                derivative_paths.setdefault(_join(item.scope, name), item.scope)
            for name in dec.produces:
                algebraic_paths.setdefault(_join(item.scope, name), item.scope)
            for local, _ in dec.algebraic:
                algebraic_paths.setdefault(resolved[local], _scope_of(resolved[local]))
            for local, _ in dec.inputs:
                input_paths.setdefault(resolved[local], _scope_of(resolved[local]))

        def order(paths):
            keyed = [(registry_index[_leaf(p)], scope_index.get(s, len(scope_index)), p)
                     for p, s in paths.items()]
            keyed.sort(key=lambda k: (k[0], k[1], k[2]))
            return [p for _, _, p in keyed]

        self.state_order = order(derivative_paths)
        self.algebraic_order = order({p: s for p, s in algebraic_paths.items()
                                      if p not in derivative_paths})
        self.input_order = order({p: s for p, s in input_paths.items()
                                  if p not in derivative_paths and p not in algebraic_paths})
        self.quantity_order = [path for item in self.placed
                               for path in (_join(item.scope, q.name)
                                            for q in item.declaration.quantities)]

    def _check_aggregation(self) -> None:
        for path in sorted(self._producers):
            declared = self.registry.get(_leaf(path))
            owners = self._producers[path]
            if declared.aggregation is sig.Aggregation.UNIQUE and len(owners) > 1:
                raise ModelError(
                    f"{len(owners)} components produce {path!r}: {sorted(owners)}. "
                    f"{_leaf(path)!r} is declared UNIQUE -- either one of them is wrong, or "
                    f"the signal should be SUM."
                )

        # A SUM signal with no producer is ZERO: that is the identity of addition, and it is
        # what lets a reduced model be expressed by dropping a component. A UNIQUE signal with
        # no producer stays an error, because there is no identity to fall back on -- an absent
        # mass is zero, and a zero mass is a division by zero inside the integrator.
        self.unproduced_sums = [
            p for p in self.algebraic_order
            if p not in self._producers
            and self.registry.get(_leaf(p)).aggregation is sig.Aggregation.SUM
        ]
        unmet = sorted(p for p in self.algebraic_order
                       if p not in self._producers and p not in self.unproduced_sums)
        if unmet:
            raise ModelError(
                f"no component produces {unmet}, and they are declared UNIQUE so there is no "
                f"zero to fall back on. Add a producer, or declare the signal SUM."
            )

    def _check_states(self) -> None:
        for item in self.placed:
            resolved = self._resolved[item.path]
            floating = sorted({resolved[local] for local, _ in item.declaration.states
                               if resolved[local] not in self.state_order})
            if floating:
                raise ModelError(
                    f"{item.path} reads state(s) {floating} that nothing integrates. A state "
                    f"with no derivative is frozen at its initial value -- if that is "
                    f"intended, it is an input, not a state."
                )

    def _topological_order(self) -> list:
        """Producers of an algebraic signal before its consumers. Kahn's, kept stable.

        Ties break by the order the components were written, so the build
        order -- and therefore the expression graph -- is a function of the
        model definition alone.
        """
        edges = {item.path: [] for item in self.placed}
        indegree = {item.path: 0 for item in self.placed}

        for item in self.placed:
            resolved = self._resolved[item.path]
            for local, _ in item.declaration.algebraic:
                for producer in self._producers.get(resolved[local], []):
                    if producer != item.path and item.path not in edges[producer]:
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
        and quantity. The values may be SX, MX or DM; nothing here inspects
        the type, which is what lets one component serve both back ends.
        """
        self._require_declared("wire")
        expected = list(self.state_order) + list(self.input_order) + list(self.quantity_order)
        missing = [p for p in expected if p not in leaves]
        if missing:
            raise ModelError(
                f"wire() was not given leaves for {missing}. Every state, input and quantity "
                f"needs one; algebraic signals are computed, not supplied."
            )
        extra = sorted(p for p in leaves if p not in set(expected))
        if extra:
            raise ModelError(
                f"wire() was given leaves for {extra}, which are not states, inputs or "
                f"quantities of this model. Check for a typo in the instance path."
            )

        values = {p: leaves[p] for p in expected}
        for path in self.unproduced_sums:
            values[path] = ca.DM.zeros(*self.registry.get(_leaf(path)).shape)

        xdot_terms = {p: [] for p in self.state_order}
        constraints, costs = [], []

        for item in self._order:
            dec = item.declaration
            resolved = self._resolved[item.path]
            built = item.component.build({h: self.helpers[h] for h in dec.helpers})
            _check_keys(item.path, built)

            # Only what it declared. A component reading an undeclared signal gets a missing
            # argument rather than a working model with an invisible dependency.
            visible = {local: values[resolved[local]] for local in dec.consumed()
                       if resolved[local] in values}

            if "g" in built:
                expected_g = tuple((n, self.registry.get(n).shape) for n in dec.produces)
                for name, expr in zip(dec.produces,
                                      self._call(item, built["g"], expected_g, visible)):
                    path = _join(item.scope, name)
                    values[path] = expr if path not in values else values[path] + expr
            if "f" in built:
                expected_f = tuple((n, self.registry.get(n).shape) for n in dec.derivatives)
                for name, expr in zip(dec.derivatives,
                                      self._call(item, built["f"], expected_f, visible)):
                    xdot_terms[_join(item.scope, name)].append(expr)
            if "h" in built:
                expected_h = tuple((c.name, c.shape) for c in dec.constraints)
                for con, expr in zip(dec.constraints,
                                     self._call(item, built["h"], expected_h, visible)):
                    constraints.append(WiredConstraint(
                        _join(item.scope, con.name), expr, con.lb, con.ub, con.shape,
                        con.doc, item.path))
            if "J" in built:
                expected_j = tuple((c.name, (1, 1)) for c in dec.costs)
                for cost, expr in zip(dec.costs,
                                      self._call(item, built["J"], expected_j, visible)):
                    costs.append(WiredCost(
                        _join(item.scope, cost.name), expr, cost.weight, cost.doc, item.path))

        xdot = {}
        for path in self.state_order:
            terms = xdot_terms[path]
            if not terms:  # pragma: no cover - _check_states makes this unreachable
                raise ModelError(f"nothing integrates {path!r}.")
            xdot[path] = _column(sum(terms[1:], terms[0]))

        return Wired(values, xdot, tuple(constraints), tuple(costs))

    def _call(self, item: Placed, fn, expected: tuple, values: dict) -> list:
        """Apply one component's Function, checking both ends against the declaration.

        Inputs are matched by name against what the component said it reads;
        every output's shape is checked against what it said it produces.
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
        ``fixed`` substitutes numeric values for named quantities.
        """
        self._require_declared("build")
        fixed = dict(fixed or {})
        unknown = sorted(p for p in fixed if p not in set(self.quantity_order))
        if unknown:
            raise ModelError(
                f"fixed={unknown} are not quantities of this model. Known quantities: "
                f"{self.quantity_order}."
            )
        free = [p for p in self.quantity_order if p not in fixed]

        x = ca.SX.sym("x", self._width(self.state_order))
        u = ca.SX.sym("u", self._width(self.input_order))
        q = ca.SX.sym("q", sum(self._quantity_of[p][1].size for p in free))

        leaves = {}
        leaves.update(self._unpack(x, self.state_order))
        leaves.update(self._unpack(u, self.input_order))
        leaves.update(self._unpack(q, free))
        for path, value in fixed.items():
            shape = self._quantity_of[path][1].shape
            leaves[path] = ca.DM(value) if shape == (1, 1) else ca.reshape(ca.DM(value), *shape)

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

    # --- layout ---------------------------------------------------------------------------

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

    def quantities(self) -> tuple:
        """``(instance path, Quantity, owning component path)``, in declaration order."""
        self._require_declared("quantities")
        return tuple((p, self._quantity_of[p][1], self._quantity_of[p][0])
                     for p in self.quantity_order)

    def constraints(self) -> tuple:
        """``(instance path, Constraint, owning component path)``, in declaration order."""
        self._require_declared("constraints")
        return tuple((_join(i.scope, c.name), c, i.path)
                     for i in self.placed for c in i.declaration.constraints)

    def costs(self) -> tuple:
        """``(instance path, Cost, owning component path)``, in declaration order."""
        self._require_declared("costs")
        return tuple((_join(i.scope, c.name), c, i.path)
                     for i in self.placed for c in i.declaration.costs)

    def producers(self) -> dict:
        """Instance path to the component paths that produce it."""
        self._require_declared("producers")
        return {p: tuple(v) for p, v in self._producers.items()}

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
        return sum(self._quantity_of[p][1].size for p in self.quantity_order)

    # --- internals ------------------------------------------------------------------------

    def _require_declared(self, what: str) -> None:
        if not self._declared:
            raise ModelError(f"call declare() before {what}().")

    def _size(self, path: str) -> int:
        if path in self._quantity_of:
            return self._quantity_of[path][1].size
        return self.registry.get(_leaf(path)).size

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


def _check_keys(component_path: str, built) -> None:
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
