"""
``Evaluator`` -- a model's whole signal graph as one function, without a solve.

A study script that tabulates a latency timeline per configuration row does
not want an optimum; it wants every signal at the leaf values the row names.
``Evaluator`` wires the graph once over MX leaves and keeps one
``ca.Function``: free leaves in, every value out. Calling it is one Function
evaluation, so a table of a thousand rows costs no rebuild.

Two ways in
-----------
``Evaluator(builder, fixed=..., values=...)``
    A declared :class:`~machina.model.Builder`; nothing is compiled, built or
    solved. The leaves are what :meth:`Builder.wire` needs -- every state,
    input and quantity, in that order -- minus the quantities named in
    ``fixed``, which are folded in as constants under the rule of
    :meth:`Builder.build` (finite, no scalar broadcast into a vector, a matrix
    read column-major). A leaf's default is ``values[path]``, else its
    ``Quantity.default``; a leaf with neither must be given at call time.
``Evaluator.from_problem(problem)``
    A compiled :class:`~machina.compiler.Problem`. The leaves are its backend
    variables then its parameters, as the MX symbols the solver itself uses;
    ``FIXED`` quantities are the constants the Problem folded. A variable
    defaults to its initial guess and a parameter to its stored value, both
    read once, when the Evaluator is made. Feeding it a result's ``x_opt``
    and ``p_opt`` reproduces :meth:`Problem.evaluate` exactly.

Outputs
-------
Values only: every leaf in :meth:`Builder.wire` order (states, inputs,
quantities -- fixed ones included, as their constants), then every algebraic
signal, each path once. For a static model that is quantities then algebraic
signals, which is the order and the set :meth:`Problem.evaluate` returns.
State derivatives are not outputs: an Evaluator reports where the model is,
not where it is going; :meth:`Builder.build` gives ``f_system`` for that.
Each value comes back as :meth:`Problem.evaluate` shapes it: a 1-D array for a
column (a scalar is ``(1,)``), ``(rows, cols)`` for a matrix.

A ``None`` value -- in ``values=``, at call time, or in a table row -- is no
value at all and falls through to the leaf's default, as it does in
``Problem.compile``.
"""

from collections.abc import Mapping

import casadi as ca
import numpy as np

from machina.compiler.problem import Problem
from machina.model.builder import Builder
from machina.model.component import numeric
from machina.model.errors import ModelError

__all__ = ["Evaluator"]


class Evaluator:
    """Every signal and quantity of a model at given leaf values, without solving.

    ``builder`` must be declared. ``fixed`` is ``{quantity path: value}``,
    folded in as constants and no longer a leaf; ``values`` is
    ``{leaf path: value}``, the default for that leaf at call time.
    """

    def __init__(self, builder, *, fixed=None, values=None):
        if isinstance(builder, Problem):
            raise TypeError(
                "Evaluator(builder) takes a Builder, and was given a Problem. Use "
                "Evaluator.from_problem(problem) to evaluate a compiled Problem, or "
                "Evaluator(problem.builder, fixed=...) to choose the fixed quantities yourself."
            )
        if not isinstance(builder, Builder):
            raise TypeError(
                f"Evaluator(builder) takes a declared machina.model.Builder, got "
                f"{type(builder).__name__}."
            )
        try:
            # Raises until declare() has run, and is needed anyway for the defaults.
            declared = {path: quantity.default for path, quantity, _ in builder.quantities()}
        except ModelError:
            raise RuntimeError(
                "Evaluator(builder): the Builder has not been declared. Call builder.declare() "
                "first (it returns the builder), or pass a compiled Problem through "
                "Evaluator.from_problem(problem)."
            ) from None

        fixed = dict(fixed or {})
        values = dict(values or {})
        states_inputs = list(builder.state_order) + list(builder.input_order)
        quantities = list(builder.quantity_order)

        for path in fixed:
            if path in states_inputs:
                raise ValueError(
                    f"Evaluator(fixed={{{path!r}: ...}}): {path!r} is a state or an input, and "
                    f"only quantities can be fixed. Pass it at call time, or give it a default "
                    f"with Evaluator(values={{{path!r}: ...}})."
                )
        unknown = sorted(path for path in fixed if path not in quantities)
        if unknown:
            raise ValueError(
                f"Evaluator(fixed=...) names {unknown}, which are not quantities of this model. "
                f"Known quantities: {quantities}."
            )
        constants = {path: _fixed_value(path, fixed[path], builder.shape_of(path))
                     for path in quantities if path in fixed}

        leaves = [path for path in states_inputs + quantities if path not in fixed]
        shapes = {path: tuple(builder.shape_of(path)) for path in leaves}
        for path in values:
            if path in fixed:
                raise ValueError(
                    f"Evaluator(values={{{path!r}: ...}}): {path!r} is also in fixed=, so it is "
                    f"a constant, not a leaf. Drop it from one of the two."
                )
            if path not in shapes:
                raise ValueError(
                    f"Evaluator(values=...) names {path!r}, which is not a leaf of this model. "
                    f"Free leaves: {leaves}."
                )

        defaults = {}
        for path in leaves:
            if values.get(path) is not None:
                defaults[path] = _leaf_value(values[path], shapes[path],
                                             f"Evaluator(values={{{path!r}: ...}})")
            elif declared.get(path) is not None:
                # The declaration's own rule, which lets a scalar default fill a vector.
                defaults[path] = numeric(declared[path], shapes[path],
                                         f"quantity {path!r}: default")

        symbols = {path: ca.MX.sym(path, *shapes[path]) for path in leaves}
        wired = builder.wire({**symbols, **constants})
        outputs = _dedupe(states_inputs + quantities + list(builder.algebraic_order),
                          wired.values)

        self._finish(
            leaves=leaves,
            inputs=[symbols[path] for path in leaves],
            outputs=outputs,
            expressions=[wired.values[path] for path in outputs],
            shapes={path: tuple(builder.shape_of(path)) for path in outputs},
            defaults=defaults,
            fixed={path: _natural(constants[path], constants[path].shape)
                   for path in constants},
            fixed_hint=("it is in fixed= of this Evaluator. Make another Evaluator without it "
                        "in fixed= to vary it."),
            missing_hint="or give it a default with Evaluator(builder, values={...})",
        )

    @classmethod
    def from_problem(cls, problem) -> "Evaluator":
        """The graph of a compiled Problem, over the symbols its solver sees.

        Leaves are the backend variables, then its parameters, in registration
        order; outputs are exactly the paths :meth:`Problem.evaluate` returns.
        """
        if isinstance(problem, Builder):
            raise TypeError(
                "Evaluator.from_problem(problem) takes a compiled Problem, and was given a "
                "Builder. Use Evaluator(builder) for a declared Builder."
            )
        if not isinstance(problem, Problem):
            raise TypeError(
                f"Evaluator.from_problem(problem) takes a machina.compiler.Problem, got "
                f"{type(problem).__name__}."
            )
        if not problem.is_compiled:
            raise RuntimeError(
                "Evaluator.from_problem(problem): the Problem is not compiled. Call "
                "problem.compile(...) first; build() and solve() are not needed."
            )
        builder, backend = problem.builder, problem.backend
        variables, parameters = backend.variable_order(), backend.parameter_order()
        leaves = list(variables) + list(parameters)
        shapes = {name: tuple(backend.shape_of(name)) for name in leaves}

        defaults = {}
        for name in variables:
            defaults[name] = numeric(backend.initial_guess(name), shapes[name],
                                     f"initial guess of {name!r}")
        for name in parameters:
            value = backend.parameter_value(name)
            if value is not None:
                defaults[name] = numeric(value, shapes[name], f"value of parameter {name!r}")

        # Problem.compile refuses states and inputs, so this is Problem.evaluate's order.
        outputs = list(builder.quantity_order)
        outputs += [path for path in builder.algebraic_order if path not in outputs]
        fixed = problem.fixed

        evaluator = cls.__new__(cls)
        evaluator._finish(
            leaves=leaves,
            inputs=[backend.symbol_of(name) for name in leaves],
            outputs=outputs,
            expressions=[problem.expr(path).symbol for path in outputs],
            shapes={path: tuple(builder.shape_of(path)) for path in outputs},
            defaults=defaults,
            fixed={path: np.asarray(value, dtype=float) for path, value in fixed.items()},
            fixed_hint=("it is FIXED in this Problem, folded in as a constant. To vary it, "
                        "evaluate the Problem's builder with the other fixed quantities: "
                        "Evaluator(problem.builder, fixed={...})."),
            missing_hint=("or store a value in the Problem with compile(values={...}) or "
                          "problem.set_value()"),
        )
        return evaluator

    def _finish(self, *, leaves, inputs, outputs, expressions, shapes, defaults, fixed,
                fixed_hint, missing_hint) -> None:
        """One Function over the leaves, and the bookkeeping every call reads."""
        self._leaves = tuple(leaves)
        self._outputs = tuple(outputs)
        self._shapes = dict(shapes)
        for leaf, symbol in zip(leaves, inputs):
            self._shapes[leaf] = (symbol.size1(), symbol.size2())
        self._defaults = defaults
        self._fixed = fixed
        self._fixed_hint = fixed_hint
        self._missing_hint = missing_hint
        # Inputs are named by leaf path, outputs 'out:<path>': CasADi keeps input and output
        # names in one namespace, every leaf is echoed as an output, and no path has a ':'.
        self._function = ca.Function("evaluator", list(inputs),
                                     [_as_mx(expr) for expr in expressions], list(leaves),
                                     [f"out:{path}" for path in outputs])

    # --- properties -----------------------------------------------------------------------

    @property
    def leaves(self) -> tuple:
        """Free leaf paths, in the order ``function`` takes them."""
        return self._leaves

    @property
    def outputs(self) -> tuple:
        """Output paths, in the order ``function`` returns them."""
        return self._outputs

    @property
    def function(self) -> ca.Function:
        """``leaves`` in, each in its declared shape; ``outputs`` out, named ``out:<path>``."""
        return self._function

    @property
    def fixed(self) -> dict:
        """``{quantity path: value}`` folded in as constants; not leaves, still outputs."""
        return {path: value.copy() for path, value in self._fixed.items()}

    # --- evaluation -----------------------------------------------------------------------

    def __call__(self, leaves=None, /, **by_path) -> dict:
        """Every output at the given leaf values, ``{path: ndarray}`` in ``outputs`` order.

        Leaves come as a mapping, as keywords, or both -- a path with a ``/``
        needs the mapping or ``**{"sat/p": ...}``. A leaf left out takes its
        default; one with no default is an error naming it.
        """
        given = {}
        if leaves is not None:
            if not isinstance(leaves, Mapping):
                raise TypeError(
                    f"Evaluator(...)(leaves): the positional argument must be a mapping "
                    f"{{leaf path: value}}, got {type(leaves).__name__}."
                )
            given.update(leaves)
        for path, value in by_path.items():
            if path in given:
                raise ValueError(
                    f"Evaluator(...)(...): {path!r} is given twice, in the mapping and as a "
                    f"keyword. Pass it once."
                )
            given[path] = value
        self._check_names(given)

        args, missing = [], []
        for path in self._leaves:
            shape = self._shapes[path]
            if given.get(path) is not None:
                flat = _leaf_value(given[path], shape, f"Evaluator value for {path!r}")
            elif path in self._defaults:
                flat = self._defaults[path]
            else:
                missing.append(path)
                continue
            args.append(ca.reshape(ca.DM(flat), shape[0], shape[1]))
        if missing:
            raise ValueError(
                f"Evaluator(...)(...): no value for {missing}; they have no default. Pass them "
                f"at call time, e.g. evaluator({{{missing[0]!r}: ...}}), {self._missing_hint}."
            )
        out = self._function.call(args)
        return {path: _natural(value, self._shapes[path])
                for path, value in zip(self._outputs, out)}

    def table(self, rows, *, outputs=None) -> list:
        """One dict per row, in row order: the row's leaf values first, then the outputs.

        ``rows`` is a sequence of ``{leaf path: value}`` mappings, and every key
        of every row must be a free leaf: a label or any other extra key is
        refused like an unknown path, naming the row. Carry labels beside the
        rows instead and pair them afterwards, e.g.
        ``for label, entry in zip(labels, evaluator.table(rows))``.
        ``outputs`` picks output paths (default: all of them, in ``outputs``
        order). A size-1 output comes back as a Python ``float``, anything
        larger as an array. An output that is also a key of the row (a leaf is
        an output too) keeps the row's position and takes the evaluated value.
        """
        names = self._pick_outputs(outputs)
        if isinstance(rows, Mapping):
            raise TypeError(
                "Evaluator.table(rows) takes a sequence of mappings, one per row, and was given "
                "a single mapping. Wrap it in a list, or call the Evaluator directly."
            )
        table = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"Evaluator.table(rows): row {index} is a {type(row).__name__}, not a "
                    f"mapping {{leaf path: value}}."
                )
            extra = [key for key in row if key not in self._shapes]
            if extra:
                raise ValueError(
                    f"Evaluator.table(rows): row {index} has {extra}, which are not leaves of "
                    f"this model. A row holds leaf values only; carry labels beside the rows "
                    f"and pair them with the result, e.g. "
                    f"zip(labels, evaluator.table(rows)). Free leaves: {list(self._leaves)}."
                )
            try:
                values = self(row)
            except ValueError as exc:
                raise ValueError(f"Evaluator.table(rows): row {index}: {exc}") from None
            entry = dict(row)
            for name in names:
                value = values[name]
                entry[name] = float(value.reshape(-1)[0]) if value.size == 1 else value
            table.append(entry)
        return table

    # --- internals ------------------------------------------------------------------------

    def _check_names(self, given: dict) -> None:
        """Every key a free leaf: a fixed quantity and a computed signal say why not."""
        for path in given:
            if path in self._fixed:
                raise ValueError(
                    f"Evaluator(...)(...): {path!r} is not a leaf here -- {self._fixed_hint}"
                )
        unknown = [path for path in given if path not in self._shapes]
        computed = [path for path in given
                    if path in self._shapes and path not in self._leaves]
        if unknown:
            raise ValueError(
                f"Evaluator(...)(...): {unknown} are not leaves of this model. Free leaves: "
                f"{list(self._leaves)}."
            )
        if computed:
            raise ValueError(
                f"Evaluator(...)(...): {computed} are algebraic signals, computed from the "
                f"leaves -- they are outputs, not inputs. Free leaves: {list(self._leaves)}."
            )

    def _pick_outputs(self, outputs) -> list:
        if outputs is None:
            return list(self._outputs)
        if isinstance(outputs, str):
            raise TypeError(
                f"Evaluator.table(outputs={outputs!r}) is a string, not a sequence of paths; "
                f"did you mean outputs=({outputs!r},)?"
            )
        names = list(outputs)
        unknown = [name for name in names if name not in self._outputs]
        if unknown:
            raise ValueError(
                f"Evaluator.table(outputs=...) names {unknown}, which are not outputs of this "
                f"Evaluator. Outputs: {list(self._outputs)}."
            )
        return names

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Evaluator {len(self._leaves)} leaf/leaves, {len(self._outputs)} output(s)>"


# --- module helpers ---------------------------------------------------------------------------


def _leaf_value(value, shape: tuple, what: str) -> np.ndarray:
    """A caller's value as a flat column-major array of ``shape``; a scalar fills only (1, 1)."""
    try:
        flat = numeric(value, shape, what)
    except ModelError as exc:
        raise ValueError(str(exc)) from None
    array = np.asarray(value.full() if isinstance(value, ca.DM) else value, dtype=float)
    if array.ndim == 0 and tuple(shape) != (1, 1):
        raise ValueError(
            f"{what} is a scalar, but the leaf is declared {tuple(shape)}. Pass "
            f"{shape[0] * shape[1]} values; nothing is broadcast."
        )
    return flat


def _fixed_value(path: str, value, shape: tuple):
    """:meth:`Builder.build`'s rule for a fixed quantity: finite, of its shape, column-major."""
    what = f"Evaluator(fixed={{{path!r}: ...}})"
    flat = _leaf_value(value, shape, what)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{what} is not finite.")
    return ca.reshape(ca.DM(flat), shape[0], shape[1])


def _dedupe(paths: list, present: dict) -> list:
    """Each path once, in first-seen order, and only those the wiring produced."""
    out = []
    for path in paths:
        if path in present and path not in out:
            out.append(path)
    return out


def _as_mx(expr):
    """MX passes through; a constant (a fixed leaf, a folded signal) becomes one."""
    return expr if isinstance(expr, ca.MX) else ca.MX(ca.DM(expr))


def _natural(value, shape) -> np.ndarray:
    """``Problem.evaluate``'s convention: 1-D for a column, ``(rows, cols)`` for a matrix."""
    array = np.asarray(ca.DM(value).full(), dtype=float)
    return array if shape[1] > 1 else array.ravel(order="F")
