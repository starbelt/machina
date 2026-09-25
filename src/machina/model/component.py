"""
``Component`` -- one piece of a model, declaring what it reads and what it
contributes -- plus the things it may own or emit.

The declare/build split is the good idea shared by both lineages: a component
states its interface without constructing any symbols, so the builder can lay
out vectors, resolve dependencies and detect conflicts *before* anything is
built. Only then does ``build()`` run with the symbols already sliced out.

This is icarus-dynamics' ``Declaration`` with three additions from machina's
April 2026 prototype:

``quantities``
    Leaves the component **owns** rather than reads: an orbital element, a
    power cap, a gain. A quantity is *private* to its component -- it is not
    a registry signal, it carries its own shape and unit, and only its owner
    sees it. A component that wants to share one produces a signal from it.
    That keeps "coupling is a signal dependency" true without exception, and
    keeps the decision-variable namespace out of the global signal namespace.

``constraints`` / ``costs``
    Declared ``h`` and ``J`` outputs. The component that knows the physics
    owns its feasibility constraints; the problem decides which cost terms
    enter the objective and with what weight (Decision Log #20 and #34).

Consumed names resolve lexically against the enclosing scopes. A consumed
entry is either a bare signal name (``"mass"``) or an explicit
``(local_name, reference)`` pair, which is how a component reads two
instances of one signal::

    Declaration(algebraic=(("r_a", "/sat_a/r_eci"), ("r_b", "/sat_b/r_eci")))

The local name is what the component's ``ca.Function`` calls its argument,
so it must be a Python identifier; the reference may be a scoped or absolute
instance path. Produced names are always bare: a component produces into its
own scope and nowhere else.
"""

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass, field
from enum import Enum

import casadi as ca
import numpy as np

from machina.model.errors import ModelError
from machina.units import is_si, is_well_formed

PROVENANCE_CODES = ("D", "P", "E", "A", "M")

__all__ = [
    "Role", "Quantity", "Constraint", "Cost", "Declaration", "Component", "Scope",
]


class Role(Enum):
    """What a quantity becomes when the problem is compiled.

    ``FLEXIBLE`` means the component has no opinion and the problem decides,
    falling back to ``Quantity.default_role``.
    """

    FLEXIBLE = "flexible"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    DISCRETE = "discrete"
    FIXED = "fixed"


@dataclass(frozen=True)
class Quantity:
    """A leaf the component owns. Replaces April's ``QuantityDeclaration``.

    ``param`` links the quantity to a :mod:`machina.params` declaration by
    name, so its value, bounds and documentation can come from the params CSV
    with provenance attached.

    ``scale`` is the quantity's nominal magnitude. The compiler passes it to
    ``SolverBackend.add_variable(scale=)``, so the solver sees ``x / scale``
    while every caller stays in physical units (vault: Solver Backend,
    Decision Log #53). It is a scalar or an array of the declared shape,
    strictly positive; the default ``1.0`` means unscaled.
    """

    name: str
    shape: tuple[int, int] = (1, 1)
    unit: str = "1"
    doc: str = ""
    role: Role = Role.FLEXIBLE
    default_role: Role = Role.VARIABLE
    default: object = None
    lb: object = -math.inf
    ub: object = math.inf
    frame: str = "none"
    provenance: str = None
    source: str = None
    param: str = None
    scale: object = 1.0

    def __post_init__(self):
        _check_identifier(self.name, "quantity name")
        object.__setattr__(self, "shape", _shape(self.shape, f"quantity {self.name!r}"))
        if not isinstance(self.role, Role) or not isinstance(self.default_role, Role):
            raise ModelError(
                f"quantity {self.name!r}: role and default_role must be Role members, got "
                f"{self.role!r} and {self.default_role!r}."
            )
        if self.default_role is Role.FLEXIBLE:
            raise ModelError(
                f"quantity {self.name!r}: default_role is the fallback when role is FLEXIBLE, "
                f"so it cannot itself be FLEXIBLE."
            )
        what = f"quantity {self.name!r}"
        _check_unit(self.unit, what)
        if not isinstance(self.frame, str) or not self.frame.isidentifier():
            raise ModelError(f"{what}: frame must be a declared frame name, got {self.frame!r}.")
        if self.provenance is not None and self.provenance not in PROVENANCE_CODES:
            raise ModelError(
                f"{what}: provenance {self.provenance!r} is not one of "
                f"{' '.join(PROVENANCE_CODES)} (documented, physics, estimate, assumption, "
                f"measured)."
            )
        for label in ("source", "param", "doc"):
            value = getattr(self, label)
            if value is not None and not isinstance(value, str):
                raise ModelError(f"{what}: {label} must be a string, got {value!r}.")
        _check_bounds(self.lb, self.ub, self.shape, what)
        _check_scale(self.scale, self.shape, what)
        if self.default is not None:
            numeric(self.default, self.shape, f"{what}: default")

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True)
class Constraint:
    """A declared ``h`` output. Bounds are in the component's own units.

    ``scale`` is the nominal magnitude of the constraint rows. The compiler
    passes it to ``SolverBackend.add_constraint(scale=)``, so the solver sees
    ``g / scale`` while every caller stays in physical units (vault: Solver
    Backend, Decision Log #53). It is a scalar or an array of the declared
    shape, strictly positive; the default ``1.0`` means unscaled.
    """

    name: str
    shape: tuple[int, int] = (1, 1)
    lb: object = -math.inf
    ub: object = math.inf
    doc: str = ""
    scale: object = 1.0

    def __post_init__(self):
        _check_identifier(self.name, "constraint name")
        object.__setattr__(self, "shape", _shape(self.shape, f"constraint {self.name!r}"))
        if self.shape[1] != 1:
            raise ModelError(
                f"constraint {self.name!r}: shape {self.shape} must be a column; the solver "
                f"stacks constraints into one vector."
            )
        _check_bounds(self.lb, self.ub, self.shape, f"constraint {self.name!r}")
        _check_scale(self.scale, self.shape, f"constraint {self.name!r}")

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True)
class Cost:
    """A declared ``J`` output: one scalar candidate term for the objective.

    ``weight`` is the component's own suggestion. The problem applies it and
    may override it; it may also be a solver parameter, so a weight can be
    swept without rebuilding.
    """

    name: str
    doc: str = ""
    weight: object = 1.0

    def __post_init__(self):
        _check_identifier(self.name, "cost name")
        weight = self.weight
        if isinstance(weight, (ca.SX, ca.MX)):
            if weight.numel() != 1:
                raise ModelError(
                    f"cost {self.name!r}: a symbolic weight must be scalar, got shape "
                    f"{weight.shape}."
                )
            return
        # A number, or anything holding exactly one number (a DM, a 0-d array).
        if isinstance(weight, (bool, np.bool_, str, bytes, list, tuple)) or weight is None:
            value = None
        else:
            try:
                array = np.asarray(weight.full() if isinstance(weight, ca.DM) else weight,
                                   dtype=float)
                value = float(array.reshape(-1)[0]) if array.size == 1 else None
            except (TypeError, ValueError):
                value = None
        if value is None:
            raise ModelError(
                f"cost {self.name!r}: weight must be a number or a scalar CasADi expression "
                f"(a solver parameter, to sweep it), got {weight!r}."
            )
        if not math.isfinite(value):
            raise ModelError(f"cost {self.name!r}: weight {weight!r} is not finite.")


@dataclass(frozen=True)
class Declaration:
    """What a component reads, owns and contributes.

    The contract with ``build()`` is positional: ``build()["f"]`` returns
    outputs in ``derivatives`` order, ``["g"]`` in ``produces`` order,
    ``["h"]`` in ``constraints`` order and ``["J"]`` in ``costs`` order.
    Inputs are matched **by name** against the local names declared here.
    """

    # Consumed. Each entry is a name, or an (local_name, reference) pair.
    states: tuple = ()
    algebraic: tuple = ()
    inputs: tuple = ()
    helpers: tuple = ()

    # Owned.
    quantities: tuple = ()

    # Produced, into this component's own scope.
    derivatives: tuple = ()
    produces: tuple = ()
    constraints: tuple = ()
    costs: tuple = ()

    # Filled by __post_init__: (local_name, reference) for states/algebraic/inputs.
    _refs: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for group in _GROUPS:
            object.__setattr__(self, group, _sequence(getattr(self, group), group))
        refs = {}
        for group in ("states", "algebraic", "inputs"):
            pairs = tuple(_ref(entry, group) for entry in getattr(self, group))
            object.__setattr__(self, group, pairs)
            for local, reference in pairs:
                if local in refs:
                    raise ModelError(
                        f"a component reads {local!r} twice. Give one of them an explicit "
                        f"local name, e.g. (\"{local}_b\", \"{reference}\")."
                    )
                refs[local] = reference
        for group in ("helpers", "derivatives", "produces"):
            for name in getattr(self, group):
                _check_identifier(name, f"{group[:-1]} name")
            _check_unique(getattr(self, group), group)
        for group, kind in (("quantities", Quantity), ("constraints", Constraint),
                            ("costs", Cost)):
            for item in getattr(self, group):
                if not isinstance(item, kind):
                    raise ModelError(
                        f"{group} must hold {kind.__name__} objects, got {item!r}. "
                        f"Write {group}=({kind.__name__}('name', ...),)."
                    )
            _check_unique([item.name for item in getattr(self, group)], group)
        for quantity in self.quantities:
            if quantity.name in refs:
                raise ModelError(
                    f"quantity {quantity.name!r} shadows a signal the component also reads. "
                    f"A quantity is owned and private; rename one of them."
                )
            refs[quantity.name] = None
        object.__setattr__(self, "_refs", refs)

    def consumed_refs(self) -> tuple:
        """``(local_name, reference)`` for every consumed signal, in declaration order."""
        return self.states + self.algebraic + self.inputs

    def consumed(self) -> tuple:
        """Local names of everything visible inside ``build()``: signals then quantities."""
        return tuple(local for local, _ in self.consumed_refs()) + \
            tuple(q.name for q in self.quantities)

    def produced(self) -> tuple:
        """Signal names this component produces, in declaration order."""
        return self.derivatives + self.produces


class Component(ABC):
    """One piece of a model.

    Subclasses implement :meth:`declare` and :meth:`build`. ``build()``
    returns any of the keys ``f`` (derivatives), ``g`` (produced signals),
    ``h`` (constraints) and ``J`` (cost terms); all are optional, and a
    component returning ``{}`` is legal and contributes nothing.

    Functions are built in SX. Whether they are wired with SX leaves (the
    simulation path) or MX leaves (the NLP path) is the builder's business.
    """

    def __init__(self, name: str = None):
        cls = type(self).__name__
        self.name = _snake(cls if name is None else f"{cls}_{name}")
        if not self.name.isidentifier():
            raise ModelError(
                f"component name {self.name!r} must be a Python identifier; names become "
                f"instance-path segments and reach generated code."
            )

    @abstractmethod
    def declare(self) -> Declaration:
        """The interface. Must construct no symbols -- it runs before layout."""

    @abstractmethod
    def build(self, helpers: dict) -> dict:
        """The physics, as ``ca.Function`` objects keyed ``f``, ``g``, ``h``, ``J``.

        ``helpers`` holds exactly the names declared under
        ``Declaration.helpers`` and nothing else, so a component cannot
        quietly grow a dependency it did not declare.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r}>"


_GROUPS = ("states", "algebraic", "inputs", "helpers", "quantities", "derivatives",
           "produces", "constraints", "costs")


@dataclass(frozen=True)
class Scope:
    """A named subtree. Its members' instance paths are prefixed with its name.

    Two satellites are ``Scope("sat_a", [...])`` and ``Scope("sat_b", [...])``
    with leaves ``sat_a/p`` and ``sat_b/p``. Scopes nest.
    """

    name: str
    components: tuple = ()

    def __post_init__(self):
        _check_identifier(self.name, "scope name")
        object.__setattr__(self, "components",
                           _sequence(self.components, f"scope {self.name!r} components"))
        if not self.components:
            raise ModelError(f"scope {self.name!r} is empty; a scope with nothing in it is a typo.")


# --- helpers ----------------------------------------------------------------------------------


def _check_identifier(name, what: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise ModelError(
            f"{what} {name!r} must be a Python identifier: these names become CasADi argument "
            f"names and instance-path segments."
        )


def _is_integer(value) -> bool:
    """An int or numpy integer, but not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _shape(shape, what: str) -> tuple:
    if _is_integer(shape):
        shape = (shape, 1)
    ok = (isinstance(shape, (tuple, list)) and len(shape) == 2
          and all(_is_integer(d) for d in shape))
    if not ok:
        raise ModelError(
            f"{what}: shape must be an int or a (rows, cols) pair of ints, got {shape!r}."
        )
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ModelError(f"{what}: shape {(rows, cols)} must be positive in both dimensions.")
    return (rows, cols)


def _sequence(value, group: str) -> tuple:
    """A declaration group as a tuple, refusing the two silent mistakes.

    A bare string is the missing one-tuple comma -- ``("mass")`` is ``"mass"``
    -- and would be iterated character by character. A set has no order, and
    outputs are matched to names by position, so a set would make the model
    change with ``PYTHONHASHSEED``.
    """
    if isinstance(value, str):
        raise ModelError(
            f"{group}={value!r} is a string, not a tuple; did you mean ({value!r},)? "
            f"A one-element tuple needs the trailing comma."
        )
    if isinstance(value, (Set, dict)):
        raise ModelError(
            f"{group} is a {type(value).__name__}. Its order would change from process to "
            f"process, and outputs are matched positionally -- pass a tuple or list in the "
            f"order build() returns them."
        )
    if not isinstance(value, (tuple, list)):
        raise ModelError(f"{group} must be a tuple or list, got {type(value).__name__}.")
    return tuple(value)


def _check_unique(names, group: str) -> None:
    seen = []
    for name in names:
        if name in seen:
            raise ModelError(f"{group} lists {name!r} twice.")
        seen.append(name)


def _check_unit(unit, what: str) -> None:
    if not isinstance(unit, str) or not unit:
        raise ModelError(f'{what}: unit is mandatory; use "1" for dimensionless.')
    if not is_well_formed(unit):
        raise ModelError(f"{what}: unit {unit!r} is not a well-formed unit string.")
    if not is_si(unit):
        raise ModelError(
            f"{what}: unit {unit!r} is not SI. Convert at the boundary that produced the "
            f"number; display units belong in plot labels. See machina/units.py."
        )


def numeric(value, shape: tuple, what: str, *, finite: bool = False) -> np.ndarray:
    """``value`` as a flat column-major float array, under the solver backend's rule.

    A scalar broadcasts; an array of the declared shape is read column-major;
    a 1-D array of the right length is taken as already column-major; a row
    given for a column vector (or the reverse) is accepted. Anything else --
    and strings, booleans and NaN -- is refused here, at the declaration,
    rather than at compile time with a message naming a solver variable.
    """
    shape = tuple(shape)
    numel = shape[0] * shape[1]
    items = value if isinstance(value, (list, tuple)) else [value]
    if any(isinstance(v, (str, bytes, bool, np.bool_)) for v in items):
        raise ModelError(f"{what} must be numeric, got {value!r}.")
    try:
        array = np.asarray(value.full() if isinstance(value, ca.DM) else value, dtype=float)
    except (TypeError, ValueError):
        raise ModelError(f"{what} must be numeric, got {value!r}.") from None
    if array.dtype == bool:
        raise ModelError(f"{what} must be numeric, got {value!r}.")
    if array.ndim == 0:
        flat = np.full(numel, float(array))
    elif array.ndim == 1 and array.size == numel:
        flat = array.copy()
    elif array.ndim == 2 and array.shape == shape:
        flat = array.ravel(order="F")
    elif array.ndim == 2 and 1 in shape and 1 in array.shape and array.size == numel:
        flat = array.ravel()
    else:
        raise ModelError(
            f"{what} has shape {array.shape}; expected a scalar, an array of shape {shape}, "
            f"or a flat column-major array of length {numel}."
        )
    if np.any(np.isnan(flat)):
        raise ModelError(f"{what} contains NaN.")
    if finite and not np.all(np.isfinite(flat)):
        raise ModelError(f"{what} is not finite.")
    return flat


def _check_bounds(lb, ub, shape: tuple, what: str) -> None:
    lo = numeric(lb, shape, f"{what}: lb")
    hi = numeric(ub, shape, f"{what}: ub")
    if np.any(lo > hi):
        raise ModelError(f"{what}: lb {lb!r} exceeds ub {ub!r}.")


def _check_scale(scale, shape: tuple, what: str) -> None:
    """A nominal magnitude: broadcastable to ``shape``, finite and strictly positive."""
    values = numeric(scale, shape, f"{what}: scale", finite=True)
    if np.any(values <= 0.0):
        raise ModelError(
            f"{what}: scale {scale!r} must be strictly positive; it divides the value the "
            f"solver sees, and the default 1.0 means unscaled."
        )


def _ref(entry, group: str) -> tuple:
    """Normalise a consumed entry to ``(local_name, reference)``."""
    if isinstance(entry, str):
        local = reference = entry
    else:
        try:
            local, reference = entry
        except (TypeError, ValueError):
            raise ModelError(
                f"{group} entry {entry!r} must be a signal name or a "
                f"(local_name, reference) pair."
            ) from None
    _check_identifier(local, f"{group} local name")
    if not isinstance(reference, str) or not reference:
        raise ModelError(f"{group} entry {entry!r}: the reference must be a non-empty string.")
    segments = (reference[1:] if reference.startswith("/") else reference).split("/")
    if not all(segment.isidentifier() for segment in segments):
        raise ModelError(
            f"{group} entry {entry!r}: {reference!r} is not a valid reference. Write a signal "
            f"name, a scoped path such as 'sat_a/r_eci', or an absolute path such as "
            f"'/sat_a/r_eci' -- no empty segments and no trailing '/'."
        )
    return (local, reference)


def _snake(name: str) -> str:
    """``SingleSatCoverage`` to ``single_sat_coverage``. Names reach generated code."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper() and name[i - 1] != "_":
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
