"""
``Component`` -- one piece of a model, declaring what it reads and what it
contributes -- plus the things it may own or emit.

The declare/build split is the good idea shared by both lineages: a component
states its interface without constructing any symbols, so the builder can lay
out vectors, resolve dependencies and detect conflicts *before* anything is
built. Only then does ``build()`` run with the symbols already sliced out.

This is icarus-dynamics' ``Declaration`` with three additions from machina's
April agent types:

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from machina.model.errors import ModelError

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

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True)
class Constraint:
    """A declared ``h`` output. Bounds are in the component's own units."""

    name: str
    shape: tuple[int, int] = (1, 1)
    lb: object = -math.inf
    ub: object = math.inf
    doc: str = ""

    def __post_init__(self):
        _check_identifier(self.name, "constraint name")
        object.__setattr__(self, "shape", _shape(self.shape, f"constraint {self.name!r}"))
        if self.shape[1] != 1:
            raise ModelError(
                f"constraint {self.name!r}: shape {self.shape} must be a column; the solver "
                f"stacks constraints into one vector."
            )

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
            names = tuple(getattr(self, group))
            for name in names:
                _check_identifier(name, f"{group[:-1]} name")
            object.__setattr__(self, group, names)
        for group in ("quantities", "constraints", "costs"):
            object.__setattr__(self, group, tuple(getattr(self, group)))
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
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise ModelError(f"scope {self.name!r} is empty; a scope with nothing in it is a typo.")


# --- helpers ----------------------------------------------------------------------------------


def _check_identifier(name, what: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise ModelError(
            f"{what} {name!r} must be a Python identifier: these names become CasADi argument "
            f"names and instance-path segments."
        )


def _shape(shape, what: str) -> tuple:
    if isinstance(shape, int):
        shape = (shape, 1)
    try:
        rows, cols = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        raise ModelError(
            f"{what}: shape must be an int or a (rows, cols) pair, got {shape!r}."
        ) from None
    if rows < 1 or cols < 1:
        raise ModelError(f"{what}: shape {(rows, cols)} must be positive in both dimensions.")
    return (rows, cols)


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
    return (local, reference)


def _snake(name: str) -> str:
    """``SingleSatCoverage`` to ``single_sat_coverage``. Names reach generated code."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper() and name[i - 1] != "_":
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
