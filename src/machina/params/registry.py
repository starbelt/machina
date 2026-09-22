"""
The param declaration registry.

Adopted from icarus-dynamics ``params/registry.py``: a table of everything
:func:`machina.params.param` has declared, keyed by name, populated by import
side effects, with duplicate detection. It is deliberately not clever.

Declaration order is preserved because it matters: a CasADi parameter
vector's element order has to come from somewhere stable, and "the order they
were declared in" is the only ordering that is both deterministic and
meaningful to the person reading the model. The CSV is sorted by name for
review. Those are two orderings of one set, both derived from this one walk.

machina makes the registry a class, as it did the signal registry, so tests
and consumers can hold an isolated one; the module-level functions act on
:data:`DEFAULT`, which is what icarus's call sites expect.
"""

from dataclasses import dataclass

__all__ = ["ParamDecl", "ParamRegistry", "DuplicateParam", "DEFAULT",
           "register", "all_params", "clear", "snapshot", "restore"]


@dataclass(frozen=True)
class ParamDecl:
    """Everything about a param except its value. The value lives in the CSV, by hand."""

    name: str
    type: str          # storage type: f32, f64, i8 ... u32
    unit: str
    lo: float
    hi: float
    mutability: str
    kind: str          # symbolic | static
    doc: str
    default: float = None
    declared_in: str = "<unknown>"   # module path, so an error can say where to go

    @property
    def is_integer(self) -> bool:
        return not self.type.startswith("f")


class DuplicateParam(Exception):
    """Two declarations of one name. Which one wins is not a question worth having."""


class ParamRegistry:
    """Declarations in declaration order."""

    def __init__(self) -> None:
        self._decls: dict = {}

    def register(self, decl: ParamDecl) -> ParamDecl:
        existing = self._decls.get(decl.name)
        if existing is not None:
            raise DuplicateParam(
                f"{decl.name} is declared twice: in {existing.declared_in} and in "
                f"{decl.declared_in}. If two copies of machina are on sys.path, remove one."
            )
        self._decls[decl.name] = decl
        return decl

    def all(self) -> dict:
        """Declaration order. Load the declaring modules first (see
        :func:`machina.params.modules.load_all`), or this is whatever happens to
        have been imported -- a subtly wrong answer rather than an obviously wrong one."""
        return dict(self._decls)

    def get(self, name: str) -> ParamDecl:
        try:
            return self._decls[name]
        except KeyError:
            raise KeyError(
                f"no param named {name!r} is declared. Declared: {list(self._decls)}. Load "
                f"the declaring modules first."
            ) from None

    def snapshot(self):
        """Tests only."""
        return dict(self._decls)

    def restore(self, state) -> None:
        """Tests only. In place, so module-level aliases keep working."""
        self._decls.clear()
        self._decls.update(state)

    def clear(self) -> None:
        """Tests only. Nothing in a pipeline should ever want this."""
        self._decls.clear()


DEFAULT = ParamRegistry()

register = DEFAULT.register
clear = DEFAULT.clear
snapshot = DEFAULT.snapshot
restore = DEFAULT.restore


def all_params() -> dict:
    """Every declaration in the default registry, in declaration order."""
    return DEFAULT.all()
