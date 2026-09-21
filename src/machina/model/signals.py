"""
The signal registry: every quantity that crosses a component boundary,
declared once with a shape, an SI unit, a frame and an aggregation policy.

Adopted from icarus-dynamics ``model/signals.py``, which exists because its
predecessor carried a hardcoded name-to-shape dict and checked nothing. Three
silent bugs lived in that gap: a ``(1, 1)`` standing in for a ``(3, 3)`` is
only an error if something knows a ``(3, 3)`` belonged.

One declaration carries four things, and each closes a specific hole:

==============  ==========================================================
``shape``       a ``build()`` that produces the wrong shape fails at build
``unit``        SI only, mandatory (see :mod:`machina.units`)
``frame``       a validated name from the frame registry
``aggregation`` what it means for two components to produce the same name
==============  ==========================================================

``SUM`` is for forces, moments and budgets: physics adds, and any number of
producers contribute. An unproduced ``SUM`` signal is zero, which is the
identity of addition and what makes a reduced model expressible. ``UNIQUE``
is for states and owned outputs: exactly one producer, and an unproduced one
is an error, because there is no identity to fall back on.

Two things differ from icarus. ``Frame`` is an extensible registry rather
than a closed enum -- the core declares ``none``, :mod:`machina.rigid`
declares ``ned`` and ``frd``, and an astro pack will declare ``eci`` and
friends. And the registry is a class with ``snapshot``/``restore``/``clear``,
so tests and packs can be isolated from each other.

**Declaration order is an ABI.** Vector layouts are the registry walk
filtered, so appending a signal is safe and inserting one shifts every offset
below it.
"""

from dataclasses import dataclass
from enum import Enum

from machina.model.errors import SignalError
from machina.units import NON_SI_UNITS, UNIT_RE, is_si, is_well_formed

__all__ = [
    "Aggregation", "Frame", "Signal", "SignalRegistry", "SignalError",
    "DEFAULT", "declare", "declare_frame", "get", "all_signals", "frame", "frames",
    "NON_SI_UNITS", "UNIT_RE",
]

_NONE_FRAME_DOC = "Frame-free scalars: a mass, a budget, an angle magnitude"


class Aggregation(Enum):
    """What it means for more than one component to produce a signal."""

    SUM = "sum"
    UNIQUE = "unique"


@dataclass(frozen=True)
class Frame:
    """A reference frame a vector signal can be expressed in.

    A registry entry rather than an enum member, so a domain pack can add its
    own frames without editing the core. ``family`` groups related frames
    (inertial, body, earth-fixed) for documentation only -- nothing in the
    model layer interprets it.
    """

    name: str
    doc: str = ""
    family: str = ""


@dataclass(frozen=True)
class Signal:
    """One declared quantity. ``frame`` is a name registered in the same registry."""

    name: str
    shape: tuple[int, int]
    unit: str
    frame: str
    aggregation: Aggregation
    doc: str

    @property
    def size(self) -> int:
        """Number of elements. Stored column-major, as ``ca.vec`` does."""
        return self.shape[0] * self.shape[1]


def _normalise_shape(shape, name: str) -> tuple[int, int]:
    """``n`` or ``(rows, cols)`` to ``(rows, cols)``, checked positive."""
    if isinstance(shape, int):
        shape = (shape, 1)
    try:
        rows, cols = shape
        rows, cols = int(rows), int(cols)
    except (TypeError, ValueError):
        raise SignalError(
            f"{name}: shape must be an int or a (rows, cols) pair, got {shape!r}."
        ) from None
    if rows < 1 or cols < 1:
        raise SignalError(f"{name}: shape {(rows, cols)} must be positive in both dimensions.")
    return (rows, cols)


class SignalRegistry:
    """Signals and frames, in declaration order.

    Process-global through the module-level :data:`DEFAULT` instance. Packs
    declare into it at import time; tests snapshot and restore it.
    """

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}
        self._frames: dict[str, Frame] = {}
        self.declare_frame("none", doc=_NONE_FRAME_DOC)

    # --- frames ---------------------------------------------------------------------------

    def declare_frame(self, name: str, doc: str = "", family: str = "") -> Frame:
        if not name.isidentifier():
            raise SignalError(
                f"frame name {name!r} must be a Python identifier; frame names reach generated "
                f"manifests and are compared by string."
            )
        if name in self._frames:
            raise SignalError(
                f"frame {name!r} is already declared. Frame names are global -- if two packs "
                f"need the same name for different things, one of them needs a new name."
            )
        declared = Frame(name, doc, family)
        self._frames[name] = declared
        return declared

    def frame(self, name: str) -> Frame:
        try:
            return self._frames[name]
        except KeyError:
            raise SignalError(
                f"unknown frame {name!r}. Declared frames: {sorted(self._frames)}. Declare it "
                f"with declare_frame() in the pack that owns it."
            ) from None

    def frames(self) -> dict[str, Frame]:
        """Declaration order."""
        return dict(self._frames)

    # --- signals --------------------------------------------------------------------------

    def declare(self, name: str, shape, unit: str, *, frame: str = "none",
                aggregation: Aggregation = Aggregation.UNIQUE, doc: str = "") -> Signal:
        """Declare a signal. Appends to the layout ABI -- never insert."""
        if not name.isidentifier():
            raise SignalError(
                f"signal name {name!r} must be a Python identifier: a component build() "
                f"function names its arguments after the signals it reads, and CasADi argument "
                f"names must be identifiers. Scoping uses instance paths, not names."
            )
        if name in self._signals:
            raise SignalError(
                f"signal {name!r} is declared twice. If two packs need the same name for "
                f"different quantities, one of them needs a new name."
            )
        if not unit:
            raise SignalError(f"{name}: unit is mandatory; use \"1\" for dimensionless.")
        if not is_well_formed(unit):
            raise SignalError(f"{name}: unit {unit!r} is not a well-formed unit string.")
        if not is_si(unit):
            raise SignalError(
                f"{name}: unit {unit!r} is not SI. Signals carry SI throughout; the conversion "
                f"belongs at the boundary that produced the number, and display units belong "
                f"in plot labels. See machina/units.py."
            )
        if not isinstance(aggregation, Aggregation):
            raise SignalError(
                f"{name}: aggregation must be an Aggregation member, got {aggregation!r}."
            )
        self.frame(frame)  # raises, listing the declared frames, if unknown
        declared = Signal(name, _normalise_shape(shape, name), unit, frame, aggregation, doc)
        self._signals[name] = declared
        return declared

    def get(self, name: str) -> Signal:
        """The declaration, or an error naming what to do about it."""
        try:
            return self._signals[name]
        except KeyError:
            raise SignalError(
                f"unknown signal {name!r}. Every quantity crossing a component boundary must "
                f"be declared with signals.declare() -- an undeclared name is a typo, not a "
                f"new signal. A value a component owns privately is a Quantity, not a signal."
            ) from None

    def has(self, name: str) -> bool:
        return name in self._signals

    def all(self) -> dict[str, Signal]:
        """Declaration order. This walk is the vector layout ABI."""
        return dict(self._signals)

    # --- test isolation -------------------------------------------------------------------

    def snapshot(self):
        """Opaque state for :meth:`restore`. Tests only."""
        return (dict(self._signals), dict(self._frames))

    def restore(self, state) -> None:
        """Put back a :meth:`snapshot`, in place. Tests only."""
        signals, declared_frames = state
        self._signals.clear()
        self._signals.update(signals)
        self._frames.clear()
        self._frames.update(declared_frames)

    def clear(self) -> None:
        """Empty the registry, keeping the ``none`` frame. Tests only.

        Packs declare at import time, so clearing does not bring their signals
        back on a later import (Python caches modules): restore a snapshot.
        """
        self._signals.clear()
        self._frames.clear()
        self.declare_frame("none", doc=_NONE_FRAME_DOC)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SignalRegistry {len(self._signals)} signal(s), {len(self._frames)} frame(s)>"


DEFAULT = SignalRegistry()

declare = DEFAULT.declare
declare_frame = DEFAULT.declare_frame
get = DEFAULT.get
frame = DEFAULT.frame
frames = DEFAULT.frames


def all_signals() -> dict[str, Signal]:
    """Every signal in the default registry, in declaration order."""
    return DEFAULT.all()
