"""
Orbital frames and signals.

Declares the ``eci``, ``ecef`` and ``lvlh`` frames and the one signal a
coverage study puts on the wire. Orbital elements themselves are *not*
signals: they are quantities owned by the component that flies them, and the
derived ``sma``/``ecc``/``inc``/``period`` are display-only (Decision Log
#33), computed from a solution rather than wired into anything.

Importing this module declares into :data:`machina.model.signals.DEFAULT`,
once, the way every pack does. :func:`declare_into` declares the same set into
any other registry, which is how tests get a clean copy.

**Layout ABI.** Registry declaration order is the vector layout. Append new
signals at the end of :func:`declare_into`; never insert.
"""

from machina.model.signals import DEFAULT, Aggregation, SignalRegistry

__all__ = ["declare_into", "FRAMES", "SIGNALS"]

FRAMES = ("eci", "ecef", "lvlh")

SIGNALS = (
    "coverage_total",
)


def declare_into(registry: SignalRegistry) -> None:
    """Declare the orbital frames and signals into ``registry``, in ABI order."""
    registry.declare_frame(
        "eci", family="inertial",
        doc="Earth-centred inertial. Z along the mean spin axis, X at the vernal equinox.")
    registry.declare_frame(
        "ecef", family="rotating",
        doc="Earth-centred Earth-fixed. Rotates with the Earth; X through the prime meridian.")
    registry.declare_frame(
        "lvlh", family="orbital",
        doc="Local-vertical local-horizontal orbital frame of one satellite.")

    # UNIQUE: exactly one component owns the coverage figure of a given satellite. A
    # constellation total is a separate signal, summed there, not this one reused.
    registry.declare("coverage_total", 1, "1", frame="none", aggregation=Aggregation.UNIQUE,
                     doc="Mean smooth coverage of the ground target over the sampled orbit, "
                         "in [0, 1]")


declare_into(DEFAULT)
