"""
SWaP-C budget signals.

Declares the two power-budget signals: ``power_load``, the electrical load a
component puts on the power subsystem, and ``battery_energy``, the battery a
power budget sizes for the eclipse. Budgets are frame-free scalars, so the pack
declares no frames. The currency ``Budget`` waits on Decision Log #74 (see
:mod:`machina.swapc`); nothing here needs it.

Importing this module declares into :data:`machina.model.signals.DEFAULT`,
once, the way every pack does. :func:`declare_into` declares the same set into
any other registry, which is how tests get a clean copy.

**Layout ABI.** Registry declaration order is the vector layout. Append new
signals at the end of :func:`declare_into`; never insert.
"""

from machina.model.signals import DEFAULT, Aggregation, SignalRegistry

__all__ = ["declare_into", "SIGNALS"]

SIGNALS = (
    "power_load",
    "battery_energy",
)


def declare_into(registry: SignalRegistry) -> None:
    """Declare the SWaP-C signals into ``registry``, in ABI order."""
    # SUM: every consumer of electrical power adds its own load, and a model with no load
    # component reads zero. Loads are summed per instance path: two producers in one scope
    # share a path; a load in another scope is read by that path.
    registry.declare("power_load", 1, "W", frame="none", aggregation=Aggregation.SUM,
                     doc="Electrical load on the power subsystem, summed over every producer")

    # UNIQUE: one power budget sizes the battery of one spacecraft.
    registry.declare("battery_energy", 1, "J", frame="none", aggregation=Aggregation.UNIQUE,
                     doc="Battery nameplate energy that carries the total load through the "
                         "longest eclipse")


declare_into(DEFAULT)
