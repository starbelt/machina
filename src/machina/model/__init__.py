"""
machina.model -- the signal registry, components and the builder.

The middle of machina: it turns declared components into one symbolic
expression graph that the compiler hands to the solver or composes into an
ODE. Nothing here knows about the solver, and nothing here knows any physics.

Public API
----------
Signals     ``Signal``, ``Frame``, ``Aggregation``, ``SignalRegistry``, ``DEFAULT``
Components  ``Component``, ``Declaration``, ``Quantity``, ``Constraint``, ``Cost``,
            ``Role``, ``Scope``
Builder     ``Builder``, ``Wired``, ``WiredConstraint``, ``WiredCost``, ``Placed``
Errors      ``SignalError``, ``ModelError``
"""

from machina.model.builder import Builder, Placed, Wired, WiredConstraint, WiredCost
from machina.model.component import (
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    Role,
    Scope,
)
from machina.model.errors import ModelError, SignalError
from machina.model.signals import (
    DEFAULT,
    Aggregation,
    Frame,
    Signal,
    SignalRegistry,
    all_signals,
    declare,
    declare_frame,
    get,
)

__all__ = [
    "Signal", "Frame", "Aggregation", "SignalRegistry", "SignalError", "DEFAULT",
    "declare", "declare_frame", "get", "all_signals",
    "Component", "Declaration", "Quantity", "Constraint", "Cost", "Role", "Scope",
    "Builder", "Wired", "WiredConstraint", "WiredCost", "Placed", "ModelError",
]
