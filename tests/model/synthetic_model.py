"""
A synthetic model the graph tests build on: a thrusting point mass in 2-D.

Small enough to check on paper, and it exercises every mechanism the builder
has -- a state integrated from another state, a ``SUM`` signal with a real
producer, a ``UNIQUE`` algebraic signal, an exogenous input, an owned
quantity, a declared constraint and a declared cost. It replaces the aircraft
model icarus's own graph tests lean on, which would drag six degrees of
freedom in for no extra coverage.

    pos_dot = vel                  Kinematics
    vel_dot = force / mass         Dynamics
    mass    = dry_mass             MassSource   (quantity)
    force   = [thrust_cmd * mass, 0]            Thruster   (input)
"""

import casadi as ca

from machina.model import (
    Aggregation,
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    SignalRegistry,
)


def make_registry() -> SignalRegistry:
    """A fresh registry. Declaration order here is the layout ABI for these tests."""
    reg = SignalRegistry()
    reg.declare_frame("world", doc="A flat 2-D inertial frame", family="inertial")
    reg.declare("pos", 2, "m", frame="world", doc="Position")
    reg.declare("vel", 2, "m/s", frame="world", doc="Velocity")
    reg.declare("force", 2, "N", frame="world", aggregation=Aggregation.SUM, doc="Applied force")
    reg.declare("mass", 1, "kg", doc="Total mass")
    reg.declare("thrust_cmd", 1, "1", doc="Commanded thrust fraction")
    reg.declare("separation", 1, "m^2", doc="Squared distance between two instances")
    return reg


class Kinematics(Component):
    """``pos_dot = vel``."""

    def declare(self):
        return Declaration(states=("vel",), derivatives=("pos",))

    def build(self, helpers):
        vel = ca.SX.sym("vel", 2)
        return {"f": ca.Function("kinematics_f", [vel], [vel], ["vel"], ["pos_dot"])}


class Dynamics(Component):
    """``vel_dot = force / mass``."""

    def declare(self):
        return Declaration(states=("vel",), algebraic=("force", "mass"), derivatives=("vel",))

    def build(self, helpers):
        vel, force, mass = ca.SX.sym("vel", 2), ca.SX.sym("force", 2), ca.SX.sym("mass")
        return {"f": ca.Function("dynamics_f", [vel, force, mass], [force / mass],
                                 ["vel", "force", "mass"], ["vel_dot"])}


class MassSource(Component):
    """Owns ``dry_mass`` and publishes it as the ``mass`` signal."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("dry_mass", unit="kg", default=2.0, lb=0.1, ub=10.0,
                                 doc="Dry mass"),),
            produces=("mass",),
        )

    def build(self, helpers):
        dry = ca.SX.sym("dry_mass")
        return {"g": ca.Function("mass_source_g", [dry], [dry], ["dry_mass"], ["mass"])}


class Thruster(Component):
    """``force = [thrust_cmd * mass, 0]`` -- a constant-acceleration thruster."""

    def declare(self):
        return Declaration(inputs=("thrust_cmd",), algebraic=("mass",), produces=("force",))

    def build(self, helpers):
        cmd, mass = ca.SX.sym("thrust_cmd"), ca.SX.sym("mass")
        return {"g": ca.Function("thruster_g", [cmd, mass], [ca.vertcat(cmd * mass, 0.0)],
                                 ["thrust_cmd", "mass"], ["force"])}


class Drag(Component):
    """A second producer of the ``SUM`` signal ``force``, to prove aggregation."""

    def declare(self):
        return Declaration(states=("vel",), produces=("force",))

    def build(self, helpers):
        vel = ca.SX.sym("vel", 2)
        return {"g": ca.Function("drag_g", [vel], [-0.5 * vel], ["vel"], ["force"])}


class Budget(Component):
    """Owns a cap, constrains the commanded thrust against it, and scores it.

    The component that knows the physics owns its feasibility constraint; the
    problem decides what the cost term is worth.
    """

    def declare(self):
        return Declaration(
            inputs=("thrust_cmd",),
            quantities=(Quantity("cap", unit="1", default=1.0, lb=0.0, ub=2.0,
                                 doc="Thrust fraction cap"),),
            constraints=(Constraint("headroom", lb=0.0, doc="cap - thrust_cmd >= 0"),),
            costs=(Cost("effort", weight=2.0, doc="Squared commanded thrust"),),
        )

    def build(self, helpers):
        cmd, cap = ca.SX.sym("thrust_cmd"), ca.SX.sym("cap")
        return {
            "h": ca.Function("budget_h", [cmd, cap], [cap - cmd],
                             ["thrust_cmd", "cap"], ["headroom"]),
            "J": ca.Function("budget_J", [cmd], [cmd ** 2], ["thrust_cmd"], ["effort"]),
        }


def point_mass():
    """The four components of the base model, freshly constructed."""
    return [Kinematics(), Dynamics(), MassSource(), Thruster()]
