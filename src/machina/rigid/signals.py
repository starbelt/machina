"""
Rigid-body frames and signals.

Declares the ``ned`` and ``frd`` frames and the nine signals a six-degree-of-
freedom rigid body needs, in the same relative order icarus-dynamics declares
them. The other aircraft signals (surfaces, throttle, air data, sensors) stay
in icarus until its components move here (Phase 6).

Importing this module declares into :data:`machina.model.signals.DEFAULT`,
once, the way every pack does. :func:`declare_into` declares the same set into
any other registry, which is how tests get a clean copy.

**Layout ABI.** Registry declaration order is the vector layout. Append new
signals at the end of :func:`declare_into`; never insert. When icarus adopts
this module its own declarations follow these nine, which moves any icarus
signal that used to sit between them -- that is a Phase 6 decision, recorded
in the vault Roadmap, not something to fix by reordering here.
"""

from machina.model.signals import DEFAULT, Aggregation, SignalRegistry

__all__ = ["declare_into", "FRAMES", "SIGNALS"]

FRAMES = ("ned", "frd")

SIGNALS = (
    "inertial_position",
    "inertial_orientation_quaternion",
    "body_linear_velocity",
    "body_angular_velocity",
    "force_body",
    "moment_body",
    "mass",
    "inertia_tensor_body",
    "body_angular_acceleration",
)


def declare_into(registry: SignalRegistry) -> None:
    """Declare the rigid-body frames and signals into ``registry``, in ABI order."""
    registry.declare_frame(
        "ned", family="inertial",
        doc="North-east-down, local-level inertial frame. Z points down.")
    registry.declare_frame(
        "frd", family="body",
        doc="Forward-right-down body frame, right-handed, at the centre of mass.")

    unique, total = Aggregation.UNIQUE, Aggregation.SUM

    # Differential states. UNIQUE: exactly one component owns each integration.
    registry.declare("inertial_position", 3, "m", frame="ned", aggregation=unique,
                     doc="Position of the centre of mass in the NED frame")
    registry.declare("inertial_orientation_quaternion", 4, "1", frame="none",
                     aggregation=unique,
                     doc="Body-to-inertial rotation as [qx, qy, qz, qw], scalar last")
    registry.declare("body_linear_velocity", 3, "m/s", frame="frd", aggregation=unique,
                     doc="Velocity of the centre of mass resolved in the body frame")
    registry.declare("body_angular_velocity", 3, "rad/s", frame="frd", aggregation=unique,
                     doc="Body angular rate [p, q, r] about the body axes")

    # What acts on the body. SUM: gravity, aero and thrust all contribute to one total, and
    # no component should have to know which others exist.
    registry.declare("force_body", 3, "N", frame="frd", aggregation=total,
                     doc="Total external force at the centre of mass, body frame")
    registry.declare("moment_body", 3, "N*m", frame="frd", aggregation=total,
                     doc="Total external moment about the centre of mass, body frame")

    # Mass properties: algebraic signals rather than constants, so a fuel-state model can
    # make them time-varying without changing any consumer.
    registry.declare("mass", 1, "kg", frame="none", aggregation=unique,
                     doc="Total vehicle mass")
    registry.declare("inertia_tensor_body", (3, 3), "kg*m^2", frame="frd", aggregation=unique,
                     doc="Inertia tensor about the centre of mass in body axes, not assumed "
                         "diagonal")

    registry.declare("body_angular_acceleration", 3, "rad/s^2", frame="frd",
                     aggregation=unique,
                     doc="Body angular acceleration; an accelerometer away from the centre "
                         "of mass reads it")


declare_into(DEFAULT)
