"""
machina.rigid -- rigid-body frames, signals and quaternion kinematics.

A skeleton pack (Decision Log #37). It carries what a six-degree-of-freedom
component needs to compile against machina -- the ``ned`` and ``frd`` frames,
the rigid-body signals and the rotation helpers -- and nothing aircraft-
specific; the aircraft components stay in icarus-dynamics until Phase 6.

Importing this package declares its frames and signals into the default
signal registry. ``import machina`` never imports it.
"""

from machina.rigid import frames, signals
from machina.rigid.frames import (
    helpers,
    quaternion_kinematics_matrix,
    quaternion_to_rotation_matrix,
)

__all__ = ["frames", "signals", "helpers", "quaternion_to_rotation_matrix",
           "quaternion_kinematics_matrix"]
