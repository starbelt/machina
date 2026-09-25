"""
Generic utility factories: ``util.sum`` and the axis rotations ``util.rotate_x``,
``util.rotate_y`` and ``util.rotate_z``.

Importing this module registers the four factories; ``import machina`` imports
it through :mod:`machina.library`.
"""

import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register("util.sum")
def make_sum(*, n_terms: int) -> FunctionDescriptor:
    """
    Sum a vector of terms.

    Factory parameters
    ------------------
    n_terms : int
        Number of terms to sum. Must be >= 1. Raises ValueError if 0 or
        negative.

    Function interface
    ------------------
    Input
        terms : (n_terms, 1)  — vector of terms to sum.
    Output
        total : (1, 1)  — scalar sum of the input terms.

    Notes
    -----
    This function is a simple wrapper around ca.sum1 for semantic clarity.
    An agent could write ca.sum1(terms) directly, but using util.sum makes
    the intent explicit and the computation appear in the function registry
    for introspection.
    """
    if n_terms < 1:
        raise ValueError(
            f"util.sum: n_terms must be >= 1, got {n_terms}."
        )
    terms = ca.SX.sym('terms', n_terms)
    total = ca.sum1(terms)
    f = ca.Function('sum', [terms], [total], ['terms'], ['total'])
    return FunctionDescriptor(
        f, description=f'Σ ({n_terms} terms)'
    )


@register("util.rotate_x")
def make_rotate_x() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the x-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the x-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_x = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c)
    )
    v_rot = R_x @ v
    f = ca.Function('rotate_x', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Rx(theta) * v'
    )


@register("util.rotate_y")
def make_rotate_y() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the y-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the y-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_y = ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c)
    )
    v_rot = R_y @ v
    f = ca.Function('rotate_y', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Ry(theta) * v'
    )


@register("util.rotate_z")
def make_rotate_z() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the z-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the z-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_z = ca.vertcat(
        ca.horzcat(c, -s, 0),
        ca.horzcat(s, c, 0),
        ca.horzcat(0, 0, 1)
    )
    v_rot = R_z @ v
    f = ca.Function('rotate_z', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Rz(theta) * v'
    )
