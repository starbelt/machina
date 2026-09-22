"""
Rotation and quaternion kinematics, as CasADi Functions.

Taken verbatim from icarus-dynamics ``model/frames.py`` (Decision Log #37: the
packs live in machina, and ``rigid`` is the skeleton the aircraft components
will move into). Every component that needs to move a vector between frames
calls in here rather than writing the algebra again -- two constructions of
one rotation is how a sign gets lost.

BOTH FUNCTIONS BELOW WERE WRONG IN THE CODE ICARUS ADOPTED, in ways that
built cleanly:

* ``quaternion_to_rotation_matrix`` had ``- 2 qw [qv]x``. Still a valid
  rotation matrix -- just the transpose, so it rotated inertial-to-body while
  being used body-to-inertial. For small angles the difference looks like a
  slightly wrong gain rather than a reversed frame.
* ``Q_matrix`` was assembled as ``vertcat(top_3x3[0,0], qT[0])``, which
  indexes two scalars out of the blocks it meant to stack, giving a ``(2,1)``
  where a ``(4,3)`` belonged. Its own docstring said 4x3.

Neither was catchable, because nothing declared a shape. The signal registry
now does, the builder checks it, and ``tests/rigid/test_frames.py`` pins the
numbers against a rotation a person can verify on paper.

Conventions: quaternions are ``[qx, qy, qz, qw]`` -- vector first, scalar
last -- and ``R_ib`` takes a vector from body to inertial. Read the subscript
right to left.
"""

import casadi as ca

__all__ = ["quaternion_to_rotation_matrix", "quaternion_kinematics_matrix", "helpers"]


def quaternion_to_rotation_matrix() -> ca.Function:
    """``R_ib(q)`` -- takes a vector FROM BODY TO INERTIAL.

        v_i = R_ib @ v_b
        R_ib = (qw^2 - qv'qv) I + 2 qv qv' + 2 qw [qv]x

    The inertial-to-body rotation is the transpose and is deliberately not
    built here.
    """
    q = ca.SX.sym("q", 4)          # [qx, qy, qz, qw] -- vector first, scalar last
    qv = q[0:3]
    qw = q[3]

    R = ((qw * qw - ca.dot(qv, qv)) * ca.DM.eye(3)
         + 2.0 * (qv @ qv.T)
         + 2.0 * qw * ca.skew(qv))

    return ca.Function("R_ib", [q], [R], ["q"], ["R_ib"])


def quaternion_kinematics_matrix() -> ca.Function:
    """``Q(q)`` in ``qdot = 0.5 * Q(q) @ omega_b``.

        Q = [ qw I + [qv]x ]   rows 0-2
            [     -qv'     ]   row 3

    The bottom row is NEGATIVE ``qv'``. It falls out of the Hamilton product
    ``q (x) (omega, 0)``: the scalar part of that product is ``-qv . omega``.
    """
    q = ca.SX.sym("q", 4)
    qv = q[0:3]
    qw = q[3]

    Q = ca.vertcat(qw * ca.DM.eye(3) + ca.skew(qv), -qv.T)

    return ca.Function("Q_kin", [q], [Q], ["q"], ["Q"])


def helpers() -> dict:
    """The helper table a rigid-body component's ``build()`` is handed.

    Keyed by the names components declare under ``Declaration.helpers``. The
    builder resolves against this and refuses a component asking for a name
    that is not here, so a typo is a build error rather than a ``KeyError``
    four frames down a stack trace.
    """
    return {
        "quaternion_to_rotation_matrix": quaternion_to_rotation_matrix(),
        "quaternion_kinematics_matrix": quaternion_kinematics_matrix(),
    }
