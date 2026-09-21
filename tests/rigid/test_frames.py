"""
The rotation and quaternion-kinematics formulas, pinned against numbers
checkable on paper.

Ported from icarus-dynamics ``tests/test_frames.py``. It is the regression for
two real bugs inherited from ``machina-sim``, both of which built cleanly and
produced plausible-looking trajectories:

* ``quaternion_to_rotation_matrix`` had ``- 2 qw [qv]x`` -- the transpose, so
  it rotated inertial-to-body while being used body-to-inertial.
* ``Q_matrix`` was ``vertcat(top_3x3[0,0], qT[0])``: a ``(2,1)`` where a
  ``(4,3)`` belonged, against its own docstring.

The lesson worth keeping is that a convention with no executable statement of
itself has already drifted, which is why every case below is one a person can
verify without running anything.
"""

import numpy as np
import pytest

from machina.rigid.frames import (
    helpers,
    quaternion_kinematics_matrix,
    quaternion_to_rotation_matrix,
)

pytestmark = pytest.mark.requires_casadi

ROOT_HALF = np.sqrt(0.5)
Q_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
Q_YAW_90 = np.array([0.0, 0.0, ROOT_HALF, ROOT_HALF])   # +90 deg about inertial Z


def normalise(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestShapes:
    """The half of this that the builder now also enforces."""

    def test_the_rotation_matrix_is_three_by_three(self):
        assert quaternion_to_rotation_matrix().size_out(0) == (3, 3)

    def test_the_kinematics_matrix_is_four_by_three(self):
        """It was (2,1). The docstring said 4x3 and nothing checked."""
        assert quaternion_kinematics_matrix().size_out(0) == (4, 3)

    def test_indexing_a_function_result_takes_an_element_not_an_output(self):
        """``R(q)`` is already the matrix; ``R(q)[0]`` is element [0, 0] of it.

        Three components used the second where they meant the first, and a
        scalar standing in for a rotation matrix is a change of gain rather
        than an error -- which is why it survived.
        """
        R = quaternion_to_rotation_matrix()
        assert R(Q_YAW_90).shape == (3, 3)
        assert R(Q_YAW_90)[0].shape == (1, 1)


class TestValuesCheckableOnPaper:

    def test_identity_attitude_is_the_identity_rotation(self):
        R = quaternion_to_rotation_matrix()
        np.testing.assert_allclose(np.array(R(Q_IDENTITY)), np.eye(3), atol=1e-15)

    def test_body_x_maps_to_inertial_y_under_a_ninety_degree_yaw(self):
        """Nose-east after a +90 degree yaw from north.

        With the sign wrong this gives [0, -1, 0] -- still unit length, still
        orthonormal, and pointing the other way.
        """
        R = quaternion_to_rotation_matrix()
        np.testing.assert_allclose(np.array(R(Q_YAW_90)) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                   atol=1e-15)

    def test_the_rotation_matrix_is_orthonormal_with_unit_determinant(self):
        """A reflection has determinant -1 and would pass an orthonormality check alone."""
        R = quaternion_to_rotation_matrix()
        for q in (Q_IDENTITY, Q_YAW_90, normalise([0.3, -0.2, 0.5, 0.8])):
            m = np.array(R(q))
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)
            assert float(np.linalg.det(m)) == pytest.approx(1.0)

    def test_a_body_rate_about_z_produces_the_expected_quaternion_rate(self):
        """At identity attitude, 1 rad/s about body Z gives qdot = [0, 0, 0.5, 0]."""
        Q = quaternion_kinematics_matrix()
        qdot = 0.5 * np.array(Q(Q_IDENTITY)) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(qdot.ravel(), [0.0, 0.0, 0.5, 0.0], atol=1e-15)

    def test_quaternion_kinematics_preserves_unit_norm(self):
        """``q . qdot`` must be zero for every attitude and every rate.

        This is where the missing minus sign on the bottom row shows up: with
        ``+qv'`` the product is ``2 qw (qv . omega)``, zero only when the
        vehicle is level or not rotating, so a straight-and-level scenario
        would never catch it.
        """
        Q = quaternion_kinematics_matrix()
        rng = np.random.default_rng(20260812)
        for _ in range(20):
            q = normalise(rng.normal(size=4))
            omega = rng.normal(size=3)
            qdot = 0.5 * np.array(Q(q)) @ omega
            assert float(q @ qdot.ravel()) == pytest.approx(0.0, abs=1e-12)


class TestTheHelperTable:

    def test_the_helper_table_is_what_components_declare(self):
        """Components name helpers as strings and the builder refuses an unknown
        one, so a rename cannot pass by leaving components asking for the old name."""
        table = helpers()
        assert list(table) == ["quaternion_to_rotation_matrix", "quaternion_kinematics_matrix"]
        assert table["quaternion_to_rotation_matrix"].size_out(0) == (3, 3)
        assert table["quaternion_kinematics_matrix"].size_out(0) == (4, 3)
