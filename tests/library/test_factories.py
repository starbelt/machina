"""
Direct numerical tests for the generic factories in ``machina.library``:
``cost.quadratic``, ``cost.rosenbrock`` and ``cost.least_squares``
(``machina.library.cost``), ``constraint.linear`` (``machina.library.constraint``),
``util.sum`` and ``util.rotate_x/y/z`` (``machina.library.util``).

Each factory's ``FunctionDescriptor.function`` is evaluated (and, for the costs,
differentiated at SX level) against a numpy reference written out here. The
registry mechanics are in ``test_registry.py`` beside this file; the solves
through ``cost.quadratic``, ``cost.rosenbrock`` and ``constraint.linear`` are its
``TestEndToEnd``.

Test classes
------------
TestQuadraticIsTheSumOfSquares          -- value x·x, gradient Qx with Q = 2I
TestRosenbrockVanishesOnlyAtItsMinimum  -- zero at (a, a²), positive elsewhere
TestLeastSquaresMatchesNumpyLstsq       -- value and gradient against np.linalg.lstsq
TestSumMatchesNumpy                     -- util.sum against np.sum
TestRotationsFollowTheRightHandRule     -- util.rotate_x/y/z against numpy matrices
TestLinearIsTheDotProduct               -- constraint.linear value c·x
"""

import casadi as ca
import numpy as np
import pytest

import machina.library  # noqa: F401  (registers the generic factories)
from machina.library import registry

pytestmark = pytest.mark.requires_casadi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value(fd, *inputs):
    """Evaluate a factory's function at numeric inputs; a numpy array, flattened."""
    return np.array(fd.function(*[ca.DM(v) for v in inputs])).flatten()


def _gradient(fd, x_value, *fixed):
    """
    Gradient of a scalar factory output with respect to its last input, the
    others held at the numeric values ``fixed``. Built at SX level through
    ``fd.function``.
    """
    x = ca.SX.sym('x', len(x_value))
    cost = fd.function(*[ca.DM(v) for v in fixed], x)
    grad = ca.Function('grad', [x], [ca.gradient(cost, x)])
    return np.array(grad(ca.DM(x_value))).flatten()


def _rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_NUMPY_ROTATION = {'x': _rx, 'y': _ry, 'z': _rz}


def _rotate(axis, v, theta):
    fd = registry.get(f'util.rotate_{axis}')()
    return _value(fd, v, theta)


def _rotation_matrix(axis, theta):
    """The factory's rotation matrix, column by column from the basis vectors."""
    return np.column_stack([_rotate(axis, e, theta) for e in np.eye(3)])


# A small overdetermined system: a straight line through five noisy points.
_T = np.arange(5.0)
_A = np.column_stack([np.ones_like(_T), _T])          # (5, 2)
_B = np.array([1.1, 2.9, 5.2, 6.8, 9.1])              # (5,)


# ---------------------------------------------------------------------------
# cost.quadratic
# ---------------------------------------------------------------------------

class TestQuadraticIsTheSumOfSquares:

    def test_the_value_is_x_dot_x(self):
        x = np.array([1.0, -2.0, 3.0])
        fd = registry.get('cost.quadratic')(n=3)
        np.testing.assert_allclose(_value(fd, x), [np.dot(x, x)], rtol=1e-14)

    def test_the_gradient_is_Qx_with_Q_twice_the_identity(self):
        """x·x = ½ xᵀQx with Q = 2I, so the gradient is Qx = 2x."""
        x = np.array([1.0, -2.0, 3.0])
        Q = 2.0 * np.eye(3)
        fd = registry.get('cost.quadratic')(n=3)
        np.testing.assert_allclose(_gradient(fd, x), Q @ x, rtol=1e-14)


# ---------------------------------------------------------------------------
# cost.rosenbrock
# ---------------------------------------------------------------------------

class TestRosenbrockVanishesOnlyAtItsMinimum:

    def test_the_value_is_zero_at_a_and_a_squared(self):
        for a, b in [(1.0, 100.0), (2.5, 10.0), (-1.5, 100.0)]:
            fd = registry.get('cost.rosenbrock')(a=a, b=b)
            np.testing.assert_allclose(_value(fd, [a, a**2]), [0.0], atol=1e-12,
                                       err_msg=f"a={a}, b={b}")

    def test_the_value_is_positive_elsewhere_and_matches_the_formula(self):
        a, b = 1.0, 100.0
        fd = registry.get('cost.rosenbrock')(a=a, b=b)
        for xy in ([0.0, 0.0], [a, a**2 + 0.1], [a + 0.1, a**2], [-1.0, 1.0]):
            value = _value(fd, xy)[0]
            expected = (a - xy[0])**2 + b * (xy[1] - xy[0]**2)**2
            assert value > 0.0, f"xy={xy}"
            np.testing.assert_allclose(value, expected, rtol=1e-12, err_msg=f"xy={xy}")


# ---------------------------------------------------------------------------
# cost.least_squares
# ---------------------------------------------------------------------------

class TestLeastSquaresMatchesNumpyLstsq:

    def test_the_value_at_the_lstsq_solution_is_half_the_residual_sum_of_squares(self):
        x_ls, residuals, _, _ = np.linalg.lstsq(_A, _B, rcond=None)
        fd = registry.get('cost.least_squares')(m=5, n=2)
        np.testing.assert_allclose(_value(fd, _A, _B, x_ls), [0.5 * residuals[0]], rtol=1e-12)

    def test_the_gradient_vanishes_at_the_lstsq_solution(self):
        x_ls = np.linalg.lstsq(_A, _B, rcond=None)[0]
        fd = registry.get('cost.least_squares')(m=5, n=2)
        np.testing.assert_allclose(_gradient(fd, x_ls, _A, _B), [0.0, 0.0], atol=1e-10)

    def test_the_gradient_elsewhere_is_A_transpose_times_the_residual(self):
        """The ½ convention: ∇ ½‖Ax − b‖² = Aᵀ(Ax − b), no factor of 2."""
        x = np.array([0.5, 1.5])
        fd = registry.get('cost.least_squares')(m=5, n=2)
        np.testing.assert_allclose(_gradient(fd, x, _A, _B), _A.T @ (_A @ x - _B), rtol=1e-12)


# ---------------------------------------------------------------------------
# util.sum
# ---------------------------------------------------------------------------

class TestSumMatchesNumpy:

    def test_the_total_is_np_sum_of_the_terms(self):
        for n_terms in (1, 4, 7):
            terms = 0.37 * np.arange(1.0, n_terms + 1.0) - 1.0
            fd = registry.get('util.sum')(n_terms=n_terms)
            np.testing.assert_allclose(_value(fd, terms), [np.sum(terms)], atol=1e-12,
                                       err_msg=f"n_terms={n_terms}")


# ---------------------------------------------------------------------------
# util.rotate_x / rotate_y / rotate_z
# ---------------------------------------------------------------------------

class TestRotationsFollowTheRightHandRule:

    def test_a_quarter_turn_carries_each_axis_onto_the_next(self):
        """Right-hand rule: about x, y → z; about y, z → x; about z, x → y."""
        e_x, e_y, e_z = np.eye(3)
        for axis, v, expected in (('x', e_y, e_z), ('y', e_z, e_x), ('z', e_x, e_y)):
            np.testing.assert_allclose(_rotate(axis, v, np.pi / 2), expected, atol=1e-15,
                                       err_msg=f"rotate_{axis}")

    @pytest.mark.parametrize('axis', ['x', 'y', 'z'])
    def test_a_generic_angle_matches_the_numpy_rotation(self, axis):
        theta = 0.7314
        v = np.array([0.3, -1.2, 2.5])
        np.testing.assert_allclose(_rotate(axis, v, theta), _NUMPY_ROTATION[axis](theta) @ v,
                                   atol=1e-14)

    def test_each_matrix_is_orthonormal_with_determinant_plus_one(self):
        theta = 0.7314
        for axis in ('x', 'y', 'z'):
            R = _rotation_matrix(axis, theta)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14, err_msg=f"rotate_{axis}")
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14,
                                       err_msg=f"rotate_{axis}")


# ---------------------------------------------------------------------------
# constraint.linear
# ---------------------------------------------------------------------------

class TestLinearIsTheDotProduct:

    def test_the_value_is_c_dot_x(self):
        c = [2.0, -1.0, 0.5, 3.0]
        x = np.array([1.0, 4.0, -2.0, 0.25])
        fd = registry.get('constraint.linear')(n=4, coefficients=c)
        np.testing.assert_allclose(_value(fd, x), [np.dot(c, x)], atol=1e-14)
