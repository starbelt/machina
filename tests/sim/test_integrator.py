"""
Fixed-step RK4, numerically and symbolically, against solutions on paper.

The builder's ``f_system(x, u)`` goes straight in: that is the point of
substituting algebraic signals out. The step function is what gets exported,
so it has to agree with the numeric stepper to the last bit a test can see.
"""

import casadi as ca
import numpy as np
import pytest

from machina.model import Builder, Component, Declaration, Quantity, SignalRegistry
from machina.rigid import frames
from machina.sim import build_step_function, rk4_step

pytestmark = pytest.mark.requires_casadi


def decay():
    """xdot = -x + u, whose exact solution with constant u is known."""
    x, u = ca.SX.sym("x"), ca.SX.sym("u")
    return ca.Function("f", [x, u], [-x + u], ["x", "u"], ["xdot"])


class TestRk4Step:

    def test_one_step_is_fourth_order_accurate(self):
        f = decay()
        step = rk4_step(lambda x, u: np.array(f(x, u)).ravel(), np.array([1.0]),
                        np.array([0.0]), 0.1)
        np.testing.assert_allclose(step, [np.exp(-0.1)], rtol=1e-6)

    def test_the_error_falls_sixteen_fold_when_the_step_halves(self):
        f = decay()

        def integrate(dt):
            x = np.array([1.0])
            for _ in range(int(round(1.0 / dt))):
                x = rk4_step(lambda x, u: np.array(f(x, u)).ravel(), x, np.array([0.0]), dt)
            return abs(x[0] - np.exp(-1.0))

        ratio = integrate(0.1) / integrate(0.05)
        assert 14.0 < ratio < 18.0

    def test_the_input_is_held_across_the_step(self):
        """Zero-order hold: the equilibrium of xdot = -x + u is u."""
        f = decay()
        x = np.array([0.5])
        for _ in range(200):
            x = rk4_step(lambda x, u: np.array(f(x, u)).ravel(), x, np.array([0.5]), 0.05)
        np.testing.assert_allclose(x, [0.5], atol=1e-12)


class TestBuildStepFunction:

    def test_it_matches_the_numeric_stepper(self):
        f = decay()
        step = build_step_function(f)
        symbolic = float(step(0.7, 0.2, 0.05))
        numeric = rk4_step(lambda x, u: np.array(f(x, u)).ravel(), np.array([0.7]),
                           np.array([0.2]), 0.05)[0]
        assert symbolic == pytest.approx(numeric, rel=1e-15)

    def test_its_signature_is_x_u_dt(self):
        step = build_step_function(decay())
        assert step.name_in() == ["x", "u", "dt"] and step.name_out() == ["x_next"]
        assert step.name() == "plant_step"

    def test_normalise_holds_a_quaternion_on_the_unit_sphere(self):
        """RK4 works in the embedding space; without renormalisation the norm drifts."""
        Q = frames.quaternion_kinematics_matrix()
        q, w = ca.SX.sym("q", 4), ca.SX.sym("w", 3)
        f = ca.Function("f", [q, w], [0.5 * (Q(q) @ w)], ["x", "u"], ["xdot"])
        raw = build_step_function(f)
        held = build_step_function(f, normalise=slice(0, 4))
        state_raw = state_held = np.array([0.0, 0.0, 0.0, 1.0])
        rate = np.array([3.0, -2.0, 5.0])
        for _ in range(100):
            state_raw = np.array(raw(state_raw, rate, 0.1)).ravel()
            state_held = np.array(held(state_held, rate, 0.1)).ravel()
        # Unnormalised RK4 drifts off the sphere (about 6e-4 here); normalised stays on it.
        assert abs(np.linalg.norm(state_raw) - 1.0) > 1e-5
        assert np.linalg.norm(state_held) == pytest.approx(1.0, abs=1e-14)

    def test_normalise_leaves_the_other_states_alone(self):
        f = ca.Function("f", [ca.SX.sym("x", 3), ca.SX.sym("u")],
                        [ca.DM([1.0, 0.0, 0.0])], ["x", "u"], ["xdot"])
        step = build_step_function(f, normalise=slice(1, 3))
        np.testing.assert_allclose(np.array(step([0.0, 3.0, 4.0], 0.0, 0.5)).ravel(),
                                   [0.5, 0.6, 0.8])

    def test_a_three_input_f_is_refused_naming_fixed(self):
        """A model with free quantities builds f_system(x, u, q)."""
        x, u, q = ca.SX.sym("x"), ca.SX.sym("u"), ca.SX.sym("q")
        f = ca.Function("f_system", [x, u, q], [x * q + u], ["x", "u", "q"], ["xdot"])
        with pytest.raises(ValueError, match="fixed="):
            build_step_function(f)

    def test_a_normalise_slice_outside_the_state_is_refused(self):
        with pytest.raises(ValueError, match="outside the state vector"):
            build_step_function(decay(), normalise=slice(0, 4))

    def test_a_strided_normalise_slice_is_refused(self):
        f = ca.Function("f", [ca.SX.sym("x", 4), ca.SX.sym("u")], [ca.DM.zeros(4)],
                        ["x", "u"], ["xdot"])
        with pytest.raises(ValueError, match="contiguous"):
            build_step_function(f, normalise=slice(0, 4, 2))


class TestTheBuilderFeedsIt:

    def test_a_built_model_steps_after_its_quantities_are_fixed(self):
        registry = SignalRegistry()
        registry.declare("level", 1, "m")
        registry.declare("inflow", 1, "m/s")

        class Tank(Component):
            def declare(self):
                return Declaration(states=("level",), inputs=("inflow",),
                                   quantities=(Quantity("leak", unit="1/s", default=1.0),),
                                   derivatives=("level",))

            def build(self, helpers):
                h, qin, k = ca.SX.sym("level"), ca.SX.sym("inflow"), ca.SX.sym("leak")
                return {"f": ca.Function("tank_f", [h, qin, k], [qin - k * h],
                                         ["level", "inflow", "leak"], ["level_dot"])}

        builder = Builder([Tank()], registry=registry).declare()
        with pytest.raises(ValueError, match="fixed="):
            build_step_function(builder.build()["f"])
        step = build_step_function(builder.build(fixed={"leak": 1.0})["f"])
        np.testing.assert_allclose(float(step(1.0, 0.0, 0.1)), np.exp(-0.1), rtol=1e-6)
