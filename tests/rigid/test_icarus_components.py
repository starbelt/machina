"""
icarus-dynamics components compile against machina unchanged.

That is the whole contract of Phase 2 with icarus (vault Component Graph):
a component written against icarus's ``Declaration`` -- only ``states``,
``algebraic``, ``inputs``, ``helpers``, ``derivatives`` and ``produces``, and
only the ``f`` and ``g`` keys -- must build with machina's ``Builder`` and give
the two-input ``f_system(x, u)`` its exporters expect.

``Kinematics`` and ``RigidBody`` below are icarus's ``model/components/
kinematics.py`` and ``rigid_body.py`` with only the import lines changed.
``MassProperties`` is a test stand-in for icarus's mass component, which
reads the params pipeline.
"""

import casadi as ca
import numpy as np
import pytest

from machina.model import Builder, Component, Declaration, SignalRegistry
from machina.rigid import frames
from machina.rigid import signals as rigid_signals

pytestmark = pytest.mark.requires_casadi


# --- icarus-dynamics model/components/kinematics.py, imports changed ----------------------------

class Kinematics(Component):
    """`p_i_dot = R_ib v_b` and `q_dot = 1/2 Q(q) w_b`."""

    def declare(self) -> Declaration:
        return Declaration(
            states=("inertial_position", "inertial_orientation_quaternion",
                    "body_linear_velocity", "body_angular_velocity"),
            helpers=("quaternion_to_rotation_matrix", "quaternion_kinematics_matrix"),
            derivatives=("inertial_position", "inertial_orientation_quaternion"),
        )

    def build(self, helpers: dict) -> dict:
        p = ca.SX.sym("inertial_position", 3)
        q = ca.SX.sym("inertial_orientation_quaternion", 4)
        v = ca.SX.sym("body_linear_velocity", 3)
        w = ca.SX.sym("body_angular_velocity", 3)
        R_ib = helpers["quaternion_to_rotation_matrix"](q)
        Q = helpers["quaternion_kinematics_matrix"](q)
        return {
            "f": ca.Function(
                "kinematics_f",
                [p, q, v, w],
                [R_ib @ v, 0.5 * (Q @ w)],
                ["inertial_position", "inertial_orientation_quaternion",
                 "body_linear_velocity", "body_angular_velocity"],
                ["inertial_position_dot", "inertial_orientation_quaternion_dot"],
            )
        }


# --- icarus-dynamics model/components/rigid_body.py, imports changed ----------------------------

class RigidBody(Component):
    """`F = m a` and `M = I w_dot + w x I w`, both resolved in body axes."""

    def declare(self) -> Declaration:
        return Declaration(
            states=("body_linear_velocity", "body_angular_velocity"),
            algebraic=("force_body", "moment_body", "mass", "inertia_tensor_body"),
            derivatives=("body_linear_velocity", "body_angular_velocity"),
            produces=("body_angular_acceleration",),
        )

    def build(self, helpers: dict) -> dict:
        v = ca.SX.sym("body_linear_velocity", 3)
        w = ca.SX.sym("body_angular_velocity", 3)
        force = ca.SX.sym("force_body", 3)
        moment = ca.SX.sym("moment_body", 3)
        mass = ca.SX.sym("mass")
        inertia = ca.SX.sym("inertia_tensor_body", 3, 3)

        v_dot = force / mass - ca.cross(w, v)
        w_dot = ca.solve(inertia, moment - ca.cross(w, inertia @ w))

        return {
            "f": ca.Function(
                "rigid_body_f",
                [v, w, force, moment, mass, inertia],
                [v_dot, w_dot],
                ["body_linear_velocity", "body_angular_velocity", "force_body", "moment_body",
                 "mass", "inertia_tensor_body"],
                ["body_linear_velocity_dot", "body_angular_velocity_dot"],
            ),
            "g": ca.Function(
                "rigid_body_g",
                [w, moment, inertia],
                [w_dot],
                ["body_angular_velocity", "moment_body", "inertia_tensor_body"],
                ["body_angular_acceleration"],
            ),
        }


class MassProperties(Component):
    """A 2 kg body with inertia diag(0.1, 0.2, 0.3)."""

    def declare(self) -> Declaration:
        return Declaration(produces=("mass", "inertia_tensor_body"))

    def build(self, helpers: dict) -> dict:
        return {"g": ca.Function("mass_properties_g", [],
                                 [ca.SX(2.0), ca.SX(ca.diag(ca.DM([0.1, 0.2, 0.3])))],
                                 [], ["mass", "inertia_tensor_body"])}


def rigid_registry() -> SignalRegistry:
    registry = SignalRegistry()
    rigid_signals.declare_into(registry)
    return registry


def six_dof(registry=None):
    return Builder([Kinematics(), RigidBody(), MassProperties()], frames.helpers(),
                   registry=registry or rigid_registry()).declare()


class TestIcarusComponentsCompile:

    def test_the_six_dof_body_builds_the_two_input_abi(self):
        model = six_dof().build()
        assert model["f"].name_in() == ["x", "u"]
        assert model["f"].name_out() == ["xdot"]

    def test_the_state_vector_follows_the_rigid_registry(self):
        builder = six_dof()
        assert builder.state_order == ["inertial_position", "inertial_orientation_quaternion",
                                       "body_linear_velocity", "body_angular_velocity"]
        assert builder.nx == 13 and builder.nu == 0

    def test_unproduced_forces_and_moments_are_zero(self):
        """No aero, no thrust, no gravity: a free body. SUM identity, not an error."""
        assert six_dof().unproduced_sums == ["force_body", "moment_body"]

    def test_a_body_moving_forward_at_identity_moves_north(self):
        builder = six_dof()
        model = builder.build()
        x = np.zeros(13)
        x[3:7] = [0.0, 0.0, 0.0, 1.0]         # identity attitude
        x[7:10] = [5.0, 0.0, 0.0]             # 5 m/s along body x
        xdot = np.array(model["f"](x, [])).ravel()
        np.testing.assert_allclose(xdot[0:3], [5.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(xdot[7:13], np.zeros(6), atol=1e-12)

    def test_a_ninety_degree_yaw_sends_body_forward_east(self):
        model = six_dof().build()
        x = np.zeros(13)
        x[3:7] = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
        x[7:10] = [5.0, 0.0, 0.0]
        xdot = np.array(model["f"](x, [])).ravel()
        np.testing.assert_allclose(xdot[0:3], [0.0, 5.0, 0.0], atol=1e-12)

    def test_the_gyroscopic_term_couples_rates_on_an_asymmetric_body(self):
        """Euler's equations with no moment: I w_dot = -w x I w."""
        model = six_dof().build()
        x = np.zeros(13)
        x[3:7] = [0.0, 0.0, 0.0, 1.0]
        x[10:13] = [1.0, 1.0, 0.0]
        w_dot = np.array(model["f"](x, [])).ravel()[10:13]
        inertia = np.diag([0.1, 0.2, 0.3])
        w = np.array([1.0, 1.0, 0.0])
        np.testing.assert_allclose(w_dot, np.linalg.solve(inertia, -np.cross(w, inertia @ w)))

    def test_the_published_angular_acceleration_matches_the_integrated_one(self):
        builder = six_dof()
        model = builder.build()
        x = np.zeros(13)
        x[3:7] = [0.0, 0.0, 0.0, 1.0]
        x[10:13] = [0.3, -0.2, 0.5]
        xdot = np.array(model["f"](x, [])).ravel()
        z = np.array(model["g"](x, [])).ravel()
        where = builder.layout(builder.algebraic_order)["body_angular_acceleration"]
        np.testing.assert_allclose(z[where], xdot[10:13])


class TestTheRigidSignals:

    def test_the_default_registry_carries_them_once_the_pack_is_imported(self):
        import machina.rigid  # noqa: F401
        from machina.model import signals
        for name in rigid_signals.SIGNALS:
            assert signals.DEFAULT.has(name)

    def test_declare_into_gives_a_clean_registry_the_same_set_in_order(self):
        assert list(rigid_registry().all()) == list(rigid_signals.SIGNALS)

    def test_forces_and_moments_sum_and_states_do_not(self):
        registry = rigid_registry()
        assert registry.get("force_body").aggregation.value == "sum"
        assert registry.get("moment_body").aggregation.value == "sum"
        assert registry.get("inertial_position").aggregation.value == "unique"

    def test_every_vector_signal_names_its_frame(self):
        registry = rigid_registry()
        assert registry.get("inertial_position").frame == "ned"
        assert registry.get("body_linear_velocity").frame == "frd"
        assert registry.frame("ned").family == "inertial"
        assert registry.frame("frd").family == "body"

    def test_the_inertia_tensor_is_a_matrix_signal(self):
        assert rigid_registry().get("inertia_tensor_body").shape == (3, 3)

    def test_importing_machina_does_not_import_the_pack(self):
        """Decision Log #37: packs are imported explicitly. The subprocess gets this
        tree's src/ first on its path, so it tests this checkout, not an installed copy."""
        import os
        import subprocess
        import sys
        from pathlib import Path
        src = str(Path(__file__).resolve().parents[2] / "src")
        env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, machina; print(machina.__file__); print('machina.rigid' in sys.modules, "
             "'machina.aero' in sys.modules)"],
            capture_output=True, text=True, check=True, env=env)
        where, flags = out.stdout.splitlines()[0], out.stdout.splitlines()[1]
        assert Path(where).resolve().is_relative_to(Path(src).resolve())
        assert flags.split() == ["False", "False"]
