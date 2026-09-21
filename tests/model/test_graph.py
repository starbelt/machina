"""
The builder: what it refuses, and what it lays out.

Ported from icarus-dynamics ``tests/test_model_graph.py`` onto the synthetic
point mass, because icarus's version leans on its aircraft model. Written as
"this must be refused" rather than "this works", because the graph working is
demonstrated by the other files in this directory; what needs pinning is the
set of mistakes it will not let through. Every one of these was possible in
the code both lineages descend from.
"""

import casadi as ca
import numpy as np
import pytest
from synthetic_model import (
    Drag,
    Dynamics,
    Kinematics,
    MassSource,
    Thruster,
    make_registry,
    point_mass,
)

from machina.model import Builder, Component, Declaration, ModelError, SignalError

pytestmark = pytest.mark.requires_casadi


def declared(components, registry=None, helpers=None):
    return Builder(components, helpers, registry=registry or make_registry()).declare()


class TestTheGraphWorks:

    def test_the_synthetic_ode_builds_the_two_input_abi(self):
        model = declared(point_mass()).build(fixed={"dry_mass": 2.0})
        assert model["f"].name_in() == ["x", "u"]
        assert model["f"].name_out() == ["xdot"]

    def test_the_derivative_is_the_physics_on_paper(self):
        # pos_dot = vel; vel_dot = force / mass = [cmd * mass, 0] / mass = [cmd, 0].
        model = declared(point_mass()).build(fixed={"dry_mass": 2.0})
        xdot = np.array(model["f"]([1.0, 2.0, 3.0, 4.0], [0.5])).ravel()
        np.testing.assert_allclose(xdot, [3.0, 4.0, 0.5, 0.0])

    def test_g_system_reports_the_algebraic_signals(self):
        builder = declared(point_mass())
        model = builder.build(fixed={"dry_mass": 2.0})
        assert builder.algebraic_order == ["force", "mass"]
        z = np.array(model["g"]([0, 0, 0, 0], [0.5])).ravel()
        np.testing.assert_allclose(z, [1.0, 0.0, 2.0])

    def test_a_sum_signal_adds_every_producer(self):
        builder = declared([*point_mass(), Drag()])
        model = builder.build(fixed={"dry_mass": 2.0})
        # force = thruster [cmd*mass, 0] + drag -0.5*vel
        z = np.array(model["g"]([0, 0, 2.0, 4.0], [0.5])).ravel()
        np.testing.assert_allclose(z[:2], [1.0 - 1.0, -2.0])

    def test_an_unproduced_sum_signal_is_zero_not_an_error(self):
        """Dropping a component should give a body with no thrust, not a build failure.

        That is the identity of addition, and it is what makes a reduced model
        expressible at all.
        """
        builder = declared([Kinematics(), Dynamics(), MassSource()])
        assert builder.unproduced_sums == ["force"]
        model = builder.build(fixed={"dry_mass": 2.0})
        # The (x, u) signature is kept even with no inputs, so u is a 0-width argument.
        assert builder.nu == 0 and model["f"].name_in() == ["x", "u"]
        np.testing.assert_allclose(
            np.array(model["f"]([0, 0, 1.0, 1.0], [])).ravel(), [1.0, 1.0, 0.0, 0.0])


class TestWhatTheBuilderRefuses:

    def test_a_shape_disagreement_fails_at_build_naming_the_signal(self):
        """The inherited bug, re-made on purpose.

        ``f(q)[0]`` takes element [0, 0] of the result -- a (1, 1) where a
        (2, 1) belongs. In the predecessor this built cleanly and integrated
        position against a scalar.
        """
        class BuggyKinematics(Kinematics):
            def build(self, helpers):
                vel = ca.SX.sym("vel", 2)
                return {"f": ca.Function("kinematics_f", [vel], [vel[0]], ["vel"], ["pos_dot"])}

        with pytest.raises(ModelError, match=r"pos is declared \(2, 1\)"):
            declared([BuggyKinematics(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_the_shape_error_names_the_likely_cause(self):
        class BuggyKinematics(Kinematics):
            def build(self, helpers):
                vel = ca.SX.sym("vel", 2)
                return {"f": ca.Function("kinematics_f", [vel], [vel[0]], ["vel"], ["pos_dot"])}

        with pytest.raises(ModelError, match=r"indexed with \[0\]"):
            declared([BuggyKinematics(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_two_producers_of_a_unique_signal_is_an_error(self):
        class SecondMass(Component):
            def declare(self):
                return Declaration(produces=("mass",))

            def build(self, helpers):
                raise AssertionError("declare() must fail before build() is reached")

        with pytest.raises(ModelError, match="declared UNIQUE"):
            declared([*point_mass(), SecondMass()])

    def test_the_unique_error_offers_sum_as_the_alternative(self):
        class SecondMass(Component):
            def declare(self):
                return Declaration(produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="the signal should be SUM"):
            declared([*point_mass(), SecondMass()])

    def test_an_undeclared_helper_is_refused_with_the_available_names(self):
        class NeedsMagic(Component):
            def declare(self):
                return Declaration(helpers=("magic",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="undeclared helper"):
            declared([NeedsMagic()], helpers={"real": ca.Function("real", [], [])})

    def test_a_unique_algebraic_signal_with_no_producer_is_refused(self):
        """Otherwise it reads as whatever the vector was initialised to -- zero,
        and a zero mass is a division by zero inside the integrator."""
        with pytest.raises(ModelError, match="no component produces"):
            declared([Kinematics(), Dynamics(), Thruster()])

    def test_a_state_nothing_integrates_is_refused(self):
        class ReadsOnly(Component):
            def declare(self):
                return Declaration(states=("pos",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="nothing integrates"):
            declared([ReadsOnly()])

    def test_the_floating_state_error_says_it_is_really_an_input(self):
        class ReadsOnly(Component):
            def declare(self):
                return Declaration(states=("pos",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="it is an input, not a state"):
            declared([ReadsOnly()])

    def test_an_algebraic_cycle_is_reported_with_the_components_in_it(self):
        """Two components each waiting on the other's output is an implicit
        system. The substitution in build() depends on there being no cycle."""
        class A(Component):
            def declare(self):
                return Declaration(algebraic=("force",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        class B(Component):
            def declare(self):
                return Declaration(algebraic=("mass",), produces=("force",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="cycle"):
            declared([Kinematics(), Dynamics(), A(), B()])

    def test_the_cycle_error_says_it_needs_a_solver_not_a_sort(self):
        class A(Component):
            def declare(self):
                return Declaration(algebraic=("force",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        class B(Component):
            def declare(self):
                return Declaration(algebraic=("mass",), produces=("force",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="needs a solver, not a sort"):
            declared([Kinematics(), Dynamics(), A(), B()])

    def test_a_component_cannot_read_what_it_did_not_declare(self):
        """Reading an undeclared signal would work by accident -- it is already
        in the value map -- and leave a dependency the sort cannot see."""
        class Sneaky(Component):
            def declare(self):
                return Declaration(states=("vel",), derivatives=("vel",))

            def build(self, helpers):
                vel, mass = ca.SX.sym("vel", 2), ca.SX.sym("mass")
                return {"f": ca.Function("sneaky_f", [vel, mass], [vel * mass],
                                         ["vel", "mass"], ["vel_dot"])}

        with pytest.raises(ModelError, match="does not declare that it reads"):
            declared([Kinematics(), Sneaky()]).build()

    def test_a_wrong_output_count_says_they_are_matched_positionally(self):
        class TooMany(Kinematics):
            def build(self, helpers):
                vel = ca.SX.sym("vel", 2)
                return {"f": ca.Function("kinematics_f", [vel], [vel, vel],
                                         ["vel"], ["pos_dot", "spare"])}

        with pytest.raises(ModelError, match="matched positionally"):
            declared([TooMany(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_two_components_may_not_share_an_instance_path(self):
        with pytest.raises(ModelError, match="share the instance path"):
            Builder([Kinematics(), Kinematics()], registry=make_registry())

    def test_a_distinguishing_suffix_makes_two_instances_legal(self):
        assert Kinematics("left").name == "kinematics_left"
        assert Kinematics("right").name == "kinematics_right"

    def test_an_undeclared_signal_names_what_to_do_about_it(self):
        class Typo(Component):
            def declare(self):
                return Declaration(states=("velocty",), derivatives=("velocty",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(SignalError, match="not a new signal"):
            declared([Typo()])

    def test_a_signal_cannot_be_both_integrated_and_computed(self):
        class Both(Component):
            def declare(self):
                return Declaration(states=("vel",), derivatives=("mass",), produces=("mass",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="either integrated or computed"):
            declared([Both()])

    def test_build_returns_a_dict_of_functions_or_says_so(self):
        class NotADict(Kinematics):
            def build(self, helpers):
                return ca.Function("f", [], [])

        with pytest.raises(ModelError, match="must return a dict"):
            declared([NotADict(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_an_unknown_build_key_lists_the_valid_ones(self):
        class Typo(Kinematics):
            def build(self, helpers):
                vel = ca.SX.sym("vel", 2)
                return {"fn": ca.Function("kinematics_f", [vel], [vel], ["vel"], ["pos_dot"])}

        with pytest.raises(ModelError, match="unknown key"):
            declared([Typo(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_a_model_is_components_and_scopes_and_nothing_else(self):
        with pytest.raises(ModelError, match="neither a Component nor a Scope"):
            Builder(["kinematics"], registry=make_registry())

    def test_wire_and_build_require_declare_first(self):
        builder = Builder(point_mass(), registry=make_registry())
        with pytest.raises(ModelError, match="call declare"):
            builder.build()
        with pytest.raises(ModelError, match="call declare"):
            builder.wire({})


class TestLayout:

    def test_the_layout_follows_the_registry_not_the_component_list(self):
        """Reordering the component list must not silently re-lay-out the state
        vector: an archived trajectory would then be unreadable against a model
        that is physically identical."""
        registry = make_registry()
        forward = declared(point_mass(), registry)
        reversed_ = declared(list(reversed(point_mass())), registry)
        assert forward.state_order == reversed_.state_order == ["pos", "vel"]
        assert forward.layout(forward.state_order) == reversed_.layout(reversed_.state_order)

    def test_the_layout_follows_the_registry_not_the_declaration_order(self):
        registry = make_registry()
        builder = declared(point_mass(), registry)
        # Dynamics declares vel_dot, Kinematics declares pos_dot; the registry
        # declares pos before vel, and that is what the vector follows.
        assert builder.state_order == ["pos", "vel"]

    def test_every_element_is_accounted_for(self):
        registry = make_registry()
        builder = declared(point_mass(), registry)
        assert sum(registry.get(n).size for n in builder.state_order) == builder.nx == 4
        assert sum(registry.get(n).size for n in builder.input_order) == builder.nu == 1

    def test_layout_gives_a_slice_per_path(self):
        builder = declared(point_mass())
        assert builder.layout(builder.state_order) == {
            "pos": slice(0, 2), "vel": slice(2, 4),
        }

    def test_the_topological_order_puts_producers_before_consumers(self):
        builder = declared(point_mass())
        order = [item.path for item in builder._order]
        assert order.index("mass_source") < order.index("thruster")
        assert order.index("thruster") < order.index("dynamics")

    def test_the_topological_tie_break_is_the_order_the_caller_wrote(self):
        registry = make_registry()
        forward = [item.path for item in declared(point_mass(), registry)._order]
        backward = [item.path for item in declared(list(reversed(point_mass())), registry)._order]
        assert forward[0] == "kinematics" and backward[0] == "mass_source"

    def test_producers_are_reported_by_instance_path(self):
        builder = declared([*point_mass(), Drag()])
        assert builder.producers()["force"] == ("thruster", "drag")
        assert builder.producers()["mass"] == ("mass_source",)
