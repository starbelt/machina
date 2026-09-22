"""
Scopes: how two of the same thing become ordinary instances.

``Scope("sat_a", [...])`` prefixes every instance path its members produce or
own. Consumed names resolve lexically -- the component's own scope, then each
parent, then the root -- so a shared signal reads naturally from inside a
nested scope and an absolute ``/sat_a/pos`` bypasses the search entirely.
That is what replaces April's rule that agents cannot see each other
(Decision Log #22, superseded by #36 and #48).
"""

import casadi as ca
import numpy as np
import pytest
from synthetic_model import Dynamics, Kinematics, MassSource, Thruster, make_registry, point_mass

from machina.model import Builder, Component, Declaration, ModelError, Quantity, Scope

pytestmark = pytest.mark.requires_casadi


def declared(components, registry=None):
    return Builder(components, registry=registry or make_registry()).declare()


class Separation(Component):
    """The cross-scope coupling April's design had no room for.

    An ordinary component that consumes two absolute paths. Decision Log #22
    ("agents cannot see each other") is superseded by exactly this.
    """

    def declare(self):
        return Declaration(states=(("pos_a", "/a/pos"), ("pos_b", "/b/pos")),
                           produces=("separation",))

    def build(self, helpers):
        pos_a, pos_b = ca.SX.sym("pos_a", 2), ca.SX.sym("pos_b", 2)
        return {"g": ca.Function("separation_g", [pos_a, pos_b], [ca.sumsqr(pos_a - pos_b)],
                                 ["pos_a", "pos_b"], ["separation"])}


class SpareMass(Component):
    """Owns a dry_mass of its own and produces nothing."""

    def declare(self):
        return Declaration(quantities=(Quantity("dry_mass", unit="kg"),))

    def build(self, helpers):
        return {}


def two_vehicles(extra=()):
    return [Scope("a", point_mass()), Scope("b", point_mass()), *extra]


class TestFlattening:

    def test_a_scope_prefixes_every_path_its_members_touch(self):
        builder = declared([Scope("sat", point_mass())])
        assert builder.state_order == ["sat/pos", "sat/vel"]
        assert builder.input_order == ["sat/thrust_cmd"]
        assert builder.quantity_order == ["sat/dry_mass"]
        assert builder.algebraic_order == ["sat/force", "sat/mass"]

    def test_two_scopes_are_two_independent_instances(self):
        builder = declared(two_vehicles())
        assert builder.state_order == ["a/pos", "b/pos", "a/vel", "b/vel"]
        assert (builder.nx, builder.nu, builder.nq) == (8, 2, 2)

    def test_the_layout_is_registry_first_then_the_order_the_scopes_were_written(self):
        # pos before vel comes from the registry; b before a from the list.
        builder = declared([Scope("b", point_mass()), Scope("a", point_mass())])
        assert builder.state_order == ["b/pos", "a/pos", "b/vel", "a/vel"]

    def test_scopes_nest(self):
        builder = declared([Scope("fleet", [Scope("sat", point_mass())])])
        assert builder.state_order == ["fleet/sat/pos", "fleet/sat/vel"]
        assert builder.placed[0].path == "fleet/sat/kinematics"

    def test_the_same_component_class_in_two_scopes_is_not_a_name_collision(self):
        builder = declared(two_vehicles())
        assert [item.path for item in builder.placed][:2] == ["a/kinematics", "a/dynamics"]
        assert "b/kinematics" in [item.path for item in builder.placed]

    def test_a_duplicate_instance_path_inside_one_scope_is_refused(self):
        with pytest.raises(ModelError, match="share the instance path"):
            Builder([Scope("a", [Kinematics(), Kinematics()])], registry=make_registry())

    def test_a_scope_name_must_be_an_identifier(self):
        with pytest.raises(ModelError, match="must be a Python identifier"):
            Scope("sat a", [Kinematics()])

    def test_an_empty_scope_is_a_typo(self):
        with pytest.raises(ModelError, match="a scope with nothing in it is a typo"):
            Scope("sat", [])

    def test_a_model_needs_at_least_one_component(self):
        with pytest.raises(ModelError, match="at least one component"):
            Builder([], registry=make_registry())


class TestLexicalResolution:

    def test_a_consumed_name_finds_the_producer_in_its_own_scope_first(self):
        builder = declared(two_vehicles())
        assert builder.resolved("a/dynamics")["mass"] == "a/mass"
        assert builder.resolved("b/dynamics")["mass"] == "b/mass"

    def test_a_consumed_name_falls_back_to_a_parent_scope(self):
        """One mass source at the root serves a scoped consumer."""
        builder = declared([MassSource(),
                            Scope("inner", [Kinematics(), Dynamics(), Thruster()])])
        assert builder.resolved("inner/dynamics")["mass"] == "mass"
        assert builder.resolved("inner/thruster")["mass"] == "mass"
        assert builder.state_order == ["inner/pos", "inner/vel"]

    def test_lookup_walks_through_intermediate_scopes(self):
        """fleet/sat/bus reads mass: not in bus, not in sat, found in fleet."""
        builder = declared([Scope("fleet", [
            MassSource(),
            Scope("sat", [Scope("bus", [Kinematics(), Dynamics(), Thruster()])]),
        ])])
        assert builder.resolved("fleet/sat/bus/dynamics")["mass"] == "fleet/mass"
        assert builder.resolved("fleet/sat/bus/thruster")["mass"] == "fleet/mass"

    def test_an_intermediate_producer_wins_over_the_root(self):
        builder = declared([MassSource(), Scope("sat", [
            MassSource(), Scope("bus", [Kinematics(), Dynamics(), Thruster()])])])
        assert builder.resolved("sat/bus/dynamics")["mass"] == "sat/mass"

    def test_the_innermost_producer_wins_over_an_outer_one(self):
        builder = declared([MassSource(),
                            Scope("inner", [Kinematics(), Dynamics(), Thruster(), MassSource()])])
        assert builder.resolved("inner/dynamics")["mass"] == "inner/mass"

    def test_an_absolute_reference_bypasses_the_search(self):
        builder = declared(two_vehicles([Separation()]))
        assert builder.resolved("separation") == {"pos_a": "a/pos", "pos_b": "b/pos"}

    def test_a_cross_scope_component_wires_and_evaluates(self):
        builder = declared(two_vehicles([Separation()]))
        model = builder.build(fixed={"a/dry_mass": 2.0, "b/dry_mass": 2.0})
        where = builder.layout(builder.algebraic_order)["separation"]
        # x = [a/pos, b/pos, a/vel, b/vel]; (0, 0) to (3, 4) is 25.
        z = np.array(model["g"]([0, 0, 3, 4, 0, 0, 0, 0], [0.0, 0.0])).ravel()
        np.testing.assert_allclose(z[where], [25.0])

    def test_an_explicit_local_name_lets_one_component_read_two_instances(self):
        builder = declared(two_vehicles([Separation()]))
        assert builder.resolved("separation")["pos_a"] != builder.resolved("separation")["pos_b"]

    def test_an_input_resolves_to_its_own_scope_so_instances_do_not_share_one(self):
        builder = declared(two_vehicles())
        assert builder.resolved("a/thruster")["thrust_cmd"] == "a/thrust_cmd"
        assert builder.resolved("b/thruster")["thrust_cmd"] == "b/thrust_cmd"

    def test_an_absolute_input_reference_shares_one_command(self):
        class SharedThruster(Thruster):
            def declare(self):
                return Declaration(inputs=(("thrust_cmd", "/thrust_cmd"),),
                                   algebraic=("mass",), produces=("force",))

        def vehicle():
            return [Kinematics(), Dynamics(), MassSource(), SharedThruster()]

        builder = declared([Scope("a", vehicle()), Scope("b", vehicle())])
        assert builder.input_order == ["thrust_cmd"]
        assert builder.nu == 1

    def test_producers_always_produce_into_their_own_scope(self):
        builder = declared(two_vehicles())
        assert builder.producers()["a/mass"] == ("a/mass_source",)
        assert builder.producers()["b/mass"] == ("b/mass_source",)

    def test_resolution_requires_declare_first(self):
        builder = Builder(point_mass(), registry=make_registry())
        with pytest.raises(ModelError, match="call declare"):
            builder.resolved("kinematics")


class TestQuantityPaths:

    def test_a_quantity_is_named_in_its_component_scope(self):
        builder = declared([Scope("sat", point_mass())])
        assert [path for path, _, _ in builder.quantities()] == ["sat/dry_mass"]

    def test_quantities_report_their_owning_component(self):
        path, quantity, owner = declared([Scope("sat", point_mass())]).quantities()[0]
        assert (path, quantity.name, owner) == ("sat/dry_mass", "dry_mass", "sat/mass_source")

    def test_two_owners_of_one_quantity_path_are_refused(self):
        with pytest.raises(ModelError, match="both own a quantity"):
            declared([*point_mass(), SpareMass()])

    def test_the_collision_error_suggests_separate_scopes(self):
        with pytest.raises(ModelError, match="separate scopes"):
            declared([*point_mass(), SpareMass()])

    def test_the_same_owner_in_a_separate_scope_is_fine(self):
        builder = declared([*point_mass(), Scope("spare", [SpareMass()])])
        assert builder.quantity_order == ["dry_mass", "spare/dry_mass"]

    def test_the_same_quantity_in_two_scopes_is_two_quantities(self):
        builder = declared(two_vehicles())
        assert builder.quantity_order == ["a/dry_mass", "b/dry_mass"]

    def test_a_quantity_may_not_collide_with_a_produced_signal(self):
        class Colliding(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("mass"),), produces=("force",))

            def build(self, helpers):
                raise AssertionError("unreachable")

        with pytest.raises(ModelError, match="collides with the signal"):
            declared([*point_mass(), Colliding()])
