"""
Quantities: the leaves a component owns.

A quantity is not a registry signal. It carries its own shape and unit, only
its owner can read it, and it becomes a decision variable, a solver parameter
or a fixed number when the problem is compiled. The builder's job is narrower:
give it a path, put it in a vector, and hand it to ``build()`` by name.

The ABI matters as much as the mechanism. ``f_system`` keeps the two-input
``(x, u)`` signature when no quantities are free, so a model made only of
icarus-style components -- and everything downstream that reads
``f.size1_in(0)`` and ``f.size1_in(1)`` -- is unaffected by the addition.
"""

import casadi as ca
import numpy as np
import pytest
from synthetic_model import Budget, Kinematics, make_registry, point_mass

from machina.model import Builder, Component, Declaration, ModelError, Quantity, Role, Scope

pytestmark = pytest.mark.requires_casadi


def declared(components, registry=None):
    return Builder(components, registry=registry or make_registry()).declare()


class TestQuantitiesAreLeaves:

    def test_a_quantity_reaches_build_by_name(self):
        builder = declared(point_mass())
        assert builder.quantity_order == ["dry_mass"]
        model = builder.build()
        # vel_dot = force / mass = [cmd * dry_mass, 0] / dry_mass = [cmd, 0], for any dry_mass.
        for dry_mass in (1.0, 7.5):
            xdot = np.array(model["f"]([0, 0, 0, 0], [0.4], [dry_mass])).ravel()
            np.testing.assert_allclose(xdot[2:], [0.4, 0.0])

    def test_a_quantity_carries_its_own_shape_and_bounds(self):
        _, quantity, _ = declared(point_mass()).quantities()[0]
        assert (quantity.shape, quantity.unit) == ((1, 1), "kg")
        assert (quantity.lb, quantity.ub, quantity.default) == (0.1, 10.0, 2.0)

    def test_a_matrix_quantity_is_unpacked_column_major(self):
        class Gains(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("gain", shape=(2, 2), unit="1"),),
                                   produces=("mass",))

            def build(self, helpers):
                gain = ca.SX.sym("gain", 2, 2)
                return {"g": ca.Function("gains_g", [gain], [gain[1, 0]], ["gain"], ["mass"])}

        builder = declared([Gains()])
        model = builder.build()
        assert builder.nq == 4
        # Column-major: element [1, 0] is the second entry of the flat vector.
        np.testing.assert_allclose(np.array(model["g"]([], [], [10, 20, 30, 40])).ravel(), [20.0])

    def test_quantity_order_is_declaration_order_within_component_order(self):
        builder = declared([*point_mass(), Budget()])
        assert builder.quantity_order == ["dry_mass", "cap"]

    def test_nq_counts_elements_not_quantities(self):
        class Gains(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("gain", shape=(2, 3), unit="1"),),
                                   produces=("mass",))

            def build(self, helpers):
                gain = ca.SX.sym("gain", 2, 3)
                return {"g": ca.Function("gains_g", [gain], [gain[0, 0]], ["gain"], ["mass"])}

        assert declared([Gains()]).nq == 6


class TestTheAbiIsPreserved:

    def test_no_quantities_means_the_two_input_signature(self):
        class Constant(Component):
            def declare(self):
                return Declaration(produces=("mass",))

            def build(self, helpers):
                return {"g": ca.Function("constant_g", [], [ca.SX(2.0)], [], ["mass"])}

        model = declared([Kinematics(), *point_mass()[1:2], Constant(),
                          *point_mass()[3:]]).build()
        assert model["f"].name_in() == ["x", "u"]

    def test_free_quantities_add_a_third_input(self):
        model = declared(point_mass()).build()
        assert model["f"].name_in() == ["x", "u", "q"]
        assert model["g"].name_in() == ["x", "u", "q"]

    def test_fixing_every_quantity_restores_the_two_input_signature(self):
        """This is the escape hatch for the exporters, which read size1_in(0) and (1)."""
        model = declared(point_mass()).build(fixed={"dry_mass": 2.0})
        assert model["f"].name_in() == ["x", "u"]

    def test_a_fixed_quantity_is_substituted_as_a_constant(self):
        builder = declared(point_mass())
        free = builder.build()
        fixed = builder.build(fixed={"dry_mass": 3.0})
        np.testing.assert_allclose(
            np.array(fixed["g"]([0, 0, 0, 0], [0.5])).ravel(),
            np.array(free["g"]([0, 0, 0, 0], [0.5], [3.0])).ravel())

    def test_fixing_only_some_quantities_keeps_the_rest_free(self):
        builder = declared([*point_mass(), Budget()])
        model = builder.build(fixed={"dry_mass": 2.0})
        assert model["f"].name_in() == ["x", "u", "q"]
        assert model["h"].name_in() == ["x", "u", "q"]

    def test_a_matrix_quantity_can_be_fixed_with_a_matrix(self):
        class Gains(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("gain", shape=(2, 2), unit="1"),),
                                   produces=("mass",))

            def build(self, helpers):
                gain = ca.SX.sym("gain", 2, 2)
                return {"g": ca.Function("gains_g", [gain], [gain[1, 0]], ["gain"], ["mass"])}

        model = declared([Gains()]).build(fixed={"gain": [[1.0, 2.0], [3.0, 4.0]]})
        np.testing.assert_allclose(np.array(model["g"]([], [])).ravel(), [3.0])

    def test_fixing_an_unknown_path_lists_the_real_ones(self):
        with pytest.raises(ModelError, match="are not quantities of this model"):
            declared(point_mass()).build(fixed={"dry_masss": 2.0})


class TestConstraintsAndCosts:

    def test_declared_constraints_and_costs_become_h_and_J(self):
        model = declared([*point_mass(), Budget()]).build(
            fixed={"dry_mass": 2.0, "cap": 0.8})
        assert model["h"].name_out() == ["h"]
        assert model["J"].name_out() == ["J"]
        # headroom = cap - thrust_cmd; effort = thrust_cmd**2
        np.testing.assert_allclose(np.array(model["h"]([0, 0, 0, 0], [0.3])).ravel(), [0.5])
        np.testing.assert_allclose(np.array(model["J"]([0, 0, 0, 0], [0.3])).ravel(), [0.09])

    def test_a_model_without_them_emits_no_h_or_J(self):
        model = declared(point_mass()).build(fixed={"dry_mass": 2.0})
        assert "h" not in model and "J" not in model

    def test_constraints_report_bounds_docs_and_owner(self):
        path, constraint, owner = declared([*point_mass(), Budget()]).constraints()[0]
        assert (path, constraint.lb, owner) == ("headroom", 0.0, "budget")
        assert constraint.ub == float("inf")
        assert constraint.doc

    def test_costs_report_their_declared_weight(self):
        path, cost, owner = declared([*point_mass(), Budget()]).costs()[0]
        assert (path, cost.weight, owner) == ("effort", 2.0, "budget")

    def test_a_scoped_constraint_is_named_in_its_scope(self):
        builder = declared([Scope("sat", [*point_mass(), Budget()])])
        assert [path for path, _, _ in builder.constraints()] == ["sat/headroom"]
        assert [path for path, _, _ in builder.costs()] == ["sat/effort"]

    def test_a_constraint_shape_disagreement_is_caught(self):
        class WrongShape(Budget):
            def build(self, helpers):
                cmd, cap = ca.SX.sym("thrust_cmd"), ca.SX.sym("cap")
                return {"h": ca.Function("budget_h", [cmd, cap], [ca.vertcat(cap - cmd, cmd)],
                                         ["thrust_cmd", "cap"], ["headroom"]),
                        "J": ca.Function("budget_J", [cmd], [cmd ** 2],
                                         ["thrust_cmd"], ["effort"])}

        with pytest.raises(ModelError, match=r"headroom is declared \(1, 1\)"):
            declared([*point_mass(), WrongShape()]).build(fixed={"dry_mass": 2.0, "cap": 1.0})

    def test_a_cost_term_must_be_scalar(self):
        class VectorCost(Budget):
            def build(self, helpers):
                cmd, cap = ca.SX.sym("thrust_cmd"), ca.SX.sym("cap")
                return {"h": ca.Function("budget_h", [cmd, cap], [cap - cmd],
                                         ["thrust_cmd", "cap"], ["headroom"]),
                        "J": ca.Function("budget_J", [cmd], [ca.vertcat(cmd, cmd)],
                                         ["thrust_cmd"], ["effort"])}

        with pytest.raises(ModelError, match=r"effort is declared \(1, 1\)"):
            declared([*point_mass(), VectorCost()]).build(fixed={"dry_mass": 2.0, "cap": 1.0})


class TestRoles:

    def test_a_quantity_defaults_to_flexible_with_a_variable_fallback(self):
        _, quantity, _ = declared(point_mass()).quantities()[0]
        assert quantity.role is Role.FLEXIBLE
        assert quantity.default_role is Role.VARIABLE

    def test_a_component_can_insist_on_a_role(self):
        class Fixed(Component):
            def declare(self):
                return Declaration(
                    quantities=(Quantity("dry_mass", unit="kg", role=Role.PARAMETER,
                                         default=2.0),),
                    produces=("mass",))

            def build(self, helpers):
                dry = ca.SX.sym("dry_mass")
                return {"g": ca.Function("fixed_g", [dry], [dry], ["dry_mass"], ["mass"])}

        _, quantity, _ = declared([Fixed()]).quantities()[0]
        assert quantity.role is Role.PARAMETER

    def test_the_builder_does_not_interpret_roles(self):
        """Roles are the compiler's business; the builder only lays quantities out."""
        builder = declared(point_mass())
        assert builder.quantity_order == ["dry_mass"]
        assert builder.build()["f"].name_in() == ["x", "u", "q"]
