"""
Every instance path has exactly one role, and the builder says so when a
declaration disagrees.

A path is a *state* (some component integrates it), an *algebraic* signal
(some component computes it, or it is read and unproduced), an *input* (read,
produced by nobody) or a *quantity* (owned by one component). Before these
checks existed, each disagreement below built cleanly and gave a wrong number:
an input replaced by zeros, a reader seeing a partial SUM because it had no
topological edge, a component reading another's private quantity, a
constraint silently dropped. Each test is the regression for one of the
adversarial-review findings on Phase 2a.
"""

import casadi as ca
import numpy as np
import pytest
from synthetic_model import (
    Budget,
    Drag,
    Dynamics,
    Kinematics,
    MassSource,
    Thruster,
    make_registry,
    point_mass,
)

from machina.model import (
    Builder,
    Component,
    Constraint,
    Cost,
    Declaration,
    ModelError,
    Quantity,
    Scope,
    SignalError,
)

pytestmark = pytest.mark.requires_casadi


def declared(components, registry=None):
    return Builder(components, registry=registry or make_registry()).declare()


def passthrough(fn_name, arg, out, shape=1):
    """A Function returning its one argument, named for binding."""
    symbol = ca.SX.sym(arg, shape)
    return ca.Function(fn_name, [symbol], [symbol], [arg], [out])


def component(declaration, built=None, name=None):
    """A one-off component with a fixed declaration and build() result."""
    class OneOff(Component):
        def declare(self):
            return declaration

        def build(self, helpers):
            return dict(built or {})

    OneOff.__name__ = name or "one_off"
    return OneOff()


class Controller(Component):
    """Computes thrust_cmd from velocity: a produced signal, not an input."""

    def declare(self):
        return Declaration(states=("vel",), produces=("thrust_cmd",))

    def build(self, helpers):
        vel = ca.SX.sym("vel", 2)
        return {"g": ca.Function("controller_g", [vel], [-0.1 * vel[0]], ["vel"],
                                 ["thrust_cmd"])}


class TestAQuantityIsPrivate:

    def test_a_reader_cannot_reach_a_quantity_through_an_unproduced_sum_path(self):
        """The owner would have received zeros instead of its own leaf."""
        owner = component(Declaration(quantities=(Quantity("force", shape=(2, 1), unit="N"),)),
                          name="Owner")
        with pytest.raises(ModelError, match="is a quantity owned by owner"):
            declared([Kinematics(), Dynamics(), MassSource(), owner])

    def test_a_reader_cannot_reach_a_quantity_through_an_input_path(self):
        """The reader would have seen the owner's private value, sized from the quantity."""
        owner = component(Declaration(quantities=(Quantity("thrust_cmd"),)), name="Owner")
        with pytest.raises(ModelError, match="A quantity is private"):
            declared([*point_mass(), owner])

    def test_nor_through_an_absolute_reference_into_another_scope(self):
        owner = component(Declaration(quantities=(Quantity("thrust_cmd"),)), name="Owner")
        reader = component(Declaration(inputs=(("cmd", "/sat/thrust_cmd"),),
                                       produces=("separation",)), name="Reader")
        with pytest.raises(ModelError, match="quantity owned by sat/owner"):
            declared([Scope("sat", [owner]), reader])


class TestAnInputIsExogenous:

    def test_an_input_that_another_component_produces_is_refused(self):
        """It would have had no topological edge: a SUM reader saw a partial sum,
        a UNIQUE reader an error that depended on the component order."""
        with pytest.raises(ModelError, match="under inputs, but 'thrust_cmd' is produced by"):
            declared([*point_mass(), Controller()])

    def test_the_error_says_which_group_it_belongs_in(self):
        with pytest.raises(ModelError, match="List it under algebraic"):
            declared([*point_mass(), Controller()])

    def test_an_input_whose_name_a_parent_scope_produces_is_refused(self):
        root = [Controller(), Kinematics(), Dynamics(), MassSource()]
        with pytest.raises(ModelError, match="'thrust_cmd' is produced by"):
            declared([*root, Scope("sat", [Kinematics(), Dynamics(), MassSource(), Thruster()])])

    def test_an_absolute_reference_is_the_escape_hatch_for_a_same_named_own_input(self):
        class OwnInput(Thruster):
            def declare(self):
                return Declaration(inputs=(("thrust_cmd", "/sat/thrust_cmd"),),
                                   algebraic=("mass",), produces=("force",))

        builder = declared([Controller(), Kinematics(), Dynamics(), MassSource(),
                            Scope("sat", [Kinematics(), Dynamics(), MassSource(), OwnInput()])])
        assert builder.input_order == ["sat/thrust_cmd"]

    def test_a_path_read_as_an_input_and_as_a_signal_is_refused(self):
        """The input used to vanish from u and be replaced by a constant zero."""
        monitor = component(Declaration(algebraic=(("cmd", "thrust_cmd"),),
                                        produces=("separation",)), name="Monitor")
        with pytest.raises(ModelError, match="A path has one role"):
            declared([*point_mass(), monitor])


class TestStatesAndSignalsAreNotInterchangeable:

    def test_a_signal_both_integrated_and_computed_is_refused(self):
        heat = component(Declaration(produces=("pos",)), name="Heat")
        with pytest.raises(ModelError, match="integrated by .* and computed by"):
            declared([*point_mass(), heat])

    def test_a_state_listed_under_algebraic_names_the_right_group(self):
        """It used to create a false dependency edge and a misleading cycle error."""
        reader = component(Declaration(algebraic=("vel",), produces=("separation",)),
                           name="Reader")
        with pytest.raises(ModelError, match="is a state integrated by .* List it under states"):
            declared([*point_mass(), reader])

    def test_a_computed_signal_listed_under_states_names_the_right_group(self):
        """It used to say 'it is an input, not a state'."""
        reader = component(Declaration(states=("mass",), produces=("separation",)),
                           name="Reader")
        with pytest.raises(ModelError, match="computed by .* List it under algebraic"):
            declared([*point_mass(), reader])

    def test_a_component_reading_what_it_produces_is_an_algebraic_loop(self):
        """It used to pass declare() and then read a partial sum or fail misleadingly."""
        loop = component(Declaration(algebraic=(("f_in", "force"),), produces=("force",)),
                         name="Loop")
        with pytest.raises(ModelError, match="algebraic loop through one component"):
            declared([*point_mass(), loop])

    def test_a_component_may_read_a_state_it_integrates(self):
        """The ordinary rigid-body case, which the loop check must not catch."""
        assert declared(point_mass()).resolved("dynamics")["vel"] == "vel"


class TestReferencesAreValidated:

    @pytest.mark.parametrize("reference", ["/sat_b//force", "sat_b/force/", "//sat_b/force",
                                           "sat b/force", "/"])
    def test_a_malformed_reference_is_refused_where_it_is_written(self, reference):
        with pytest.raises(ModelError, match="is not a valid reference"):
            Declaration(algebraic=(("f", reference),))

    def test_a_reference_into_a_scope_that_does_not_exist_is_refused(self):
        """A typo in a SUM reference used to read as a silent zero."""
        reader = component(Declaration(algebraic=(("f", "/satb/force"),),
                                       produces=("separation",)), name="Reader")
        with pytest.raises(ModelError, match="names a scope that does not exist"):
            declared([Scope("sat_b", point_mass()), reader])

    def test_the_scope_error_lists_the_scopes_there_are(self):
        reader = component(Declaration(algebraic=(("f", "/satb/force"),),
                                       produces=("separation",)), name="Reader")
        with pytest.raises(ModelError, match="sat_b"):
            declared([Scope("sat_b", point_mass()), reader])

    def test_an_explicit_path_that_nothing_produces_is_refused_even_for_sum(self):
        reader = component(Declaration(algebraic=(("f", "/sat_b/force"),),
                                       produces=("separation",)), name="Reader")
        # sat_b exists but has no thruster, so nothing produces sat_b/force.
        with pytest.raises(ModelError, match="names a path nothing produces"):
            declared([Scope("sat_b", [Kinematics(), Dynamics(), MassSource()]), reader])

    def test_a_bare_unproduced_sum_name_is_still_zero(self):
        """Dropping a producer is how a reduced model is written; that must keep working."""
        builder = declared([Kinematics(), Dynamics(), MassSource()])
        assert builder.unproduced_sums == ["force"]

    def test_an_unknown_signal_names_the_component_and_the_entry(self):
        typo = component(Declaration(algebraic=(("r", "/sat/r_ecii"),),
                                     produces=("separation",)), name="Typo")
        with pytest.raises(SignalError, match=r"typo: algebraic entry 'r' -> '/sat/r_ecii'"):
            declared([Scope("sat", point_mass()), typo])


class TestDeclaredOutputsAreUniqueAndBuilt:

    def test_two_constraints_with_one_path_are_refused(self):
        def guard(name):
            return component(Declaration(inputs=("thrust_cmd",),
                                         constraints=(Constraint("limit", ub=1.0),)),
                             {"h": passthrough("h", "thrust_cmd", "limit")}, name=name)
        with pytest.raises(ModelError, match="both declare the constraint 'limit'"):
            declared([*point_mass(), guard("GuardA"), guard("GuardB")])

    def test_two_costs_with_one_path_are_refused(self):
        def scorer(name):
            return component(Declaration(inputs=("thrust_cmd",), costs=(Cost("effort"),)),
                             name=name)
        with pytest.raises(ModelError, match="both declare the cost 'effort'"):
            declared([*point_mass(), scorer("ScorerA"), scorer("ScorerB")])

    def test_the_same_constraint_name_in_two_scopes_is_fine(self):
        builder = declared([Scope("a", [*point_mass(), Budget()]),
                            Scope("b", [*point_mass(), Budget()])])
        assert [p for p, _, _ in builder.constraints()] == ["a/headroom", "b/headroom"]

    @pytest.mark.parametrize("group,declaration,key", [
        ("constraints", Declaration(inputs=("thrust_cmd",),
                                    constraints=(Constraint("limit", ub=1.0),)), "h"),
        ("costs", Declaration(inputs=("thrust_cmd",), costs=(Cost("effort"),)), "J"),
        ("produces", Declaration(produces=("separation",)), "g"),
    ])
    def test_a_declared_output_that_build_never_returns_is_an_error(self, group, declaration,
                                                                    key):
        """A declared constraint used to vanish from the NLP without a word."""
        silent = component(declaration, {}, name="Silent")
        with pytest.raises(ModelError, match=rf"declares {group} .* no '{key}' Function"):
            declared([*point_mass(), silent]).build(fixed={"dry_mass": 2.0})

    def test_a_declared_derivative_that_build_never_returns_is_an_error(self):
        class NoF(Kinematics):
            def build(self, helpers):
                return {}

        with pytest.raises(ModelError, match="declares derivatives .* no 'f' Function"):
            declared([NoF(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})


class TestConstraintAndCostOrder:

    def test_h_and_j_rows_follow_declaration_order_not_execution_order(self):
        """The checker is declared first but runs after MassSource; h must still list its
        constraint first, the way constraints() does, or row i is the wrong constraint."""
        checker = component(
            Declaration(algebraic=("mass",), constraints=(Constraint("mass_ok", lb=0.5),),
                        costs=(Cost("heavy"),)),
            {"h": passthrough("h", "mass", "mass_ok"),
             "J": passthrough("J", "mass", "heavy")},
            name="Checker")
        builder = declared([checker, Budget(), *point_mass()])
        assert builder.order.index("checker") > builder.order.index("budget")
        declared_order = [p for p, _, _ in builder.constraints()]
        assert declared_order == ["mass_ok", "headroom"]
        model = builder.build(fixed={"dry_mass": 2.0, "cap": 0.8})
        h = np.array(model["h"]([0, 0, 0, 0], [0.3])).ravel()
        np.testing.assert_allclose(h, [2.0, 0.5])          # mass_ok = mass, headroom = cap - cmd
        J = np.array(model["J"]([0, 0, 0, 0], [0.3])).ravel()
        np.testing.assert_allclose(J, [2.0, 0.09])          # heavy = mass, effort = cmd**2

    def test_wired_constraints_follow_the_same_order(self):
        checker = component(
            Declaration(algebraic=("mass",), constraints=(Constraint("mass_ok", lb=0.5),)),
            {"h": passthrough("h", "mass", "mass_ok")},
            name="Checker")
        builder = declared([checker, Budget(), *point_mass()])
        leaves = {p: ca.MX.sym(p) if builder.shape_of(p) == (1, 1)
                  else ca.MX.sym(p, *builder.shape_of(p))
                  for p in builder.state_order + builder.input_order + builder.quantity_order}
        wired = builder.wire(leaves)
        assert [c.path for c in wired.constraints] == [p for p, _, _ in builder.constraints()]


class TestShapesAreCheckedAtBothEnds:

    def test_a_scalar_leaf_for_a_vector_state_is_refused(self):
        builder = declared(point_mass())
        leaves = {"pos": ca.MX.sym("pos"), "vel": ca.MX.sym("vel", 2),
                  "thrust_cmd": ca.MX.sym("u"), "dry_mass": ca.MX.sym("m")}
        with pytest.raises(ModelError, match=r"leaf for 'pos' has shape \(1, 1\).*\(2, 1\)"):
            builder.wire(leaves)

    def test_a_numeric_leaf_of_the_right_shape_is_accepted(self):
        builder = declared(point_mass())
        wired = builder.wire({"pos": np.zeros(2), "vel": np.array([1.0, 2.0]),
                              "thrust_cmd": 0.5, "dry_mass": 2.0})
        np.testing.assert_allclose(np.array(ca.evalf(wired.xdot["pos"])).ravel(), [1.0, 2.0])

    def test_a_function_argument_wider_than_its_signal_is_refused(self):
        """CasADi would broadcast the (1, 1) mass into the 3-vector argument."""
        class WideMass(Dynamics):
            def build(self, helpers):
                vel, force, mass = ca.SX.sym("vel", 2), ca.SX.sym("force", 2), ca.SX.sym("mass", 2)
                return {"f": ca.Function("dynamics_f", [vel, force, mass], [force / mass],
                                         ["vel", "force", "mass"], ["vel_dot"])}

        with pytest.raises(ModelError, match=r"takes 'mass' as \(2, 1\), but it is declared "
                                             r"\(1, 1\)"):
            declared([Kinematics(), WideMass(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_a_function_argument_that_disagrees_with_its_quantity_is_refused(self):
        class WideSource(MassSource):
            def build(self, helpers):
                dry = ca.SX.sym("dry_mass", 3)
                return {"g": ca.Function("mass_source_g", [dry], [dry[0]], ["dry_mass"],
                                         ["mass"])}

        with pytest.raises(ModelError, match=r"takes 'dry_mass' as \(3, 1\)"):
            declared([Kinematics(), Dynamics(), WideSource(), Thruster()]).build()

    def test_a_row_where_a_column_is_declared_is_refused(self):
        class RowKinematics(Kinematics):
            def build(self, helpers):
                vel = ca.SX.sym("vel", 1, 2)
                return {"f": ca.Function("kinematics_f", [vel], [vel.T], ["vel"], ["pos_dot"])}

        with pytest.raises(ModelError, match=r"takes 'vel' as \(1, 2\)"):
            declared([RowKinematics(), Dynamics(), MassSource(), Thruster()]).build(
                fixed={"dry_mass": 2.0})

    def test_a_fixed_value_of_the_wrong_size_names_the_quantity(self):
        with pytest.raises(ModelError, match=r"fixed\['dry_mass'\] has shape \(2,\)"):
            declared(point_mass()).build(fixed={"dry_mass": [1.0, 2.0]})


class TestTheBuilderItself:

    def test_declare_is_idempotent(self):
        """A second call used to raise a false 'X and X both own a quantity'."""
        builder = declared(point_mass())
        assert builder.declare() is builder
        assert builder.quantity_order == ["dry_mass"]

    def test_a_set_of_components_is_refused(self):
        with pytest.raises(ModelError, match="Its order would change"):
            Builder({Kinematics()}, registry=make_registry())

    def test_the_topological_order_is_public(self):
        builder = declared([*point_mass(), Drag()])
        assert builder.order[0] == "kinematics" and len(builder.order) == 5

    def test_order_requires_declare(self):
        with pytest.raises(ModelError, match="call declare"):
            _ = Builder(point_mass(), registry=make_registry()).order
