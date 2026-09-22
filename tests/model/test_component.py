"""
What a ``Declaration``, ``Quantity``, ``Constraint`` and ``Cost`` refuse at the
line where they are written.

The two refusals that matter most are the silent ones. A bare string where a
tuple belongs -- ``("mass")`` is ``"mass"``, the missing one-tuple comma --
used to be iterated character by character. A set used to be accepted, and
since outputs are matched to names by position, its per-process order made
the model change with ``PYTHONHASHSEED``.
"""

import math

import casadi as ca
import numpy as np
import pytest

from machina.model import Component, Constraint, Cost, Declaration, ModelError, Quantity, Role
from machina.model.component import Scope

pytestmark = pytest.mark.requires_casadi


class TestDeclarationGroups:

    @pytest.mark.parametrize("group", ["states", "algebraic", "inputs", "helpers",
                                       "derivatives", "produces"])
    def test_a_bare_string_is_the_missing_comma_and_says_so(self, group):
        with pytest.raises(ModelError, match=r"did you mean \('mass',\)"):
            Declaration(**{group: "mass"})

    @pytest.mark.parametrize("group", ["states", "derivatives", "produces", "quantities"])
    def test_a_set_is_refused_because_its_order_changes_per_process(self, group):
        value = {Quantity("a"), Quantity("b")} if group == "quantities" else {"pos", "vel"}
        with pytest.raises(ModelError, match="Its order would change"):
            Declaration(**{group: value})

    def test_a_frozenset_is_refused_too(self):
        with pytest.raises(ModelError, match="Its order would change"):
            Declaration(produces=frozenset({"mass"}))

    def test_a_dict_is_refused(self):
        with pytest.raises(ModelError, match="Its order would change"):
            Declaration(produces={"mass": 1})

    def test_a_list_is_accepted_and_kept_in_order(self):
        declaration = Declaration(states=["vel", "pos"], produces=["mass"])
        assert declaration.states == (("vel", "vel"), ("pos", "pos"))
        assert declaration.produces == ("mass",)

    def test_something_else_entirely_is_refused(self):
        with pytest.raises(ModelError, match="must be a tuple or list"):
            Declaration(produces=42)

    @pytest.mark.parametrize("group,item,kind", [
        ("quantities", "a", "Quantity"),
        ("constraints", "c", "Constraint"),
        ("costs", Constraint("c"), "Cost"),
    ])
    def test_the_elements_of_quantities_constraints_and_costs_are_type_checked(self, group,
                                                                               item, kind):
        with pytest.raises(ModelError, match=f"must hold {kind} objects"):
            Declaration(**{group: (item,)})

    @pytest.mark.parametrize("group", ["derivatives", "produces", "helpers"])
    def test_a_name_listed_twice_is_refused(self, group):
        with pytest.raises(ModelError, match="lists 'mass' twice"):
            Declaration(**{group: ("mass", "mass")})

    def test_two_quantities_with_one_name_are_refused(self):
        with pytest.raises(ModelError, match="lists 'gain' twice"):
            Declaration(quantities=(Quantity("gain"), Quantity("gain")))

    def test_a_scope_refuses_a_set_of_components(self):
        class Empty(Component):
            def declare(self):
                return Declaration()

            def build(self, helpers):
                return {}

        with pytest.raises(ModelError, match="Its order would change"):
            Scope("sat", {Empty()})


class TestQuantity:

    def test_the_defaults_describe_a_flexible_scalar_variable(self):
        quantity = Quantity("gain")
        assert (quantity.shape, quantity.unit, quantity.role, quantity.default_role) == \
            ((1, 1), "1", Role.FLEXIBLE, Role.VARIABLE)
        assert (quantity.lb, quantity.ub) == (-math.inf, math.inf)

    @pytest.mark.parametrize("unit", ["deg", "deg/s", "ft", "rpm"])
    def test_a_non_si_unit_is_refused(self, unit):
        with pytest.raises(ModelError, match="is not SI"):
            Quantity("inc", unit=unit)

    def test_a_missing_unit_is_refused(self):
        with pytest.raises(ModelError, match="unit is mandatory"):
            Quantity("a", unit="")

    def test_lb_above_ub_is_refused(self):
        with pytest.raises(ModelError, match="exceeds ub"):
            Quantity("a", lb=5.0, ub=1.0)

    def test_elementwise_bounds_are_checked_elementwise(self):
        with pytest.raises(ModelError, match="exceeds ub"):
            Quantity("v", shape=2, lb=[0.0, 3.0], ub=[1.0, 2.0])

    def test_bounds_of_the_wrong_length_are_refused(self):
        with pytest.raises(ModelError, match="has 2 elements; expected 1 or 3"):
            Quantity("v", shape=3, lb=[0.0, 0.0])

    def test_a_non_numeric_bound_is_refused(self):
        with pytest.raises(ModelError, match="must be numeric"):
            Quantity("a", ub="1.0")

    def test_a_boolean_bound_is_refused(self):
        with pytest.raises(ModelError, match="must be numeric"):
            Quantity("a", ub=True)

    def test_a_nan_bound_is_refused(self):
        with pytest.raises(ModelError, match="contains NaN"):
            Quantity("a", lb=math.nan)

    def test_a_default_of_the_wrong_size_is_refused(self):
        with pytest.raises(ModelError, match="default has 2 elements; expected 1 or 3"):
            Quantity("a", shape=3, default=[1.0, 2.0])

    def test_a_matrix_default_of_the_right_size_is_accepted(self):
        assert Quantity("gain", shape=(2, 2), default=np.eye(2)).size == 4

    def test_an_unknown_provenance_code_is_refused(self):
        with pytest.raises(ModelError, match="provenance 'X' is not one of D P E A M"):
            Quantity("a", provenance="X")

    @pytest.mark.parametrize("code", ["D", "P", "E", "A", "M"])
    def test_every_provenance_code_is_accepted(self, code):
        assert Quantity("a", provenance=code).provenance == code

    def test_a_frame_must_look_like_a_frame_name(self):
        with pytest.raises(ModelError, match="frame must be a declared frame name"):
            Quantity("r", frame="earth centred")

    @pytest.mark.parametrize("shape", [(3, 1, 1), (2.0, 1), (0, 1), "3", True])
    def test_a_malformed_shape_is_refused(self, shape):
        with pytest.raises(ModelError, match="shape"):
            Quantity("a", shape=shape)

    def test_a_flexible_default_role_is_refused(self):
        with pytest.raises(ModelError, match="cannot itself be FLEXIBLE"):
            Quantity("a", default_role=Role.FLEXIBLE)

    def test_a_role_given_as_a_string_is_refused(self):
        with pytest.raises(ModelError, match="must be Role members"):
            Quantity("a", role="variable")

    def test_source_must_be_text(self):
        with pytest.raises(ModelError, match="source must be a string"):
            Quantity("a", source=3)


class TestConstraint:

    def test_lb_above_ub_is_refused(self):
        with pytest.raises(ModelError, match="exceeds ub"):
            Constraint("c", lb=1.0, ub=0.0)

    def test_an_equality_is_lb_equal_to_ub(self):
        assert Constraint("c", lb=0.0, ub=0.0).lb == 0.0

    def test_a_matrix_constraint_is_refused(self):
        with pytest.raises(ModelError, match="must be a column"):
            Constraint("c", shape=(2, 2))

    def test_vector_bounds_must_match_the_shape(self):
        with pytest.raises(ModelError, match="expected 1 or 3"):
            Constraint("c", shape=3, lb=[0.0, 0.0])


class TestCost:

    @pytest.mark.parametrize("weight", ["heavy", None, True, [1.0], math.nan, math.inf])
    def test_a_weight_that_is_not_a_finite_number_is_refused(self, weight):
        with pytest.raises(ModelError, match="weight"):
            Cost("j", weight=weight)

    def test_a_numeric_weight_is_accepted(self):
        assert Cost("j", weight=2).weight == 2

    def test_a_scalar_casadi_weight_is_accepted_so_it_can_be_swept(self):
        weight = ca.MX.sym("w")
        assert Cost("j", weight=weight).weight is weight

    def test_a_vector_casadi_weight_is_refused(self):
        with pytest.raises(ModelError, match="weight"):
            Cost("j", weight=ca.MX.sym("w", 2))
