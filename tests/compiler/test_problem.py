"""
``machina.compiler.Problem``: role and value precedence, and the compiled NLP.

The fleet below is the two-satellite power budget of ``tests/model/test_wire_mx.py``
-- the same closed form, re-created here so this file stands alone:

    maximise  50 (d_a + d_b) - 0.1 (d_a^2 + d_b^2)
    subject to  100 (d_a + d_b) <= limit,  0 <= d <= 1

At ``limit = 150`` the cap binds, symmetry gives ``d_a = d_b = 0.75``, the
objective is ``-74.8875`` and ``d f*/d limit = -0.4985``. ``test_wire_mx``
drives the backend by hand; everything here goes through ``Problem``, so the
numbers double as a check that the compiler's extra layer changed nothing.

``Knob`` is the minimal component for the precedence tables: one owned
quantity, one cost term pulling it towards a target, optionally one
constraint. A ``Problem`` made of two of them has exactly the leaf under test
and one free variable to keep the NLP well posed.
"""

import warnings

import casadi as ca
import numpy as np
import pytest

from machina.compiler import Problem
from machina.model import (
    Aggregation,
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    Role,
    Scope,
    SignalRegistry,
)
from machina.params.values import values_from

pytestmark = pytest.mark.requires_casadi


# --- the fleet ---------------------------------------------------------------------------------


def make_registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.declare("power", 1, "W", aggregation=Aggregation.SUM, doc="Electrical load")
    reg.declare("rate", 1, "1/s", aggregation=Aggregation.SUM, doc="Useful data rate")
    return reg


class Payload(Component):
    """``power = 100 duty``, ``rate = 50 duty``, with a small effort penalty."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("duty", unit="1", lb=0.0, ub=1.0, default=0.1,
                                 doc="Duty cycle", provenance="A", source="test fixture"),),
            produces=("power", "rate"),
            costs=(Cost("effort", weight=0.1, doc="Squared duty cycle"),),
        )

    def build(self, helpers):
        duty = ca.SX.sym("duty")
        return {
            "g": ca.Function("payload_g", [duty], [100.0 * duty, 50.0 * duty],
                             ["duty"], ["power", "rate"]),
            "J": ca.Function("payload_J", [duty], [duty ** 2], ["duty"], ["effort"]),
        }


class FleetBudget(Component):
    """Caps the fleet's total power and scores its total rate."""

    def declare(self):
        return Declaration(
            algebraic=(("power_a", "/a/power"), ("power_b", "/b/power"),
                       ("rate_a", "/a/rate"), ("rate_b", "/b/rate")),
            quantities=(Quantity("limit", unit="W", role=Role.PARAMETER, default=150.0,
                                 doc="Fleet power cap", provenance="D", source="bus spec"),),
            constraints=(Constraint("margin", lb=0.0, doc="limit - total power >= 0"),),
            costs=(Cost("neg_rate", weight=1.0, doc="Negated fleet data rate"),),
        )

    def build(self, helpers):
        power_a, power_b = ca.SX.sym("power_a"), ca.SX.sym("power_b")
        rate_a, rate_b = ca.SX.sym("rate_a"), ca.SX.sym("rate_b")
        limit = ca.SX.sym("limit")
        return {
            "h": ca.Function("budget_h", [power_a, power_b, limit],
                             [limit - (power_a + power_b)],
                             ["power_a", "power_b", "limit"], ["margin"]),
            "J": ca.Function("budget_J", [rate_a, rate_b], [-(rate_a + rate_b)],
                             ["rate_a", "rate_b"], ["neg_rate"]),
        }


def fleet(**kwargs) -> Problem:
    return Problem([Scope("a", [Payload()]), Scope("b", [Payload()]), FleetBudget()],
                   registry=make_registry(), verbose=False, **kwargs)


def solved_fleet(**compile_kwargs):
    problem = fleet().compile(**compile_kwargs).build()
    return problem, problem.solve()


# --- the one-quantity probe --------------------------------------------------------------------


class Knob(Component):
    """One owned quantity, pulled towards ``target``; optionally one constraint."""

    def __init__(self, quantity, *, name=None, target=2.0, constraint=None):
        super().__init__(name)
        self._quantity = quantity
        self._target = target
        self._constraint = constraint

    def declare(self):
        return Declaration(
            quantities=(self._quantity,),
            constraints=() if self._constraint is None else (self._constraint,),
            costs=(Cost(f"miss_{self._quantity.name}", weight=1.0,
                        doc="Squared distance to the target"),),
        )

    def build(self, helpers):
        name = self._quantity.name
        knob = ca.SX.sym(name, *self._quantity.shape)
        out = {"J": ca.Function("knob_J", [knob], [ca.sumsqr(knob - self._target)],
                                [name], [f"miss_{name}"])}
        if self._constraint is not None:
            out["h"] = ca.Function("knob_h", [knob], [10.0 - ca.sum1(ca.vec(knob))], [name],
                                   [self._constraint.name])
        return out


FREE = Quantity("free", unit="1", default=0.0, doc="A plain decision variable",
                provenance="A", source="fixture")


def probe(quantity, *, target=2.0, constraint=None, **kwargs) -> Problem:
    """A Problem holding ``quantity`` plus one free variable, so the NLP is well posed."""
    return Problem([Knob(quantity, name="probe", target=target, constraint=constraint),
                    Knob(FREE, name="free")],
                   registry=SignalRegistry(), verbose=False, **kwargs)


def record_for(problem: Problem, path: str):
    """The backend's variable or parameter record for ``path``."""
    for record in list(problem.backend.variables()) + list(problem.backend.parameters()):
        if record.name == path:
            return record
    raise AssertionError(f"{path!r} is registered as neither a variable nor a parameter")


PARAMS_CSV = (
    "name,value,type,unit,min,max,default,mutability,provenance,source,comment\n"
    "COMPUTE_W,30.0,f32,W,0.0,500.0,,design,A,fixture table,Onboard compute power draw\n"
)

UNLINKED_ROW = "RADIO_W,12.0,f32,W,0.0,100.0,,design,E,fixture table,Radio power draw\n"


def params_table(tmp_path, *, extra_rows=""):
    """``{name: ParamValue}`` from a minimal, valid params CSV."""
    csv = tmp_path / "params.csv"
    csv.write_bytes((PARAMS_CSV + extra_rows).encode("utf-8"))
    return values_from(csv)


class Drifter(Component):
    """``x_dot = -x``: one state, which the Phase 3 compiler has nowhere to put."""

    def declare(self):
        return Declaration(states=("x",), derivatives=("x",))

    def build(self, helpers):
        x = ca.SX.sym("x")
        return {"f": ca.Function("drifter_f", [x], [-x], ["x"], ["x_dot"])}


def drifting_problem() -> Problem:
    registry = SignalRegistry()
    registry.declare("x", 1, "m", doc="A drifting state")
    return Problem([Drifter(), Knob(FREE, name="free")], registry=registry, verbose=False)


# --- tests -------------------------------------------------------------------------------------


class TestTheCompiledProblemSolves:

    def test_the_optimum_is_the_closed_form_one(self):
        _, result = solved_fleet()
        assert result.success
        np.testing.assert_allclose(result["a/duty"], [0.75], rtol=1e-6)
        np.testing.assert_allclose(result["b/duty"], [0.75], rtol=1e-6)
        np.testing.assert_allclose(result.f_opt, -74.8875, rtol=1e-8)

    def test_the_power_cap_is_active_and_priced(self):
        _, result = solved_fleet()
        margin = result.constraint("margin")
        np.testing.assert_allclose(margin.value, [0.0], atol=1e-6)
        assert margin.active
        assert abs(margin.multiplier[0]) > 1e-6

    def test_the_parameter_sensitivity_matches_the_derivative_on_paper(self):
        """d f*/d limit = -1/2 + limit / 100000, which is -0.4985 at 150 W."""
        _, result = solved_fleet()
        np.testing.assert_allclose(result.sensitivity("limit"), [-0.4985], rtol=1e-4)

    def test_scope_names_prefix_the_quantity_paths(self):
        problem = fleet().compile()
        assert list(problem.roles) == ["a/duty", "b/duty", "limit"]
        assert [v.name for v in problem.backend.variables()] == ["a/duty", "b/duty"]
        assert [p.name for p in problem.backend.parameters()] == ["limit"]

    def test_declared_metadata_reaches_the_backend_records(self):
        problem = fleet().compile()
        duty = record_for(problem, "a/duty")
        assert (duty.unit, duty.provenance, duty.source) == ("1", "A", "test fixture")
        assert duty.component == "a/payload" and duty.doc == "Duty cycle"
        assert [c.name for c in problem.backend.constraints()] == ["margin"]
        assert [c.name for c in problem.backend.cost_terms()] == \
            ["a/effort", "b/effort", "neg_rate"]

    def test_evaluate_reports_every_signal_and_quantity_in_path_order(self):
        problem, result = solved_fleet()
        values = problem.evaluate(result)
        assert list(values) == ["a/duty", "b/duty", "limit", "a/power", "b/power",
                                "a/rate", "b/rate"]
        np.testing.assert_allclose(values["a/power"], [75.0], rtol=1e-6)
        np.testing.assert_allclose(values["b/rate"], [37.5], rtol=1e-6)
        np.testing.assert_allclose(values["limit"], [150.0], rtol=1e-12)


class TestRolePrecedence:

    def test_an_override_beats_the_roles_dict(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(roles={"probe": Role.VARIABLE},
                                          overrides={"probe": {"role": Role.PARAMETER}})
        assert problem.roles["probe"] is Role.PARAMETER

    def test_the_roles_dict_beats_the_declared_role(self):
        quantity = Quantity("probe", role=Role.PARAMETER, default=1.0,
                            provenance="A", source="fixture")
        problem = probe(quantity).compile(roles={"probe": Role.VARIABLE})
        assert problem.roles["probe"] is Role.VARIABLE

    def test_the_declared_role_beats_the_default_role(self):
        quantity = Quantity("probe", role=Role.PARAMETER, default_role=Role.VARIABLE,
                            default=1.0, provenance="A", source="fixture")
        assert probe(quantity).compile().roles["probe"] is Role.PARAMETER

    def test_a_flexible_declaration_falls_through_to_the_default_role(self):
        quantity = Quantity("probe", role=Role.FLEXIBLE, default_role=Role.PARAMETER,
                            default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile()
        assert problem.roles["probe"] is Role.PARAMETER
        assert [p.name for p in problem.backend.parameters()] == ["probe"]

    def test_flexible_never_survives_the_resolution(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="FLEXIBLE"):
            probe(quantity).compile(roles={"probe": Role.FLEXIBLE})

    def test_a_role_name_is_accepted_as_a_string(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(roles={"probe": "parameter"})
        assert problem.roles["probe"] is Role.PARAMETER

    def test_an_unknown_role_name_names_the_valid_ones(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'variable'"):
            probe(quantity).compile(roles={"probe": "continuous"})


class TestValuePrecedence:

    def test_an_override_beats_the_values_dict(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(values={"probe": 2.0},
                                          overrides={"probe": {"value": 3.0}})
        np.testing.assert_allclose(record_for(problem, "probe").x0, [3.0], rtol=1e-12)

    def test_the_values_dict_beats_the_declared_default(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(values={"probe": 2.0})
        np.testing.assert_allclose(record_for(problem, "probe").x0, [2.0], rtol=1e-12)

    def test_a_params_name_supplies_the_value_through_the_quantity_link(self, tmp_path):
        quantity = Quantity("probe", unit="W", role=Role.PARAMETER, param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path))
        np.testing.assert_allclose(record_for(problem, "probe").value, [30.0], rtol=1e-12)

    def test_an_explicit_path_beats_the_params_link(self, tmp_path):
        quantity = Quantity("probe", unit="W", role=Role.PARAMETER, param="COMPUTE_W")
        values = params_table(tmp_path)
        values["probe"] = 7.0
        problem = probe(quantity).compile(values=values)
        np.testing.assert_allclose(record_for(problem, "probe").value, [7.0], rtol=1e-12)

    def test_the_declared_default_is_the_last_source(self):
        quantity = Quantity("probe", default=1.5, provenance="A", source="fixture")
        problem = probe(quantity).compile()
        np.testing.assert_allclose(record_for(problem, "probe").x0, [1.5], rtol=1e-12)

    def test_a_param_value_carries_its_provenance_onto_a_parameter_record(self, tmp_path):
        quantity = Quantity("probe", unit="W", role=Role.PARAMETER, param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path))
        record = record_for(problem, "probe")
        assert (record.provenance, record.source) == ("A", "fixture table")

    def test_a_param_value_carries_its_provenance_onto_a_variable_record(self, tmp_path):
        quantity = Quantity("probe", unit="W", param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path))
        record = record_for(problem, "probe")
        assert (record.provenance, record.source) == ("A", "fixture table")
        np.testing.assert_allclose(record.x0, [30.0], rtol=1e-12)

    def test_param_limits_become_the_bounds_of_an_open_quantity(self, tmp_path):
        quantity = Quantity("probe", unit="W", param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path))
        record = record_for(problem, "probe")
        np.testing.assert_allclose([record.lb[0], record.ub[0]], [0.0, 500.0], rtol=1e-12)

    def test_two_quantities_linked_to_one_params_name_both_take_the_table_value(self, tmp_path):
        problem = Problem([Knob(Quantity("probe", unit="W", param="COMPUTE_W"), name="one"),
                           Knob(Quantity("other", unit="W", param="COMPUTE_W"), name="two")],
                          registry=SignalRegistry(), verbose=False)
        problem.compile(values=params_table(tmp_path))
        np.testing.assert_allclose(record_for(problem, "probe").x0, [30.0], rtol=1e-12)
        np.testing.assert_allclose(record_for(problem, "other").x0, [30.0], rtol=1e-12)

    def test_a_param_value_provenance_beats_the_one_the_quantity_declared(self, tmp_path):
        quantity = Quantity("probe", unit="W", param="COMPUTE_W", default=1.0,
                            provenance="D", source="bus spec")
        record = record_for(probe(quantity).compile(values=params_table(tmp_path)), "probe")
        assert (record.provenance, record.source) == ("A", "fixture table")

    def test_a_matrix_parameter_reaches_the_backend_column_major(self):
        value = [[1.0, 2.0], [3.0, 4.0]]
        quantity = Quantity("probe", shape=(2, 2), role=Role.PARAMETER,
                            provenance="A", source="fixture")
        problem = probe(quantity, target=0.0).compile(values={"probe": value})
        np.testing.assert_allclose(
            np.asarray(problem.backend.parameter_value("probe")).ravel(order="F"),
            np.asarray(value).ravel(order="F"), rtol=1e-12)

    def test_an_override_value_becomes_the_stored_value_of_a_parameter(self):
        quantity = Quantity("probe", role=Role.PARAMETER, default=1.0,
                            provenance="A", source="fixture")
        problem = probe(quantity).compile(overrides={"probe": {"value": 4.0}})
        np.testing.assert_allclose(record_for(problem, "probe").value, [4.0], rtol=1e-12)
        np.testing.assert_allclose(problem.backend.parameter_value("probe"), [4.0], rtol=1e-12)

    def test_param_limits_do_not_overwrite_bounds_the_quantity_declared(self, tmp_path):
        quantity = Quantity("probe", unit="W", lb=1.0, ub=2.0, param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path))
        record = record_for(problem, "probe")
        np.testing.assert_allclose([record.lb[0], record.ub[0]], [1.0, 2.0], rtol=1e-12)

    def test_a_param_limit_fills_only_the_bound_the_quantity_left_open(self, tmp_path):
        quantity = Quantity("probe", unit="W", lb=1.0, param="COMPUTE_W")
        record = record_for(probe(quantity).compile(values=params_table(tmp_path)), "probe")
        np.testing.assert_allclose([record.lb[0], record.ub[0]], [1.0, 500.0], rtol=1e-12)

    def test_an_override_beats_the_param_limits(self, tmp_path):
        quantity = Quantity("probe", unit="W", param="COMPUTE_W")
        problem = probe(quantity).compile(values=params_table(tmp_path),
                                          overrides={"probe": {"lb": -3.0, "x0": 4.0}})
        record = record_for(problem, "probe")
        np.testing.assert_allclose([record.lb[0], record.ub[0], record.x0[0]],
                                   [-3.0, 500.0, 4.0], rtol=1e-12)

    def test_a_parameter_with_no_value_anywhere_names_the_quantity(self):
        quantity = Quantity("probe", role=Role.PARAMETER)
        with pytest.raises(ValueError, match="no value for quantity 'probe'"):
            probe(quantity).compile()

    def test_a_variable_with_no_value_anywhere_names_the_quantity_too(self):
        quantity = Quantity("probe", role=Role.VARIABLE)
        with pytest.raises(ValueError, match="no value for quantity 'probe'") as err:
            probe(quantity).compile()
        assert "silent zero" in str(err.value)

    def test_a_none_in_the_values_dict_is_no_value_for_a_parameter(self):
        quantity = Quantity("probe", role=Role.PARAMETER)
        with pytest.raises(ValueError, match="no value for quantity 'probe'"):
            probe(quantity).compile(values={"probe": None})

    def test_a_none_override_is_no_value_for_a_variable(self):
        quantity = Quantity("probe", role=Role.VARIABLE)
        with pytest.raises(ValueError, match="no value for quantity 'probe'") as err:
            probe(quantity).compile(overrides={"probe": {"value": None}})
        assert "None from any source" in str(err.value)

    def test_a_wrong_shape_override_bound_names_the_quantity(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'probe'") as err:
            probe(quantity).compile(overrides={"probe": {"lb": [0.0, 1.0, 2.0]}})
        assert "lb bound" in str(err.value)
        assert "compile(overrides=" in str(err.value)


class TestTheUnsourcedDefaultWarning:

    def test_one_warning_names_every_quantity_that_took_an_unsourced_default(self):
        problem = Problem([Knob(Quantity("first", default=1.0), name="first"),
                           Knob(Quantity("second", default=2.0), name="second"),
                           Knob(FREE, name="free")],
                          registry=SignalRegistry(), verbose=False)
        with pytest.warns(UserWarning) as caught:
            problem.compile()
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "'first'" in message and "'second'" in message and "'free'" not in message

    def test_strict_turns_the_warning_into_an_error(self):
        problem = probe(Quantity("probe", default=1.0))
        with pytest.raises(ValueError, match="provenance"):
            problem.compile(strict=True)

    def test_the_strict_error_says_what_to_do_instead_of_naming_the_flag_again(self):
        problem = probe(Quantity("probe", default=1.0))
        with pytest.raises(ValueError) as err:
            problem.compile(strict=True)
        assert "strict=True makes" not in str(err.value)
        assert "provenance=" in str(err.value) and "compile(values=" in str(err.value)

    def test_strict_compiles_silently_when_every_default_is_sourced(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            problem = fleet().compile(strict=True)
        assert [str(w.message) for w in caught] == []
        assert list(problem.roles) == ["a/duty", "b/duty", "limit"]

    def test_a_sourced_default_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fleet().compile()
        assert [str(w.message) for w in caught] == []

    def test_a_supplied_value_is_silent_even_without_provenance(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probe(Quantity("probe", default=1.0)).compile(values={"probe": 2.0})
        assert [str(w.message) for w in caught] == []


class TestOverrideKeysMustFitTheResolvedRole:

    @pytest.mark.parametrize("key, value",
                             [("x0", 0.5), ("lb", 0.0), ("ub", 9.0), ("scale", 2.0)])
    def test_a_parameter_refuses_the_keys_only_a_variable_uses(self, key, value):
        quantity = Quantity("probe", role=Role.PARAMETER, default=1.0,
                            provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'parameter'") as err:
            probe(quantity).compile(overrides={"probe": {key: value}})
        assert repr(key) in str(err.value) and "'probe'" in str(err.value)

    @pytest.mark.parametrize("key, value",
                             [("x0", 0.5), ("lb", 0.0), ("ub", 9.0), ("scale", 2.0),
                              ("provenance", "D"), ("source", "a note")])
    def test_a_fixed_quantity_refuses_everything_but_role_and_value(self, key, value):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'fixed'") as err:
            probe(quantity).compile(roles={"probe": Role.FIXED},
                                    overrides={"probe": {"value": 0.5, key: value}})
        assert repr(key) in str(err.value) and "'probe'" in str(err.value)

    def test_a_variable_takes_every_quantity_key(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(overrides={"probe": {
            "role": Role.VARIABLE, "value": 1.0, "x0": 2.0, "lb": 0.0, "ub": 3.0,
            "scale": 2.0, "provenance": "E", "source": "an override"}})
        record = record_for(problem, "probe")
        assert (record.provenance, record.source) == ("E", "an override")
        np.testing.assert_allclose([record.x0[0], record.lb[0], record.ub[0], record.scale[0]],
                                   [2.0, 0.0, 3.0, 2.0], rtol=1e-12)

    def test_a_discrete_quantity_takes_the_variable_keys_too(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(
            roles={"probe": Role.DISCRETE}, discrete_mode="relax",
            overrides={"probe": {"x0": 2.0, "lb": 0.0, "ub": 3.0}})
        record = record_for(problem, "probe")
        assert bool(record.discrete[0])
        np.testing.assert_allclose([record.x0[0], record.ub[0]], [2.0, 3.0], rtol=1e-12)

    def test_a_parameter_takes_role_value_provenance_and_source(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(overrides={"probe": {
            "role": Role.PARAMETER, "value": 5.0, "provenance": "M", "source": "a bench run"}})
        record = record_for(problem, "probe")
        assert (record.provenance, record.source) == ("M", "a bench run")
        np.testing.assert_allclose(record.value, [5.0], rtol=1e-12)

    def test_a_fixed_quantity_takes_role_and_value(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        problem = probe(quantity).compile(overrides={"probe": {"role": Role.FIXED,
                                                               "value": 0.5}})
        np.testing.assert_allclose(problem.fixed["probe"], [0.5], rtol=1e-12)

    def test_an_unknown_provenance_code_names_the_allowed_ones(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'probe'") as err:
            probe(quantity).compile(overrides={"probe": {"provenance": "X"}})
        assert "'D'" in str(err.value) and "'M'" in str(err.value)

    def test_a_source_that_is_not_a_string_is_refused(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        with pytest.raises(ValueError, match="'probe'") as err:
            probe(quantity).compile(overrides={"probe": {"source": 3}})
        assert "string" in str(err.value)


class TestOnlyStaticModelsCompile:

    def test_a_declared_state_is_refused_with_the_state_named(self):
        with pytest.raises(ValueError, match="machina.sim") as err:
            drifting_problem().compile()
        assert "'x'" in str(err.value) and "states" in str(err.value)


class TestFixedQuantities:

    def test_a_fixed_quantity_is_registered_nowhere_but_recorded(self):
        problem = fleet().compile(roles={"a/duty": Role.FIXED},
                                  overrides={"a/duty": {"value": 0.5}})
        assert [v.name for v in problem.backend.variables()] == ["b/duty"]
        assert [p.name for p in problem.backend.parameters()] == ["limit"]
        np.testing.assert_allclose(problem.fixed["a/duty"], [0.5], rtol=1e-12)

    def test_a_fixed_quantity_still_comes_back_from_evaluate(self):
        problem = fleet().compile(roles={"a/duty": Role.FIXED},
                                  overrides={"a/duty": {"value": 0.5}}).build()
        values = problem.evaluate(problem.solve())
        np.testing.assert_allclose(values["a/duty"], [0.5], rtol=1e-12)
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=1e-6)

    def test_fixing_by_role_and_fixing_by_bounds_reach_the_same_optimum(self):
        by_role = fleet().compile(roles={"a/duty": Role.FIXED},
                                  overrides={"a/duty": {"value": 0.5}}).build().solve()
        by_bounds = fleet().compile().build()
        by_bounds.fix("a/duty", 0.5)
        pinned = by_bounds.solve()
        np.testing.assert_allclose(by_role.f_opt, pinned.f_opt, rtol=1e-6)
        np.testing.assert_allclose(by_role["b/duty"], pinned["b/duty"], rtol=1e-6)


class TestScale:

    def test_a_declared_quantity_scale_reaches_the_variable_record(self):
        quantity = Quantity("probe", default=1.0, scale=100.0, provenance="A", source="fixture")
        problem = probe(quantity).compile()
        np.testing.assert_allclose(record_for(problem, "probe").scale, [100.0], rtol=1e-12)

    def test_a_declared_constraint_scale_reaches_the_constraint_record(self):
        quantity = Quantity("probe", default=1.0, provenance="A", source="fixture")
        constraint = Constraint("room", lb=0.0, scale=50.0, doc="10 - probe >= 0")
        problem = probe(quantity, constraint=constraint).compile()
        record = next(c for c in problem.backend.constraints() if c.name == "room")
        np.testing.assert_allclose(record.scale, [50.0], rtol=1e-12)

    def test_an_override_replaces_both_scales(self):
        quantity = Quantity("probe", default=1.0, scale=100.0, provenance="A", source="fixture")
        constraint = Constraint("room", lb=0.0, scale=50.0, doc="10 - probe >= 0")
        problem = probe(quantity, constraint=constraint).compile(
            overrides={"probe": {"scale": 2.0}, "room": {"scale": 4.0}})
        np.testing.assert_allclose(record_for(problem, "probe").scale, [2.0], rtol=1e-12)
        record = next(c for c in problem.backend.constraints() if c.name == "room")
        np.testing.assert_allclose(record.scale, [4.0], rtol=1e-12)

    def test_a_scaled_problem_reaches_the_same_optimum(self):
        problem = fleet().compile(overrides={"a/duty": {"scale": 0.5},
                                             "margin": {"scale": 100.0}}).build()
        result = problem.solve()
        assert problem.backend.is_scaled
        np.testing.assert_allclose(result["a/duty"], [0.75], rtol=1e-5)
        np.testing.assert_allclose(result.f_opt, -74.8875, rtol=1e-6)


class TestKeysAreChecked:

    def test_an_unknown_role_path_lists_the_quantities(self):
        with pytest.raises(ValueError, match=r"a/dutyy") as err:
            fleet().compile(roles={"a/dutyy": Role.FIXED})
        assert "'a/duty'" in str(err.value)

    def test_an_unknown_value_key_lists_the_quantities_and_the_params_names(self):
        with pytest.raises(ValueError, match="COMPUTE_W") as err:
            fleet().compile(values={"COMPUTE_W": 30.0})
        assert "'limit'" in str(err.value)

    def test_an_unknown_override_key_names_the_allowed_ones(self):
        with pytest.raises(ValueError, match="'provenance'") as err:
            fleet().compile(overrides={"a/duty": {"guess": 0.5}})
        assert "'guess'" in str(err.value)

    def test_a_constraint_override_accepts_only_scale(self):
        with pytest.raises(ValueError, match=r"\['scale'\]") as err:
            fleet().compile(overrides={"margin": {"role": Role.FIXED}})
        assert "constraint" in str(err.value)

    def test_a_constraint_path_is_not_a_role_key(self):
        with pytest.raises(ValueError, match="margin") as err:
            fleet().compile(roles={"margin": Role.FIXED})
        assert "not a quantity" in str(err.value)

    def test_a_params_table_row_nothing_links_to_is_ignored(self, tmp_path):
        quantity = Quantity("probe", unit="W", role=Role.PARAMETER, param="COMPUTE_W")
        values = params_table(tmp_path, extra_rows=UNLINKED_ROW)
        assert list(values) == ["COMPUTE_W", "RADIO_W"]
        problem = probe(quantity).compile(values=values)
        np.testing.assert_allclose(record_for(problem, "probe").value, [30.0], rtol=1e-12)

    def test_an_unknown_value_key_that_is_not_a_param_value_is_still_an_error(self):
        with pytest.raises(ValueError, match="RADIO_W") as err:
            fleet().compile(values={"RADIO_W": 12.0})
        assert "not one" in str(err.value)

    def test_an_override_that_is_not_a_dict_says_what_to_write(self):
        with pytest.raises(ValueError, match="dict of override keys"):
            fleet().compile(overrides={"a/duty": 0.5})


class TestExpr:

    def test_a_signal_comes_back_as_an_mx_backed_descriptor(self):
        problem = fleet().compile()
        descriptor = problem.expr("a/power")
        assert isinstance(descriptor.symbol, ca.MX)
        assert (descriptor.name, descriptor.shape) == ("a/power", (1, 1))
        assert (descriptor.semantic_type, descriptor.units) == ("scalar", "W")

    def test_a_quantity_comes_back_with_its_own_unit(self):
        problem = fleet().compile()
        descriptor = problem.expr("a/duty")
        assert isinstance(descriptor.symbol, ca.MX)
        assert (descriptor.units, descriptor.semantic_type) == ("1", "scalar")

    def test_a_vector_quantity_is_a_vector_and_a_matrix_is_a_matrix(self):
        vector = Quantity("probe", shape=(3, 1), default=[1.0, 2.0, 3.0],
                          provenance="A", source="fixture")
        assert probe(vector).compile().expr("probe").semantic_type == "vector"
        matrix = Quantity("probe", shape=(2, 2), default=1.0,
                          provenance="A", source="fixture")
        descriptor = probe(matrix, target=0.0).compile().expr("probe")
        assert (descriptor.semantic_type, descriptor.shape) == ("matrix", (2, 2))

    def test_a_fixed_quantity_comes_back_as_a_constant(self):
        problem = fleet().compile(roles={"a/duty": Role.FIXED},
                                  overrides={"a/duty": {"value": 0.25}})
        descriptor = problem.expr("a/duty")
        assert not isinstance(descriptor.symbol, ca.MX)
        np.testing.assert_allclose(float(descriptor.symbol), 0.25, rtol=1e-12)

    def test_an_unknown_path_lists_the_known_ones(self):
        with pytest.raises(ValueError, match=r"a/powerr") as err:
            fleet().compile().expr("a/powerr")
        assert "'a/power'" in str(err.value)


class TestProblemLevelTerms:

    def test_a_problem_level_cost_moves_the_optimum(self):
        baseline = fleet().compile().build().solve()
        problem = fleet().compile()
        problem.add_cost(-problem.expr("a/rate").symbol, "extra_rate", weight=1.0,
                         doc="Pay twice for satellite a")
        result = problem.build().solve()
        assert result.success
        assert result.f_opt < baseline.f_opt - 1.0
        assert "extra_rate" in result.cost_terms
        assert result["a/duty"][0] > result["b/duty"][0]

    def test_a_problem_level_constraint_binds(self):
        problem = fleet().compile()
        problem.add_constraint(problem.expr("a/duty").symbol, ub=0.5, name="duty_cap",
                               doc="Satellite a is throttled")
        result = problem.build().solve()
        np.testing.assert_allclose(result["a/duty"], [0.5], rtol=1e-5)
        assert result.constraint("duty_cap").active

    def test_registration_is_refused_after_build(self):
        problem = fleet().compile().build()
        with pytest.raises(RuntimeError, match="registration is locked"):
            problem.add_cost(ca.MX(1.0), "late")


class TestResolvingWithoutARebuild:

    def test_solve_values_overrides_a_parameter_for_one_call(self):
        problem = fleet().compile().build()
        tight = problem.solve()
        loose = problem.solve(values={"limit": 100.0})
        np.testing.assert_allclose(loose["a/duty"], [0.5], rtol=1e-6)
        assert loose.f_opt > tight.f_opt
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.75], rtol=1e-6)

    def test_solve_values_refuses_a_path_that_is_not_a_parameter(self):
        problem = fleet().compile().build()
        with pytest.raises(ValueError, match="'variable'") as err:
            problem.solve(values={"a/duty": 0.5})
        assert "'limit'" in str(err.value)

    def test_evaluate_after_a_resolve_reports_the_value_that_solve_was_given(self):
        problem = fleet().compile().build()
        values = problem.evaluate(problem.solve(values={"limit": 100.0}))
        np.testing.assert_allclose(values["limit"], [100.0], rtol=1e-12)
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=1e-5)

    def test_a_warm_start_reaches_the_backend(self):
        problem = fleet().compile().build()
        base = problem.solve()
        cold = problem.solve(values={"limit": 152.0})
        warm = problem.solve(values={"limit": 152.0}, warm_start=base)
        assert not cold.warm_started and warm.warm_started
        assert warm.success and warm.iterations < cold.iterations
        np.testing.assert_allclose(warm["a/duty"], cold["a/duty"], rtol=1e-5)


class TestDiscreteQuantities:

    def test_relax_mode_records_the_relaxed_quantity(self):
        quantity = Quantity("probe", lb=0.0, ub=10.0, default=0.0,
                            provenance="A", source="fixture")
        problem = probe(quantity, target=2.6).compile(roles={"probe": Role.DISCRETE},
                                                      discrete_mode="relax").build()
        result = problem.solve()
        assert problem.roles["probe"] is Role.DISCRETE
        assert result.relaxed_discrete == ("probe",)
        np.testing.assert_allclose(result["probe"], [2.6], atol=1e-5)

    def test_a_discrete_quantity_is_flagged_on_the_variable_record(self):
        quantity = Quantity("probe", lb=0.0, ub=10.0, default=0.0,
                            provenance="A", source="fixture")
        problem = probe(quantity).compile(roles={"probe": "discrete"}, discrete_mode="relax")
        assert bool(record_for(problem, "probe").discrete[0])

    @pytest.mark.requires_bonmin
    def test_native_mode_reaches_an_integer_optimum(self):
        quantity = Quantity("probe", lb=0.0, ub=10.0, default=0.0,
                            provenance="A", source="fixture")
        problem = probe(quantity, target=2.6, solver="bonmin").compile(
            roles={"probe": Role.DISCRETE}).build()
        result = problem.solve()
        assert result.relaxed_discrete == ()
        np.testing.assert_allclose(result["probe"], [3.0], atol=1e-6)


class TestEditableData:

    def test_set_value_changes_the_stored_parameter(self):
        problem = fleet().compile().build()
        problem.set_value("limit", 100.0)
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.5], rtol=1e-6)

    def test_set_value_refuses_a_variable_and_names_the_role(self):
        problem = fleet().compile().build()
        with pytest.raises(ValueError, match="'variable'"):
            problem.set_value("a/duty", 0.5)

    def test_set_bounds_narrows_a_variable(self):
        problem = fleet().compile().build()
        problem.set_bounds("a/duty", ub=0.4)
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.4], rtol=1e-5)

    def test_set_bounds_refuses_a_parameter(self):
        problem = fleet().compile().build()
        with pytest.raises(ValueError, match="'parameter'"):
            problem.set_bounds("limit", ub=200.0)

    def test_fix_then_unfix_restores_the_declared_bounds(self):
        problem = fleet().compile().build()
        problem.fix("a/duty", 0.25)
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.25], rtol=1e-6)
        problem.unfix("a/duty")
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.75], rtol=1e-5)

    def test_an_unknown_path_lists_the_eligible_ones(self):
        problem = fleet().compile().build()
        with pytest.raises(ValueError, match="not a quantity") as err:
            problem.fix("a/dutyy", 0.25)
        assert "'a/duty'" in str(err.value)


class TestLifecycle:

    def test_compiling_twice_is_refused(self):
        problem = fleet().compile()
        with pytest.raises(RuntimeError, match="already run"):
            problem.compile()

    def test_expr_before_compile_names_compile(self):
        with pytest.raises(RuntimeError, match=r"compile\(\) before expr"):
            fleet().expr("a/duty")

    def test_add_cost_before_compile_names_compile(self):
        with pytest.raises(RuntimeError, match=r"compile\(\) before add_cost"):
            fleet().add_cost(ca.MX(1.0), "early")

    def test_build_before_compile_names_compile(self):
        with pytest.raises(RuntimeError, match=r"compile\(\) before build"):
            fleet().build()

    def test_solve_before_build_names_build(self):
        with pytest.raises(RuntimeError, match=r"build\(\) before solve"):
            fleet().compile().solve()

    def test_roles_before_compile_names_compile(self):
        with pytest.raises(RuntimeError, match=r"compile\(\) before roles"):
            assert fleet().roles

    def test_the_backend_before_compile_names_compile(self):
        with pytest.raises(RuntimeError, match=r"compile\(\) before backend"):
            assert fleet().backend

    def test_a_compile_that_failed_halfway_can_be_retried_on_the_same_problem(self):
        problem = probe(Quantity("probe", default=1.0, provenance="A", source="fixture"))
        with pytest.raises(ValueError, match="free"):
            problem.compile(values={"free": [1.0, 2.0]})      # the second quantity, wrong shape
        problem.compile(values={"free": 0.5})
        assert [v.name for v in problem.backend.variables()] == ["probe", "free"]
        np.testing.assert_allclose(record_for(problem, "free").x0, [0.5], rtol=1e-12)

    def test_compile_and_build_return_the_problem_itself(self):
        problem = fleet()
        assert problem.compile() is problem
        assert problem.build() is problem
        assert problem.backend.is_built
