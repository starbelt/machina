"""
The swapc power budget: the three ``util.*`` power factories
(``machina.swapc.power``), the ``power_load``/``battery_energy`` signals and the
``ComputeLoad``/``PowerBudget`` components, alone and compiled through
``Problem``.

Sections
--------
TestSignals        -- what importing the pack declares; every unit is SI
TestFactories      -- each factory against its formula, and its guards
TestComputeLoad    -- declaration and the produced load
TestPowerBudget    -- declaration, and the loads= argument
TestAreaSizing     -- minimise array_area: closed-form area, active margin, battery,
                      sensitivity to a load
TestLoadsAdd       -- loads in two scopes, and two producers of one path
TestEclipseSweep   -- machina.study.sweep over eclipse_duration
"""

import casadi as ca
import numpy as np
import pytest

import machina.swapc  # noqa: F401
from machina.compiler import Problem
from machina.library import registry
from machina.model import (
    Aggregation,
    Builder,
    Component,
    Declaration,
    ModelError,
    Role,
    Scope,
    SignalRegistry,
)
from machina.model import signals as model_signals
from machina.study import sweep
from machina.swapc import ComputeLoad, PowerBudget
from machina.swapc import signals as swapc_signals
from machina.units import is_si

pytestmark = pytest.mark.requires_casadi

# PowerBudget's defaults.
S, ETA, L_D = 1361.0, 0.30, 0.85
X_E, X_D = 0.65, 0.85
T_E, T_ORBIT = 4320.0, 86164.0
DOD, ETA_B = 0.8, 0.9
FRONT, BUS, COMM = 5.0, 20.0, 10.0
# ComputeLoad's defaults.
E_SCENE, T_REFRESH = 10.0, 300.0

POWER_BUDGET_QUANTITIES = (
    "array_area", "front_power", "bus_power", "comm_power", "solar_flux", "cell_efficiency",
    "degradation", "X_e", "X_d", "eclipse_duration", "orbit_period", "depth_of_discharge",
    "battery_efficiency")


def area_per_watt(T_e: float = T_E) -> float:
    """dA*/dP: array area per watt of orbit-average load, (T_e/X_e + T_d/X_d) / (T_d S eta L_d)."""
    T_d = T_ORBIT - T_e
    return (T_e / X_E + T_d / X_D) / (T_d * S * ETA * L_D)


def array_power(area, T_e=T_E):
    """SMAD orbit-average array power, the reference for util.solar_array_power."""
    T_d = T_ORBIT - T_e
    return S * ETA * L_D * area * T_d / (T_e / X_E + T_d / X_D)


def total_load(*scene_energies) -> float:
    """Front + bus + comm, plus one E_scene / T_refresh per compute load."""
    return FRONT + BUS + COMM + sum(e / T_REFRESH for e in scene_energies)


def sized(components, **compile_kwargs) -> tuple:
    """``(problem, result)``: compile, minimise the root ``array_area``, solve."""
    problem = Problem(components, verbose=False).compile(**compile_kwargs)
    problem.add_cost(problem.expr("array_area").symbol, name="area")
    result = problem.build().solve()
    assert result.success, result.status
    return problem, result


def evaluate(descriptor, *args) -> float:
    return float(descriptor.function(*[ca.DM(a) for a in args]))


def quantities(component) -> dict:
    return {q.name: q for q in component.declare().quantities}


class TestSignals:

    def test_power_load_is_a_summed_watt_budget(self):
        signal = model_signals.DEFAULT.get("power_load")
        assert (signal.shape, signal.unit, signal.frame) == ((1, 1), "W", "none")
        assert signal.aggregation is Aggregation.SUM

    def test_battery_energy_is_a_unique_joule_figure(self):
        signal = model_signals.DEFAULT.get("battery_energy")
        assert (signal.shape, signal.unit, signal.frame) == ((1, 1), "J", "none")
        assert signal.aggregation is Aggregation.UNIQUE

    def test_declare_into_gives_a_clean_registry_the_same_set_in_order(self):
        registry_ = SignalRegistry()
        swapc_signals.declare_into(registry_)
        assert list(registry_.all()) == list(swapc_signals.SIGNALS)

    def test_the_swapc_signals_follow_every_earlier_pack_in_the_default_layout(self):
        names = list(model_signals.DEFAULT.all())
        assert names[-len(swapc_signals.SIGNALS):] == list(swapc_signals.SIGNALS) or \
            names.index("power_load") > names.index("coverage_total")

    def test_every_signal_and_quantity_unit_is_si(self):
        units = [model_signals.DEFAULT.get(name).unit for name in swapc_signals.SIGNALS]
        units += [q.unit for q in ComputeLoad().declare().quantities]
        units += [q.unit for q in PowerBudget().declare().quantities]
        assert units and all(is_si(unit) for unit in units), units


class TestFactories:

    @pytest.mark.parametrize("name", ["util.scene_compute_power", "util.solar_array_power",
                                      "util.battery_energy"])
    def test_registered_by_the_power_module(self, name):
        assert registry.registered_in(name) == "machina.swapc.power"

    @pytest.mark.parametrize("E, T", [(10.0, 300.0), (0.0, 60.0), (2.5e3, 600.0)])
    def test_scene_compute_power_is_energy_over_cadence(self, E, T):
        power = registry.get("util.scene_compute_power")()
        np.testing.assert_allclose(evaluate(power, E, T), E / T, rtol=1e-15)

    def test_scene_compute_power_stays_finite_at_zero_cadence(self):
        power = registry.get("util.scene_compute_power")()
        E, T = ca.SX.sym("E"), ca.SX.sym("T")
        grad = ca.Function("g", [E, T], [ca.gradient(power.function(E, T), ca.vertcat(E, T))])
        assert np.all(np.isfinite(np.asarray(grad(1.0, 0.0))))

    @pytest.mark.parametrize("area, T_e", [(1.0, 4320.0), (0.37, 0.0), (2.0, 2000.0)])
    def test_solar_array_power_matches_smad(self, area, T_e):
        power = registry.get("util.solar_array_power")()
        got = evaluate(power, area, S, ETA, L_D, T_e, T_ORBIT, X_E, X_D)
        np.testing.assert_allclose(got, array_power(area, T_e), rtol=1e-14)

    def test_solar_array_power_closes_the_orbit_energy_balance(self):
        """S eta L_d A T_d = P_bar (T_e / X_e + T_d / X_d): the derivation, checked."""
        power = registry.get("util.solar_array_power")()
        area = 0.8
        P_bar = evaluate(power, area, S, ETA, L_D, T_E, T_ORBIT, X_E, X_D)
        T_d = T_ORBIT - T_E
        np.testing.assert_allclose(P_bar * (T_E / X_E + T_d / X_D), S * ETA * L_D * area * T_d,
                                   rtol=1e-14)

    def test_no_daylight_generates_nothing(self):
        power = registry.get("util.solar_array_power")()
        for T_e in (T_ORBIT, T_ORBIT + 100.0):
            assert evaluate(power, 1.0, S, ETA, L_D, T_e, T_ORBIT, X_E, X_D) == 0.0

    @pytest.mark.parametrize("P, T_e, dod, eta", [(35.0, 4320.0, 0.8, 0.9), (1.0, 60.0, 1.0, 1.0)])
    def test_battery_energy_is_eclipse_energy_over_dod_and_efficiency(self, P, T_e, dod, eta):
        battery = registry.get("util.battery_energy")()
        np.testing.assert_allclose(evaluate(battery, P, T_e, dod, eta), P * T_e / (dod * eta),
                                   rtol=1e-15)


class TestComputeLoad:

    def test_declaration(self):
        q = quantities(ComputeLoad())
        assert list(q) == ["E_scene", "T_refresh"]
        assert (q["E_scene"].unit, q["E_scene"].role, q["E_scene"].default_role) == \
            ("J", Role.FLEXIBLE, Role.PARAMETER)
        assert (q["T_refresh"].unit, q["T_refresh"].role) == ("s", Role.FIXED)
        assert (q["E_scene"].default, q["T_refresh"].default) == (E_SCENE, T_REFRESH)
        assert (q["E_scene"].provenance, q["T_refresh"].provenance) == ("E", "D")
        assert ComputeLoad().declare().produces == ("power_load",)

    def test_produces_energy_over_cadence(self):
        builder = Builder([ComputeLoad()]).declare()
        wired = builder.wire({"E_scene": ca.DM(12.0), "T_refresh": ca.DM(60.0)})
        np.testing.assert_allclose(float(wired.values["power_load"]), 0.2, rtol=1e-15)


class TestPowerBudget:

    def test_declaration_order_roles_and_units(self):
        q = quantities(PowerBudget())
        assert tuple(q) == POWER_BUDGET_QUANTITIES
        assert q["array_area"].unit == "m^2"
        assert q["solar_flux"].unit == "W/m^2"
        assert all(q[n].unit == "W" for n in ("front_power", "bus_power", "comm_power"))
        assert all(q[n].unit == "s" for n in ("eclipse_duration", "orbit_period"))
        assert (q["array_area"].role, q["array_area"].default_role) == \
            (Role.FLEXIBLE, Role.VARIABLE)
        flexible_parameters = ("front_power", "bus_power", "comm_power", "eclipse_duration")
        assert all(q[n].role is Role.FLEXIBLE and q[n].default_role is Role.PARAMETER
                   for n in flexible_parameters)
        fixed = [n for n in POWER_BUDGET_QUANTITIES
                 if n not in flexible_parameters and n != "array_area"]
        assert all(q[n].role is Role.FIXED for n in fixed)
        assert all(q[n].provenance in ("D", "P", "E", "A", "M") and q[n].source
                   for n in POWER_BUDGET_QUANTITIES)

    def test_every_default_is_sourced(self):
        """compile(strict=True) refuses an unsourced default."""
        Problem([ComputeLoad(), PowerBudget()], verbose=False).compile(strict=True)

    def test_reads_power_load_and_produces_battery_energy(self):
        dec = PowerBudget().declare()
        assert dec.algebraic == (("load_0", "power_load"),)
        assert dec.produces == ("battery_energy",)
        assert [c.name for c in dec.constraints] == ["power_margin"]
        assert dec.costs == ()

    def test_loads_become_one_read_each(self):
        dec = PowerBudget(loads=("a/power_load", "/b/power_load")).declare()
        assert dec.algebraic == (("load_0", "a/power_load"), ("load_1", "/b/power_load"))
        assert PowerBudget(loads=()).declare().algebraic == ()

    @pytest.mark.parametrize("loads, match", [
        ("power_load", "tuple"),
        (("a/mass",), "path to a power_load"),
        (("a/power_load", "a/power_load"), "twice"),
    ])
    def test_bad_loads_are_refused_naming_the_fix(self, loads, match):
        with pytest.raises(ModelError, match=match):
            PowerBudget(loads=loads)


class TestAreaSizing:
    """minimise array_area s.t. power_margin >= 0: the margin is active, A* = P_tot dA/dP."""

    def test_optimum_area_is_the_closed_form(self):
        problem, result = sized([ComputeLoad(), PowerBudget()])
        area = problem.evaluate(result)["array_area"].item()
        np.testing.assert_allclose(area, total_load(E_SCENE) * area_per_watt(), rtol=1e-8)
        np.testing.assert_allclose(result.f_opt, area, rtol=1e-12)

    def test_the_margin_is_active(self):
        _, result = sized([ComputeLoad(), PowerBudget()])
        margin = result.constraint("power_margin")
        assert bool(margin.active.item())
        np.testing.assert_allclose(margin.value.item(), 0.0, atol=1e-7)

    def test_battery_energy_carries_the_total_load_through_the_eclipse(self):
        problem, result = sized([ComputeLoad(), PowerBudget()])
        values = problem.evaluate(result)
        np.testing.assert_allclose(values["battery_energy"].item(),
                                   total_load(E_SCENE) * T_E / (DOD * ETA_B), rtol=1e-12)
        np.testing.assert_allclose(values["power_load"].item(), E_SCENE / T_REFRESH, rtol=1e-15)

    def test_sensitivity_to_a_load_is_area_per_watt(self):
        """d f*/d bus_power = -lam_p = dA*/dP > 0: one more watt costs this much array."""
        _, result = sized([ComputeLoad(), PowerBudget()])
        np.testing.assert_allclose(result.sensitivity("bus_power").item(), area_per_watt(),
                                   rtol=1e-6)
        np.testing.assert_allclose(result.sensitivity("E_scene").item(),
                                   area_per_watt() / T_REFRESH, rtol=1e-6)

    def test_the_margin_multiplier_is_minus_area_per_watt(self):
        """CasADi's sign: an active lower bound has a negative multiplier."""
        _, result = sized([ComputeLoad(), PowerBudget()])
        np.testing.assert_allclose(result.constraint("power_margin").multiplier.item(),
                                   -area_per_watt(), rtol=1e-6)


class ExtraLoad(Component):
    """A second producer of power_load with no quantities, so it can share a scope."""

    def declare(self):
        return Declaration(produces=("power_load",))

    def build(self, helpers):
        return {"g": ca.Function("extra_g", [], [ca.SX(7.0)], [], ["power_load"])}


class TestLoadsAdd:

    def test_two_compute_loads_in_scopes_sum(self):
        components = [Scope("a", [ComputeLoad()]), Scope("b", [ComputeLoad()]),
                      PowerBudget(loads=("a/power_load", "b/power_load"))]
        problem, result = sized(components, values={"a/E_scene": 10.0, "b/E_scene": 40.0})
        values = problem.evaluate(result)
        np.testing.assert_allclose(values["a/power_load"].item(), 10.0 / T_REFRESH, rtol=1e-15)
        np.testing.assert_allclose(values["b/power_load"].item(), 40.0 / T_REFRESH, rtol=1e-15)
        P_tot = total_load(10.0, 40.0)
        np.testing.assert_allclose(values["array_area"].item(), P_tot * area_per_watt(),
                                   rtol=1e-8)
        np.testing.assert_allclose(values["battery_energy"].item(),
                                   P_tot * T_E / (DOD * ETA_B), rtol=1e-12)

    def test_two_producers_of_one_path_add(self):
        """power_load is SUM: a ComputeLoad and another producer in one scope add."""
        problem, result = sized([ComputeLoad(), ExtraLoad(), PowerBudget()])
        values = problem.evaluate(result)
        np.testing.assert_allclose(values["power_load"].item(), E_SCENE / T_REFRESH + 7.0,
                                   rtol=1e-15)
        np.testing.assert_allclose(values["array_area"].item(),
                                   (total_load(E_SCENE) + 7.0) * area_per_watt(), rtol=1e-8)

    def test_a_child_scope_load_is_not_read_by_the_bare_name(self):
        """Lexical lookup goes outward only; the unproduced root SUM reads zero."""
        builder = Builder([Scope("a", [ComputeLoad()]), PowerBudget()]).declare()
        assert builder.resolved("power_budget")["load_0"] == "power_load"
        assert builder.unproduced_sums == ["power_load"]

    def test_the_budget_in_the_loads_scope_reads_it_by_the_bare_name(self):
        builder = Builder([Scope("sc", [ComputeLoad(), PowerBudget()])]).declare()
        assert builder.resolved("sc/power_budget")["load_0"] == "sc/power_load"
        assert builder.unproduced_sums == []


class TestEclipseSweep:
    """The eclipse is a parameter: one warm-started solve per value, no rebuild."""

    ECLIPSES = [0.0, 1800.0, 3600.0, 4320.0]

    def test_area_grows_with_the_eclipse(self):
        problem = Problem([ComputeLoad(), PowerBudget()], verbose=False).compile()
        problem.add_cost(problem.expr("array_area").symbol, name="area")
        problem.build()
        table = sweep(problem, {"eclipse_duration": self.ECLIPSES},
                      collect=("array_area", "battery_energy"))
        assert all(table["success"])
        areas = np.asarray(table["array_area"])
        assert np.all(np.diff(areas) > 0.0)
        expected = [total_load(E_SCENE) * area_per_watt(T_e) for T_e in self.ECLIPSES]
        np.testing.assert_allclose(areas, expected, rtol=1e-8)
        np.testing.assert_allclose(
            table["battery_energy"],
            [total_load(E_SCENE) * T_e / (DOD * ETA_B) for T_e in self.ECLIPSES],
            rtol=1e-12, atol=1e-9)
