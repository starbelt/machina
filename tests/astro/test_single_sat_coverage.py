"""
The astro pack's ``SingleSatCoverage``: what it declares, what importing the
pack registers, what it builds through the ``Builder``, and the oracles.

``TestOracles`` at the foot of the file compiles the motivating problem of
Math/MEE Numerics through ``Problem`` and pins what IPOPT does with it: April's
documented stop point at unit scale, the strict solve failing unscaled, and
the strict solve converging to the apogee bound once the declared scales are
in place.
"""

import math
import warnings

import casadi as ca
import numpy as np
import pytest

from machina.astro import SingleSatCoverage
from machina.astro import signals as astro_signals
from machina.model import Aggregation, Builder, Role, Scope, SignalRegistry
from machina.model import signals as model_signals

pytestmark = pytest.mark.requires_casadi

R_EARTH = 6378.137
MU = 398600.4418

# April's declaration order, which is the decision-vector layout.
QUANTITY_NAMES = ("p", "f", "g", "h", "k", "L", "r_target", "min_elevation", "sigmoid_k")


def quantities(component) -> dict:
    """The component's quantities by name."""
    return {q.name: q for q in component.declare().quantities}


def leaves(component, scope: str, overrides: dict = None) -> dict:
    """A ``ca.DM`` leaf per quantity, from its declared default, with overrides."""
    overrides = overrides or {}
    out = {}
    for quantity in component.declare().quantities:
        value = overrides.get(quantity.name, quantity.default)
        path = f"{scope}/{quantity.name}" if scope else quantity.name
        out[path] = ca.DM(np.asarray(value, dtype=float).reshape(quantity.shape))
    return out


def iss_like(f: float = 0.01, g: float = 0.0, raan_deg: float = 240.0) -> dict:
    """An ISS-like state: 500 km, i = 51.6 deg, at the given RAAN."""
    tan_half_i = math.tan(math.radians(51.6) / 2.0)
    raan = math.radians(raan_deg)
    return {"p": R_EARTH + 500.0, "f": f, "g": g,
            "h": tan_half_i * math.cos(raan), "k": tan_half_i * math.sin(raan)}


def evaluate(component, overrides: dict) -> list:
    """``[coverage_total, perigee row, apogee row]`` from the component, wired with DM."""
    builder = Builder([Scope("sat", [component])]).declare()
    wired = builder.wire(leaves(component, "sat", overrides))
    return [float(wired.values["sat/coverage_total"]),
            float(wired.constraints[0].expr),
            float(wired.constraints[1].expr)]


class TestDeclaration:
    """Nine quantities, in April's order, each one documented and sourced."""

    def test_declares_the_nine_quantities_in_aprils_order(self):
        declared = SingleSatCoverage().declare().quantities
        assert tuple(q.name for q in declared) == QUANTITY_NAMES

    def test_shapes_units_and_frames(self):
        q = quantities(SingleSatCoverage(n_sample_points=24))
        assert [q[name].shape for name in QUANTITY_NAMES] == \
            [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (24, 1), (3, 1), (1, 1), (1, 1)]
        assert [q[name].unit for name in QUANTITY_NAMES] == \
            ["km", "1", "1", "1", "1", "rad", "km", "rad", "1/rad"]
        assert q["r_target"].frame == "eci"
        assert all(q[name].frame == "none" for name in QUANTITY_NAMES if name != "r_target")

    def test_the_five_elements_are_flexible_and_the_four_parameters_are_not(self):
        q = quantities(SingleSatCoverage())
        for name in ("p", "f", "g", "h", "k"):
            assert q[name].role is Role.FLEXIBLE
            assert q[name].default_role is Role.VARIABLE
        for name in ("L", "r_target", "min_elevation", "sigmoid_k"):
            assert q[name].role is Role.PARAMETER

    def test_every_default_carries_a_provenance_code_and_a_source(self):
        for quantity in SingleSatCoverage().declare().quantities:
            assert quantity.default is not None
            assert quantity.provenance in ("D", "P", "E", "A", "M")
            assert quantity.source
            assert quantity.doc

    def test_element_bounds_follow_the_mee_numerics_rules(self):
        component = SingleSatCoverage()
        q = quantities(component)
        assert (q["p"].lb, q["p"].ub) == (100.0, R_EARTH + component.apogee_max_km)
        assert (q["f"].lb, q["f"].ub) == (-1.0, 1.0)
        assert (q["g"].lb, q["g"].ub) == (-1.0, 1.0)
        assert (q["h"].lb, q["h"].ub) == (-1.5, 1.5)
        assert (q["k"].lb, q["k"].ub) == (-1.5, 1.5)
        assert q["f"].default == 0.01   # non-zero, per Rule 3

    def test_p_and_the_altitude_rows_carry_the_mee_numerics_scaling(self):
        """The nominal is the body radius, so it follows ``R_earth`` rather than Earth."""
        component = SingleSatCoverage()
        declaration = component.declare()
        assert quantities(component)["p"].scale == component.R_earth == 6378.137
        assert [c.name for c in declaration.constraints] == \
            ["perigee_altitude", "apogee_altitude"]
        for constraint in declaration.constraints:
            assert constraint.scale == component.R_earth ** 2
            assert constraint.lb == 0.0
            assert constraint.shape == (1, 1)
        mars = SingleSatCoverage(R_earth=3396.2, mu=42828.37)
        assert quantities(mars)["p"].scale == 3396.2
        assert mars.declare().constraints[0].scale == 3396.2 ** 2

    def test_target_default_is_on_the_sphere_and_points_at_the_target(self):
        component = SingleSatCoverage(target_lat_deg=38.9, target_lon_deg=-77.0)
        r_target = np.asarray(quantities(component)["r_target"].default, dtype=float)
        assert r_target.shape == (3, 1)
        np.testing.assert_allclose(np.linalg.norm(r_target), R_EARTH, rtol=1e-6)
        np.testing.assert_allclose(float(r_target[2, 0]),
                                   R_EARTH * math.sin(math.radians(38.9)), rtol=1e-12)
        np.testing.assert_allclose(
            float(r_target[0, 0]),
            R_EARTH * math.cos(math.radians(38.9)) * math.cos(math.radians(-77.0)),
            rtol=1e-12)

    def test_sample_grid_is_a_column_of_n_longitudes_across_one_revolution(self):
        component = SingleSatCoverage(n_sample_points=24)
        grid = np.asarray(quantities(component)["L"].default, dtype=float)
        assert grid.shape == (24, 1)
        np.testing.assert_allclose(grid.ravel(), np.linspace(0.0, 2.0 * math.pi, 25)[:-1],
                                   atol=1e-12)

    def test_n_sample_points_changes_the_declared_shape_of_L(self):
        assert quantities(SingleSatCoverage(n_sample_points=12))["L"].shape == (12, 1)
        assert quantities(SingleSatCoverage(n_sample_points=72))["L"].shape == (72, 1)

    def test_produces_coverage_total_and_leaves_the_sign_to_the_problem(self):
        declaration = SingleSatCoverage().declare()
        assert declaration.produces == ("coverage_total",)
        assert declaration.costs == ()      # the problem adds -coverage_total itself
        assert declaration.derivatives == ()

    def test_default_name_is_the_snake_case_class_name(self):
        assert SingleSatCoverage().name == "single_sat_coverage"
        assert SingleSatCoverage("a").name == "single_sat_coverage_a"


class TestPack:
    """Importing the pack is what declares its frames and its one signal."""

    def test_import_declares_the_orbital_frames(self):
        for name in ("eci", "ecef", "lvlh"):
            assert model_signals.DEFAULT.frame(name).name == name

    def test_import_declares_coverage_total(self):
        signal = model_signals.DEFAULT.get("coverage_total")
        assert signal.shape == (1, 1)
        assert signal.unit == "1"
        assert signal.aggregation is Aggregation.UNIQUE
        assert signal.frame == "none"

    def test_importing_the_pack_twice_declares_nothing_twice(self):
        import machina.astro as first
        import machina.astro as second

        assert first is second
        assert model_signals.DEFAULT.get("coverage_total").unit == "1"

    def test_declare_into_gives_a_clean_registry_the_same_set(self):
        registry = SignalRegistry()
        astro_signals.declare_into(registry)
        assert [name for name in registry.all() if name in astro_signals.SIGNALS] == \
            list(astro_signals.SIGNALS)
        for name in astro_signals.FRAMES:
            assert registry.frame(name).name == name


class TestBuildThroughTheBuilder:
    """The component is a Phase 2 component: it declares, wires and builds."""

    def test_declare_lays_out_the_nine_quantities_in_order(self):
        builder = Builder([Scope("sat", [SingleSatCoverage()])]).declare()
        assert builder.quantity_order == [f"sat/{name}" for name in QUANTITY_NAMES]
        assert builder.state_order == []
        assert builder.input_order == []
        assert builder.algebraic_order == ["sat/coverage_total"]

    def test_wiring_the_defaults_gives_a_coverage_fraction(self):
        component = SingleSatCoverage()
        builder = Builder([Scope("sat", [component])]).declare()
        wired = builder.wire(leaves(component, "sat"))
        coverage = float(wired.values["sat/coverage_total"])
        assert 0.0 <= coverage <= 1.0

    def test_the_constraint_rows_are_the_squared_altitude_forms(self):
        component = SingleSatCoverage()
        state = iss_like(f=0.1, g=-0.05)
        _, perigee, apogee = evaluate(component, state)

        e_squared = state["f"] ** 2 + state["g"] ** 2
        R_min = R_EARTH + component.perigee_min_km
        R_max = R_EARTH + component.apogee_max_km
        np.testing.assert_allclose(
            perigee, (state["p"] - R_min) ** 2 - R_min ** 2 * e_squared, rtol=1e-12)
        np.testing.assert_allclose(
            apogee, (R_max - state["p"]) ** 2 - R_max ** 2 * e_squared, rtol=1e-12)

    def test_coverage_rises_when_the_elevation_mask_is_relaxed(self):
        state = iss_like()
        strict = evaluate(SingleSatCoverage(min_elevation_deg=40.0), state)[0]
        relaxed = evaluate(SingleSatCoverage(min_elevation_deg=0.0), state)[0]
        assert relaxed > strict

    def test_it_also_builds_at_root_scope(self):
        builder = Builder([SingleSatCoverage()]).declare()
        assert builder.quantity_order == list(QUANTITY_NAMES)
        assert builder.algebraic_order == ["coverage_total"]

    def test_build_returns_an_f_g_h_bundle_over_the_quantity_vector(self):
        builder = Builder([SingleSatCoverage(n_sample_points=24)]).declare()
        assert (builder.nx, builder.nu, builder.nq) == (0, 0, 5 + 24 + 3 + 1 + 1)

        bundle = builder.build()
        assert sorted(bundle) == ["f", "g", "h"]     # no costs declared, so no "J"
        for key in ("f", "g", "h"):
            assert [bundle[key].name_in(i) for i in range(bundle[key].n_in())] == \
                ["x", "u", "q"]
            assert [bundle[key].size1_in(i) for i in range(3)] == [0, 0, builder.nq]
        assert bundle["h"].size1_out(0) == 2         # perigee row, then apogee row
        assert bundle["g"].size1_out(0) == 1         # coverage_total


# --- the oracles ------------------------------------------------------------------------------
#
# The motivating problem of Math/MEE Numerics, compiled through ``Problem``: maximise coverage
# of Washington DC over a 200-1600 km altitude box from an ISS-like start. Two oracles are
# pinned -- April's documented stop point at unit scale and the strict scaled optimum on the
# apogee bound -- together with the strict unscaled solve whose failure is why the scaling exists.

APRIL_OPTS = {"ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 3}
P_SCALE = R_EARTH                 # the component's nominal for p: the body radius
ALTITUDE_SCALE = P_SCALE**2

# x0 and the box, in the declaration order that is the decision-vector layout.
ELEMENT_BOUNDS = (("p", R_EARTH + 200.0, R_EARTH + 1600.0), ("f", -0.30, 0.30),
                  ("g", -0.30, 0.30), ("h", -1.5, 1.5), ("k", -1.5, 1.5))


def coverage_component() -> SingleSatCoverage:
    """The component the motivating problem is built from."""
    return SingleSatCoverage(target_lat_deg=38.9, target_lon_deg=-77.0, n_sample_points=24,
                             min_elevation_deg=10.0, sigmoid_k=20.0, perigee_min_km=200.0,
                             apogee_max_km=1600.0, mu=MU, R_earth=R_EARTH)


def coverage_overrides(*, unit_scale: bool) -> dict:
    """Initial guess and box for the five elements; optionally scales back to 1.

    The component declares ``scale = R_earth`` on ``p`` and ``R_earth^2`` on both altitude
    rows (the MEE Numerics recipe, derived from the body), so the *default* compile is the
    scaled one. April's unscaled problem is the override.
    """
    state = iss_like()                               # p = R + 500, f = 0.01, g = 0, RAAN 240 deg
    overrides = {f"sat/{name}": {"x0": state[name], "lb": lb, "ub": ub}
                 for name, lb, ub in ELEMENT_BOUNDS}
    if unit_scale:
        overrides["sat/p"]["scale"] = 1.0
        overrides["sat/perigee_altitude"] = {"scale": 1.0}
        overrides["sat/apogee_altitude"] = {"scale": 1.0}
    return overrides


def coverage_problem(solver_opts: dict, *, unit_scale: bool, build: bool = True):
    """The compiled problem. Warnings are left alone: a default compile must not raise any."""
    from machina.compiler import Problem

    problem = Problem([Scope("sat", [coverage_component()])],
                      solver_opts=solver_opts, verbose=False)
    problem.compile(overrides=coverage_overrides(unit_scale=unit_scale))
    problem.add_cost(-problem.expr("sat/coverage_total").symbol, name="neg_coverage")
    return problem.build() if build else problem


class TestOracles:
    """The pinned numbers of Math/MEE Numerics, reproduced through ``Problem``."""

    def test_april_recipe_at_unit_scale(self):
        """Unscaled, with April's loose acceptable tolerance: her documented stop point."""
        res = coverage_problem(APRIL_OPTS, unit_scale=True).solve()
        assert res.status == "Solved_To_Acceptable_Level"
        assert res.iterations == 13
        np.testing.assert_allclose(res.f_opt, -0.12583632, atol=1e-7)

    def test_unscaled_strict_solve_fails(self):
        """Why the scaling exists: strict defaults on the unscaled rows hit a NaN."""
        res = coverage_problem({}, unit_scale=True).solve()
        assert not res.success
        assert res.status == "Invalid_Number_Detected"

    def test_scaled_strict_solve_converges_to_the_apogee_bound(self):
        """With the declared scales, the same defaults converge, with apogee active."""
        res = coverage_problem({}, unit_scale=False).solve()
        assert res.status == "Solve_Succeeded"
        np.testing.assert_allclose(res.f_opt, -0.16394, atol=1e-4)
        np.testing.assert_allclose(res["sat/p"], [R_EARTH + 1600.0], rtol=1e-6)
        assert res.bound_multiplier("sat/p")[0] > 0.0
        # Physical units (km^2); the solver's tolerance applies to the scaled row.
        assert res.constraint("sat/apogee_altitude").value[0] / ALTITUDE_SCALE >= -1e-7

    def test_the_declared_scales_reach_the_backend(self):
        """The component's scaling recipe survives compile, and an override replaces it."""
        backend = coverage_problem({}, unit_scale=False, build=False).backend
        variables = {record.name: record for record in backend.variables()}
        constraints = {record.name: record for record in backend.constraints()}
        np.testing.assert_array_equal(variables["sat/p"].scale, [P_SCALE])
        assert constraints["sat/perigee_altitude"].scale == ALTITUDE_SCALE
        assert constraints["sat/apogee_altitude"].scale == ALTITUDE_SCALE

        unit = coverage_problem({}, unit_scale=True, build=False).backend
        unit_variables = {record.name: record for record in unit.variables()}
        np.testing.assert_array_equal(unit_variables["sat/p"].scale, [1.0])
        assert all(record.scale == 1.0 for record in unit.constraints())

    def test_the_default_compile_warns_about_nothing(self):
        """Every declared default carries a provenance code, so no value is unsourced."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coverage_problem({}, unit_scale=False, build=False)
        assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []
