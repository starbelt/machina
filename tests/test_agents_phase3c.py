"""
tests/test_agents_phase3c.py
------------------------------
Phase 3c unit tests for the SingleSatCoverage agent.

Test classes
------------
TestSingleSatCoverageDeclare  -- declare() output correctness
TestSingleSatCoverageBuild    -- build() namespace and constraints
TestSingleSatCoverageResolve  -- resolve() / list_paths()
TestSingleSatCoverageEndToEnd -- full compile -> solve optimization
"""

import math
import warnings

import casadi as ca
import numpy as np
import pytest

from machina.agents import SingleSatCoverage
from machina.agents.agent_type import QuantityDeclaration, ConstraintDeclaration
from machina.blocks.descriptor import SymbolDescriptor
from machina.compiler.compiler_stub import CompilerStub

pytestmark = pytest.mark.requires_casadi


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SMALL_N = 12   # N=12 sample points for fast unit tests

_BASE_CONFIG = {
    'n_sample_points': _SMALL_N,
    'ground_target': {'lat_deg': 38.9, 'lon_deg': -77.0},
    'coverage': {'min_elevation_deg': 10.0, 'sigmoid_k': 20.0},
    'altitude_bounds': {'perigee_min_km': 200.0, 'apogee_max_km': 40000.0},
    'constants': {'mu': 398600.4418, 'R_earth': 6378.137},
}

R_EARTH = 6378.137
MU      = 398600.4418


def _make_agent(config=None, name='sat'):
    """Create a SingleSatCoverage agent with optional config overrides."""
    cfg = {**_BASE_CONFIG, **(config or {})}
    return SingleSatCoverage(name, cfg)


def _compile_agent(agent, suppress_warnings=True):
    """
    Run the full declare -> compile lifecycle.

    Returns (compiler, symbols_dict) where symbols_dict mirrors what the
    agent's build() method received.
    """
    compiler = CompilerStub(solver_opts={'ipopt.print_level': 0,
                                         'ipopt.sb': 'yes'})
    compiler.add_agent(agent)

    # Provide explicit overrides so compiler suppresses default-value warnings
    overrides = {}
    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter('ignore', UserWarning)
        compiler.compile(overrides=overrides)

    return compiler


# ---------------------------------------------------------------------------
# TestSingleSatCoverageDeclare
# ---------------------------------------------------------------------------

class TestSingleSatCoverageDeclare:

    def test_returns_list_of_quantity_declarations(self):
        agent = _make_agent()
        decls = agent.declare()
        assert isinstance(decls, list)
        assert all(isinstance(d, QuantityDeclaration) for d in decls)

    def test_returns_nine_quantities(self):
        agent = _make_agent()
        decls = agent.declare()
        assert len(decls) == 9

    def test_declared_paths(self):
        agent = _make_agent()
        paths = {d.path for d in agent.declare()}
        expected = {
            'orbital/p', 'orbital/f', 'orbital/g', 'orbital/h', 'orbital/k',
            'sample_points/L',
            'target/position',
            'coverage/min_elevation',
            'coverage/sigmoid_k',
        }
        assert paths == expected

    def test_orbital_elements_flexible_variable(self):
        agent = _make_agent()
        orbital_decls = [d for d in agent.declare()
                         if d.path.startswith('orbital/')]
        for d in orbital_decls:
            assert d.role == 'flexible', f"{d.path} should be flexible"
            assert d.default_role == 'variable', f"{d.path} default_role should be variable"

    def test_parameters_always_parameter(self):
        agent = _make_agent()
        param_paths = {'sample_points/L', 'target/position',
                       'coverage/min_elevation', 'coverage/sigmoid_k'}
        param_decls = [d for d in agent.declare() if d.path in param_paths]
        assert len(param_decls) == 4
        for d in param_decls:
            assert d.role == 'always_parameter', f"{d.path} should be always_parameter"

    def test_sample_points_L_shape(self):
        N = 18
        agent = _make_agent({'n_sample_points': N})
        L_decl = next(d for d in agent.declare() if d.path == 'sample_points/L')
        assert L_decl.shape == (N, 1)

    def test_sample_points_L_default_value(self):
        """L samples are linspace(0, 2*pi, N+1)[:-1]."""
        N = _SMALL_N
        agent = _make_agent()
        L_decl = next(d for d in agent.declare() if d.path == 'sample_points/L')
        expected = np.linspace(0.0, 2 * math.pi, N + 1)[:-1]
        np.testing.assert_allclose(L_decl.default_value, expected, atol=1e-12)

    def test_target_position_default_value_shape(self):
        agent = _make_agent()
        t_decl = next(d for d in agent.declare() if d.path == 'target/position')
        assert np.asarray(t_decl.default_value).shape == (3,)

    def test_target_position_magnitude(self):
        """Target ECI position is on Earth's surface."""
        agent = _make_agent()
        t_decl = next(d for d in agent.declare() if d.path == 'target/position')
        mag = np.linalg.norm(t_decl.default_value)
        np.testing.assert_allclose(mag, R_EARTH, rtol=1e-6)

    def test_min_elevation_converted_to_radians(self):
        agent = _make_agent({'coverage': {'min_elevation_deg': 10.0, 'sigmoid_k': 20.0}})
        me_decl = next(d for d in agent.declare()
                       if d.path == 'coverage/min_elevation')
        np.testing.assert_allclose(float(me_decl.default_value),
                                   math.radians(10.0), atol=1e-12)

    def test_orbital_p_bounds_reasonable(self):
        agent = _make_agent()
        p_decl = next(d for d in agent.declare() if d.path == 'orbital/p')
        assert p_decl.lb > 0
        assert p_decl.ub > p_decl.lb

    def test_default_role_consistent_with_docs(self):
        """Orbital elements (flexible) default to variable role."""
        agent = _make_agent()
        for d in agent.declare():
            if d.path.startswith('orbital/'):
                assert d.default_role == 'variable'
            # always_parameter quantities: default_role has no effect, skip check


# ---------------------------------------------------------------------------
# TestSingleSatCoverageBuild
# ---------------------------------------------------------------------------

class TestSingleSatCoverageBuild:

    def test_built_flag_false_before_build(self):
        agent = _make_agent()
        assert not agent._built

    def test_build_sets_built_flag(self):
        agent = _make_agent()
        _compile_agent(agent)
        assert agent._built

    def test_namespace_contains_orbital_elements(self):
        agent = _make_agent()
        _compile_agent(agent)
        for path in ('orbital/p', 'orbital/f', 'orbital/g', 'orbital/h', 'orbital/k'):
            assert path in agent._namespace

    def test_namespace_contains_derived_quantities(self):
        agent = _make_agent()
        _compile_agent(agent)
        for path in ('orbital/sma', 'orbital/ecc', 'orbital/inc', 'orbital/period'):
            assert path in agent._namespace

    def test_namespace_contains_coverage_total(self):
        agent = _make_agent()
        _compile_agent(agent)
        assert 'coverage/total' in agent._namespace

    def test_namespace_has_per_point_keys(self):
        agent = _make_agent()
        _compile_agent(agent)
        for i in range(_SMALL_N):
            assert f'coverage/per_point/{i}' in agent._namespace

    def test_coverage_total_is_symbol_descriptor(self):
        agent = _make_agent()
        _compile_agent(agent)
        sd = agent._namespace['coverage/total']
        assert isinstance(sd, SymbolDescriptor)

    def test_build_returns_two_constraints(self):
        # Build manually to inspect return value
        agent = _make_agent()
        decls = agent.declare()
        symbols = {}
        for d in decls:
            if d.role == 'always_parameter':
                sym = ca.MX.sym(d.path.replace('/', '_'), *d.shape)
            else:
                sym = ca.MX.sym(d.path.replace('/', '_'), *d.shape)
            symbols[d.path] = sym

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            constraints = agent.build(symbols)

        assert len(constraints) == 2
        assert all(isinstance(c, ConstraintDeclaration) for c in constraints)

    def test_constraint_names(self):
        agent = _make_agent()
        decls = agent.declare()
        symbols = {d.path: ca.MX.sym(d.path.replace('/', '_'), *d.shape)
                   for d in decls}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            constraints = agent.build(symbols)
        names = {c.name for c in constraints}
        assert names == {'perigee_altitude', 'apogee_altitude'}

    def test_perigee_constraint_has_correct_bounds(self):
        """Perigee constraint uses squared form: (p-R_min)^2 - R_min^2*e^2 >= 0."""
        agent = _make_agent()
        decls = agent.declare()
        symbols = {d.path: ca.MX.sym(d.path.replace('/', '_'), *d.shape)
                   for d in decls}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            constraints = agent.build(symbols)
        peri = next(c for c in constraints if c.name == 'perigee_altitude')
        assert peri.lb == 0.0
        assert peri.ub == np.inf

    def test_apogee_constraint_has_correct_bounds(self):
        """Apogee constraint uses squared form: (R_max-p)^2 - R_max^2*e^2 >= 0."""
        agent = _make_agent()
        decls = agent.declare()
        symbols = {d.path: ca.MX.sym(d.path.replace('/', '_'), *d.shape)
                   for d in decls}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            constraints = agent.build(symbols)
        apo = next(c for c in constraints if c.name == 'apogee_altitude')
        assert apo.lb == 0.0
        assert apo.ub == np.inf

    def test_all_namespace_entries_are_symbol_descriptors(self):
        agent = _make_agent()
        _compile_agent(agent)
        for path, sd in agent._namespace.items():
            assert isinstance(sd, SymbolDescriptor), \
                f"Namespace entry '{path}' is not a SymbolDescriptor"


# ---------------------------------------------------------------------------
# TestSingleSatCoverageResolve
# ---------------------------------------------------------------------------

class TestSingleSatCoverageResolve:

    def test_resolve_before_build_raises(self):
        agent = _make_agent()
        with pytest.raises(RuntimeError, match='before build'):
            agent.resolve('orbital/p')

    def test_resolve_after_build_returns_symbol_descriptor(self):
        agent = _make_agent()
        _compile_agent(agent)
        sd = agent.resolve('orbital/p')
        assert isinstance(sd, SymbolDescriptor)

    def test_resolve_unknown_path_raises_key_error(self):
        agent = _make_agent()
        _compile_agent(agent)
        with pytest.raises(KeyError):
            agent.resolve('orbital/nonexistent')

    def test_list_paths_before_build_returns_empty(self):
        agent = _make_agent()
        assert agent.list_paths() == []

    def test_list_paths_after_build_is_sorted(self):
        agent = _make_agent()
        _compile_agent(agent)
        paths = agent.list_paths()
        assert paths == sorted(paths)

    def test_list_paths_count(self):
        """5 orbital vars + 4 derived + 3 param entries + 1 total + N per-point."""
        # sample_points/L is a parameter but not added to namespace (it's internal)
        # Namespace: p,f,g,h,k + sma,ecc,inc,period + target/position,
        #            coverage/min_elevation, coverage/sigmoid_k + total + N per-point
        agent = _make_agent()
        _compile_agent(agent)
        paths = agent.list_paths()
        expected_min = 5 + 4 + 3 + 1 + _SMALL_N
        assert len(paths) >= expected_min

    def test_repr_before_build(self):
        agent = _make_agent()
        r = repr(agent)
        assert 'not built' in r
        assert 'SingleSatCoverage' in r

    def test_repr_after_build(self):
        agent = _make_agent()
        _compile_agent(agent)
        r = repr(agent)
        assert 'built' in r


# ---------------------------------------------------------------------------
# TestSingleSatCoverageEndToEnd
# ---------------------------------------------------------------------------

class TestSingleSatCoverageEndToEnd:

    def _run_optimization(self, n_pts=12, initial_inc_deg=51.6, extra_config=None):
        """
        Build and solve the coverage maximization problem.

        Uses a LEO-bounded search space (500-1600 km altitude) for fast,
        stable convergence in unit tests.  The full apogee_max=40000 km range
        is exercised in the example script.

        Returns (result, agent, compiler).
        """
        # Bound to LEO for test stability: p in [R_earth+500, R_earth+1600] km
        p0 = R_EARTH + 500.0
        p_lb = R_EARTH + 200.0    # 200 km altitude
        p_ub = R_EARTH + 1600.0   # 1600 km altitude
        h0 = math.tan(math.radians(initial_inc_deg) / 2.0)

        config = {
            'n_sample_points': n_pts,
            'ground_target': {'lat_deg': 38.9, 'lon_deg': -77.0},
            'coverage': {'min_elevation_deg': 10.0, 'sigmoid_k': 20.0},
            'altitude_bounds': {'perigee_min_km': 200.0, 'apogee_max_km': 1600.0},
            'constants': {'mu': MU, 'R_earth': R_EARTH},
        }
        if extra_config:
            config.update(extra_config)

        agent = SingleSatCoverage('sat', config)

        compiler = CompilerStub(solver_opts={
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
        })
        compiler.add_agent(agent)

        overrides = {
            'sat/orbital/p': {'value': p0, 'lb': p_lb, 'ub': p_ub},
            # f=0.01: small non-zero eccentricity to avoid degenerate Jacobian
            # of sqrt(f^2+g^2) at the circular orbit initial point.
            'sat/orbital/f': {'value': 0.01, 'lb': -0.3, 'ub': 0.3},
            'sat/orbital/g': {'value': 0.0,  'lb': -0.3, 'ub': 0.3},
            'sat/orbital/h': {'value': h0,   'lb': -1.5, 'ub': 1.5},
            'sat/orbital/k': {'value': 0.0,  'lb': -1.5, 'ub': 1.5},
        }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            compiler.compile(overrides=overrides)

        # Maximize coverage/total
        cov_sd = compiler.resolve('sat', 'coverage/total')
        compiler.add_cost(-cov_sd.symbol, name='neg_coverage')

        compiler.build_solver()

        result = compiler.solve()
        return result, agent, compiler

    def test_optimizer_finds_valid_solution(self):
        """Full pipeline runs without error and solver succeeds."""
        result, _, _ = self._run_optimization(n_pts=12)
        assert result.success

    def test_solution_satisfies_perigee_constraint(self):
        """Optimal orbit has perigee altitude >= 200 km."""
        result, _, _ = self._run_optimization(n_pts=12)
        p_opt = float(result['sat/orbital/p'][0])
        f_opt = float(result['sat/orbital/f'][0])
        g_opt = float(result['sat/orbital/g'][0])
        e_opt = math.sqrt(f_opt**2 + g_opt**2)
        r_p   = p_opt / (1 + e_opt)
        alt_p = r_p - R_EARTH
        assert alt_p >= 200.0 - 1.0  # 1 km tolerance for solver precision

    def test_solution_satisfies_apogee_constraint(self):
        """Optimal orbit has apogee altitude <= 1600 km (test LEO bounds)."""
        result, _, _ = self._run_optimization(n_pts=12)
        p_opt = float(result['sat/orbital/p'][0])
        f_opt = float(result['sat/orbital/f'][0])
        g_opt = float(result['sat/orbital/g'][0])
        e_opt = math.sqrt(f_opt**2 + g_opt**2)
        if e_opt < 1.0:
            r_a   = p_opt / (1 - e_opt)
            alt_a = r_a - R_EARTH
            assert alt_a <= 1600.0 + 1.0  # 1 km tolerance

    def test_coverage_total_in_unit_interval(self):
        """Coverage fraction is in [0, 1]."""
        result, agent, _ = self._run_optimization(n_pts=12)
        p_opt = float(result['sat/orbital/p'][0])
        f_opt = float(result['sat/orbital/f'][0])
        g_opt = float(result['sat/orbital/g'][0])
        h_opt = float(result['sat/orbital/h'][0])
        k_opt = float(result['sat/orbital/k'][0])

        # Build a standalone ca.Function to evaluate coverage/total
        p_sym = ca.MX.sym('p', 1, 1)
        f_sym = ca.MX.sym('f', 1, 1)
        g_sym = ca.MX.sym('g', 1, 1)
        h_sym = ca.MX.sym('h', 1, 1)
        k_sym = ca.MX.sym('k', 1, 1)

        cov_expr = agent.resolve('coverage/total').symbol
        # Create function: but we need the original MX symbols.
        # Instead, just check that the optimal cost is a valid fraction.
        # The optimizer minimizes -coverage, so f_opt should be in [-1, 0].
        assert result.f_opt >= -1.0 - 1e-6
        assert result.f_opt <= 0.0 + 1e-6

    def test_default_config_compiles_and_solves(self):
        """Default config with LEO bounds compiles and solves without error."""
        # Use LEO-bounded config so optimizer converges in reasonable time
        config = {
            'n_sample_points': 12,
            'altitude_bounds': {'perigee_min_km': 200.0, 'apogee_max_km': 1600.0},
        }
        agent = SingleSatCoverage('sat', config)

        compiler = CompilerStub(solver_opts={'ipopt.print_level': 0,
                                              'ipopt.sb': 'yes'})
        compiler.add_agent(agent)

        overrides = {
            'sat/orbital/p': {'value': R_EARTH + 500.0,
                               'lb': R_EARTH + 200.0, 'ub': R_EARTH + 1600.0},
            'sat/orbital/f': {'value': 0.01, 'lb': -0.3, 'ub': 0.3},
            'sat/orbital/g': {'value': 0.0,  'lb': -0.3, 'ub': 0.3},
            'sat/orbital/h': {'value': 0.484, 'lb': -1.5, 'ub': 1.5},
            'sat/orbital/k': {'value': 0.0,   'lb': -1.5, 'ub': 1.5},
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            compiler.compile(overrides=overrides)

        cov_sd = compiler.resolve('sat', 'coverage/total')
        compiler.add_cost(-cov_sd.symbol, name='neg_coverage')
        compiler.build_solver()
        result = compiler.solve()
        assert result.success

    def test_compiler_stub_repr(self):
        """CompilerStub repr reports 1 agent after compile."""
        agent = _make_agent()
        compiler = CompilerStub()
        compiler.add_agent(agent)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            compiler.compile()
        r = repr(compiler)
        assert '1 agent' in r
        assert 'compiled' in r
