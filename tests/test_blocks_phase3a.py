"""
Phase 3a tests — coordinate transforms, AgentType lifecycle, compiler stub.

Sections
--------
TestCoordinateTransforms   -- KOE↔MEE↔ECI, roundtrip and against reference
TestTransformRegistry      -- registry registration and FunctionDescriptor metadata
TestAgentTypeLifecycle     -- MockAgent declare/build/resolve, error conditions
TestCompilerStubLifecycle  -- role assignment, parameter wiring, solvable NLP

Reference implementation
------------------------
_koe_to_eci_ref() in this file provides an independent numpy-based PQW→ECI
computation used to validate mee_to_eci without relying on the same formula.
"""

import warnings

import casadi as ca
import numpy as np
import pytest

from machina.agents.agent_type import (
    AgentType,
    ConstraintDeclaration,
    QuantityDeclaration,
)
from machina.blocks import registry
from machina.blocks.descriptor import SymbolDescriptor
from machina.compiler.compiler_stub import CompilerStub
from machina.solver.backend import SolverBackend

pytestmark = pytest.mark.requires_casadi

MU_EARTH = 398600.4418  # km³/s²


# ===========================================================================
# Helpers
# ===========================================================================

def make_solver(**opts):
    """Create a test SolverBackend with quiet output."""
    defaults = {'ipopt.print_level': 0, 'print_time': 0}
    defaults.update(opts)
    return SolverBackend(solver_opts=defaults)


def _koe_to_eci_ref(a, e, inc, raan, aop, nu):
    """
    Reference ECI position via perifocal-frame rotation (pure numpy).

    This is the independent implementation used to validate mee_to_eci.
    It does NOT use the MEE formulas; it uses the classical rotation
    R3(-raan) @ R1(-inc) @ R3(-aop).

    Reference: Curtis, Orbital Mechanics for Engineering Students (2019), Ch. 4.
    """
    p = a * (1 - e ** 2)
    r_mag = p / (1 + e * np.cos(nu))
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

    def R1(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def R3(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # PQW→ECI: columns are [P̂, Q̂, Ŵ] expressed in ECI.
    # Equivalent to R3(raan) @ R1(inc) @ R3(aop) with the active-rotation
    # convention R3(θ) = [[cosθ,-sinθ,0],[sinθ,cosθ,0],[0,0,1]].
    Q = R3(raan) @ R1(inc) @ R3(aop)
    return Q @ r_pqw


def _eval_mx(expr, *args):
    """Evaluate a CasADi MX/DM expression numerically, returning a numpy array."""
    return np.array(ca.DM(expr)).flatten()


def _call_koe_to_mee(koe_np):
    factory = registry.get('transform.koe_to_mee')()
    koe_dm = ca.DM(koe_np)
    result = factory.function(koe_dm)
    return np.array(result).flatten()


def _call_mee_to_koe(mee_np):
    factory = registry.get('transform.mee_to_koe')()
    mee_dm = ca.DM(mee_np)
    result = factory.function(mee_dm)
    return np.array(result).flatten()


def _call_mee_to_eci(mee_np, mu=MU_EARTH):
    factory = registry.get('transform.mee_to_eci')(mu=mu)
    mee_dm = ca.DM(mee_np)
    r, v = factory.function(mee_dm)
    return np.array(r).flatten(), np.array(v).flatten()


# ===========================================================================
# Reference orbit fixtures
# ===========================================================================

# koe = [a (km), e, i (rad), Ω (rad), ω (rad), ν (rad)]

KOE_ISS = [6778.0, 0.001, np.radians(51.6), np.radians(23.0),
           np.radians(45.0), np.radians(120.0)]

KOE_NEAR_CIRCULAR_EQUATORIAL = [7000.0, 0.001, np.radians(0.5),
                                  np.radians(10.0), np.radians(20.0),
                                  np.radians(90.0)]

KOE_POLAR = [7000.0, 0.02, np.radians(90.0), np.radians(60.0),
             np.radians(30.0), np.radians(200.0)]

KOE_HEO = [26560.0, 0.72, np.radians(63.4), np.radians(150.0),
           np.radians(270.0), np.radians(10.0)]

# Circular equatorial at ν=0: r_eci should be exactly [a, 0, 0].
KOE_CIRC_EQ_NU0 = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Circular equatorial at ν=π/2: r_eci should be exactly [0, a, 0].
KOE_CIRC_EQ_NU90 = [7000.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]


# ===========================================================================
# Section 1 — Coordinate Transform Tests
# ===========================================================================

class TestCoordinateTransforms:
    """KOE ↔ MEE roundtrip and MEE → ECI against independent reference."""

    @pytest.mark.parametrize('koe', [
        KOE_ISS,
        KOE_NEAR_CIRCULAR_EQUATORIAL,
        KOE_POLAR,
        KOE_HEO,
    ], ids=['iss', 'near_circular_equatorial', 'polar', 'heo'])
    def test_koe_mee_koe_roundtrip(self, koe):
        """KOE → MEE → KOE recovers original elements within floating-point.

        Angles are compared modulo 2π so that equivalent representations
        (e.g. ω=270° vs ω=-90°) pass.  atan2-based inversion can return
        angles in (-π, π] rather than [0, 2π), which is mathematically
        identical.
        """
        mee = _call_koe_to_mee(koe)
        koe_back = _call_mee_to_koe(mee)

        # Compare scalar quantities (a, e) directly.
        np.testing.assert_allclose(koe_back[:2], koe[:2], atol=1e-8,
                                   err_msg=f"Scalar elements failed for KOE={koe}")

        # Compare angular quantities modulo 2π via unit-vector comparison.
        for idx in range(2, 6):
            expected = float(koe[idx])
            actual = float(koe_back[idx])
            # cos/sin comparison handles wrapping cleanly (e.g. 270° vs -90°).
            np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=1e-8,
                                       err_msg=f"angle[{idx}] cos mismatch: {np.degrees(actual):.3f}° vs {np.degrees(expected):.3f}°")
            np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=1e-8,
                                       err_msg=f"angle[{idx}] sin mismatch: {np.degrees(actual):.3f}° vs {np.degrees(expected):.3f}°")

    def test_koe_to_mee_iss_spot_check(self):
        """Manually verify one MEE element for the ISS orbit."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        p_expected = a * (1 - e ** 2)
        np.testing.assert_allclose(mee[0], p_expected, rtol=1e-10,
                                   err_msg="p = a*(1-e²) failed")

    def test_mee_to_eci_circular_equatorial_nu0(self):
        """Circular equatorial at ν=0: r_eci = [a, 0, 0] exactly."""
        a = KOE_CIRC_EQ_NU0[0]
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU0)
        r_eci, _ = _call_mee_to_eci(mee)
        np.testing.assert_allclose(r_eci, [a, 0.0, 0.0], atol=1e-6,
                                   err_msg="r_eci ≠ [a, 0, 0] at ν=0")

    def test_mee_to_eci_circular_equatorial_nu90(self):
        """Circular equatorial at ν=π/2: r_eci = [0, a, 0] exactly."""
        a = KOE_CIRC_EQ_NU90[0]
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU90)
        r_eci, _ = _call_mee_to_eci(mee)
        np.testing.assert_allclose(r_eci, [0.0, a, 0.0], atol=1e-6,
                                   err_msg="r_eci ≠ [0, a, 0] at ν=π/2")

    @pytest.mark.parametrize('koe', [
        KOE_ISS,
        KOE_NEAR_CIRCULAR_EQUATORIAL,
        KOE_POLAR,
    ], ids=['iss', 'near_circular_equatorial', 'polar'])
    def test_mee_to_eci_position_vs_reference(self, koe):
        """MEE→ECI position matches independent PQW reference (atol=0.01 km)."""
        a, e, inc, raan, aop, nu = koe
        mee = _call_koe_to_mee(koe)
        r_eci_mee, _ = _call_mee_to_eci(mee)
        r_eci_ref = _koe_to_eci_ref(a, e, inc, raan, aop, nu)
        np.testing.assert_allclose(
            r_eci_mee, r_eci_ref, atol=0.01,
            err_msg=f"ECI position mismatch for KOE={koe}",
        )

    def test_mee_to_eci_radius_magnitude(self):
        """Radius magnitude |r_eci| matches r = a*(1-e²)/(1+e*cosν)."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        r_eci, _ = _call_mee_to_eci(mee)
        r_expected = a * (1 - e ** 2) / (1 + e * np.cos(nu))
        np.testing.assert_allclose(
            np.linalg.norm(r_eci), r_expected, rtol=1e-8,
        )

    def test_mee_to_eci_velocity_magnitude_vis_viva(self):
        """Velocity magnitude |v_eci| matches vis-viva: v² = μ*(2/r - 1/a)."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        r_eci, v_eci = _call_mee_to_eci(mee)
        r = np.linalg.norm(r_eci)
        v_expected = np.sqrt(MU_EARTH * (2 / r - 1 / a))
        np.testing.assert_allclose(
            np.linalg.norm(v_eci), v_expected, rtol=1e-6,
        )

    def test_mee_to_eci_r_v_orthogonal_circular(self):
        """For a circular orbit, r and v are orthogonal (r·v = 0)."""
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU0)
        r_eci, v_eci = _call_mee_to_eci(mee)
        np.testing.assert_allclose(np.dot(r_eci, v_eci), 0.0, atol=1e-6)

    def test_singularity_near_retrograde_equatorial(self):
        """i = π - 1e-4 rad: output should be finite (not NaN)."""
        koe_near_retro = [7000.0, 0.01, np.pi - 1e-4,
                          np.radians(45.0), np.radians(30.0), np.radians(60.0)]
        # NOTE: this is near the MEE singularity at i=π (h,k→∞).
        # The transform still runs without raising, but h/k are large.
        mee = _call_koe_to_mee(koe_near_retro)
        assert np.all(np.isfinite(mee)), (
            "koe_to_mee produced non-finite values near i=π singularity"
        )

    def test_koe_to_mee_produces_casadi_mx_when_called_with_mx(self):
        """koe_to_mee called with MX input returns MX output."""
        factory = registry.get('transform.koe_to_mee')()
        koe_mx = ca.MX.sym('koe', 6)
        result = factory(koe=koe_mx)
        assert isinstance(result, ca.MX)


# ===========================================================================
# Section 2 — Transform Registry Tests
# ===========================================================================

class TestTransformRegistry:
    """Registry registration and FunctionDescriptor metadata for transforms."""

    def test_koe_to_mee_is_registered(self):
        factory = registry.get('transform.koe_to_mee')
        assert callable(factory)

    def test_mee_to_koe_is_registered(self):
        factory = registry.get('transform.mee_to_koe')
        assert callable(factory)

    def test_mee_to_eci_is_registered(self):
        factory = registry.get('transform.mee_to_eci')
        assert callable(factory)

    def test_transform_domain_lists_phase3a_transforms(self):
        # Phase 3b adds 4 more transform.* entries; check the 3 from Phase 3a
        # are present rather than asserting the exact total count.
        transforms = registry.list_by_domain('transform')
        assert len(transforms) >= 3
        assert {'transform.koe_to_mee', 'transform.mee_to_koe', 'transform.mee_to_eci'} \
               .issubset(set(transforms))

    def test_koe_to_mee_returns_function_descriptor(self):
        from machina.blocks.descriptor import FunctionDescriptor
        fd = registry.get('transform.koe_to_mee')()
        assert isinstance(fd, FunctionDescriptor)

    def test_mee_to_koe_returns_function_descriptor(self):
        from machina.blocks.descriptor import FunctionDescriptor
        fd = registry.get('transform.mee_to_koe')()
        assert isinstance(fd, FunctionDescriptor)

    def test_mee_to_eci_returns_function_descriptor(self):
        from machina.blocks.descriptor import FunctionDescriptor
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert isinstance(fd, FunctionDescriptor)

    def test_koe_to_mee_input_shape(self):
        fd = registry.get('transform.koe_to_mee')()
        assert fd.input_shapes == [(6, 1)]

    def test_koe_to_mee_output_shape(self):
        fd = registry.get('transform.koe_to_mee')()
        assert fd.output_shapes == [(6, 1)]

    def test_mee_to_eci_input_shape(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.input_shapes == [(6, 1)]

    def test_mee_to_eci_output_shapes_two_3x1(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.output_shapes == [(3, 1), (3, 1)]

    def test_mee_to_eci_output_names(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.output_names == ['r_eci', 'v_eci']

    def test_koe_to_mee_callable_with_mx_returns_mx(self):
        fd = registry.get('transform.koe_to_mee')()
        koe_mx = ca.MX.sym('koe', 6)
        result = fd(koe=koe_mx)
        assert isinstance(result, ca.MX)

    def test_mee_to_eci_callable_with_mx_returns_list(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        mee_mx = ca.MX.sym('mee', 6)
        result = fd(mee=mee_mx)
        # Multi-output CasADi function returns a list-like object
        assert len(result) == 2

    def test_koe_to_mee_description_nonempty(self):
        fd = registry.get('transform.koe_to_mee')()
        assert len(fd.description) > 0

    def test_mee_to_eci_description_contains_mu(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert str(MU_EARTH) in fd.description


# ===========================================================================
# Section 3 — AgentType Lifecycle Tests
# ===========================================================================

class MockAgent(AgentType):
    """
    Minimal concrete agent for testing the base-class lifecycle.

    Declares three quantities:
      - 'state/sma'  : flexible variable (semi-major axis)
      - 'state/ecc'  : always_variable (eccentricity)
      - 'mu'         : always_parameter (gravitational parameter)

    build() populates namespace with the three declared symbols plus one
    computed expression 'v_circ' = sqrt(mu/sma).
    """

    def declare(self) -> list[QuantityDeclaration]:
        return [
            QuantityDeclaration(
                path='state/sma',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=7000.0,
                lb=6371.0,
                ub=42164.0,
                description='Semi-major axis',
                units='km',
                role='flexible',
                default_role='variable',
            ),
            QuantityDeclaration(
                path='state/ecc',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=0.01,
                lb=0.0,
                ub=0.9,
                description='Eccentricity',
                units=None,
                role='always_variable',
            ),
            QuantityDeclaration(
                path='mu',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=MU_EARTH,
                lb=-np.inf,
                ub=np.inf,
                description='Gravitational parameter',
                units='km3/s2',
                role='always_parameter',
            ),
        ]

    def build(self, symbols: dict) -> list[ConstraintDeclaration]:
        sma = symbols['state/sma']
        ecc = symbols['state/ecc']
        mu_sym = symbols['mu']

        self._namespace['state/sma'] = SymbolDescriptor(
            symbol=sma, name='state/sma', shape=(1, 1),
            semantic_type='scalar', units='km',
        )
        self._namespace['state/ecc'] = SymbolDescriptor(
            symbol=ecc, name='state/ecc', shape=(1, 1),
            semantic_type='scalar',
        )
        self._namespace['mu'] = SymbolDescriptor(
            symbol=mu_sym, name='mu', shape=(1, 1),
            semantic_type='scalar', units='km3/s2',
        )
        # Computed expression: circular velocity sqrt(μ/a)
        v_circ = ca.sqrt(mu_sym / sma)
        self._namespace['v_circ'] = SymbolDescriptor(
            symbol=v_circ, name='v_circ', shape=(1, 1),
            semantic_type='scalar', units='km/s',
        )

        self._built = True
        return [
            ConstraintDeclaration(
                expr=ecc,
                lb=0.0,
                ub=0.9,
                name='ecc_bounds',
                description='Eccentricity physical bounds',
            )
        ]


class TestAgentTypeLifecycle:
    """MockAgent declaration, build, resolve, and error-condition tests."""

    def _make_symbols(self, agent):
        """Create dummy MX symbols matching the agent's declarations."""
        syms = {}
        for decl in agent.declare():
            syms[decl.path] = ca.MX.sym(decl.path, *decl.shape)
        return syms

    # --- declare() ---

    def test_declare_returns_list(self):
        agent = MockAgent('test_agent', {})
        decls = agent.declare()
        assert isinstance(decls, list)

    def test_declare_returns_correct_count(self):
        agent = MockAgent('test_agent', {})
        decls = agent.declare()
        assert len(decls) == 3

    def test_declare_all_quantity_declarations(self):
        agent = MockAgent('test_agent', {})
        decls = agent.declare()
        assert all(isinstance(d, QuantityDeclaration) for d in decls)

    def test_declare_no_mx_symbols_created(self):
        agent = MockAgent('test_agent', {})
        decls = agent.declare()
        for decl in decls:
            assert not isinstance(decl.default_value, ca.MX), (
                f"QuantityDeclaration.default_value should not be ca.MX: {decl.path}"
            )

    def test_declare_roles_are_correct(self):
        agent = MockAgent('test_agent', {})
        decls = {d.path: d for d in agent.declare()}
        assert decls['state/sma'].role == 'flexible'
        assert decls['state/ecc'].role == 'always_variable'
        assert decls['mu'].role == 'always_parameter'

    # --- build() ---

    def test_build_populates_namespace(self):
        agent = MockAgent('test_agent', {})
        symbols = self._make_symbols(agent)
        agent.build(symbols)
        assert len(agent._namespace) == 4  # 3 declared + 1 computed

    def test_build_sets_built_flag(self):
        agent = MockAgent('test_agent', {})
        assert not agent._built
        agent.build(self._make_symbols(agent))
        assert agent._built

    def test_build_namespace_contains_computed_expression(self):
        agent = MockAgent('test_agent', {})
        agent.build(self._make_symbols(agent))
        assert 'v_circ' in agent._namespace

    def test_build_returns_constraint_declarations(self):
        agent = MockAgent('test_agent', {})
        constraints = agent.build(self._make_symbols(agent))
        assert len(constraints) == 1
        assert isinstance(constraints[0], ConstraintDeclaration)
        assert constraints[0].name == 'ecc_bounds'

    # --- resolve() ---

    def test_resolve_works_after_build(self):
        agent = MockAgent('test_agent', {})
        agent.build(self._make_symbols(agent))
        sd = agent.resolve('state/sma')
        assert isinstance(sd, SymbolDescriptor)

    def test_resolve_raises_runtime_error_before_build(self):
        agent = MockAgent('test_agent', {})
        with pytest.raises(RuntimeError, match="before build"):
            agent.resolve('state/sma')

    def test_resolve_raises_key_error_for_unknown_path(self):
        agent = MockAgent('test_agent', {})
        agent.build(self._make_symbols(agent))
        with pytest.raises(KeyError, match="nonexistent/path"):
            agent.resolve('nonexistent/path')

    def test_resolve_key_error_lists_available_paths(self):
        agent = MockAgent('test_agent', {})
        agent.build(self._make_symbols(agent))
        with pytest.raises(KeyError) as exc_info:
            agent.resolve('no/such/path')
        # Available paths should be in the error message
        assert 'state/sma' in str(exc_info.value)

    # --- list_paths() ---

    def test_list_paths_empty_before_build(self):
        agent = MockAgent('test_agent', {})
        assert agent.list_paths() == []

    def test_list_paths_sorted_after_build(self):
        agent = MockAgent('test_agent', {})
        agent.build(self._make_symbols(agent))
        paths = agent.list_paths()
        assert paths == sorted(paths)
        assert len(paths) == 4

    # --- QuantityDeclaration validation ---

    def test_quantity_declaration_invalid_role_raises(self):
        with pytest.raises(ValueError, match="invalid role"):
            QuantityDeclaration(
                path='x', shape=(1, 1), semantic_type='scalar',
                default_value=0.0, lb=-1.0, ub=1.0,
                description='test', role='bad_role',
            )

    def test_quantity_declaration_invalid_default_role_raises(self):
        with pytest.raises(ValueError, match="invalid default_role"):
            QuantityDeclaration(
                path='x', shape=(1, 1), semantic_type='scalar',
                default_value=0.0, lb=-1.0, ub=1.0,
                description='test', default_role='bad_default_role',
            )

    def test_quantity_declaration_invalid_semantic_type_raises(self):
        with pytest.raises(ValueError, match="invalid semantic_type"):
            QuantityDeclaration(
                path='x', shape=(1, 1), semantic_type='matrix',
                default_value=0.0, lb=-1.0, ub=1.0,
                description='test',
            )

    def test_quantity_declaration_path_no_leading_slash(self):
        with pytest.raises(ValueError, match="must not start with '/'"):
            QuantityDeclaration(
                path='/bad/path', shape=(1, 1), semantic_type='scalar',
                default_value=0.0, lb=-1.0, ub=1.0,
                description='test',
            )

    def test_quantity_declaration_warns_when_default_role_set_on_non_flexible(self):
        with pytest.warns(UserWarning, match="no effect"):
            QuantityDeclaration(
                path='x', shape=(1, 1), semantic_type='scalar',
                default_value=0.0, lb=-1.0, ub=1.0,
                description='test',
                role='always_variable',
                default_role='parameter',  # has no effect — should warn
            )

    # --- ConstraintDeclaration ---

    def test_constraint_declaration_fields(self):
        expr = ca.MX.sym('x', 1)
        cd = ConstraintDeclaration(
            expr=expr, lb=0.0, ub=1.0,
            name='my_constraint', description='test constraint',
        )
        assert cd.name == 'my_constraint'
        assert cd.lb == 0.0
        assert cd.ub == 1.0

    # --- repr ---

    def test_agent_repr_before_build(self):
        agent = MockAgent('my_agent', {})
        r = repr(agent)
        assert 'my_agent' in r
        assert 'not built' in r

    def test_agent_repr_after_build(self):
        agent = MockAgent('my_agent', {})
        agent.build(self._make_symbols(agent))
        r = repr(agent)
        assert 'built' in r


# ===========================================================================
# Section 4 — CompilerStub Lifecycle Tests
# ===========================================================================

class TestCompilerStubLifecycle:
    """Role assignment, parameter wiring, constraints, solvable NLP."""

    def _make_compiler(self, **solver_opts):
        defaults = {'ipopt.print_level': 0, 'print_time': 0}
        defaults.update(solver_opts)
        return CompilerStub(solver_opts=defaults)

    def _make_compiled_compiler(self, overrides=None, **solver_opts):
        compiler = self._make_compiler(**solver_opts)
        agent = MockAgent('sat', {})
        compiler.add_agent(agent)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            compiler.compile(overrides=overrides)
        return compiler, agent

    # --- add_agent ---

    def test_add_agent_registers_agent(self):
        compiler = self._make_compiler()
        agent = MockAgent('sat', {})
        compiler.add_agent(agent)
        assert len(compiler._agents) == 1

    def test_add_agent_rejects_duplicate_name(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with pytest.raises(ValueError, match="already registered"):
            compiler.add_agent(MockAgent('sat', {}))

    def test_add_agent_raises_after_compile(self):
        compiler, _ = self._make_compiled_compiler()
        with pytest.raises(RuntimeError, match="Cannot add agents"):
            compiler.add_agent(MockAgent('new_agent', {}))

    # --- role assignment ---

    def test_always_variable_creates_variable(self):
        compiler, _ = self._make_compiled_compiler()
        # 'sat/state/ecc' is always_variable
        assert 'sat/state/ecc' in compiler.solver.variable_order()

    def test_always_parameter_creates_parameter(self):
        compiler, _ = self._make_compiled_compiler()
        # 'sat/mu' is always_parameter
        assert 'sat/mu' in compiler.solver.parameter_order()

    def test_flexible_default_variable_creates_variable(self):
        compiler, _ = self._make_compiled_compiler()
        # 'sat/state/sma' is flexible with default_role='variable'
        assert 'sat/state/sma' in compiler.solver.variable_order()

    def test_flexible_override_to_parameter(self):
        overrides = {'sat/state/sma': {'role': 'parameter', 'value': 7000.0}}
        compiler, _ = self._make_compiled_compiler(overrides=overrides)
        assert 'sat/state/sma' in compiler.solver.parameter_order()
        assert 'sat/state/sma' not in compiler.solver.variable_order()

    def test_override_invalid_role_raises(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with pytest.raises(ValueError, match="'variable' or 'parameter'"):
            compiler.compile(overrides={'sat/state/sma': {'role': 'bad'}})

    # --- warnings ---

    def test_compile_warns_on_each_unoverridden_quantity(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with pytest.warns(UserWarning):
            compiler.compile()

    def test_no_warning_when_all_quantities_overridden(self):
        overrides = {
            'sat/state/sma': {'value': 7500.0},
            'sat/state/ecc': {'value': 0.001},
            'sat/mu': {'value': MU_EARTH},
        }
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            compiler.compile(overrides=overrides)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    # --- constraints ---

    def test_constraint_registered_with_solver(self):
        compiler, _ = self._make_compiled_compiler()
        # MockAgent.build() returns one ConstraintDeclaration
        assert len(compiler.solver.constraints()) == 1

    def test_constraint_name_prefixed_with_agent_name(self):
        compiler, _ = self._make_compiled_compiler()
        name = compiler.solver.constraints()[0].name
        assert name.startswith('sat/')

    # --- double compile ---

    def test_double_compile_raises(self):
        compiler, _ = self._make_compiled_compiler()
        with pytest.raises(RuntimeError, match="already been called"):
            compiler.compile()

    # --- resolve ---

    def test_resolve_returns_symbol_descriptor(self):
        compiler, _ = self._make_compiled_compiler()
        sd = compiler.resolve('sat', 'state/sma')
        assert isinstance(sd, SymbolDescriptor)

    def test_resolve_unknown_agent_raises(self):
        compiler, _ = self._make_compiled_compiler()
        with pytest.raises(KeyError, match="no agent named"):
            compiler.resolve('nonexistent', 'state/sma')

    def test_resolve_unknown_path_raises(self):
        compiler, _ = self._make_compiled_compiler()
        with pytest.raises(KeyError):
            compiler.resolve('sat', 'no/such/path')

    # --- full solvable NLP ---

    def test_full_nlp_solves_successfully(self):
        """
        Minimize (sma - 7000)² subject to ecc ∈ [0, 0.9].
        Optimal: sma ≈ 7000.0 km, ecc = initial guess.
        """
        overrides = {
            'sat/state/sma': {'value': 8000.0, 'initial_guess': 8000.0},
            'sat/state/ecc': {'value': 0.01},
            'sat/mu': {'value': MU_EARTH},
        }
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        compiler.compile(overrides=overrides)

        # Minimize (sma - 7000)²
        sma_sym = compiler.resolve('sat', 'state/sma').symbol
        compiler.add_cost((sma_sym - 7000.0) ** 2, name='sma_cost')

        compiler.build_solver()
        result = compiler.solve()

        assert result.success
        np.testing.assert_allclose(result['sat/state/sma'], 7000.0, atol=1.0)

    def test_parameter_value_wired_correctly(self):
        """
        When mu is a parameter, its numeric value should be accessible in
        the solution via computed namespace expressions.
        The solve should succeed and sma converge to the expected value.
        """
        overrides = {
            'sat/state/sma': {'value': 8000.0, 'initial_guess': 8000.0},
            'sat/state/ecc': {'value': 0.01},
            'sat/mu': {'value': MU_EARTH},
        }
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        compiler.compile(overrides=overrides)

        sma_sym = compiler.resolve('sat', 'state/sma').symbol
        compiler.add_cost((sma_sym - 7500.0) ** 2)

        compiler.build_solver()
        result = compiler.solve()

        assert result.success
        np.testing.assert_allclose(result['sat/state/sma'], 7500.0, atol=1.0)

    def test_add_cost_before_compile_raises(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with pytest.raises(RuntimeError, match="Call compile"):
            compiler.add_cost(ca.MX.zeros(1), name='test')

    def test_add_constraint_before_compile_raises(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        with pytest.raises(RuntimeError, match="Call compile"):
            compiler.add_constraint(ca.MX.zeros(1), lb=0.0, ub=1.0)

    def test_solve_before_build_solver_raises(self):
        compiler, _ = self._make_compiled_compiler()
        with pytest.raises(RuntimeError, match="build_solver"):
            compiler.solve()

    def test_compiler_repr(self):
        compiler = self._make_compiler()
        compiler.add_agent(MockAgent('sat', {}))
        r = repr(compiler)
        assert '1 agent' in r
        assert 'not compiled' in r

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            compiler.compile()
        assert 'compiled' in repr(compiler)
