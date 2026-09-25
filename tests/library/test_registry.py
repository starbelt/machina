"""
Tests for the factory registry in ``machina.library.registry``.

Sections
--------
TestRegistry                             -- register, get, list_registered, list_by_domain
TestRegistryCounts                       -- domain contents once the astro pack is imported
TestEachNameRecordsItsRegisteringModule  -- registered_in names the library or pack module
TestEndToEnd                             -- registry → factory → descriptor → IPOPT

Snapshot/restore isolation lives in ``test_registry_isolation.py`` beside this
file; the "exactly these names after ``import machina``" checks need a fresh
interpreter and live in ``tests/test_import_graph.py``.
"""

import numpy as np
import pytest

import machina.astro  # noqa: F401  (packs register their factories on import)
import machina.swapc  # noqa: F401
from machina.library import registry
from machina.model import FunctionDescriptor
from machina.solver.backend import SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-5


def make_solver(**opts):
    base = {'ipopt.print_level': 0, 'print_time': False}
    base.update(opts)
    return SolverBackend(solver_opts=base)


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistry:

    def test_registered_factories_are_callable(self):
        factory = registry.get('cost.quadratic')
        assert callable(factory)

    def test_get_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found"):
            registry.get('cost.does_not_exist')

    def test_get_unknown_name_lists_available(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get('cost.does_not_exist')
        assert 'cost.quadratic' in str(exc_info.value)

    def test_list_registered_returns_sorted_list(self):
        names = registry.list_registered()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_list_registered_contains_known_factories(self):
        names = registry.list_registered()
        assert 'cost.quadratic' in names
        assert 'cost.rosenbrock' in names
        assert 'constraint.linear' in names

    def test_list_by_domain_cost(self):
        cost_names = registry.list_by_domain('cost')
        assert all(n.startswith('cost.') for n in cost_names)
        assert 'cost.quadratic' in cost_names

    def test_list_by_domain_constraint(self):
        names = registry.list_by_domain('constraint')
        assert all(n.startswith('constraint.') for n in names)
        assert 'constraint.linear' in names

    def test_list_by_domain_empty_for_unknown(self):
        assert registry.list_by_domain('nonexistent_domain') == []

    def test_duplicate_registration_raises(self):
        """Re-registering an existing name must raise ValueError."""
        from machina.library.registry import register as reg_register
        with pytest.raises(ValueError, match="already registered"):
            @reg_register('cost.quadratic')
            def _duplicate():
                pass

    def test_factory_returns_function_descriptor(self):
        fd = registry.get('cost.quadratic')(n=2)
        assert isinstance(fd, FunctionDescriptor)


# ===========================================================================
# Registry counts
# ===========================================================================

class TestRegistryCounts:

    def test_geometry_domain_has_two_functions(self):
        """geometry domain has ground_target_eci and elevation_angle."""
        geo_funcs = registry.list_by_domain('geometry')
        assert set(geo_funcs) == {
            'geometry.ground_target_eci',
            'geometry.elevation_angle',
        }

    def test_cost_domain_includes_smooth_coverage(self):
        """cost domain includes smooth_coverage alongside existing functions."""
        cost_funcs = registry.list_by_domain('cost')
        assert 'cost.smooth_coverage' in cost_funcs

    def test_total_registered_count(self):
        """Registry has at least 10 entries (7 transforms + 2 geometry + smooth_coverage + others)."""
        all_funcs = registry.list_registered()
        assert len(all_funcs) >= 10


# ===========================================================================
# Provenance
# ===========================================================================

class TestEachNameRecordsItsRegisteringModule:
    """The generic factories register from ``machina.library``; domain ones from their pack."""

    @pytest.mark.parametrize('name, module', [
        ('cost.quadratic', 'machina.library.cost'),
        ('transform.mee_to_eci', 'machina.astro.transforms'),
        ('cost.sigmoid_goodput', 'machina.swapc.goodput'),
        ('util.ttp_computation', 'machina.swapc.latency'),
    ])
    def test_registered_in_names_the_module_that_holds_the_factory(self, name, module):
        assert registry.registered_in(name) == module


# ===========================================================================
# End-to-end pipeline tests
# ===========================================================================

class TestEndToEnd:
    """
    Prove that the full pipeline — registry lookup → factory → FunctionDescriptor
    → MX call → solver.add_cost() → IPOPT — works correctly with AD intact.
    """

    def test_quadratic_pipeline_converges(self):
        """
        minimize  x^T x   via cost.quadratic
        Solution: x* = 0,  f* = 0
        """
        b = make_solver()
        x = b.add_variable('x', 2, initial_guess=3.0)

        quadratic = registry.get('cost.quadratic')(n=2)
        cost = quadratic(x=x)
        b.add_cost(cost)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res['x'], [0.0, 0.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 0.0, atol=ATOL)

    def test_rosenbrock_pipeline_converges(self):
        """
        minimize  (1 - xy[0])^2 + 100*(xy[1] - xy[0]^2)^2   via cost.rosenbrock
        Solution: xy* = [1, 1],  f* = 0

        This is the primary AD correctness test: if automatic differentiation
        does not flow correctly through the FunctionDescriptor wrapper, IPOPT
        will fail to converge or converge to the wrong point.
        """
        b = make_solver()
        xy = b.add_variable('xy', 2, initial_guess=0.0)

        rosenbrock = registry.get('cost.rosenbrock')(a=1.0, b=100.0)
        cost = rosenbrock(xy=xy)
        b.add_cost(cost)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res['xy'], [1.0, 1.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 0.0, atol=ATOL)

    def test_rosenbrock_default_params(self):
        """Factory default parameters (a=1, b=100) should work without explicit kwargs."""
        b = make_solver()
        xy = b.add_variable('xy', 2, initial_guess=0.0)
        rosenbrock = registry.get('cost.rosenbrock')()
        cost = rosenbrock(xy=xy)
        b.add_cost(cost)
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res['xy'], [1.0, 1.0], atol=ATOL)

    def test_linear_constraint_pipeline(self):
        """
        minimize  x^T x
        s.t.      x[0] + x[1] >= 3   (c = [1, 1], lb=3)
        Solution: x* = [1.5, 1.5],  f* = 4.5
        """
        b = make_solver()
        x = b.add_variable('x', 2, initial_guess=0.0)

        quadratic = registry.get('cost.quadratic')(n=2)
        linear    = registry.get('constraint.linear')(n=2, coefficients=[1.0, 1.0])

        b.add_cost(quadratic(x=x))
        b.add_constraint(linear(x=x), lb=3.0)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res['x'], [1.5, 1.5], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 4.5, atol=ATOL)

    def test_descriptor_reuse_same_function_object(self):
        """
        Calling a factory once and using the descriptor at two call sites
        must reuse the same underlying ca.Function object.
        """
        quadratic = registry.get('cost.quadratic')(n=1)
        assert quadratic.function is quadratic.function  # identity, not copy
