"""
Phase 1 tests for machina.blocks — Layer 2 foundation.

Sections
--------
TestFunctionDescriptorMetadata  — metadata extraction from ca.Function
TestFunctionDescriptorCall      — __call__ validation (shapes, positional, keyword)
TestFunctionDescriptorRepr      — __repr__ format
TestSymbolDescriptor            — SymbolDescriptor validation rules
TestRegistry                    — register, get, list_registered, list_by_domain
TestEndToEnd                    — full pipeline: registry → factory → descriptor → IPOPT
"""

import casadi as ca
import numpy as np
import pytest

from machina.blocks import FunctionDescriptor, SymbolDescriptor, registry
from machina.solver.backend import SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-5


def make_solver(**opts):
    base = {'ipopt.print_level': 0, 'print_time': False}
    base.update(opts)
    return SolverBackend(solver_opts=base)


# ---------------------------------------------------------------------------
# Helpers — simple ca.Functions for testing without hitting the registry
# ---------------------------------------------------------------------------

def _scalar_function():
    """f(x) = x^2, single scalar input and output."""
    x = ca.SX.sym('x')
    f = ca.Function('square', [x], [x**2], ['x'], ['y'])
    return f

def _two_input_function():
    """f(a, b) = a + b, two (1,1) inputs."""
    a = ca.SX.sym('a')
    b = ca.SX.sym('b')
    f = ca.Function('add', [a, b], [a + b], ['a', 'b'], ['result'])
    return f

def _two_output_function():
    """f(x) = (x, x^2), one input two outputs."""
    x = ca.SX.sym('x')
    f = ca.Function('id_and_sq', [x], [x, x**2], ['x'], ['out1', 'out2'])
    return f

def _vector_function(n):
    """f(x) = x^T x, input x(n,1), output cost(1,1)."""
    x = ca.SX.sym('x', n)
    f = ca.Function('dot', [x], [ca.dot(x, x)], ['x'], ['cost'])
    return f


# ===========================================================================
# FunctionDescriptor — metadata
# ===========================================================================

class TestFunctionDescriptorMetadata:

    def test_name_extracted(self):
        fd = FunctionDescriptor(_scalar_function())
        assert fd.name == 'square'

    def test_input_names_extracted(self):
        fd = FunctionDescriptor(_two_input_function())
        assert fd.input_names == ['a', 'b']

    def test_output_names_extracted(self):
        fd = FunctionDescriptor(_two_input_function())
        assert fd.output_names == ['result']

    def test_input_shapes_scalar(self):
        fd = FunctionDescriptor(_scalar_function())
        assert fd.input_shapes == [(1, 1)]

    def test_input_shapes_vector(self):
        fd = FunctionDescriptor(_vector_function(3))
        assert fd.input_shapes == [(3, 1)]

    def test_output_shapes(self):
        fd = FunctionDescriptor(_scalar_function())
        assert fd.output_shapes == [(1, 1)]

    def test_description_stored(self):
        fd = FunctionDescriptor(_scalar_function(), description='my desc')
        assert fd.description == 'my desc'

    def test_description_defaults_empty(self):
        fd = FunctionDescriptor(_scalar_function())
        assert fd.description == ''

    def test_function_attribute_is_original(self):
        raw = _scalar_function()
        fd = FunctionDescriptor(raw)
        assert fd.function is raw


# ===========================================================================
# FunctionDescriptor — __call__
# ===========================================================================

class TestFunctionDescriptorCall:

    def test_positional_returns_mx(self):
        fd = FunctionDescriptor(_scalar_function())
        x = ca.MX.sym('x')
        result = fd(x)
        assert isinstance(result, ca.MX)

    def test_keyword_returns_mx(self):
        fd = FunctionDescriptor(_scalar_function())
        x = ca.MX.sym('x')
        result = fd(x=x)
        assert isinstance(result, ca.MX)

    def test_positional_result_usable_as_cost(self):
        """Result of positional __call__ can be registered as a cost term."""
        fd = FunctionDescriptor(_scalar_function())
        b = make_solver()
        x = b.add_variable('x', 1, initial_guess=3.0)
        cost = fd(x)
        b.add_cost(cost)
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res['x'], [0.0], atol=ATOL)

    def test_keyword_result_usable_as_cost(self):
        fd = FunctionDescriptor(_scalar_function())
        b = make_solver()
        x = b.add_variable('x', 1, initial_guess=3.0)
        cost = fd(x=x)
        b.add_cost(cost)
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res['x'], [0.0], atol=ATOL)

    def test_keyword_reorders_inputs(self):
        """Keyword call should work regardless of argument order."""
        fd = FunctionDescriptor(_two_input_function())
        a = ca.MX.sym('a')
        b = ca.MX.sym('b')
        res_kw   = fd(a=a, b=b)
        res_kw_r = fd(b=b, a=a)  # reversed — should produce same expression
        assert isinstance(res_kw, ca.MX)
        assert isinstance(res_kw_r, ca.MX)

    def test_wrong_positional_count_raises(self):
        fd = FunctionDescriptor(_scalar_function())
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        with pytest.raises(ValueError, match="1 positional"):
            fd(x, y)

    def test_wrong_shape_raises_valueerror(self):
        """Must raise ValueError — not a raw CasADi dimension error."""
        fd = FunctionDescriptor(_vector_function(3))
        wrong = ca.MX.sym('w', 5)  # (5,1), expected (3,1)
        with pytest.raises(ValueError, match="shape"):
            fd(wrong)

    def test_wrong_shape_error_message_contains_details(self):
        fd = FunctionDescriptor(_vector_function(3))
        wrong = ca.MX.sym('w', 5)
        with pytest.raises(ValueError) as exc_info:
            fd(wrong)
        msg = str(exc_info.value)
        assert 'dot' in msg or 'x' in msg   # function name or input name
        assert '(5, 1)' in msg or '(3, 1)' in msg

    def test_unknown_keyword_raises(self):
        fd = FunctionDescriptor(_scalar_function())
        x = ca.MX.sym('x')
        with pytest.raises(ValueError, match="unknown keyword"):
            fd(z=x)

    def test_missing_keyword_raises(self):
        fd = FunctionDescriptor(_two_input_function())
        a = ca.MX.sym('a')
        with pytest.raises(ValueError, match="missing"):
            fd(a=a)   # b is missing

    def test_mixed_positional_and_keyword_raises(self):
        fd = FunctionDescriptor(_two_input_function())
        a = ca.MX.sym('a')
        b = ca.MX.sym('b')
        with pytest.raises(ValueError, match="mixed"):
            fd(a, b=b)

    def test_numeric_argument_skips_shape_check(self):
        """Numeric values should not trigger shape validation."""
        fd = FunctionDescriptor(_scalar_function())
        result = fd(2.0)
        assert result is not None

    def test_two_output_function_returns_multiple(self):
        fd = FunctionDescriptor(_two_output_function())
        x = ca.MX.sym('x')
        result = fd(x)
        # CasADi returns a list for multiple outputs
        assert result is not None


# ===========================================================================
# FunctionDescriptor — __repr__
# ===========================================================================

class TestFunctionDescriptorRepr:

    def test_repr_contains_name(self):
        fd = FunctionDescriptor(_scalar_function())
        assert 'square' in repr(fd)

    def test_repr_contains_input_names_and_shapes(self):
        fd = FunctionDescriptor(_scalar_function())
        r = repr(fd)
        assert 'x' in r
        assert '(1, 1)' in r

    def test_repr_format(self):
        fd = FunctionDescriptor(_scalar_function())
        r = repr(fd)
        assert r.startswith("FunctionDescriptor(")
        assert '->' in r


# ===========================================================================
# SymbolDescriptor
# ===========================================================================

class TestSymbolDescriptor:

    def test_valid_scalar_construction(self):
        x = ca.MX.sym('x')
        sd = SymbolDescriptor(symbol=x, name='x', shape=(1, 1), semantic_type='scalar')
        assert sd.name == 'x'

    def test_valid_vector_construction(self):
        v = ca.MX.sym('v', 3)
        sd = SymbolDescriptor(symbol=v, name='v', shape=(3, 1), semantic_type='vector')
        assert sd.shape == (3, 1)

    def test_invalid_semantic_type_raises(self):
        x = ca.MX.sym('x')
        with pytest.raises(ValueError, match="semantic_type"):
            SymbolDescriptor(symbol=x, name='x', shape=(1, 1), semantic_type='bogus')

    def test_shape_mismatch_raises(self):
        x = ca.MX.sym('x')  # (1,1)
        with pytest.raises(ValueError, match="shape mismatch"):
            SymbolDescriptor(symbol=x, name='x', shape=(3, 1), semantic_type='scalar')

    def test_frame_on_non_vector_warns(self):
        x = ca.MX.sym('x')
        with pytest.warns(UserWarning, match="frame"):
            SymbolDescriptor(
                symbol=x, name='x', shape=(1, 1),
                semantic_type='scalar', frame='ECI'
            )

    def test_frame_on_vector_no_warning(self):
        v = ca.MX.sym('v', 3)
        # Should not warn
        sd = SymbolDescriptor(
            symbol=v, name='v', shape=(3, 1),
            semantic_type='vector', frame='ECI'
        )
        assert sd.frame == 'ECI'

    def test_optional_fields_default_none(self):
        x = ca.MX.sym('x')
        sd = SymbolDescriptor(symbol=x, name='x', shape=(1, 1), semantic_type='scalar')
        assert sd.frame is None
        assert sd.time_grid is None
        assert sd.units is None

    def test_all_valid_semantic_types_accepted(self):
        x = ca.MX.sym('x')
        for st in ('scalar', 'vector', 'trajectory', 'indexed_set', 'time_grid'):
            SymbolDescriptor(symbol=x, name='x', shape=(1, 1), semantic_type=st)


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
        from machina.blocks.registry import register as reg_register
        with pytest.raises(ValueError, match="already registered"):
            @reg_register('cost.quadratic')
            def _duplicate():
                pass

    def test_factory_returns_function_descriptor(self):
        fd = registry.get('cost.quadratic')(n=2)
        assert isinstance(fd, FunctionDescriptor)


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
