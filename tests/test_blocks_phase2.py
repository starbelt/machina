"""
Phase 2 tests for machina.blocks — Flyby-relevant function library.

Sections
--------
TestSigmoidGoodput       — hand-validated numerical correctness
TestAggregateGoodput     — hand-validated for toy 2-product case
TestTtpComputation       — util.ttp_computation correctness
TestFunctionReuse        — same FunctionDescriptor called at multiple NLP sites
TestFlyboyToyProblem     — full end-to-end: sigmoid + aggregate + ttp_computation + IPOPT
"""

import casadi as ca
import numpy as np
import pytest

from machina.blocks import registry
from machina.solver.backend import SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-5


def make_solver(**opts):
    base = {'ipopt.print_level': 0, 'print_time': False}
    base.update(opts)
    return SolverBackend(solver_opts=base)


def sigmoid_formula(ttp, k, t50):
    """Reference Python implementation of the sigmoid goodput formula."""
    return 1.0 / (1.0 + np.exp(k * (ttp - t50)))


# ===========================================================================
# Sigmoid goodput — hand-validated numerical correctness
# ===========================================================================

class TestSigmoidGoodput:
    """
    Validate cost.sigmoid_goodput against the formula G(ttp) = 1/(1+exp(k*(ttp-t50)))
    at several known TTP values.
    """

    @pytest.fixture
    def sigmoid_k1_t50_0(self):
        """Sigmoid with k=1, t50=0: G(0)=0.5, G(-5)≈0.993, G(5)≈0.007."""
        return registry.get('cost.sigmoid_goodput')(k=1.0, t50=0.0)

    @pytest.fixture
    def sigmoid_k01_t50_100(self):
        """Sigmoid with k=0.1, t50=100: midpoint at ttp=100."""
        return registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)

    def _eval(self, descriptor, ttp_val):
        """Numerically evaluate a descriptor at a scalar TTP value."""
        result = descriptor.function(ca.DM(ttp_val))
        return float(result)

    def test_midpoint_is_half(self, sigmoid_k1_t50_0):
        """At ttp=t50, goodput must be exactly 0.5."""
        assert abs(self._eval(sigmoid_k1_t50_0, 0.0) - 0.5) < ATOL

    def test_below_midpoint_above_half(self, sigmoid_k1_t50_0):
        """For ttp < t50, goodput > 0.5 (lower TTP → higher value)."""
        g = self._eval(sigmoid_k1_t50_0, -5.0)
        assert g > 0.5

    def test_above_midpoint_below_half(self, sigmoid_k1_t50_0):
        """For ttp > t50, goodput < 0.5."""
        g = self._eval(sigmoid_k1_t50_0, 5.0)
        assert g < 0.5

    def test_matches_formula_at_minus_five(self, sigmoid_k1_t50_0):
        g = self._eval(sigmoid_k1_t50_0, -5.0)
        np.testing.assert_allclose(g, sigmoid_formula(-5.0, k=1.0, t50=0.0), atol=ATOL)

    def test_matches_formula_at_plus_five(self, sigmoid_k1_t50_0):
        g = self._eval(sigmoid_k1_t50_0, 5.0)
        np.testing.assert_allclose(g, sigmoid_formula(5.0, k=1.0, t50=0.0), atol=ATOL)

    def test_matches_formula_shifted_midpoint(self, sigmoid_k01_t50_100):
        """Validate at ttp=150 with k=0.1, t50=100."""
        g = self._eval(sigmoid_k01_t50_100, 150.0)
        np.testing.assert_allclose(g, sigmoid_formula(150.0, k=0.1, t50=100.0), atol=ATOL)

    def test_monotone_decreasing(self, sigmoid_k1_t50_0):
        """Goodput must strictly decrease as TTP increases."""
        ttps = [-10.0, -5.0, 0.0, 5.0, 10.0]
        values = [self._eval(sigmoid_k1_t50_0, t) for t in ttps]
        assert all(values[i] > values[i+1] for i in range(len(values) - 1))

    def test_steeper_k_steeper_slope(self):
        """Higher k means steeper sigmoid — larger absolute change per unit TTP."""
        low_k  = registry.get('cost.sigmoid_goodput')(k=0.01, t50=0.0)
        high_k = registry.get('cost.sigmoid_goodput')(k=1.00, t50=0.0)
        # Compare G(−10) − G(+10) for both
        def spread(desc):
            lo = float(desc.function(ca.DM(-10.0)))
            hi = float(desc.function(ca.DM( 10.0)))
            return lo - hi
        assert spread(high_k) > spread(low_k)

    def test_returns_mx_when_called_with_mx(self):
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        ttp = ca.MX.sym('ttp')
        result = sigmoid(ttp=ttp)
        assert isinstance(result, ca.MX)

    def test_mx_result_usable_as_cost(self):
        """MX expression from sigmoid can be added as a cost term without error."""
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        b = make_solver()
        ttp = b.add_variable('ttp', 1, lb=0.0, ub=600.0, initial_guess=100.0)
        b.add_cost(-sigmoid(ttp=ttp))
        b.build()
        res = b.solve()
        assert res.success


# ===========================================================================
# Aggregate goodput — hand-validated for 2-product toy case
# ===========================================================================

class TestAggregateGoodput:
    """
    Validate cost.aggregate_goodput against the formula:
        agg = sum(w_i * G(ttp_i))
    for known TTP vectors and weights.
    """

    def _eval_agg(self, descriptor, ttp_vector):
        """Numerically evaluate aggregate goodput at a concrete TTP vector."""
        result = descriptor.function(ca.DM(ttp_vector))
        return float(result)

    def test_equal_weights_equal_ttps_equals_single_goodput(self):
        """
        With equal weights and equal TTPs, aggregate goodput = per-product goodput.
        agg([t, t], [0.5, 0.5]) = 0.5*G(t) + 0.5*G(t) = G(t)
        """
        sigmoid = registry.get('cost.sigmoid_goodput')(k=1.0, t50=0.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=2, weights=[0.5, 0.5]
        )
        for ttp_val in [-5.0, 0.0, 5.0]:
            g_single = sigmoid_formula(ttp_val, k=1.0, t50=0.0)
            g_agg    = self._eval_agg(agg, [ttp_val, ttp_val])
            np.testing.assert_allclose(g_agg, g_single, atol=ATOL,
                                       err_msg=f"Failed at ttp={ttp_val}")

    def test_weights_sum_does_not_need_to_be_one(self):
        """Weights are arbitrary; the output is a weighted sum, not an average."""
        sigmoid = registry.get('cost.sigmoid_goodput')(k=1.0, t50=0.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=2, weights=[1.0, 1.0]
        )
        g = self._eval_agg(agg, [0.0, 0.0])
        # weights=[1,1], G(0)=0.5 each: total = 1.0*0.5 + 1.0*0.5 = 1.0
        np.testing.assert_allclose(g, 1.0, atol=ATOL)

    def test_matches_formula_asymmetric_case(self):
        """
        Hand-computed: weights=[0.7, 0.3], k=1, t50=0.
        ttp = [0.0, 5.0]:
          G(0.0) = 0.5
          G(5.0) = 1/(1+exp(5)) ≈ 0.006693
          agg = 0.7*0.5 + 0.3*0.006693 ≈ 0.35201
        """
        sigmoid = registry.get('cost.sigmoid_goodput')(k=1.0, t50=0.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=2, weights=[0.7, 0.3]
        )
        expected = 0.7 * sigmoid_formula(0.0, 1.0, 0.0) + 0.3 * sigmoid_formula(5.0, 1.0, 0.0)
        g = self._eval_agg(agg, [0.0, 5.0])
        np.testing.assert_allclose(g, expected, atol=ATOL)

    def test_three_products(self):
        """
        3-product case: weights=[0.5, 0.3, 0.2], k=0.1, t50=100.
        Hand-compute expected value and compare.
        """
        k, t50 = 0.1, 100.0
        weights = [0.5, 0.3, 0.2]
        ttps = [50.0, 100.0, 200.0]
        sigmoid = registry.get('cost.sigmoid_goodput')(k=k, t50=t50)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=3, weights=weights
        )
        expected = sum(w * sigmoid_formula(t, k, t50) for w, t in zip(weights, ttps))
        g = self._eval_agg(agg, ttps)
        np.testing.assert_allclose(g, expected, atol=ATOL)

    def test_returns_mx_when_called_with_mx(self):
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=2, weights=[0.5, 0.5]
        )
        ttp_vec = ca.MX.sym('ttp_vector', 2)
        result = agg(ttp_vector=ttp_vec)
        assert isinstance(result, ca.MX)


# ===========================================================================
# TTP computation
# ===========================================================================

class TestTtpComputation:

    def test_single_component_is_identity(self):
        """util.ttp_computation(1) with one delay returns that delay."""
        ttp_func = registry.get('util.ttp_computation')(n_components=1)
        result = float(ttp_func.function(ca.DM(37.5)))
        np.testing.assert_allclose(result, 37.5, atol=ATOL)

    def test_two_component_sum(self):
        ttp_func = registry.get('util.ttp_computation')(n_components=2)
        result = float(ttp_func.function(ca.DM([30.0, 70.0])))
        np.testing.assert_allclose(result, 100.0, atol=ATOL)

    def test_four_component_sum(self):
        ttp_func = registry.get('util.ttp_computation')(n_components=4)
        delays = [10.0, 25.0, 15.0, 50.0]
        result = float(ttp_func.function(ca.DM(delays)))
        np.testing.assert_allclose(result, sum(delays), atol=ATOL)

    def test_zero_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            registry.get('util.ttp_computation')(n_components=0)

    def test_output_shape_is_scalar(self):
        ttp_func = registry.get('util.ttp_computation')(n_components=3)
        assert ttp_func.output_shapes == [(1, 1)]

    def test_input_shape_matches_n_components(self):
        ttp_func = registry.get('util.ttp_computation')(n_components=5)
        assert ttp_func.input_shapes == [(5, 1)]

    def test_returns_mx_when_called_with_mx(self):
        ttp_func = registry.get('util.ttp_computation')(n_components=2)
        delays = ca.MX.sym('delays', 2)
        result = ttp_func(delays=delays)
        assert isinstance(result, ca.MX)

    def test_in_nlp_sum_is_correct(self):
        """
        minimize  (ttp - 60)^2  where ttp = a + b, a+b=60 at optimum.
        Verifies ttp_computation wires correctly into an NLP.
        """
        ttp_func = registry.get('util.ttp_computation')(n_components=2)
        b = make_solver()
        ab = b.add_variable('ab', 2, lb=0.0, ub=100.0, initial_guess=30.0)
        ttp_expr = ttp_func(delays=ab)
        b.add_cost((ttp_expr - 60.0)**2)
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res['ab'][0] + res['ab'][1], 60.0, atol=ATOL)


# ===========================================================================
# Function reuse — same FunctionDescriptor at multiple NLP sites
# ===========================================================================

class TestFunctionReuse:
    """
    Verify that calling the same FunctionDescriptor at multiple points
    in an NLP reuses the underlying ca.Function object (not copies).
    """

    def test_same_function_object_after_multiple_calls(self):
        """
        The .function attribute on a FunctionDescriptor is always the same
        object regardless of how many times __call__ has been invoked.
        """
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        ttp1 = ca.MX.sym('ttp1')
        ttp2 = ca.MX.sym('ttp2')
        _ = sigmoid(ttp=ttp1)
        _ = sigmoid(ttp=ttp2)
        assert sigmoid.function is sigmoid.function

    def test_two_call_sites_both_produce_mx(self):
        """Both call sites return valid MX expressions."""
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        ttp1 = ca.MX.sym('ttp1')
        ttp2 = ca.MX.sym('ttp2')
        g1 = sigmoid(ttp=ttp1)
        g2 = sigmoid(ttp=ttp2)
        assert isinstance(g1, ca.MX)
        assert isinstance(g2, ca.MX)

    def test_descriptor_called_at_three_sites_in_nlp(self):
        """
        A single sigmoid descriptor called on 3 independent variables.
        Verifies that multi-site usage is compatible with IPOPT and AD.

        minimize  -( G(x1) + G(x2) + G(x3) )
        Solution: x1*, x2*, x3* all at lb=0 (lowest TTP = best goodput)
        """
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=100.0)
        b = make_solver()
        x1 = b.add_variable('x1', 1, lb=0.0, ub=600.0, initial_guess=100.0)
        x2 = b.add_variable('x2', 1, lb=0.0, ub=600.0, initial_guess=100.0)
        x3 = b.add_variable('x3', 1, lb=0.0, ub=600.0, initial_guess=100.0)
        b.add_cost(-(sigmoid(ttp=x1) + sigmoid(ttp=x2) + sigmoid(ttp=x3)))
        b.build()
        res = b.solve()
        assert res.success
        # Unconstrained: all three should be at lower bound (0)
        np.testing.assert_allclose(res['x1'], [0.0], atol=ATOL)
        np.testing.assert_allclose(res['x2'], [0.0], atol=ATOL)
        np.testing.assert_allclose(res['x3'], [0.0], atol=ATOL)


# ===========================================================================
# Flyby toy problem — full end-to-end
# ===========================================================================

class TestFlyboyToyProblem:
    """
    Simplified flyby goodput optimization: manually wired (no agent type).

    Problem (2 products):
    ----------------------
    Products A and B have fixed observation delays (obs_A=50s, obs_B=80s)
    representing the time the flyby satellite must spend acquiring each product
    before processing can begin. These are not optimizable (geometry-determined).

    A shared compute budget of exactly 150s must be allocated as processing
    time between A and B. Processing time is a decision variable.

        TTP_i = obs_delay_i + proc_time_i          (via util.ttp_computation)
        maximize  0.7 * G(TTP_A) + 0.3 * G(TTP_B) (via cost.aggregate_goodput)
        s.t.      proc_time_A + proc_time_B = 150  (equality, equality constraint)
                  proc_time_i >= 0

    Sigmoid: k=0.1, t50=150s

    Expected behaviour:
        Product A (weight 0.7) should receive LESS processing time, giving it
        a lower TTP and higher goodput. Product B absorbs the rest.
    """

    @pytest.fixture(scope='class')
    def result(self):
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=150.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=2, weights=[0.7, 0.3]
        )
        ttp_func = registry.get('util.ttp_computation')(n_components=2)

        b = make_solver()
        proc_time = b.add_variable('proc_time', 2, lb=0.0, ub=300.0, initial_guess=75.0)

        # Fixed observation delays (baked in as MX constants)
        obs_A = ca.MX(50.0)
        obs_B = ca.MX(80.0)

        # TTP for each product: obs_delay + proc_time
        ttp_A = ttp_func(delays=ca.vertcat(obs_A, proc_time[0]))
        ttp_B = ttp_func(delays=ca.vertcat(obs_B, proc_time[1]))
        ttp_vector = ca.vertcat(ttp_A, ttp_B)

        b.add_cost(-agg(ttp_vector=ttp_vector))

        # Equality constraint: must allocate exactly 150s total
        b.add_equality(proc_time[0] + proc_time[1] - 150.0, name='compute_budget')

        b.build()
        return b.solve()

    def test_converges(self, result):
        assert result.success

    def test_proc_time_sums_to_budget(self, result):
        """The equality constraint must be satisfied at the optimum."""
        total = result['proc_time'][0] + result['proc_time'][1]
        np.testing.assert_allclose(total, 150.0, atol=ATOL)

    def test_proc_times_are_nonnegative(self, result):
        assert result['proc_time'][0] >= -ATOL
        assert result['proc_time'][1] >= -ATOL

    def test_objective_is_negative(self, result):
        """Minimizing -goodput means f_opt should be negative (goodput > 0)."""
        assert result.f_opt < 0.0
