"""
Phase 3 tests for machina.viz — NLP visualization tool.

Sections
--------
TestBuildNlpGraph          — node presence, attributes, and counts
TestEdgeInference          — ca.symvar-based dependency detection
TestCanonicalLabels        — J/h/g canonical letters, subscripts for multiples
TestAnonymousLabels        — None-named terms get auto-generated display names
TestConstraintTypes        — equality (h) vs inequality (g) discrimination
TestDrawNlpGraph           — matplotlib rendering smoke tests
TestFlyboyIntegration      — end-to-end wiring check using the flyby problem
"""

import casadi as ca
import matplotlib
import pytest

matplotlib.use('Agg')  # non-interactive backend; must be set before pyplot import

import machina.swapc  # noqa: F401
from machina.library import registry
from machina.solver.backend import SolverBackend
from machina.viz import build_nlp_graph, draw_nlp_graph

pytestmark = pytest.mark.requires_casadi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solver():
    return SolverBackend(solver_opts={'ipopt.print_level': 0, 'print_time': False})


def _simple_solver_with_cost():
    """Solver with two variables, one named cost term referencing x only."""
    s = make_solver()
    x = s.add_variable('x', 2, lb=-10, ub=10)
    y = s.add_variable('y', 1, lb=-10, ub=10)
    cost_expr = ca.sumsqr(x)  # references x, not y
    s.add_cost(cost_expr, name='cost_x')
    return s, x, y


# ---------------------------------------------------------------------------
# TestBuildNlpGraph — node presence and attributes
# ---------------------------------------------------------------------------

class TestBuildNlpGraph:

    def test_variable_nodes_present(self):
        s = make_solver()
        s.add_variable('alpha', 3)
        s.add_variable('beta', 1)
        s.add_cost(ca.MX.zeros(1, 1), name='dummy')
        G = build_nlp_graph(s)
        assert 'var:alpha' in G.nodes
        assert 'var:beta' in G.nodes

    def test_parameter_nodes_present(self):
        s = make_solver()
        s.add_variable('x', 1)
        p = s.add_parameter('my_param', 2)
        s.add_cost(p[0] * s.symbol_of('x'), name='param_cost')
        G = build_nlp_graph(s)
        assert 'param:my_param' in G.nodes

    def test_cost_nodes_present(self):
        s, x, y = _simple_solver_with_cost()
        G = build_nlp_graph(s)
        assert 'cost:0' in G.nodes

    def test_constraint_nodes_present(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0, ub=1, name='x_bound')
        G = build_nlp_graph(s)
        assert 'con:0' in G.nodes

    def test_variable_node_type_and_size(self):
        s = make_solver()
        s.add_variable('q', 4)
        s.add_cost(ca.MX.zeros(1, 1), name='dummy')
        G = build_nlp_graph(s)
        data = G.nodes['var:q']
        assert data['type'] == 'variable'
        assert data['size'] == 4

    def test_variable_label_contains_name_and_dimension(self):
        s = make_solver()
        s.add_variable('q', 4)
        s.add_cost(ca.MX.zeros(1, 1), name='dummy')
        G = build_nlp_graph(s)
        label = G.nodes['var:q']['label']
        assert 'q' in label
        assert '4' in label      # dimension appears somewhere in mathtext label
        assert 'mathbb' in label  # confirms mathtext is present

    def test_constraint_node_n_rows(self):
        s = make_solver()
        x = s.add_variable('x', 3)
        s.add_cost(ca.sumsqr(x), name='sq')
        s.add_constraint(x, lb=0, ub=1, name='x_bounds')  # 3-row constraint
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['n_rows'] == 3

    def test_no_extra_nodes(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        G = build_nlp_graph(s)
        assert set(G.nodes) == {'var:x', 'cost:0'}

    def test_multiple_cost_terms_indexed(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        y = s.add_variable('y', 1)
        s.add_cost(x ** 2, name='cost_a')
        s.add_cost(y ** 2, name='cost_b')
        G = build_nlp_graph(s)
        assert 'cost:0' in G.nodes
        assert 'cost:1' in G.nodes
        assert 'cost_a' in G.nodes['cost:0']['label']
        assert 'cost_b' in G.nodes['cost:1']['label']

    def test_cost_node_in_size(self):
        s = make_solver()
        x = s.add_variable('x', 3)
        s.add_cost(ca.sumsqr(x), name='sq')
        G = build_nlp_graph(s)
        assert G.nodes['cost:0']['in_size'] == 3
        assert G.nodes['cost:0']['out_size'] == 1


# ---------------------------------------------------------------------------
# TestEdgeInference — dependency detection
# ---------------------------------------------------------------------------

class TestEdgeInference:

    def test_variable_to_cost_edge(self):
        s, x, y = _simple_solver_with_cost()
        G = build_nlp_graph(s)
        assert G.has_edge('var:x', 'cost:0')

    def test_unused_variable_has_no_cost_edge(self):
        s, x, y = _simple_solver_with_cost()
        G = build_nlp_graph(s)
        assert not G.has_edge('var:y', 'cost:0')

    def test_variable_to_constraint_edge(self):
        s = make_solver()
        x = s.add_variable('x', 2)
        y = s.add_variable('y', 1)
        s.add_cost(ca.sumsqr(x), name='sq')
        s.add_constraint(y, lb=0, ub=1, name='y_bound')
        G = build_nlp_graph(s)
        assert G.has_edge('var:y', 'con:0')
        assert not G.has_edge('var:x', 'con:0')

    def test_parameter_to_cost_edge(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        p = s.add_parameter('scale', 1)
        s.add_cost(p * x ** 2, name='scaled_sq')
        G = build_nlp_graph(s)
        assert G.has_edge('param:scale', 'cost:0')
        assert G.has_edge('var:x', 'cost:0')

    def test_shared_variable_in_cost_and_constraint(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0, name='x_pos')
        G = build_nlp_graph(s)
        assert G.has_edge('var:x', 'cost:0')
        assert G.has_edge('var:x', 'con:0')

    def test_equality_constraint_edges(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        y = s.add_variable('y', 1)
        s.add_cost(x ** 2 + y ** 2, name='sq')
        s.add_equality(x + y - 1, name='sum_eq')
        G = build_nlp_graph(s)
        assert G.has_edge('var:x', 'con:0')
        assert G.has_edge('var:y', 'con:0')

    def test_edge_label_scalar(self):
        """Scalar variable: edge label is just the name (no [1] suffix)."""
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        G = build_nlp_graph(s)
        label = G.edges['var:x', 'cost:0']['label']
        assert label == 'x'

    def test_edge_label_vector(self):
        """Vector variable: edge label includes [n] suffix."""
        s = make_solver()
        x = s.add_variable('proc_time', 3)
        s.add_cost(ca.sumsqr(x), name='sq')
        G = build_nlp_graph(s)
        label = G.edges['var:proc_time', 'cost:0']['label']
        assert label == 'proc_time [3]'


# ---------------------------------------------------------------------------
# TestCanonicalLabels — J/h/g letters and subscripts
# ---------------------------------------------------------------------------

class TestCanonicalLabels:

    def test_single_cost_canonical_is_J(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        G = build_nlp_graph(s)
        assert G.nodes['cost:0']['canonical'] == 'J'

    def test_multiple_costs_get_subscripts(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        y = s.add_variable('y', 1)
        s.add_cost(x ** 2, name='a')
        s.add_cost(y ** 2, name='b')
        G = build_nlp_graph(s)
        assert G.nodes['cost:0']['canonical'] == 'J₀'
        assert G.nodes['cost:1']['canonical'] == 'J₁'

    def test_single_equality_canonical_is_h(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_equality(x - 1, name='x_eq')
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['canonical'] == 'h'

    def test_single_inequality_canonical_is_g(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0, ub=1, name='x_ineq')
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['canonical'] == 'g'

    def test_multiple_equalities_get_subscripts(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        y = s.add_variable('y', 1)
        s.add_cost(x ** 2 + y ** 2, name='sq')
        s.add_equality(x - 1, name='x_eq')
        s.add_equality(y - 2, name='y_eq')
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['canonical'] == 'h₀'
        assert G.nodes['con:1']['canonical'] == 'h₁'

    def test_mixed_constraints_subscript_separately(self):
        """Equalities and inequalities each get their own subscript sequence."""
        s = make_solver()
        x = s.add_variable('x', 1)
        y = s.add_variable('y', 1)
        s.add_cost(x ** 2 + y ** 2, name='sq')
        s.add_equality(x - 1, name='x_eq')      # → h (single equality)
        s.add_constraint(y, lb=0, name='y_pos')  # → g (single inequality)
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['canonical'] == 'h'
        assert G.nodes['con:1']['canonical'] == 'g'

    def test_canonical_appears_in_label(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        G = build_nlp_graph(s)
        assert G.nodes['cost:0']['label'].startswith('J:')


# ---------------------------------------------------------------------------
# TestAnonymousLabels — None names produce auto-generated display names
# ---------------------------------------------------------------------------

class TestAnonymousLabels:

    def test_anonymous_cost_display_name(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2)  # no name
        G = build_nlp_graph(s)
        assert 'cost_0' in G.nodes['cost:0']['label']

    def test_anonymous_constraint_display_name(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0)  # no name
        G = build_nlp_graph(s)
        assert 'con_0' in G.nodes['con:0']['label']

    def test_mixed_named_and_anonymous(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2)            # anonymous
        s.add_cost(x ** 3, name='cube')  # named
        G = build_nlp_graph(s)
        assert 'cost_0' in G.nodes['cost:0']['label']
        assert 'cube'   in G.nodes['cost:1']['label']


# ---------------------------------------------------------------------------
# TestConstraintTypes — equality vs inequality
# ---------------------------------------------------------------------------

class TestConstraintTypes:

    def test_equality_node_type(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_equality(x - 1, name='x_eq')
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['type'] == 'equality'
        assert G.nodes['con:0']['is_equality'] is True

    def test_inequality_node_type(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0, ub=1, name='x_bound')
        G = build_nlp_graph(s)
        assert G.nodes['con:0']['type'] == 'inequality'
        assert G.nodes['con:0']['is_equality'] is False

    def test_equality_label_contains_equals_zero(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_equality(x - 1, name='x_eq')
        G = build_nlp_graph(s)
        assert '= 0' in G.nodes['con:0']['label']

    def test_inequality_label_does_not_contain_equals_zero(self):
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        s.add_constraint(x, lb=0, ub=1, name='x_bound')
        G = build_nlp_graph(s)
        assert '= 0' not in G.nodes['con:0']['label']


# ---------------------------------------------------------------------------
# TestDrawNlpGraph — matplotlib rendering smoke tests
# ---------------------------------------------------------------------------

class TestDrawNlpGraph:

    def test_returns_axes(self):
        import matplotlib.pyplot as plt
        s = make_solver()
        x = s.add_variable('x', 2)
        s.add_cost(ca.sumsqr(x), name='sq')
        s.add_constraint(x[0], lb=0, name='pos')
        G = build_nlp_graph(s)
        ax = draw_nlp_graph(G, title='test graph')
        assert ax is not None
        plt.close('all')

    def test_accepts_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        s = make_solver()
        x = s.add_variable('x', 1)
        s.add_cost(x ** 2, name='sq')
        G = build_nlp_graph(s)
        returned_ax = draw_nlp_graph(G, ax=ax)
        assert returned_ax is ax
        plt.close('all')

    def test_draws_without_error_on_disconnected_nodes(self):
        """A variable with no associated cost/constraint renders without error."""
        import matplotlib.pyplot as plt
        s = make_solver()
        s.add_variable('orphan', 1)
        used = s.add_variable('used', 1)
        s.add_cost(used ** 2, name='sq')
        G = build_nlp_graph(s)
        ax = draw_nlp_graph(G)
        assert ax is not None
        plt.close('all')


# ---------------------------------------------------------------------------
# TestFlyboyIntegration — wiring check using the flyby problem
# ---------------------------------------------------------------------------

class TestFlyboyIntegration:
    """
    Reconstruct the flyby_goodput example's solver setup and verify that
    the graph correctly reflects the proc_time → cost and proc_time →
    constraint dependencies, with correct canonical labels.
    """

    @pytest.fixture
    def flyby_solver(self):
        N_PRODUCTS     = 3
        OBS_DELAYS     = [60.0, 90.0, 120.0]
        WEIGHTS        = [0.50, 0.30, 0.20]
        COMPUTE_BUDGET = 240.0
        SIGMOID_K      = 0.08
        SIGMOID_T50    = 200.0

        ttp_func    = registry.get('util.ttp_computation')(n_components=2)
        sigmoid     = registry.get('cost.sigmoid_goodput')(k=SIGMOID_K, t50=SIGMOID_T50)
        agg_goodput = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid, n_products=N_PRODUCTS, weights=WEIGHTS)
        budget_fn   = registry.get('constraint.linear')(
            n=N_PRODUCTS, coefficients=[1.0, 1.0, 1.0])

        s = make_solver()
        proc_time = s.add_variable('proc_time', N_PRODUCTS,
                                   lb=0.0, ub=300.0,
                                   initial_guess=COMPUTE_BUDGET / N_PRODUCTS)

        ttp_exprs = []
        for i in range(N_PRODUCTS):
            delays_i = ca.vertcat(ca.MX(OBS_DELAYS[i]), proc_time[i])
            ttp_exprs.append(ttp_func(delays=delays_i))
        ttp_vector = ca.vertcat(*ttp_exprs)

        s.add_cost(-agg_goodput(ttp_vector=ttp_vector), name='neg_aggregate_goodput')
        s.add_equality(budget_fn(x=proc_time) - COMPUTE_BUDGET, name='compute_budget')
        return s

    def test_proc_time_connected_to_cost(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert G.has_edge('var:proc_time', 'cost:0')

    def test_proc_time_connected_to_constraint(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert G.has_edge('var:proc_time', 'con:0')

    def test_cost_label_contains_name(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert 'neg_aggregate_goodput' in G.nodes['cost:0']['label']

    def test_cost_canonical_is_J(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert G.nodes['cost:0']['canonical'] == 'J'

    def test_constraint_label_contains_name(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert 'compute_budget' in G.nodes['con:0']['label']

    def test_constraint_canonical_is_h(self, flyby_solver):
        """compute_budget is registered via add_equality → should be h."""
        G = build_nlp_graph(flyby_solver)
        assert G.nodes['con:0']['canonical'] == 'h'
        assert G.nodes['con:0']['type'] == 'equality'

    def test_constraint_label_has_equals_zero(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        assert '= 0' in G.nodes['con:0']['label']

    def test_edge_label_shows_proc_time(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        # proc_time is a 3-vector, so edge label should include [3]
        assert G.edges['var:proc_time', 'cost:0']['label'] == 'proc_time [3]'
        assert G.edges['var:proc_time', 'con:0']['label']  == 'proc_time [3]'

    def test_graph_node_count(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        # 1 variable + 1 cost + 1 constraint = 3 nodes (no parameters)
        assert len(G.nodes) == 3

    def test_graph_edge_count(self, flyby_solver):
        G = build_nlp_graph(flyby_solver)
        # proc_time → cost, proc_time → constraint = 2 edges
        assert len(G.edges) == 2
