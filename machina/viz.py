"""
machina/viz.py
--------------
Standalone debugging utility: inspect the wiring of a SolverBackend as a
directed graph.

Given a SolverBackend (post-registration; does not need to be built or solved),
``build_nlp_graph`` produces a NetworkX DiGraph whose nodes are the variables,
parameters, cost terms, and constraints that have been registered, and whose
edges record which symbols appear in which expressions.

This is a development tool, not a framework component. It does not affect
the NLP formulation in any way.

Usage
-----
    from machina.viz import build_nlp_graph, draw_nlp_graph
    import matplotlib.pyplot as plt

    G = build_nlp_graph(solver)
    draw_nlp_graph(G, title='My NLP')
    plt.show()

Node types and canonical labels
--------------------------------
  variable   — decision variable (from add_variable).    Left column.
  parameter  — fixed parameter   (from add_parameter).   Left column.
  cost       — cost term         (from add_cost).         Right column. Label: J
  equality   — equality constraint (lb == ub == 0).       Right column. Label: h
  inequality — inequality constraint (lb < ub or mixed).  Right column. Label: g

Node labels use matplotlib mathtext (``$...$``) to render ∈ ℝⁿ and ℝⁿ → ℝᵐ.
Variable and parameter nodes are drawn as circles; cost and constraint nodes
are drawn as rectangles.

Edge labels
-----------
Each edge is labeled with the decision variable or parameter name and its
dimension.  The named port of the receiving expression (e.g., ``ttp_vector``)
cannot be recovered from the MX graph without deep CasADi introspection; only
the leaf symbolic variables are accessible via ``ca.symvar()``.
"""

import numpy as np
import casadi as ca
import networkx as nx


# ---------------------------------------------------------------------------
# Mathtext helpers
# ---------------------------------------------------------------------------

_SUBSCRIPT_MAP = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def _subscript(n: int) -> str:
    """Return n as unicode subscript digits: 0 → '₀', 12 → '₁₂'."""
    return str(n).translate(_SUBSCRIPT_MAP)


def _r(size) -> str:
    r"""
    Return a bare mathtext fragment for ℝ, ℝⁿ, or ℝ^{m×n} (no surrounding
    $ signs).

    Args:
        size: An integer (scalar/vector) or a ``(rows, cols)`` tuple (matrix).

    Examples:
        _r(1)      → r'\mathbb{R}'
        _r(3)      → r'\mathbb{R}^{3}'
        _r((1, 1)) → r'\mathbb{R}'
        _r((3, 1)) → r'\mathbb{R}^{3}'
        _r((2, 4)) → r'\mathbb{R}^{2 \times 4}'
    """
    if isinstance(size, tuple):
        m, n = size
        if m == 1 and n == 1:
            return r'\mathbb{R}'
        elif n == 1:
            return r'\mathbb{R}' + '^{' + str(m) + '}'
        elif m == 1:
            return r'\mathbb{R}' + '^{' + str(n) + '}'
        else:
            return r'\mathbb{R}' + '^{' + str(m) + r' \times ' + str(n) + '}'
    else:
        if size == 1:
            return r'\mathbb{R}'
        return r'\mathbb{R}' + '^{' + str(size) + '}'


def _r_label(size) -> str:
    r"""Return a complete mathtext label: '$\in \mathbb{R}$' or '$\in \mathbb{R}^{n}$'.

    Args:
        size: An integer or ``(rows, cols)`` tuple — passed directly to ``_r``.
    """
    return '$\\in ' + _r(size) + '$'


def _r_arrow(in_size: int, out_size: int, suffix: str = '') -> str:
    r"""
    Return a complete mathtext label: '$\mathbb{R}^{n} \to \mathbb{R}^{m}$'.

    Args:
        in_size:  Input dimension (total number of scalar inputs).
        out_size: Output dimension.
        suffix:   Optional plain-text suffix appended inside the dollars,
                  e.g. ' = 0' for equality constraints.
    """
    return '$' + _r(in_size) + ' \\to ' + _r(out_size) + suffix + '$'


def _canonical(letter: str, idx: int, total: int) -> str:
    """
    Return the canonical label for a cost/constraint term.

    Single term of its type → bare letter (``J``, ``h``, ``g``).
    Multiple terms of the same type → subscripted (``J₀``, ``h₁``).
    """
    return letter if total == 1 else letter + _subscript(idx)


# ---------------------------------------------------------------------------
# Dependency inference
# ---------------------------------------------------------------------------

def _symvar_deps(expr, var_map, param_map):
    """
    Return a list of ``(node_id, name, shape)`` for each named symbol in expr.

    Uses ``ca.symvar()`` to find all leaf MX symbolic variables, then looks
    them up in ``var_map`` and ``param_map`` by their ``.name()`` attribute.
    Each symbol is returned at most once (deduplication via a seen set).

    ``shape`` is the ``(rows, cols)`` tuple stored in the map.
    """
    seen = set()
    result = []
    for mx in ca.symvar(expr):
        name = mx.name()
        if name in seen:
            continue
        seen.add(name)
        if name in var_map:
            _, _, shape = var_map[name]
            result.append((f'var:{name}', name, shape))
        elif name in param_map:
            _, _, shape = param_map[name]
            result.append((f'param:{name}', name, shape))
    return result


def _edge_label(name: str, shape) -> str:
    """Return a concise edge label showing the variable/parameter name and shape.

    Args:
        name:  Symbol name.
        shape: Either an ``int`` (scalar count) or a ``(rows, cols)`` tuple.

    Examples:
        _edge_label('x', (1, 1))  → 'x'
        _edge_label('v', (3, 1))  → 'v [3]'
        _edge_label('A', (2, 4))  → 'A [2×4]'
    """
    if isinstance(shape, tuple):
        m, n = shape
        if m == 1 and n == 1:
            return name
        elif n == 1:
            return f'{name} [{m}]'
        elif m == 1:
            return f'{name} [{n}]'
        else:
            return f'{name} [{m}×{n}]'
    else:
        return name if shape == 1 else f'{name} [{shape}]'


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_nlp_graph(solver) -> nx.DiGraph:
    """
    Build a directed dependency graph from a SolverBackend's registered state.

    Reads the solver's internal maps and expression lists directly. The solver
    does not need to be built or solved before calling this function.

    Node types and their attributes
    --------------------------------
    variable / parameter
        ``type``, ``label`` (name + ∈ ℝ^{shape}), ``size``

    cost
        ``type='cost'``, ``label`` (J: name + ℝⁿ → ℝ), ``canonical`` (J / J₀ / …),
        ``in_size``, ``out_size``

    equality / inequality constraint
        ``type='equality'`` or ``'inequality'``,
        ``label`` (h/g: name + ℝⁿ → ℝᵐ [= 0]), ``canonical``,
        ``n_rows``, ``in_size``, ``is_equality``

    Edge attributes
    ---------------
    ``label``: variable/parameter name with dimension suffix (``[n]`` or
    ``[m×n]`` for non-scalar shapes).

    Args:
        solver: A ``SolverBackend`` instance with at least one variable registered.

    Returns:
        A ``networkx.DiGraph``.
    """
    G = nx.DiGraph()
    var_map   = solver._var_map
    param_map = solver._param_map

    # ------------------------------------------------------------------
    # Variable nodes (left column)
    # ------------------------------------------------------------------
    for name, (start, end, shape) in var_map.items():
        G.add_node(f'var:{name}',
                   type='variable',
                   label=f'{name}\n{_r_label(shape)}',
                   size=end - start)

    # ------------------------------------------------------------------
    # Parameter nodes (left column)
    # ------------------------------------------------------------------
    for name, (start, end, shape) in param_map.items():
        G.add_node(f'param:{name}',
                   type='parameter',
                   label=f'{name}\n{_r_label(shape)}',
                   size=end - start)

    # ------------------------------------------------------------------
    # Cost nodes (right column)
    # ------------------------------------------------------------------
    n_costs = len(solver._cost_terms)
    for i, (name, expr) in enumerate(solver._cost_terms):
        node_id      = f'cost:{i}'
        display_name = name if name is not None else f'cost_{i}'
        canonical    = _canonical('J', i, n_costs)

        deps    = _symvar_deps(expr, var_map, param_map)
        in_size = sum(m * n for _, _, (m, n) in deps)

        node_label = (f'{canonical}: {display_name}\n'
                      + _r_arrow(in_size, 1))
        G.add_node(node_id,
                   type='cost',
                   label=node_label,
                   canonical=canonical,
                   in_size=in_size,
                   out_size=1)

        for dep_id, dep_name, dep_shape in deps:
            G.add_edge(dep_id, node_id, label=_edge_label(dep_name, dep_shape))

    # ------------------------------------------------------------------
    # Constraint nodes (right column)
    # First pass: determine equality/inequality for each constraint so
    # we can compute per-type totals for correct subscript numbering.
    # ------------------------------------------------------------------
    constraint_is_eq = []
    lbg_offset = 0
    for _, (_, n_rows) in zip(solver._g, solver._constraint_names):
        lbg_i = solver._lbg[lbg_offset: lbg_offset + n_rows]
        ubg_i = solver._ubg[lbg_offset: lbg_offset + n_rows]
        constraint_is_eq.append(all(lb == ub for lb, ub in zip(lbg_i, ubg_i)))
        lbg_offset += n_rows

    n_equalities   = sum(constraint_is_eq)
    n_inequalities = len(constraint_is_eq) - n_equalities

    # Second pass: build nodes and edges.
    eq_idx   = 0
    ineq_idx = 0
    lbg_offset = 0
    for i, (g_expr, (cname, n_rows)) in enumerate(
            zip(solver._g, solver._constraint_names)):
        node_id      = f'con:{i}'
        display_name = cname if cname is not None else f'con_{i}'
        is_eq        = constraint_is_eq[i]

        if is_eq:
            canonical = _canonical('h', eq_idx, n_equalities)
            eq_idx += 1
        else:
            canonical = _canonical('g', ineq_idx, n_inequalities)
            ineq_idx += 1
        lbg_offset += n_rows

        deps    = _symvar_deps(g_expr, var_map, param_map)
        in_size = sum(m * n for _, _, (m, n) in deps)
        suffix  = ' = 0' if is_eq else ''

        node_label = (f'{canonical}: {display_name}\n'
                      + _r_arrow(in_size, n_rows, suffix))
        G.add_node(node_id,
                   type='equality' if is_eq else 'inequality',
                   label=node_label,
                   canonical=canonical,
                   n_rows=n_rows,
                   in_size=in_size,
                   is_equality=is_eq)

        for dep_id, dep_name, dep_shape in deps:
            G.add_edge(dep_id, node_id, label=_edge_label(dep_name, dep_shape))

    return G


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _node_size(label: str) -> int:
    """Estimate a node_size (scatter-plot area in display pt²) from label text.

    Strips mathtext markers before measuring, then scales by character count
    and line count to ensure the node is wide enough for its text content.
    """
    lines = label.split('\n')
    # Strip mathtext delimiters and backslash sequences for length estimation.
    clean = [ln.replace('$', '').replace('\\', '') for ln in lines]
    max_chars = max(len(ln) for ln in clean)
    n_lines   = len(lines)
    return max(3000, max_chars * n_lines * 130)




# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_NODE_COLORS = {
    'variable':   '#4C8EDA',   # blue
    'parameter':  '#F4A460',   # sandy orange
    'cost':       '#5AAD5A',   # green   (J)
    'equality':   '#9B59B6',   # purple  (h)
    'inequality': '#D9534F',   # red     (g)
}

_LEGEND_LABELS = {
    'variable':   'variable (decision)',
    'parameter':  'parameter (fixed)',
    'cost':       'J — cost term',
    'equality':   'h — equality constraint',
    'inequality': 'g — inequality constraint',
}


def draw_nlp_graph(G: nx.DiGraph,
                   ax=None,
                   figsize=(14, 8),
                   title: str = None):
    """
    Draw the NLP dependency graph using matplotlib + NetworkX.

    Decision variables and parameters are placed on the left as circles;
    cost terms and constraints are placed on the right as rectangles.
    Edges point left → right and are labeled with the variable/parameter
    name and dimension.  Node labels use mathtext to render ∈ ℝⁿ,
    ℝ^{m×n}, and ℝⁿ → ℝᵐ.  Each node is auto-sized to accommodate its
    label text.

    Args:
        G:       DiGraph returned by ``build_nlp_graph``.
        ax:      Existing matplotlib Axes to draw on. If None, a new figure
                 and axes are created.
        figsize: Figure size in inches, used only when ``ax`` is None.
        title:   Optional figure title.

    Returns:
        The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Assign a 'layer' attribute for multipartite layout (left = 0, right = 1).
    # Work on a copy to avoid mutating the caller's graph.
    H = G.copy()
    for node_id, data in H.nodes(data=True):
        H.nodes[node_id]['layer'] = (
            0 if data.get('type') in ('variable', 'parameter') else 1
        )

    pos = nx.multipartite_layout(H, subset_key='layer', align='vertical')

    node_ids = list(H.nodes)
    labels   = {n: H.nodes[n].get('label', n) for n in node_ids}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        H, pos, ax=ax,
        node_color=[_NODE_COLORS.get(H.nodes[n].get('type'), '#cccccc')
                    for n in node_ids],
        node_size=[_node_size(H.nodes[n].get('label', n)) for n in node_ids],
        node_shape='s',
        alpha=0.92,
    )

    # Edges — slight curve so parallel edges (same variable → two targets)
    # don't overlap perfectly.
    nx.draw_networkx_edges(H, pos, ax=ax,
                           edge_color='#555555',
                           width=1.5,
                           arrows=True,
                           arrowsize=18,
                           connectionstyle='arc3,rad=0.08',
                           min_source_margin=25,
                           min_target_margin=25)

    # Node labels (mathtext rendered by matplotlib).
    nx.draw_networkx_labels(H, pos, labels, ax=ax,
                            font_size=8,
                            font_color='black',
                            font_weight='bold')

    # Edge labels — variable name + dimension.
    edge_labels = {(u, v): data.get('label', '')
                   for u, v, data in H.edges(data=True)
                   if data.get('label')}
    if edge_labels:
        nx.draw_networkx_edge_labels(H, pos, edge_labels, ax=ax,
                                     font_size=7,
                                     bbox=dict(boxstyle='round,pad=0.25',
                                               facecolor='white',
                                               edgecolor='#aaaaaa',
                                               alpha=0.85))

    # Legend — only include types that are actually present in the graph.
    present_types = {data.get('type') for _, data in H.nodes(data=True)}
    legend_patches = [
        mpatches.Patch(color=_NODE_COLORS[t], label=_LEGEND_LABELS[t])
        for t in ('variable', 'parameter', 'cost', 'equality', 'inequality')
        if t in present_types
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
                  framealpha=0.9)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11)

    return ax


# ---------------------------------------------------------------------------
# DOT export (optional — requires pydot)
# ---------------------------------------------------------------------------

def export_dot(G: nx.DiGraph, path: str) -> None:
    """
    Export the graph to a Graphviz DOT file.

    Requires the ``pydot`` package (``pip install pydot``).

    Args:
        G:    DiGraph returned by ``build_nlp_graph``.
        path: File path to write (e.g. ``'nlp.dot'``).

    Raises:
        ImportError: If ``pydot`` is not installed.
    """
    try:
        from networkx.drawing.nx_pydot import write_dot
    except ImportError:
        raise ImportError(
            "export_dot requires 'pydot'. Install it with: pip install pydot"
        ) from None

    _DOT_FILL = {
        'variable':   'lightblue',
        'parameter':  'lightyellow',
        'cost':       'lightgreen',
        'equality':   'plum',
        'inequality': 'lightsalmon',
    }

    H = G.copy()
    for node_id, data in H.nodes(data=True):
        node_type = data.get('type', '')
        H.nodes[node_id]['fillcolor'] = _DOT_FILL.get(node_type, 'white')
        H.nodes[node_id]['style']     = 'filled'
        # DOT label: use canonical + display name only (no mathtext).
        canonical = data.get('canonical', '')
        raw_label = data.get('label', node_id).split('\n')[0]
        H.nodes[node_id]['label'] = raw_label

    write_dot(H, path)
