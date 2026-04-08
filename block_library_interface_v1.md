# Layer 2 — Block Library Interface Specification

**Status:** Draft v1.1 — Phase 1 and Phase 2 implemented
**Last Updated:** 2026-04-02

---

## Overview

Layer 2 is a **function library** — a catalog of reusable mathematical computations, each wrapped in a thin metadata descriptor. Every computation is a `ca.Function` object. Agent types (Layer 3) rent decision variables from the solver backend (Layer 1), call library functions with those variables, and register the resulting MX expressions as costs or constraints.

Blocks do not interact with the solver backend. They are pure math.

### Design Rationale

CasADi's `ca.Function` is the natural unit of reuse. A `ca.Function`:
- Is stateless and deterministic (safe for optimization)
- Composes by symbolic evaluation (call with MX args → get MX expression nodes)
- Shares efficiently (one definition, N call sites, AD reuses the internal derivative graph)
- Is what `nlpsol` itself returns (the architecture is uniform top to bottom)

The original design used a `Block` ABC with a two-phase declare/build lifecycle. This was motivated by the need for pre-build graph inspection and validation. In the function-centric model, functions don't touch the solver backend, so there is nothing to validate before building — constructing a `ca.Function` is cheap (SX expression graph construction), and the cost of building an unused function is negligible.

The declare phase is replaced by the `FunctionDescriptor`'s metadata (queryable from the built `ca.Function`). Structural decisions (what variables to create, what constraints to register) belong to the agent type, not to the function library.

---

## Core Types

### FunctionDescriptor

Thin wrapper around a `ca.Function`. All metadata is extracted from the wrapped function — no redundant manual specification.

```python
class FunctionDescriptor:
    """Wraps a ca.Function with metadata and validated calling."""

    function: ca.Function      # The underlying CasADi Function
    description: str           # What this function computes (factory-provided)
    name: str                  # Extracted from function.name()
    input_names: list[str]     # Extracted from function.name_in(i)
    output_names: list[str]    # Extracted from function.name_out(i)
    input_shapes: list[tuple]  # Extracted from function.size_in(i)
    output_shapes: list[tuple] # Extracted from function.size_out(i)
```

**Construction:**

```python
def __init__(self, function: ca.Function, description: str = ''):
    self.function = function
    self.description = description
    self.name = function.name()
    self.input_names = [function.name_in(i) for i in range(function.n_in())]
    self.output_names = [function.name_out(i) for i in range(function.n_out())]
    self.input_shapes = [function.size_in(i) for i in range(function.n_in())]
    self.output_shapes = [function.size_out(i) for i in range(function.n_out())]
```

**`__call__` (validated calling):**

```python
def __call__(self, *args, **kwargs) -> ca.MX | tuple[ca.MX, ...]:
    """
    Call the underlying ca.Function with shape validation.

    Supports both positional and keyword calling:
        descriptor(ttp_mx, k_mx)              # positional
        descriptor(ttp=ttp_mx, k=k_mx)        # keyword

    Raises ValueError with a descriptive message if:
        - Wrong number of arguments (positional)
        - Unknown keyword argument name
        - Argument shape does not match expected shape

    Returns:
        Single MX expression if one output, tuple of MX expressions
        if multiple outputs. Matches ca.Function behavior.
    """
```

**Validation behavior:**

- **Positional call:** Check `len(args) == function.n_in()`. For each argument, check `args[i].shape == self.input_shapes[i]` (if the argument is an MX; skip check for numeric values, which CasADi promotes automatically).
- **Keyword call:** Check that all keys are in `self.input_names`. Check that all required inputs are provided. Reorder to positional based on `input_names` ordering, then validate shapes as above.
- **Mixed positional/keyword:** Not supported. Raise `ValueError` if both `args` and `kwargs` are non-empty.
- **Error messages** include the function name, expected input name, expected shape, and received shape. Example: `"Function 'sigmoid_goodput': input 'ttp' expected shape (1, 1), got (3, 1)"`

**`__repr__`:**

```python
def __repr__(self) -> str:
    inputs = ', '.join(f'{n}{s}' for n, s in zip(self.input_names, self.input_shapes))
    outputs = ', '.join(f'{n}{s}' for n, s in zip(self.output_names, self.output_shapes))
    return f"FunctionDescriptor('{self.name}', [{inputs}] -> [{outputs}])"
```

---

### SymbolDescriptor

Wraps a CasADi MX variable or expression with semantic metadata. Used by agent types to track what symbols mean and to expose them for path resolution. Unchanged from the original design doc.

```python
@dataclass
class SymbolDescriptor:
    symbol: ca.MX              # The CasADi MX symbolic variable or expression
    name: str                  # Human-readable name (e.g., 'sma')
    shape: tuple               # Shape of the symbol, e.g., (3, 1), (1, 1)
    semantic_type: str         # 'scalar', 'vector', 'trajectory',
                               # 'indexed_set', 'time_grid'
    frame: str = None          # Reference frame if applicable ('ECI', 'LVLH', 'body')
    time_grid: object = None   # Reference to time grid if time-varying
    units: str = None          # Optional, advisory (e.g., 'km', 'rad/s')
```

**Validation (enforced):**

- `semantic_type` must be one of: `'scalar'`, `'vector'`, `'trajectory'`, `'indexed_set'`, `'time_grid'`. Raises `ValueError` otherwise.
- `shape` must match `symbol.shape`. Raises `ValueError` on mismatch.
- `frame` should only be set when `semantic_type` is `'vector'`. Issue `UserWarning` if set on a scalar or other non-spatial type.

**Design notes:**

- `symbol` may be a raw MX variable (from `solver.add_variable()`) or an MX expression (result of algebraic operations or a Function call). Both are valid MX objects.
- `units` is advisory and not enforced programmatically. Exists for documentation and debugging.

---

## Function Registry

A global dictionary mapping string names to factory callables. Provides lookup by name (for YAML-driven assembly in Phase 4) and a decorator for registration.

### Registry Interface

```python
# Module-level registry
_registry: dict[str, Callable] = {}

def register(name: str):
    """
    Decorator to register a factory function.

    Usage:
        @register('cost.sigmoid_goodput')
        def make_sigmoid_goodput(*, k: float, t50: float) -> FunctionDescriptor:
            ...

    Raises ValueError if name is already registered.
    """

def get(name: str) -> Callable:
    """
    Look up a factory by name.

    Raises KeyError with a message listing available names if not found.
    """

def list_registered() -> list[str]:
    """Return sorted list of all registered factory names."""

def list_by_domain(domain: str) -> list[str]:
    """
    Return registered names under a domain prefix.

    Example: list_by_domain('cost') returns
    ['cost.sigmoid_goodput', 'cost.aggregate_goodput', ...]
    """
```

### Naming Convention

Names follow the pattern `domain.function_name`:

| Domain | Purpose | Examples |
|--------|---------|----------|
| `cost` | Objective function components | `cost.sigmoid_goodput`, `cost.aggregate_goodput` |
| `dynamics` | Equations of motion | `dynamics.keplerian`, `dynamics.j2` |
| `constraint` | Constraint computations | `constraint.power_budget`, `constraint.downlink_capacity` |
| `transform` | Coordinate/unit transforms | `transform.koe_to_cartesian`, `transform.frame_rotation` |
| `util` | General-purpose math utilities | `util.ttp_computation`, `util.weighted_sum` |

Domain prefixes are convention, not enforced structure. The registry is a flat dict. `list_by_domain()` is a convenience filter on the prefix.

---

## Factory Functions

### Standard Signature

All factory functions use keyword-only arguments and return a `FunctionDescriptor`:

```python
@register('domain.function_name')
def make_function_name(*, arg1: type, arg2: type, ...) -> FunctionDescriptor:
    """
    Brief description of what this function computes.

    Args:
        arg1: Description.
        arg2: Description.

    Returns:
        FunctionDescriptor wrapping a ca.Function with:
            Inputs: name1(shape1), name2(shape2), ...
            Outputs: name1(shape1), name2(shape2), ...
    """
    # 1. Create SX symbols for inputs
    # 2. Build SX expression for the computation
    # 3. Wrap in ca.Function with named I/O
    # 4. Return FunctionDescriptor
```

**Rules:**

- All arguments are keyword-only (use `*` separator). This ensures uniform dispatch from YAML via `factory(**params_dict)`.
- Return type is always `FunctionDescriptor`.
- The factory constructs SX symbols internally, builds the computation in SX, wraps it in `ca.Function`, and returns the descriptor. The caller never sees SX — only the `FunctionDescriptor` (which the caller invokes with MX arguments).
- Configuration constants (steepness coefficients, gravitational parameter, etc.) can either be baked into the `ca.Function` at construction time or exposed as additional function inputs. The choice depends on whether the constant needs to vary at solve time (→ function input, backed by an MX parameter) or is fixed for the problem (→ baked in).
- If a factory needs another `FunctionDescriptor` as input (function composition), it accepts it as a keyword argument. The composed function is called symbolically during SX construction to embed it.

### Example Factory

```python
@register('cost.sigmoid_goodput')
def make_sigmoid_goodput(*, k: float, t50: float) -> FunctionDescriptor:
    """
    Sigmoid goodput function: G(ttp) = 1 / (1 + exp(k * (ttp - t50))).

    Higher goodput for lower TTP. Steepness k and midpoint t50 are
    baked into the function at construction time.

    Args:
        k: Steepness parameter (positive = goodput decreases with TTP).
        t50: TTP value at which goodput = 0.5.

    Returns:
        FunctionDescriptor wrapping a ca.Function with:
            Inputs: ttp(1,1)
            Outputs: goodput(1,1)
    """
    ttp = ca.SX.sym('ttp')
    goodput = 1.0 / (1.0 + ca.exp(k * (ttp - t50)))
    f = ca.Function('sigmoid_goodput', [ttp], [goodput], ['ttp'], ['goodput'])
    return FunctionDescriptor(f, description=f'Sigmoid goodput (k={k}, t50={t50})')
```

### Example: Factory with Function Composition

```python
@register('cost.aggregate_goodput')
def make_aggregate_goodput(
    *,
    per_product_func: FunctionDescriptor,
    n_products: int,
    weights: list[float]
) -> FunctionDescriptor:
    """
    Weighted sum of per-product goodput values.

    Calls per_product_func on each element of a TTP vector and
    computes the weighted sum.

    Args:
        per_product_func: FunctionDescriptor for per-product goodput.
        n_products: Number of products.
        weights: Importance weight for each product.

    Returns:
        FunctionDescriptor wrapping a ca.Function with:
            Inputs: ttp_vector(n_products, 1)
            Outputs: aggregate_goodput(1, 1)
    """
    ttp_vec = ca.SX.sym('ttp_vector', n_products)
    total = 0
    for i in range(n_products):
        gi = per_product_func.function(ttp_vec[i])
        total += weights[i] * gi
    f = ca.Function(
        'aggregate_goodput',
        [ttp_vec], [total],
        ['ttp_vector'], ['aggregate_goodput']
    )
    return FunctionDescriptor(f, description=f'Aggregate goodput ({n_products} products)')
```

**Note on composition:** In the example above, `per_product_func.function` is called (not `per_product_func()`) to bypass `FunctionDescriptor` validation during SX-level construction. The validation wrapper operates on MX arguments at NLP assembly time. During factory construction, we're working in SX, where the inner function is being symbolically embedded — shape validation at this level would be redundant and the SX symbols may not match the MX-oriented checks. Direct `.function` access during factory construction is the intended pattern.

---

## Usage Pattern

How Layer 2 functions are used by an agent type (Layer 3) or a manual test script:

```python
from machina.solver import SolverBackend
from machina.blocks import registry, SymbolDescriptor

# 1. Set up solver backend (Layer 1)
solver = SolverBackend()

# 2. Rent decision variables from the solver
ttp = solver.add_variable('flyby/ttp', 1, lb=0, ub=600, initial_guess=60)
weight = 0.8  # fixed, not a decision variable

# 3. Look up and construct functions from the registry
sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=30.0)

# 4. Call the function with MX arguments
goodput_expr = sigmoid(ttp=ttp)  # returns MX expression

# 5. Register with the solver backend
solver.add_cost(-goodput_expr, name='neg_goodput')  # minimize negative goodput

# 6. Build and solve
solver.build()
result = solver.solve()
```

---

## Relationship to Other Layers

| Layer | Relationship to Layer 2 |
|-------|------------------------|
| **Layer 1 (SolverBackend)** | Layer 2 functions produce MX expressions that Layer 1 accepts via `add_cost()` and `add_constraint()`. Layer 2 never calls Layer 1 directly. |
| **Layer 3 (Agent Types)** | Agent types are the primary consumers. They rent variables from Layer 1, call Layer 2 functions, and register results with Layer 1. They decide *what* to compute and *what* to optimize; Layer 2 decides *how* to compute it. |
| **Layer 4 (Problem Compiler)** | Uses the registry to look up factory functions by name from YAML configuration. Passes YAML params as `**kwargs` to the factory. |
| **Layer 5 (YAML)** | References function names (e.g., `cost.sigmoid_goodput`) and provides configuration params. |

---

## Development Phases

### Phase 1 — Foundation (FunctionDescriptor, SymbolDescriptor, Registry) ✓ COMPLETE

**Goal:** Prove the function-centric pattern works end-to-end with Layer 1.

**Deliverables:**
- [x] `FunctionDescriptor` class with `__call__` validation and `__repr__` — `machina/blocks/descriptor.py`
- [x] `SymbolDescriptor` dataclass with enforced validation — `machina/blocks/descriptor.py`
- [x] Registry module (`register`, `get`, `list_registered`, `list_by_domain`) — `machina/blocks/registry.py`
- [x] Toy factories: `cost.quadratic`, `cost.rosenbrock`, `constraint.linear` — `machina/blocks/library/cost.py`, `constraint.py`

**Tests:** `tests/test_blocks_phase1.py` — 48 tests, all passing.
- [x] Metadata extraction from `ca.Function`
- [x] Validated calling (positional, keyword, mixed-raises, wrong shape, unknown/missing keyword)
- [x] `__repr__` format
- [x] `SymbolDescriptor` validation (semantic_type, shape, frame warning)
- [x] Registry: duplicate rejection, `list_by_domain`, `KeyError` with available names
- [x] End-to-end NLP pipeline (quadratic, Rosenbrock, linear constraint) — proves AD works through `__call__`

**Implementation notes:**
- `cost.rosenbrock` was added alongside `cost.quadratic` as the primary AD correctness test (Phase 1 spec called for "a quadratic cost function"; Rosenbrock was chosen for the end-to-end test as it is a more demanding AD check).
- Resource constraint factories (`power_budget`, `downlink_capacity`, `compute_throughput`) were evaluated and determined to be structurally identical to weighted linear sums. They are **not implemented as dedicated functions** — callers compose `constraint.linear` or use inline MX expressions. See Phase 2 notes.

### Phase 2 — Flyby-Relevant Functions ✓ COMPLETE

**Goal:** Build the function library needed for the goodput optimization problem.

**Deliverables:**
- [x] `cost.sigmoid_goodput` factory — `machina/blocks/library/cost.py`
- [x] `cost.aggregate_goodput` factory (with function composition) — `machina/blocks/library/cost.py`
- [x] `util.ttp_computation` factory (sum of N delay components) — `machina/blocks/library/util.py`
- [x] Resource constraint factories — **decision: not implemented as dedicated factories**

**Resource constraint factory decision:** `constraint.power_budget`, `constraint.downlink_capacity`, and `constraint.compute_throughput` are all of the form `c^T x <= limit`, structurally identical to `constraint.linear`. Introducing dedicated named factories adds no mathematical expressiveness. Agent types (Layer 3) are the appropriate place for domain-specific resource constraint wiring, using `constraint.linear` or inline MX expressions directly. This avoids a redundant layer of domain naming in the function library, which is supposed to be pure math.

**Tests:** `tests/test_blocks_phase2.py` — 30 tests, all passing.
- [x] Sigmoid goodput: numerical correctness at known TTP values, monotonicity, steepness parameter effect
- [x] Aggregate goodput: equal-weight identity, asymmetric hand-computed case (2- and 3-product)
- [x] `util.ttp_computation`: identity (n=1), multi-component sum, shape contracts, n=0 rejection, NLP wiring
- [x] Function reuse: same `FunctionDescriptor` at multiple NLP call sites, AD still correct
- [x] Flyby toy problem (2 products, shared compute budget equality constraint): converges, constraint satisfied, bounds respected, objective sign correct

**Tests intentionally not included:** Economic optimality of the allocation (which product gets more processing time). These would test IPOPT's solver correctness, not our system's wiring.

**Stop and validate against the baseline spreadsheet.** This is the gate for moving to Layer 3. The function library must be able to express the real problem before the agent layer automates its assembly.

### Phase 3 — Visualization Tool ✓ COMPLETE

**Goal:** Debug and inspect function/variable/constraint wiring.

**Deliverables:**
- [x] Standalone utility that takes a `SolverBackend` (post-registration) and produces a NetworkX directed graph — `machina/viz.py`
- [x] Node types: variable (from `_var_map`), parameter (from `_param_map`), cost term (from `_cost_terms`), constraint (from `_constraint_names`)
- [x] Edge inference from the MX expression graph (which variables appear in which cost/constraint expressions)
- [x] Rendering: matplotlib (`draw_nlp_graph`) and Graphviz DOT export (`export_dot`)

**This phase is not gated.** Build it when you need it for debugging. It's a development tool, not a framework component.

---

## Future Extensions (documented, not implemented)

### StructuralBlock (Phase 5)

For trajectory optimization, some computations require creating decision variables and constraints as part of their internal logic (e.g., collocation schemes with defect constraints at each node). These cannot be pure functions.

When this need arises, introduce a `StructuralBlock` class that *does* interact with the solver backend. It would coexist with the function library — the function library handles pure math, `StructuralBlock` handles NLP structure generation. The agent type would use both.

This is explicitly deferred. The flyby problem does not require it.

### Factory Argument Metadata (Phase 4)

For YAML-driven assembly, the problem compiler needs to know what arguments a factory expects. Options:
- Inspect `inspect.signature()` at registration time and store parameter names/types/defaults alongside the callable.
- Require factories to provide a schema (e.g., a dict of `{param_name: {type: ..., default: ..., description: ...}}`).
- Rely on Python's built-in `TypeError` for missing/extra arguments (simplest, least informative).

Defer decision until Phase 4 implementation.

### Function Caching

If the same factory is called multiple times with identical arguments (e.g., the same sigmoid for all products), the registry could cache and return the same `FunctionDescriptor`. This avoids creating structurally identical `ca.Function` objects. Low priority — the cost of duplicate Functions is small for the expected problem scale.
