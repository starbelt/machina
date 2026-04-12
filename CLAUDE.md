# Flyby Satellite Mission Optimizer — Design Document

**Status:** Draft v0.8 (Layers 1–2 complete, Layer 3 in progress — Phases 3a–3c done)
**Last Updated:** 2026-04-09
**Author:** Nathan

---

## 1. Project Overview

### 1.1 Goal

Build a generic, extensible space mission optimization framework in Python using CasADi/IPOPT. The framework allows users to define optimization problems via YAML files that reference a library of reusable building blocks — dynamics models, cost functions, constraint templates, and mission components — without writing CasADi code directly.

### 1.2 Near-Term Application

Optimize a system of flyby satellites (near-GEO orbit) that augment the GOES-19 weather satellite. The optimizer varies the flyby system's design parameters to maximize **aggregate goodput** — the timeliness-weighted value of data products delivered to users. The flyby system is purely additive; the existing GOES-19 pipeline remains untouched.

### 1.3 Long-Term Vision

Support a broad class of space mission optimization problems:
- Constellation design and coverage optimization
- Trajectory optimization (impulsive and low-thrust)
- Maneuver planning and station-keeping
- Optimal control problems (OCPs)
- Moving horizon estimation (MHEs)
- Any problem reducible to an NLP over composed building blocks

The framework's philosophy is inspired by **STK Astrogator**: the user assembles known mission components from a library, and the optimizer operates underneath. The YAML is a bill of materials, not a math language. New capabilities require writing Python blocks, not complicating the YAML schema.

---

## 2. Architecture Overview

The framework is organized into five layers, ordered from lowest-level (solver) to highest-level (user interface).

```
┌─────────────────────────────────────┐
│  Layer 5: YAML Problem Definition   │  ← User writes this
├─────────────────────────────────────┤
│  Layer 4: Problem Compiler          │  ← Reads YAML, assembles problem
├─────────────────────────────────────┤
│  Layer 3: Agent Types               │  ← Domain-aware composition classes
├─────────────────────────────────────┤
│  Layer 2: Block Library             │  ← Reusable math building blocks
├─────────────────────────────────────┤
│  Layer 1: Solver Backend            │  ← CasADi/IPOPT wrapper
└─────────────────────────────────────┘
```

**Data flow:** The YAML defines a problem in mission-design terms. The problem compiler reads it, instantiates agent types (Layer 3), which internally compose blocks (Layer 2). Blocks register variables, constraints, and cost terms with the solver backend (Layer 1). The solver backend calls IPOPT and returns results.

**Key principle:** Domain knowledge flows downward. The YAML knows nothing about CasADi. Agent types know about mission composition but not solver internals. Blocks know CasADi but not mission context. The solver backend knows only CasADi.

---

## 3. Layer Specifications

### 3.1 Layer 1 — Solver Backend

**Purpose:** Thin, domain-agnostic wrapper around CasADi's `nlpsol` interface. Owns all CasADi symbolic variables and manages the mapping between named variables and their positions in the global decision vector.

**Why `nlpsol` instead of `Opti`:** The `Opti` interface is convenient for one-off problems but acts as a black box — it owns variable indexing internally and makes it difficult to inspect or control the structure of the global decision vector. `nlpsol` gives the framework explicit control over the decision vector layout, bound vectors, and constraint assembly. Since the solver backend is already providing the bookkeeping abstraction that `Opti` would provide (named variables, structured registration, result extraction by name), building on `nlpsol` avoids a redundant abstraction layer and keeps the framework closer to the math.

**Responsibilities:**
- **Variable registration:** Creates MX symbolic variables, tracks their index range in the global decision vector, stores bounds and initial guesses. Returns the MX symbol to the caller.
- **Parameter registration:** Creates MX symbolic parameters (fixed values set at solve time, not optimized). Enables re-solving with different problem data without rebuilding the solver.
- **Constraint collection:** Accepts MX expressions with lower/upper bounds. Accumulates into constraint vector `g` and bound vectors `lbg`/`ubg`. Equality constraints use `lb == ub`.
- **Cost term collection:** Accepts MX cost expressions and sums them into a single scalar objective `f`.
- **Build:** Calls `ca.vertcat()` on accumulated variable and constraint lists, assembles the NLP dict `{'x': x, 'f': f, 'g': g}` (with `'p'` added only when parameters are registered), and creates the `ca.nlpsol` solver object.
- **Solve:** Calls the solver with bounds, initial guesses, and parameter values. Returns a `SolutionResult`.
- **Result extraction:** Uses the stored index map to slice named variables out of `sol['x']`.
- **Solver configuration:** IPOPT by default, with options overridable (tolerances, max iterations, linear solver, print level, etc.).

**Does not contain:** Any knowledge of orbital mechanics, mission design, or problem structure. Does not know about blocks, agents, or descriptors.

**Key interface:**

```python
class SolverBackend:
    def __init__(self, solver: str = 'ipopt', solver_opts: dict = None):
        """
        Initialize the solver backend.

        Args:
            solver: Solver plugin name ('ipopt', 'sqpmethod', etc.)
            solver_opts: Solver-specific options. For IPOPT, these are
                         passed via dot notation (e.g., 'ipopt.tol': 1e-8)
                         or nested dict (e.g., {'ipopt': {'tol': 1e-8}}).
        """
        # Decision vector accumulators
        self._w = []        # MX variable symbols
        self._w0 = []       # Initial guess values (flat list)
        self._lbw = []      # Lower bounds on variables (flat list)
        self._ubw = []      # Upper bounds on variables (flat list)
        # Constraint accumulators
        self._g = []        # MX constraint expressions
        self._lbg = []      # Lower bounds on constraints (flat list)
        self._ubg = []      # Upper bounds on constraints (flat list)
        # Parameter accumulators
        self._p = []        # MX parameter symbols
        # Objective
        self._J = ca.MX.zeros(1, 1)  # Accumulated cost (MX scalar, not Python 0)
        # Index tracking
        self._offset = 0    # Current position in global decision vector
        self._p_offset = 0  # Current position in global parameter vector
        self._var_map = {}  # name → (start_index, end_index, (rows, cols))
        self._param_map = {}# name → (start_index, end_index, (rows, cols))
        # Namespace and metadata
        self._names = set() # Shared collision namespace (variables + parameters only)
        self._cost_terms = []        # (name, expr) pairs for reporting
        self._constraint_names = []  # (name, n_rows) pairs for reporting
        # Lifecycle flags
        self._solver = None # ca.nlpsol object, created at build() time
        self._built = False # True after build() completes
        self._solved = False# True after the first solve() completes

    def add_variable(self, name: str, n: int | tuple[int, int],
                     lb=-np.inf, ub=np.inf,
                     initial_guess=0.0) -> ca.MX:
        """
        Create a decision variable and register it in the global
        decision vector.

        Args:
            name: Unique name for this variable (used for result extraction)
            n: int for an n×1 vector, or (rows, cols) tuple for a matrix.
               Matrix variables are flattened column-major (ca.vec) into the
               decision vector and reshaped back on extraction.
            lb: Lower bound (scalar broadcasts to all elements)
            ub: Upper bound (scalar broadcasts to all elements)
            initial_guess: Starting value for solver (scalar broadcasts)

        Returns:
            MX symbolic variable with the requested shape — use this to
            build expressions.
        """
        ...

    def add_parameter(self, name: str, n: int | tuple[int, int]) -> ca.MX:
        """
        Create a fixed parameter (not optimized, value set at solve time).
        Enables re-solving with different data without rebuilding solver.

        Args:
            n: int for an n×1 vector, or (rows, cols) tuple for a matrix.
               Matrix parameters are flattened column-major into the parameter
               vector; p_val must supply values in the same column-major order.

        Returns:
            MX symbolic parameter with the requested shape.
        """
        ...

    def add_constraint(self, expr: ca.MX,
                       lb=-np.inf, ub=np.inf,
                       name: str = None):
        """
        Add a constraint to the problem.

        For equality: set lb == ub (or use add_equality convenience method).
        For double-sided inequality: set finite lb and ub.
        For one-sided: leave one as +/-inf.

        Args:
            expr: MX expression for the constraint (can be vector-valued)
            lb: Lower bound (scalar broadcasts)
            ub: Upper bound (scalar broadcasts)
            name: Optional name for debugging
        """
        ...

    def add_equality(self, expr: ca.MX, name: str = None):
        """Convenience: add_constraint(expr, lb=0, ub=0)."""
        ...

    def add_cost(self, expr: ca.MX, name: str = None):
        """
        Add a cost term. All terms are summed into a single scalar
        objective. The term must be a scalar MX expression.
        """
        ...

    def build(self, opts: dict = None) -> ca.Function:
        """
        Assemble the NLP and create the solver.

        Calls ca.vertcat on accumulated variables and constraints,
        builds the NLP dict, creates the ca.nlpsol solver object.

        Default IPOPT options:
            tol: 1e-8, max_iter: 2000, linear_solver: 'mumps',
            mu_strategy: 'adaptive', print_level: 5

        Args:
            opts: Override or extend default solver options.

        Returns:
            The ca.nlpsol solver Function (also stored internally).
        """
        ...

    def solve(self, p_val=None) -> 'SolutionResult':
        """
        Call the solver with stored bounds, initial guesses, and
        parameter values.

        Args:
            p_val: Numeric values for all registered parameters,
                   as a flat list or numpy array, in registration order.

        Returns:
            SolutionResult with named access to optimal values.
        """
        ...

    def extract(self, sol, name: str) -> np.ndarray:
        """
        Slice a named variable's optimal value out of the raw
        solution vector.

        Args:
            sol: Raw CasADi solution dict (sol['x'])
            name: Variable name as registered with add_variable

        Returns:
            numpy array of optimal values for this variable.
        """
        ...

    def stats(self) -> dict:
        """
        Return solver statistics: return_status, iter_count,
        t_wall_total, etc.
        """
        ...
```

**`SolutionResult`:**

```python
@dataclass
class SolutionResult:
    success: bool                        # True if 'Solve_Succeeded' or 'Solved_To_Acceptable_Level'
    x_opt: Dict[str, np.ndarray]         # Named variable → optimal value (numpy arrays)
    f_opt: float                         # Optimal objective value
    g_opt: np.ndarray                    # Constraint values at optimum
    lam_x: np.ndarray                    # Lagrange multipliers for variable bounds
    lam_g: np.ndarray                    # Lagrange multipliers for constraints
    stats: dict                          # Solver statistics (return_status, iter_count, t_wall_total)
    raw_sol: dict                        # Raw CasADi solution dict for advanced use
    lam_p: Optional[np.ndarray] = None  # Parameter sensitivities; None if no parameters registered

    def __getitem__(self, name: str) -> np.ndarray:
        return self.x_opt[name]          # Shorthand: result['x'] → result.x_opt['x']
```

All numeric fields are converted from `ca.DM` to numpy via `.full().flatten()` (or `float()` for `f_opt`). Callers never interact with CasADi types after `solve()` returns. `lam_p` is `None` rather than an empty array when no parameters are registered, so callers can test `if result.lam_p is not None` cleanly. Nonzero `lam_p` values also require `{'calc_lam_p': True}` in solver options.

`x_opt` values: vector variables produce 1-D numpy arrays; matrix variables (registered with a `(rows, cols)` tuple) produce `(rows, cols)` 2-D arrays, reshaped using column-major order to match the `ca.vec` layout in the decision vector.

**CasADi implementation notes:**
- Variables are MX (not SX). The NLP is assembled in MX space, which allows embedding `ca.Function` call nodes from blocks that define reusable computations (e.g., dynamics models).
- The `expand=True` option can be set in solver options to convert the MX graph to SX at build time for faster evaluation, provided no non-expandable operations (callbacks, SUNDIALS integrators) are present.
- IPOPT auto-detects equality constraints (where `lbg[i] == ubg[i]`).
- CasADi auto-generates sparse Jacobians and Hessians from the symbolic graph via source-code-transformation AD. No manual derivative code is needed.
- For trajectory problems (Phase 5), interleaving state/control variables per timestep produces banded Jacobian structure that sparse linear solvers (MUMPS, MA57) factor efficiently.

**Resolved design decisions:**
- **`SolutionResult` fields:** Finalized — see dataclass above. Added `lam_p` for parameter sensitivities and `__getitem__` for named access. Full specification in `machina/solver/solver_backend_interface_v2.md`.
- **Warm-starting:** Deferred. `SolutionResult` already stores `lam_x` and `lam_g`, so the round-trip data is available. Will be added as optional `lam_x0`/`lam_g0` arguments to `solve()` when needed.
- **Iteration callback:** Deferred. Will be added via the `iteration_callback` solver option (a `ca.Callback` subclass) when convergence debugging is needed on the real flyby problem. Incompatible with `expand=True`.
- **Matrix-shaped variables and parameters:** `add_variable()` and `add_parameter()` now accept `n` as either an `int` (n×1 column vector) or a `(rows, cols)` tuple (matrix). Matrix variables are stored column-major in the flat decision vector via `ca.vec()`; extraction automatically reshapes back to `(rows, cols)`. The internal index maps now store `(start, end, shape)` 3-tuples instead of `(start, end)`, which the visualization tool uses to render dimension labels. See design decision #16.

---

### 3.2 Layer 2 — Function Library

**Purpose:** A catalog of reusable mathematical computations. Each computation is a `ca.Function` object wrapped in a thin `FunctionDescriptor`. Agent types (Layer 3) rent decision variables from the solver backend, call library functions with those variables, and register the resulting MX expressions as costs or constraints.

**Design pattern:** Function-centric. No Block ABC, no declare/build lifecycle. Each reusable computation is a factory function that constructs a `ca.Function` in SX, wraps it in a `FunctionDescriptor`, and returns it. The caller invokes the descriptor with MX arguments to get MX expressions for the NLP.

**Why not a Block class:** The original design used a `Block` ABC with a two-phase declare/build lifecycle, where blocks could register variables and constraints directly with the solver backend. Analysis showed that for all near-term problems (through the flyby goodput optimization), every block is pure math — it transforms inputs to outputs with no structural side effects on the NLP. The Block ABC's declare phase existed for pre-build graph validation, but since functions don't touch the solver backend, building a `ca.Function` is cheap and there's no expensive commitment to undo. The structural decisions (what variables to create, what constraints to register) belong to the agent type, not to the function library.

**Key types:**

- **`FunctionDescriptor`** — Thin wrapper around `ca.Function`. All metadata (input/output names, shapes) extracted automatically from the wrapped function via `name_in()`, `name_out()`, `size_in()`, `size_out()`. Provides a `__call__` method that validates argument shapes before delegating to the underlying function, producing clear error messages instead of CasADi internal errors.

- **`SymbolDescriptor`** — Wraps an MX variable or expression with semantic metadata (name, shape, semantic_type, frame, units). Used by agent types to track what symbols mean and to expose them for path resolution. Validation is enforced (semantic_type must be from a fixed set, shape must match symbol).

- **Function Registry** — A global dictionary mapping string names (e.g., `'cost.sigmoid_goodput'`) to factory callables. Provides `register` decorator, `get` lookup, and `list_by_domain` filtering. Enables YAML-driven assembly in Phase 4.

**Factory functions** follow a standard signature: keyword-only arguments, returns `FunctionDescriptor`. This enables uniform dispatch from YAML via `factory(**params_dict)`. Factories that compose other functions accept `FunctionDescriptor` objects as arguments.

**Example:**

```python
@register('cost.sigmoid_goodput')
def make_sigmoid_goodput(*, k: float, t50: float) -> FunctionDescriptor:
    ttp = ca.SX.sym('ttp')
    goodput = 1.0 / (1.0 + ca.exp(k * (ttp - t50)))
    f = ca.Function('sigmoid_goodput', [ttp], [goodput], ['ttp'], ['goodput'])
    return FunctionDescriptor(f, description=f'Sigmoid goodput (k={k}, t50={t50})')
```

**Usage by an agent type:**

```python
# Agent rents variables from solver backend
ttp = solver.add_variable('flyby/ttp', 1, lb=0, ub=600, initial_guess=60)

# Agent looks up and constructs function from registry
sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=30.0)

# Agent calls function with MX arguments → gets MX expression
goodput_expr = sigmoid(ttp=ttp)

# Agent registers with solver backend
solver.add_cost(-goodput_expr, name='neg_goodput')
```

**Efficiency:** A `ca.Function` is created once by the factory. When called at multiple points in the NLP (e.g., once per weather product), CasADi creates multiple MX call nodes referencing the same underlying Function. AD reuses the function's internal derivative graph across all call sites.

**Implemented library functions (as of v0.8):**
- **Cost functions:** `cost.quadratic`, `cost.rosenbrock`, `cost.sigmoid_goodput`, `cost.aggregate_goodput`, `cost.least_squares`
- **Cost (Phase 3c):** `cost.smooth_coverage` — differentiable sigmoid coverage indicator `1/(1+exp(-k*(ε−ε_min)))` — `machina/blocks/library/cost.py`
- **Constraints:** `constraint.linear`
- **Utilities:** `util.ttp_computation`, `util.sum`, `util.rotate_x/y/z`
- **Transforms (Phase 3a):** `transform.koe_to_mee`, `transform.mee_to_koe`, `transform.mee_to_eci` — `machina/blocks/library/transforms.py`
- **Transforms (Phase 3b):** `transform.stumpff_cs`, `transform.lagrange_coefficients`, `transform.universal_kepler`, `transform.propagate_universal` — `machina/blocks/library/transforms.py`
- **Geometry (Phase 3c):** `geometry.ground_target_eci`, `geometry.elevation_angle` — `machina/blocks/library/geometry.py`

**Planned but not yet implemented:**
- **Dynamics (Phase 5):** `dynamics.keplerian`, `dynamics.j2`, `dynamics.cartesian_two_body`
- **Resource constraints:** `constraint.power_budget`, `constraint.compute_throughput`, `constraint.downlink_capacity` — determined to be structurally identical to `constraint.linear`; no dedicated factories will be added. Agent types wire these directly.

**MEE → ECI velocity formula note:** The velocity formula in `transform.mee_to_eci` was derived from first principles as `v = (dr/dL) · (dL/dt)` rather than transcribed from a reference. Several online sources carry a sign error in the vy component and an incorrect extra `1/s²` factor on vz. The implemented formula (verified against vis-viva and orthogonality checks) is:
```
vx = √(μ/p)/s² · [−(1+α²)(g+sinL) + 2hk(f+cosL)]
vy = √(μ/p)/s² · [(1−α²)(f+cosL) − 2hk(g+sinL)]
vz = √(μ/p)/s² · 2(fh + gk + h·cosL + k·sinL)
```
where α² = h²−k², s² = 1+h²+k². There is **no** additional `1/s²` on the z component.

**Debugging utility — `machina/viz.py`:** A standalone NLP visualization tool (not a framework component). `build_nlp_graph(solver)` produces a NetworkX `DiGraph` from a `SolverBackend`'s registered state (variables, parameters, costs, constraints, and their symbolic dependencies). `draw_nlp_graph(G)` renders it with matplotlib. `export_dot(G, path)` exports to Graphviz DOT format. Does not affect the NLP formulation. The solver does not need to be built or solved before calling it.

**Future extension (Phase 5):** For trajectory optimization, some computations require creating decision variables and constraints as part of their internal logic (e.g., collocation schemes). When this need arises, a `StructuralBlock` class will be introduced that interacts with the solver backend. It will coexist with the function library. This is explicitly deferred.

**Full specification:** `machina/blocks/block_library_interface_v1.md`

---

### 3.3 Layer 3 — Agent Types

**Purpose:** Domain-aware composition classes that translate high-level configurations into CasADi symbolic expressions, constraints, and resolvable namespaces. Agents are the "smart middle layer" between the function library (Layer 2) and the problem compiler (Layer 4).

**Key property:** Agents are fully decoupled from the solver backend. They never call `solver.add_variable()`, `solver.add_parameter()`, `solver.add_constraint()`, or `solver.add_cost()` directly. All interaction with the solver flows through the problem compiler.

**Lifecycle:** Two-phase declare/build.

1. **`declare()`** — Agent examines its config and returns a list of `QuantityDeclaration` objects describing every quantity it needs (decision variables, parameters, any named inputs). No CasADi symbols, no solver interaction. Pure data. The compiler uses this to assign variable vs. parameter roles based on YAML overrides and to generate default-value warnings.

2. **`build(symbols)`** — Compiler passes a dict of `str → ca.MX` mapping declared quantity paths to MX symbols (already created as variables or parameters by the compiler via the solver backend). Agent builds internal expressions using Layer 2 functions, populates its resolvable namespace, and returns a list of `ConstraintDeclaration` objects. The compiler registers these constraints with the solver backend.

**Design approach:** Agent types are Python classes with hardcoded structural knowledge (Option A from design decision #2). Each agent type knows what blocks it composes internally, what quantities it exposes, and how to resolve path queries. Adding a new agent type requires writing a new Python class.

**Rationale for full solver decoupling:** Returning constraints (and having the compiler create all symbols) gives the compiler complete visibility into problem structure. This enables the YAML override system, the default-warning system, the visualization tool, and optional constraint suppression — all without agents needing to cooperate. It also allows quantities like orbital elements to be either decision variables or fixed parameters based on YAML input, without modifying agent code.

**Key data types:**

- **`QuantityDeclaration`** — Describes a quantity the agent needs. Fields: path, shape, semantic_type, default_value, lb, ub, description, units, frame, role (`'flexible'` | `'always_variable'` | `'always_parameter'`), default_role (`'variable'` | `'parameter'`). The `role` field controls whether the YAML can override the quantity's variable/parameter assignment. The `default_role` field determines the fallback when the YAML is silent.

- **`ConstraintDeclaration`** — Describes a constraint the agent produces during build. Fields: expr (ca.MX), lb, ub, name, description. The compiler registers these with the solver backend.

**Key interface:**

```python
class AgentType(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._namespace: dict[str, SymbolDescriptor] = {}
        self._built = False

    @abstractmethod
    def declare(self) -> list[QuantityDeclaration]:
        """Declare all quantities this agent needs. No CasADi, no solver."""
        pass

    @abstractmethod
    def build(self, symbols: dict[str, ca.MX]) -> list[ConstraintDeclaration]:
        """
        Build internal expressions, populate namespace, return constraints.
        symbols maps declared paths to MX objects created by the compiler.
        """
        pass

    def resolve(self, path: str) -> SymbolDescriptor:
        """Look up a path in this agent's namespace (concrete, not overridden)."""
        ...

    def list_paths(self) -> list[str]:
        """Return all resolvable paths (post-build)."""
        return list(self._namespace.keys())
```

**Variable naming convention:** The compiler uses `{agent_name}/{quantity_path}` as the registered name with the solver backend. This name appears in `SolutionResult.x_opt`, the visualization tool, and all debug output. Segments are snake_case, separated by `/`.

**Resolvable namespace:** The agent's `_namespace` is a flat `dict[str, SymbolDescriptor]` populated during `build()`. Contains three categories: raw decision variables (MX symbols assigned the variable role), parameters (MX symbols assigned the parameter role), and computed expressions (MX expressions from Layer 2 function calls). The namespace is shaped by the config — only quantities relevant to the current problem instance are present.

**Agent-to-agent coupling:** Agents do not see each other. Cross-agent references go through the problem compiler, which resolves full paths `agent_name/internal_path` by dispatching to the appropriate agent. Coupling constraints are expressed at the YAML/compiler level.

**Planned agent types:**
- `FlybySatellite` — flyby vehicle with product catalog, resource budgets, optional orbital state, TTP chain, goodput computation
- `Constellation` — N spacecraft with orbital states (deferred until needed)
- `GroundStation` — ground segment with receive capacity, processing (stretch goal)

**Full specification:** `machina/agents/agent_types_interface_v1.md`

---

### 3.4 Layer 4 — Problem Compiler

**Purpose:** Reads the YAML problem definition, instantiates agent types, resolves references in cost functions and constraints, assembles the global CasADi problem, manages initial guesses, invokes the solver, and handles output.

**Responsibilities:**
- YAML parsing and validation
- Agent type instantiation based on `type` field
- Cost function assembly (weighted sum of named cost blocks, with source paths resolved through agent types)
- Constraint assembly (referencing agent quantities by path)
- Initial guess management:
    - User-provided guesses from YAML override everything
    - Missing guesses filled from agent-type suggested defaults
    - **Warning printed for every default used** (e.g., "Using default initial guess for main_constellation/spacecraft_1/sma = 42164.0 km — override in initial_guess section")
- Solver invocation
- Result output:
    - Terminal: final decision variable values, final cost function value
    - File: detailed results dump (format TBD — YAML, JSON, CSV, or combination)

**Does not contain:** Domain knowledge about what a constellation or spacecraft is. The compiler only knows how to ask agents to resolve paths and how to look up block types from the library registry.

**Key interface (sketch):**

```python
class ProblemCompiler:
    def __init__(self, yaml_path: str): ...
    def compile(self) -> SolverBackend:
        """
        Parse YAML, instantiate agents, resolve references,
        assemble the CasADi problem. Returns a ready-to-solve
        solver backend.
        """
        pass
    def solve(self) -> SolutionResult: ...
    def dump_results(self, output_path: str): ...
```

---

### 3.5 Layer 5 — YAML Problem Definition

**Purpose:** User-facing problem description. Describes what agents exist, how they're configured, what's being optimized, and what constraints apply. The YAML is a **bill of materials**, not a declarative math language.

**Philosophy:**
- YAML is descriptive: "I have these things, configured this way"
- No algebra except lightweight cost function composition (naming blocks and listing weights)
- New features require writing Python (new blocks or agent types), not extending YAML syntax
- A user who understands the available agent types and blocks can write a problem without understanding CasADi

**Example structure (illustrative, not finalized):**

```yaml
problem_type: optimize

agents:
  - name: main_constellation
    type: constellation
    count: 4
    dynamics: keplerian
    params:
      altitude_range: [400, 2000]  # km
      inclination_range: [0, 90]   # deg

  - name: flyby_sat
    type: flyby_satellite
    params:
      power_budget: 150            # W
      compute_throughput: 500      # Mbps
      downlink_rate: 100           # Mbps

cost_function:
  type: weighted_sum
  terms:
    - block: aggregate_goodput
      weight: 1.0
      sources:
        ttp: flyby_sat/products/ttp
        weights: flyby_sat/products/importance
    - block: total_delta_v
      weight: -0.01
      source: main_constellation/delta_v

constraints:
  - block: power_budget
    source: flyby_sat/power_consumption
    limit: flyby_sat/params/power_budget
  - block: compute_throughput
    source: flyby_sat/compute_load
    limit: flyby_sat/params/compute_throughput

initial_guess:
  main_constellation/spacecraft_1/sma: 42164.0
  main_constellation/spacecraft_1/ecc: 0.001
  flyby_sat/downlink_rate: 75.0

solver:
  type: ipopt
  max_iter: 1000
  tol: 1e-6

output:
  terminal: true
  file: results/flyby_optimization.json
```

---

## 4. Interface Contracts

### 4.1 SymbolDescriptor

The primary currency exchanged between blocks and agent types for data. Wraps a CasADi MX symbolic variable (or expression) with metadata that enables validation at assembly time.

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

**Design notes:**
- All symbols are MX. The NLP is assembled in MX space to allow embedding `ca.Function` call nodes.
- The `symbol` field may hold either a raw MX variable (from `solver.add_variable()`) or an MX expression (the result of algebraic operations or a Function call). Both are valid MX objects and both work identically in downstream expressions.
- Validation is enforced: `semantic_type` must be from the fixed set, `shape` must match `symbol.shape`. `frame` on non-spatial types issues a warning.
- `semantic_type` is drawn from a small fixed set: `scalar`, `vector`, `trajectory`, `indexed_set`, `time_grid`. New types added only when an existing type genuinely doesn't fit.
- `frame` is relevant for spatial vectors. Non-spatial quantities leave it as `None`.
- `units` is optional and not enforced programmatically. It exists for documentation and debugging.

### 4.2 FunctionDescriptor

Thin wrapper around a `ca.Function`. All metadata is extracted automatically from the wrapped function — no redundant manual specification. Provides a `__call__` method with shape validation for clear error messages.

```python
class FunctionDescriptor:
    function: ca.Function      # The CasADi Function object
    description: str           # What this function computes (factory-provided)
    name: str                  # Extracted from function.name()
    input_names: list[str]     # Extracted from function.name_in(i)
    output_names: list[str]    # Extracted from function.name_out(i)
    input_shapes: list[tuple]  # Extracted from function.size_in(i)
    output_shapes: list[tuple] # Extracted from function.size_out(i)

    def __call__(self, *args, **kwargs):
        """Validated call: checks shapes, then delegates to self.function."""
        ...
```

**Design notes:**
- Only `function` and `description` are provided by the caller. All other fields are derived in the constructor from the `ca.Function`'s own introspection methods (`name()`, `name_in()`, `name_out()`, `size_in()`, `size_out()`).
- The `__call__` wrapper validates argument shapes and names before delegating to the underlying `ca.Function`. Errors reference the function name and port names (e.g., `"Function 'sigmoid_goodput': input 'ttp' expected shape (1, 1), got (3, 1)"`). The wrapper is pure Python logic and does not insert anything into the CasADi symbolic graph — the returned MX expressions are identical to a direct `function()` call.
- A single `FunctionDescriptor` can be called at multiple points in the NLP. Each call creates a new MX call node referencing the same underlying `ca.Function`. CasADi reuses the function's internal derivative graph across all call sites.
- During factory construction (SX level), factories call `descriptor.function()` directly to bypass MX-oriented validation. Direct `.function` access during composition is the intended pattern.

**Lifecycle of a library function:**

1. **Factory call:** A factory function (registered in the function registry) creates SX symbols, builds the SX expression, wraps it in `ca.Function` with named I/O, and returns a `FunctionDescriptor`.
2. **Usage:** An agent type calls the descriptor with MX arguments. The validated call returns MX expressions that the agent registers as costs or constraints with the solver backend.
3. **Composition:** A factory that depends on another function accepts its `FunctionDescriptor` as a keyword argument and calls `.function()` symbolically during SX construction.

### 4.3 QuantityDeclaration

Describes a quantity an agent needs. Returned by `declare()`. The problem compiler uses these to assign variable/parameter roles and to create MX symbols.

```python
@dataclass
class QuantityDeclaration:
    path: str                          # Agent-relative path (e.g., 'orbital/sma')
    shape: tuple                       # (rows, cols), e.g., (1, 1) for scalar
    semantic_type: str                 # From SymbolDescriptor fixed set
    default_value: float | np.ndarray  # Initial guess (if variable) or value (if parameter)
    lb: float | np.ndarray             # Lower bound (meaningful for variables only)
    ub: float | np.ndarray             # Upper bound (meaningful for variables only)
    description: str                   # Human-readable description
    units: str = None                  # Optional, advisory
    frame: str = None                  # Reference frame if applicable
    role: str = 'flexible'             # 'flexible', 'always_variable', 'always_parameter'
    default_role: str = 'variable'     # Fallback when role='flexible' and YAML is silent
```

**Design notes:**
- `role='flexible'` means the YAML (via the problem compiler) can assign this quantity as either a variable or a parameter. `default_role` determines the fallback.
- `role='always_variable'` is for internal optimization DOFs (e.g., per-product resource allocations) that the compiler cannot fix.
- `role='always_parameter'` is for constants the agent uses internally that should never be optimized.
- `default_value` is interpreted based on the assigned role: as initial guess for variables, as the fixed numeric value for parameters.
- Bounds (`lb`, `ub`) are only meaningful when the quantity is assigned the variable role. They are ignored for parameters.

### 4.4 ConstraintDeclaration

Describes a constraint an agent produces during `build()`. The problem compiler registers it with the solver backend.

```python
@dataclass
class ConstraintDeclaration:
    expr: ca.MX        # Constraint expression (scalar or vector)
    lb: float | np.ndarray
    ub: float | np.ndarray
    name: str          # Agent-relative name; compiler prepends agent name
    description: str   # Human-readable description
```

### 4.5 Path Resolution

Agents resolve string paths to `SymbolDescriptor` objects. Paths are hierarchical, using `/` as separator.

**Resolution rules:**
1. The problem compiler splits the path at the first `/` to identify the agent name.
2. The remainder is passed to the agent's `resolve()` method.
3. The agent looks up the path in its flat `_namespace` dict.
4. If the path is invalid, the agent raises a `KeyError` with a descriptive message listing available paths.

**Example:**
- `flyby_sat/products/cmi_conus/ttp` → compiler finds agent `flyby_sat`, calls `resolve('products/cmi_conus/ttp')`, which returns the `SymbolDescriptor` wrapping the computed TTP expression for the CONUS CMI product.

**Namespace contents:** The namespace contains `SymbolDescriptor` entries for raw decision variables, parameters, and computed MX expressions. All are valid targets for path resolution. The `SymbolDescriptor`'s metadata (semantic_type, frame, units) enables validation at assembly time.

**Note:** Path resolution returns `SymbolDescriptor` objects only. `FunctionDescriptor` objects are managed by agent types internally and looked up via the function registry, not through the path system.

---

## 5. Design Decisions Log

| # | Decision | Alternatives Considered | Rationale |
|---|----------|------------------------|-----------|
| 1 | YAML is descriptive ("bill of materials"), not declarative | Declarative YAML with full expression support | Keeps YAML simple; new features are Python, not schema extensions. Inspired by Astrogator paradigm. |
| 2 | Agent types are Python classes with hardcoded structural knowledge (Option A) | Schema-driven agent type definitions (Option B) | Small number of agent types, primary user knows Python, arbitrary logic needed in composition. Option B can layer on later if needed. |
| 3 | Agent types communicate via `SymbolDescriptor` objects for data and `FunctionDescriptor` objects for computations | Raw CasADi symbols without metadata | Raw symbols lose semantic metadata, causing silent errors. Descriptors are lightweight and provide validation. |
| 4 | Decision variables are continuous; mixed-integer deferred | Integrating MINLP solver (e.g., Bonmin) | CasADi handles continuous NLPs natively. Discrete decisions can likely be relaxed or avoided. MINLP adds significant solver complexity. |
| 5 | IPOPT as default solver | SNOPT, custom solvers | IPOPT is free, handles large sparse NLPs well. Other solvers can be swapped in later via solver backend configuration. |
| 6 | Initial guesses: user-provided override agent defaults; warnings printed for every default used | Silent defaults only; require all guesses upfront | Silent defaults risk garbage convergence. Requiring all guesses upfront is hostile for exploration. Warnings balance usability with transparency. |
| 7 | Dynamics models are swappable library functions (different f in ẋ = f(x,u)) | Hardcoded dynamics | Extensibility is a core requirement. The function registry naturally supports this. |
| 8 | ~~Two-phase declare/build pattern for blocks~~ **SUPERSEDED by #13.** | — | — |
| 9 | Solver backend uses `nlpsol`, not `Opti` | CasADi `Opti` interface | `nlpsol` gives explicit control over decision vector layout, bound vectors, and constraint assembly. The solver backend provides the named-variable convenience of `Opti` without the black-box indexing. Closer to the math. |
| 10 | NLP assembled in MX space; SX used inside `ca.Function` objects for reusable computations | Pure SX throughout; pure MX throughout | MX allows embedding `ca.Function` call nodes, keeping the graph compact when a computation is called many times (e.g., dynamics at every collocation point). SX gives the fastest per-element evaluation inside those Functions. Hybrid pattern is CasADi's recommended approach for large problems. |
| 11 | `FunctionDescriptor` extracts all metadata from the wrapped `ca.Function` automatically | Manual specification of input/output names and shapes | `ca.Function` exposes `name_in()`, `name_out()`, `size_in()`, `size_out()`. Extracting automatically eliminates redundancy and the possibility of metadata-function mismatch. |
| 12 | Solver backend owns all MX variables; agent types rent variables, call library functions, and register results | Blocks own their own variables; solver collects at the end | Central ownership gives the backend explicit control over the global decision vector layout and index mapping. Library functions don't need to know about global indexing. Clean separation of concerns. |
| 13 | Layer 2 is a function library (factory functions → `FunctionDescriptor`), not a Block ABC with declare/build lifecycle | Block ABC with two-phase declare/build pattern (original design) | Every near-term computation is pure math with no solver side effects. `ca.Function` is the natural unit of reuse — stateless, composable, efficient. The Block ABC's declare phase existed for pre-build validation, but building a `ca.Function` is cheap and there's nothing to undo. Structural decisions (variable creation, constraint registration) belong to the agent type. The function-centric model enforces statelessness by construction. `StructuralBlock` can be added in Phase 5 if needed for trajectory optimization. |
| 14 | `FunctionDescriptor.__call__` validates shapes before delegating to `ca.Function` | Direct `ca.Function` calls without validation; separate `validate()` method | CasADi's native dimension mismatch errors reference internal indexing and are hard to debug. The `__call__` wrapper adds clear error messages with function and port names. The wrapper is pure Python and does not affect the CasADi symbolic graph. |
| 15 | `SymbolDescriptor` validation is enforced (not advisory) | Advisory validation with warnings | Catching type/shape mismatches early produces clearer errors. The cost of enforcement is negligible. Advisory mode was appropriate during initial design exploration but is no longer needed. |
| 16 | `add_variable`/`add_parameter` accept `int` or `(rows, cols)` tuple; index maps store `(start, end, shape)` 3-tuples | Separate `add_matrix_variable()` method; always require flat size | Agent types need to register matrix-valued variables (e.g., attitude matrices, A/b matrices for least-squares) without reshaping boilerplate. Storing shape in the index map enables the visualization tool and future result extraction to produce correctly-shaped outputs without extra state. |
| 17 | Two-phase declare/build lifecycle for agents | Single-phase build | The declare phase enables the compiler to decide variable vs. parameter roles based on YAML input. Quantities like orbital elements can be either optimized or fixed without modifying agent code. Distinct from the Layer 2 declare/build pattern (#8/#13) which was rejected because Layer 2 functions have no solver side effects. |
| 18 | Agents are fully decoupled from the solver backend | Agents call solver.add_variable() and solver.add_constraint() directly | Full decoupling gives the compiler complete visibility into problem structure. Enables the viz tool, YAML override system, default-warning system, and optional constraint suppression without agents needing to cooperate. |
| 19 | QuantityDeclaration has role/default_role fields for flexible variable/parameter assignment | All quantities are always variables; fixed values use tight bounds | Role flexibility lets the YAML control whether a quantity is optimized or fixed. Avoids polluting the NLP with extraneous equality constraints to pin "variables" to fixed values. |
| 20 | build() returns list[ConstraintDeclaration]; compiler registers all constraints | Agents register constraints directly with solver | Returning constraints gives the compiler full visibility and a single point of control for constraint registration, logging, and optional suppression. |
| 21 | resolve() returns SymbolDescriptor (with metadata), not raw ca.MX | Return raw ca.MX for simplicity | Metadata (shape, semantic_type, frame, units) enables catching errors like plugging a position vector into a velocity port at assembly time. Debugging value outweighs wrapper cost. |
| 22 | Agent namespace is a flat dict[str, SymbolDescriptor] with slash-separated keys | Nested tree structure | Flat dict is simpler to implement, query, and debug. Hierarchical structure is a naming convention only. |
| 23 | Agent config determines agent structure; namespace reflects config | Fixed agent structure with all quantities always present | Config-driven structure avoids creating irrelevant quantities. Reduces NLP size and keeps namespaces clean. |
| 24 | Kepler's equation handled as ca.rootfinder-based Layer 2 function | Manual Newton iterations; external propagator | ca.rootfinder wraps an implicit solve inside a ca.Function, enabling differentiation via IFT. Transparent to the agent. No dynamics or propagation needed for static orbital geometry. |
| 25 | Modified equinoctial elements (MEE) as internal orbital state representation | Classical KOE as decision variables | Classical KOE have singularities at e=0 (circular) and i=0 (equatorial) that cause Jacobian singularity in gradient-based optimization. MEE are continuous and differentiable across these boundaries. Classical KOE used for user I/O via transform functions. |
| 26 | Universal variable formulation for Kepler solver | Separate elliptic/hyperbolic/parabolic solvers | Universal variables + Stumpff functions handle all orbit types (elliptic, parabolic, hyperbolic) in a single formulation. Enables the optimizer to vary eccentricity across regime boundaries without solver failure. |
| 27 | MEE→ECI velocity derived from `v = (dr/dL)·(dL/dt)`, not transcribed from reference | Transcribe from Schaub & Junkins Appendix F | Multiple online sources and secondary references carry a sign error in vy and a spurious extra `1/s²` on vz. Deriving from `dL/dt = √(μ/p³)·w²` and computing `dr/dL` analytically is unambiguous and was verified against vis-viva, orthogonality, and reference orbit tests. |
| 28 | `QuantityDeclaration`, `ConstraintDeclaration`, and `AgentType` placed in `machina/agents/agent_type.py` (single file) rather than split across `declarations.py` and `base.py` | Separate files per class | All three are tightly coupled (declarations are produced/consumed as part of the AgentType lifecycle). Single-file placement keeps the public API import simple and avoids circular-import risk. |
| 29 | `mu` is a factory parameter (baked in) for all Phase 3b propagation functions | `mu` as a runtime function argument | Consistent with the `mee_to_eci` pattern established in Phase 3a. `mu` is a physical constant for a given problem, not an optimization variable. Baking it in at factory time keeps the rootfinder's parameter vector smaller and the function signatures cleaner. |
| 30 | Stumpff Taylor threshold `EPS = 1e-4`; `TINY = 1e-32` denominator guard | Various | Taylor series with 4 terms is accurate to double precision for |ψ|<0.1; `1e-4` leaves a comfortable margin. `TINY = 1e-32` prevents NaN in the non-selected `ca.if_else` branch without affecting the selected branch's precision. |
| 31 | `transform.lagrange_coefficients` computes α internally from r0, v0 rather than taking α as an input | Accept pre-computed α as input | The interface spec in A.4 listed α as an input, but computing it internally from r0 and v0 keeps the public signature cleaner and avoids the caller needing to pre-compute and pass a derived quantity. The function also needs r0 and v0 anyway (to compute r_mag and the final position vector). |
| 32 | MEE altitude constraints use squared form: `(p-R_min)² - R_min²·(f²+g²) ≥ 0` | Direct form: `p/(1+e) - R_earth ≥ h_min` where `e = sqrt(f²+g²)` | `sqrt(f²+g²)` has zero gradient at f=g=0 (circular orbit). In constraint expressions this causes a degenerate constraint Jacobian at the circular orbit — LICQ fails, IPOPT declares infeasibility. The direct form is algebraically equivalent but the squared form is smooth everywhere including at e=0, with a well-defined Jacobian. **This is a load-bearing design choice for all MEE-based agents** — not a workaround. Any future agent that constrains perigee or apogee altitude must use this form. |
| 33 | MEE namespace derived quantities (`orbital/ecc`, `orbital/sma`, `orbital/period`) are display-only; never wire them into NLP objectives or constraints | Use them as general-purpose MX expressions | These quantities contain `sqrt(f²+g²)`, which has a singular gradient at e=0. They are safe to read from `result.x_opt` after solving. If you were to pass `agent.resolve('orbital/ecc').symbol` into another agent's constraint expression, you would pull the singularity back into the NLP Jacobian. Future agent-to-agent coupling must stay in f/g space (raw MEE components), not derived eccentricity. |

---

## 6. Phase Plan

### Phase 1 — Foundation and Proof of Concept ✓ COMPLETE

**Goal:** Prove the function-centric Layer 2 design works end to end with Layer 1 / CasADi / IPOPT.

**Tasks:**
- [x] Implement `SolverBackend` wrapper around CasADi `nlpsol` (variable registration, constraint/cost collection, build, solve, extract)
- [x] Implement `SolutionResult` dataclass
- [x] Implement `SymbolDescriptor` dataclass with enforced validation — `machina/blocks/descriptor.py`
- [x] Implement `FunctionDescriptor` class with `__call__` validation and `__repr__` — `machina/blocks/descriptor.py`
- [x] Implement function registry module (`register`, `get`, `list_registered`, `list_by_domain`) — `machina/blocks/registry.py`
- [x] Write toy factory functions (`cost.quadratic`, `cost.rosenbrock`, `constraint.linear`) — `machina/blocks/library/`
- [x] Tests: registry lookup → factory → `FunctionDescriptor` → solver pipeline; Rosenbrock convergence proves AD works end-to-end — `tests/test_blocks_phase1.py` (48 tests)
- [x] Verify shape validation produces clear error messages
- [x] Verify registry rejects duplicate names

**No YAML, no agent types.** Pure Python wiring of functions to solver.

### Phase 2 — Function Library for the Flyby Problem ✓ FUNCTION LIBRARY COMPLETE — real-problem integration deferred to Layer 3

**Goal:** Solve the actual flyby goodput optimization problem.

**Tasks:**
- [x] Implement `cost.sigmoid_goodput` factory — `machina/blocks/library/cost.py`
- [x] Implement `cost.aggregate_goodput` factory (with function composition) — `machina/blocks/library/cost.py`
- [x] Implement `util.ttp_computation` factory (assembles TTP from component delays) — `machina/blocks/library/util.py`
- [x] Resource constraint factories — **decision: not implemented.** `constraint.power_budget`, `constraint.compute_throughput`, and `constraint.downlink_capacity` are all `c^T x <= limit`, structurally identical to `constraint.linear`. Domain-specific wiring belongs in the agent type (Layer 3).
- [x] Tests: numerical correctness (sigmoid, aggregate), TTP computation, flyby toy problem end-to-end — `tests/test_blocks_phase2.py` (30 tests)
- [x] Examples: `examples/flyby_goodput.py` (3-product goodput toy problem, full pipeline), `examples/rosenbrock_blocks.py` (Layer 2 wiring pattern), `examples/least_squares.py`
- [ ] Implement flyby geometry / contact window modeling — deferred; geometry is computed externally and provided as fixed obs_delay parameters
- [ ] Write Excel data ingestion to load baseline product data — deferred to integration work
- [ ] Validate results against baseline spreadsheet — ~~gate for moving to Layer 3~~ will be validated by hand

**Still no YAML or agent types.** Manual Python assembly, but solving the real problem.

### Phase 2.5 — Debugging Utility ✓ COMPLETE (ungated)

**Goal:** Provide a visualization tool for inspecting NLP wiring during development.

- [x] `machina/viz.py`: `build_nlp_graph(solver)` → NetworkX `DiGraph`; `draw_nlp_graph(G)` → matplotlib; `export_dot(G, path)` → Graphviz DOT. Node types: variable, parameter, cost term, equality constraint, inequality constraint. Edges inferred via `ca.symvar()`. Integrated into `examples/rosenbrock_blocks.py`.

### Phase 3 — Agent Types and Path Resolution ← IN PROGRESS

**Goal:** Replace manual block wiring with domain-aware agent types that are fully decoupled from the solver backend. Validate with a coverage optimization example problem. Full specification in `machina/agents/agent_type_interface_v1.md`.

**Phase 3a — Core Transforms and Agent Base ✓ COMPLETE**
- [x] Implement `QuantityDeclaration` and `ConstraintDeclaration` dataclasses — `machina/agents/agent_type.py`
- [x] Implement `AgentType` base class with declare/build/resolve — `machina/agents/agent_type.py`
- [x] Implement `transform.koe_to_mee` and `transform.mee_to_koe` factories — `machina/blocks/library/transforms.py`
- [x] Implement `transform.mee_to_eci` factory — `machina/blocks/library/transforms.py`
- [x] Implement compiler stub (declare → assign → build → register lifecycle, Python-driven) — `machina/compiler/compiler_stub.py`
- [x] Tests: element transforms roundtrip, MEE → ECI against known orbits — `tests/test_blocks_phase3a.py` (76 tests)

**Phase 3b — Universal Kepler Solver ✓ COMPLETE**
- [x] Implement `transform.stumpff_cs` factory (`ca.if_else` + Taylor expansion near ψ=0) — `machina/blocks/library/transforms.py`
- [x] Implement `transform.universal_kepler` factory (`ca.rootfinder`) — `machina/blocks/library/transforms.py`
- [x] Implement `transform.lagrange_coefficients` and `transform.propagate_universal` — `machina/blocks/library/transforms.py`
- [x] Tests: 38 tests across 4 classes — `tests/test_blocks_phase3b.py`
- [x] Math reference document: `universal_kepler_math_v1.md`

**Phase 3b implementation notes:**
- `mu` is a factory parameter (baked in), not a runtime function argument — consistent with `mee_to_eci` pattern.
- `transform.universal_kepler` uses a stacked 4-element p-vector `[r₀_mag, σ₀, α, Δt]` as the rootfinder parameter. Pre-computing scalars reduces rootfinder complexity.
- Initial guess: `χ₀ = √μ·Δt/r₀` (exact for circular orbits). Negative for Δt<0 (backward propagation).
- **`expand=True` is compatible with `ca.rootfinder`** in the tested CasADi version. The `test_expand_true_compatibility` test passes.
- **Stumpff Taylor threshold:** `EPS = 1e-4`. Four terms are sufficient for double precision for |ψ|<0.1; 1e-4 leaves a comfortable margin.
- **TINY guard:** `1e-32` in all denominators of non-selected `ca.if_else` branches prevents NaN without affecting the selected branch.
- **Correct Stumpff identity:** The planning note `ψ·S(ψ)+C(ψ)=1/2` was incorrect. The correct recursion identity is `1 − ψ·C(ψ) = cos(√ψ)` (elliptic) / `cosh(√(−ψ))` (hyperbolic). This is tested in `test_identity_one_minus_psi_C`.
- **`transform.lagrange_coefficients` actual signature:** `r0(3,1), v0(3,1), chi(1,1), dt(1,1) → F, G, Fdot, Gdot`. Computes α internally; does not take α as an input.

**Note on ordering:** Phase 3b is not required to unblock Phase 3c. `SingleSatCoverage` uses static orbital geometry — MEE elements as decision variables, true longitude L sampled at fixed points — so `transform.mee_to_eci` (already implemented) is sufficient. Phase 3b (dynamic propagation) is needed before Phase 3d when the flyby agent requires orbital state evolution. The recommended sequence is 3c → 3b → 3d → 3e, or 3b → 3c → 3d → 3e if propagation tests are prioritized.

**Phase 3c — Coverage Geometry and Example Agent ✓ COMPLETE**
- [x] Implement `geometry.ground_target_eci` and `geometry.elevation_angle` factories — `machina/blocks/library/geometry.py`
- [x] Implement `cost.smooth_coverage` factory — `machina/blocks/library/cost.py`
- [x] Implement `SingleSatCoverage` agent type — `machina/agents/single_sat_coverage.py`
- [x] Tests: 35 Layer 2 tests (geometry + smooth_coverage) — `tests/test_blocks_phase3c.py`
- [x] Tests: 36 agent lifecycle + end-to-end tests — `tests/test_agents_phase3c.py`
- [x] Example: `examples/coverage_optimization.py` — elevation profile, optimization, inclination sweep
- [x] Solve coverage optimization end-to-end, validate against analytical expectations

**Phase 3c implementation notes:**
- `SingleSatCoverage` declares 9 quantities: 5 orbital MEE elements (flexible/variable), 4 always_parameter (sample_points/L, target/position, coverage/min_elevation, coverage/sigmoid_k). The `sample_points/L` parameter is internal to build() and not exposed in the namespace.
- **Circular orbit Jacobian singularity:** `d/df[sqrt(f²+g²)] = f/sqrt(f²+g²)` is 0/0 at f=g=0. Mitigated by: (a) `ca.fmax(f²+g², TINY)` guard inside `ca.sqrt()` in derived quantities, (b) default initial guess f=0.01 (not 0.0).
- **Squared altitude constraint formulation:** Original constraints `p/(1+e) >= R_min` contain `sqrt(f²+g²)` which has zero gradient at circular orbit. Reformulated as `(p-R_min)² - R_min²*(f²+g²) >= 0` — algebraically equivalent, smooth everywhere, no sqrt. Same for apogee. Constraint `lb=0, ub=inf`.
- **h,k bounds [-1.5, 1.5]:** Covers inclinations 0–123°. Wider bounds (e.g., ±3) allow near-retrograde inclinations where IPOPT's Hessian becomes ill-conditioned and produces NaN.
- **IPOPT convergence:** Use `acceptable_tol=1e-2, acceptable_iter=3` for reliable convergence. At the optimal solution (near circular orbit), the squared altitude constraint Jacobian is degenerate in f,g (zero gradient at e=0), causing IPOPT's multiplier computation to return NaN at strict tolerance. With `acceptable_tol`, IPOPT stops before reaching the degenerate region.
- **Snapshot ECI geometry:** The coverage model holds the target fixed in ECI (no Earth rotation). RAAN determines which orbit passes over a fixed-ECI target. For Washington DC, RAAN≈240° gives ~65° max elevation for an ISS-like (i=51.6°, 500 km) orbit. RAAN=0° gives max elevation of −38° (never visible in snapshot geometry).
- **N sample points:** N=24 (every 15° of true longitude) is sufficient for smooth optimization and reliable IPOPT convergence. Larger N (N=36+) can cause IPOPT to take more iterations into the numerically problematic region near optimal.

**Phase 3d — FlybySatellite Agent:**
- [ ] Implement `FlybySatellite` agent type — `machina/agents/flyby_satellite.py`
- [ ] Port Phase 2 examples to agent framework via compiler stub, verify identical results
- [ ] Add orbital state path with link budget functions
- [ ] Integration test: flyby goodput optimization with orbital freedom

**Phase 3e — Consolidation:**
- [ ] Extract shared orbital utilities into composition module (not inheritance)
- [ ] Update viz tool if needed
- [ ] Full Layer 3 test suite — `tests/test_agents_phase3.py`
- [ ] Update design documents

### Phase 4 — Problem Compiler and YAML

**Goal:** Full YAML-driven problem definition and solution.

**Tasks:**
- [ ] Define YAML schema (document the expected structure and valid fields)
- [ ] Implement YAML parser and validator
- [ ] Implement `ProblemCompiler` class
- [ ] Implement cost function assembly (weighted sum of named blocks with source resolution)
- [ ] Implement constraint assembly via path references
- [ ] Implement initial guess management with default-warning system
- [ ] Implement terminal output (decision variable values, cost function value)
- [ ] Implement results file dump (JSON or YAML)
- [ ] Solve the flyby problem entirely from a YAML file
- [ ] Write user-facing documentation for the YAML format

### Phase 5 — Extensibility Toward OCPs (Future)

**Goal:** Extend framework to trajectory optimization and optimal control.

**Tasks:**
- [ ] Implement collocation-based dynamics block (e.g., direct collocation for Keplerian propagation)
- [ ] Implement mission segment concept with boundary continuity constraints
- [ ] Implement maneuver blocks (impulsive delta-v)
- [ ] Implement segment sequencing in agent types
- [ ] Solve at least one trajectory optimization test case end to end
- [ ] Evaluate whether MHE/OCP patterns fit the block/agent architecture or need extensions

---

## 7. Open Questions

1. ~~**SolverBackend encapsulation:**~~ **RESOLVED.** Backend fully encapsulates `nlpsol`. Agent types are fully decoupled — they never interact with the solver backend directly. The compiler mediates all solver access.

2. ~~**Result packaging:**~~ **RESOLVED.** `SolutionResult` fully specified in Layer 1 interface spec v2.

3. **Output file format:** JSON, YAML, CSV, or multiple? JSON is easiest to parse programmatically. CSV is useful for tabular results. YAML is consistent with the input format. Defer decision until Phase 4.

4. ~~**SymbolDescriptor validation strictness:**~~ **RESOLVED.** Enforced from day one. `semantic_type` must be from a fixed set, `shape` must match `symbol.shape`.

5. **Time grid management:** For trajectory problems (Phase 5), who owns the time grid — the dynamics function, the agent type, or the solver backend? How do multiple functions share a common time discretization?

6. ~~**Block registry:**~~ **RESOLVED.** Function registry: a global dictionary mapping `'domain.function_name'` strings to factory callables, with a `@register` decorator. Used by the problem compiler in Phase 4 for YAML-driven lookup.

7. **GUI/interface for YAML generation:** Deferred, but worth noting as a future usability feature. A tool that presents available agent types, required parameters, and decision variables, then produces a valid YAML. The `declare()` manifest provides the data needed to generate this automatically.

8. ~~**Function namespace scoping:**~~ **RESOLVED.** `ca.Function` objects are stateless. A single function created by a factory is shared across all agent instances that call it. Each call creates a new MX call node referencing the same Function — CasADi reuses the internal derivative graph.

9. ~~**`expand=True` as default:**~~ **PARTIALLY RESOLVED.** `ca.rootfinder` is compatible with `expand=True` in the tested CasADi version (confirmed by `test_expand_true_compatibility` in Phase 3b). The open question remains: should the solver backend *default* to `expand=True`? Still lean: default True, disable when `ca.Callback` is attached. Defer the backend default change until Phase 3c/3d when a real NLP with orbital elements needs it.

10. **Variable scaling:** CasADi/IPOPT works best when variables are in the 0.01–100 range. Orbital mechanics variables span many orders of magnitude (km, rad/s, kg). Should the solver backend handle scaling automatically (user provides nominal values), or should agent types be responsible for their own scaling? **Phase 3c update:** `SingleSatCoverage` mixes `p` (~6000–8000 km) with `f,g,h,k` (dimensionless, ~0–1.5). IPOPT converges acceptably without explicit scaling using `ipopt.acceptable_tol=1e-2`. Gradient-based NLP scaling (`nlp_scaling_method='gradient-based'`) was tried but worsened convergence. Defer explicit scaling until Phase 3d when the altitude/velocity combination in flyby optimization is tested.

11. **VP effect reduction models.** The functional relationship between resource allocation and TTP component reduction needs empirical models (from image processing compute benchmarks). The Layer 2 factories for these models will be specified when the data is available.

12. **Parameter value delivery at solve time.** When the compiler assigns a quantity as a parameter, the numeric value (from config default or YAML override) must reach the solver at solve time via `p_val`. The compiler must track which parameter symbols map to which numeric values. Layer 4 implementation detail.

13. **Constellation agent type.** Not specified. Will follow the same `AgentType` base class. May share orbital state utilities with `FlybySatellite` via composition (shared utility functions or mixins), not inheritance.

---

## 8. Reference Material

- **CasADi documentation:** https://web.casadi.org/docs/
- **CasADi `nlpsol` API reference:** https://web.casadi.org/api/internal/d4/d89/group__nlpsol.html
- **CasADi paper (Andersson et al.):** https://optimization-online.org/wp-content/uploads/2018/01/6420.pdf
- **IPOPT documentation:** https://coin-or.github.io/Ipopt/
- **IPOPT options reference:** https://coin-or.github.io/Ipopt/OPTIONS.html
- **GOES-R Ground Segment FPS Appendix G (Mode 6):** Baseline VAGL values
- **Baseline data spreadsheet:** Excel workbook with system constants, instrument specs, product baselines, and planned goodput function coefficients
- **Prior code reference:** Original `Component` base class and `SimpleGravityForce` implementation (see conversation history for source)
