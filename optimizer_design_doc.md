# Flyby Satellite Mission Optimizer — Design Document

**Status:** Draft v0.2
**Last Updated:** 2026-03-29
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
- **Build:** Calls `ca.vertcat()` on accumulated variable and constraint lists, assembles the NLP dict `{'x': x, 'f': f, 'g': g, 'p': p}`, and creates the `ca.nlpsol` solver object.
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
        # Internal storage — populated during build phase
        self._w = []        # List of MX variable symbols
        self._w0 = []       # Initial guess values (flat list)
        self._lbw = []      # Lower bounds on variables (flat list)
        self._ubw = []      # Upper bounds on variables (flat list)
        self._g = []        # List of MX constraint expressions
        self._lbg = []      # Lower bounds on constraints (flat list)
        self._ubg = []      # Upper bounds on constraints (flat list)
        self._p = []        # List of MX parameter symbols
        self._J = 0         # Accumulated cost expression (MX scalar)
        self._offset = 0    # Current position in global decision vector
        self._var_map = {}  # name → (start_index, end_index)
        self._solver = None # Created at build() time

    def add_variable(self, name: str, n: int,
                     lb=-np.inf, ub=np.inf,
                     initial_guess=0.0) -> ca.MX:
        """
        Create a decision variable and register it in the global
        decision vector.

        Args:
            name: Unique name for this variable (used for result extraction)
            n: Number of elements (scalar = 1, 3-vector = 3, etc.)
            lb: Lower bound (scalar broadcasts to all elements)
            ub: Upper bound (scalar broadcasts to all elements)
            initial_guess: Starting value for solver (scalar broadcasts)

        Returns:
            MX symbolic variable — use this to build expressions.
        """
        ...

    def add_parameter(self, name: str, n: int) -> ca.MX:
        """
        Create a fixed parameter (not optimized, value set at solve time).
        Enables re-solving with different data without rebuilding solver.

        Returns:
            MX symbolic parameter.
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

**`SolutionResult` (sketch):**

```python
@dataclass
class SolutionResult:
    success: bool                  # True if solver converged
    x_opt: Dict[str, np.ndarray]   # Named variable → optimal value
    f_opt: float                   # Optimal objective value
    g_opt: np.ndarray              # Constraint values at optimum
    lam_x: np.ndarray              # Lagrange multipliers for variable bounds
    lam_g: np.ndarray              # Lagrange multipliers for constraints
    stats: dict                    # Solver statistics (iterations, wall time, status)
    raw_sol: dict                  # Raw CasADi solution dict for advanced use
```

**CasADi implementation notes:**
- Variables are MX (not SX). The NLP is assembled in MX space, which allows embedding `ca.Function` call nodes from blocks that define reusable computations (e.g., dynamics models).
- The `expand=True` option can be set in solver options to convert the MX graph to SX at build time for faster evaluation, provided no non-expandable operations (callbacks, SUNDIALS integrators) are present.
- IPOPT auto-detects equality constraints (where `lbg[i] == ubg[i]`).
- CasADi auto-generates sparse Jacobians and Hessians from the symbolic graph via source-code-transformation AD. No manual derivative code is needed.
- For trajectory problems (Phase 5), interleaving state/control variables per timestep produces banded Jacobian structure that sparse linear solvers (MUMPS, MA57) factor efficiently.

**Open questions:**
- Exact `SolutionResult` fields — the sketch above is a starting point.
- Whether to support warm-starting in Phase 1 or defer. (Lean: defer, easy to add later by accepting `lam_x0`/`lam_g0` in `solve()`.)
- Whether to include an iteration callback mechanism in Phase 1 or defer. (Lean: defer, add when needed for convergence debugging.)

---

### 3.2 Layer 2 — Block Library

**Purpose:** Reusable mathematical building blocks. Each block encapsulates a specific computation (a dynamics model, a cost function, a constraint set, a coordinate transform, etc.).

**Design pattern:** Two-phase declare/build.

- **Declare phase:** Block declares its input ports, output ports, and function dependencies/productions with metadata. No CasADi symbols are created. The assembly layer inspects the full graph structure — including function dependency edges — before any symbolic math happens.
- **Build phase:** Block receives resolved input symbols (as `SymbolDescriptor` objects) and resolved function dependencies (as `FunctionDescriptor` objects), creates its CasADi expressions, registers variables/constraints/cost terms with the solver backend, and produces output descriptors and/or function descriptors.

**Expression blocks vs. Function-producing blocks:**

Not every block needs to create a `ca.Function` object. The distinction:

- **Expression blocks** receive MX symbols, do algebraic math on them (add, multiply, apply sigmoid, etc.), and register the resulting MX expressions as costs or constraints. The expressions become nodes in the global MX graph. Examples: `SigmoidGoodput`, `PowerBudget`, `AggregateGoodput`. These are the majority of blocks for static NLP problems.

- **Function-producing blocks** define a reusable computation as a `ca.Function` with named inputs and outputs. Other blocks that need this computation declare it as a function dependency and receive the pre-built Function at build time, calling it with their own MX symbols. This avoids rebuilding the same symbolic math at every call site, keeping the graph compact. Examples: `KeplerianDynamics`, `FrameTransform`, `QuaternionToRotationMatrix`.

The solver backend does not care about this distinction — it receives MX expressions for objectives and constraints regardless of how they were constructed (raw algebra or Function call nodes).

**Key interface:**

```python
class Block(ABC):
    def __init__(self, name: str, params: dict = None):
        """
        Args:
            name: Instance name (used for scoping symbols)
            params: Static configuration (not optimized). E.g., steepness
                    coefficients, fixed physical constants, mode flags.
        """
        ...

    @abstractmethod
    def declare(self) -> BlockDeclaration:
        """
        Declare this block's interface. No CasADi symbols created here.

        Returns:
            BlockDeclaration with input ports, output ports,
            required functions, and produced functions.
        """
        pass

    @abstractmethod
    def build(self, inputs: Dict[str, SymbolDescriptor],
              functions: Dict[str, FunctionDescriptor],
              solver: SolverBackend) -> BlockOutputs:
        """
        Build the block's CasADi expressions.

        Args:
            inputs: Resolved input descriptors, keyed by port name.
            functions: Resolved function descriptors, keyed by declared
                       function dependency name.
            solver: Solver backend for registering variables, constraints,
                    and cost terms.

        Returns:
            BlockOutputs containing output symbol descriptors and/or
            produced function descriptors.
        """
        pass
```

**Declaration types:**

```python
@dataclass
class PortDeclaration:
    name: str
    direction: str          # 'input' or 'output'
    shape: tuple            # e.g., (3,), (4,4), (1,)
    semantic_type: str      # 'scalar', 'vector', 'trajectory',
                            # 'indexed_set', 'time_grid'
    frame: str = None       # 'ECI', 'LVLH', 'body', None for non-spatial
    description: str = ''

@dataclass
class FunctionDependency:
    name: str               # Name of the function needed (e.g., 'keplerian_dynamics')
    input_names: List[str]  # Expected input argument names
    output_names: List[str] # Expected output argument names
    description: str = ''

@dataclass
class FunctionProduction:
    name: str               # Name of the function produced
    input_names: List[str]  # Input argument names of the produced Function
    output_names: List[str] # Output argument names of the produced Function
    description: str = ''

@dataclass
class BlockDeclaration:
    inputs: List[PortDeclaration]
    outputs: List[PortDeclaration]
    required_functions: List[FunctionDependency] = field(default_factory=list)
    produced_functions: List[FunctionProduction] = field(default_factory=list)

@dataclass
class BlockOutputs:
    symbols: Dict[str, SymbolDescriptor] = field(default_factory=dict)
    functions: Dict[str, FunctionDescriptor] = field(default_factory=dict)
```

**Assembly layer behavior for functions:**

1. During the declare phase, the assembly layer collects all function productions and dependencies across all blocks.
2. It matches producers to consumers by function name, verifying that input/output argument names are compatible.
3. It topologically sorts blocks so that function producers build before consumers.
4. During the build phase, produced `FunctionDescriptor` objects are passed to consuming blocks via their `functions` argument.
5. If a function dependency has no producer, the assembly layer raises a descriptive error.
6. If multiple blocks produce a function with the same name, the assembly layer raises an ambiguity error.

**Example blocks (not exhaustive):**
- **Function producers (dynamics/utilities):** `KeplerianDynamics`, `J2Dynamics`, `CartesianTwoBody`, `FrameTransform`, `QuaternionToRotationMatrix`
- **Expression blocks (cost/constraints):** `SigmoidGoodput`, `AggregateGoodput`, `DeltaVCost`, `PowerBudget`, `ComputeThroughput`, `DownlinkCapacity`
- **Hybrid (produce functions AND register constraints):** `CollocationDynamics` (Phase 5 — produces integrator Function, also registers defect constraints)

---

### 3.3 Layer 3 — Agent Types

**Purpose:** Domain-aware composition classes that know how to assemble blocks into meaningful mission subsystems. These are the "smart middle layer" that translates high-level YAML descriptions into specific block instantiations and internal wiring.

**Design approach:** Option A — agent types are Python classes with hardcoded structural knowledge. Each agent type knows:
- What blocks it composes internally
- What quantities it exposes by name for external reference
- How to resolve path queries (e.g., `my_constellation/spacecraft_1/sma`)

Adding a new agent type requires writing a new Python class. The YAML vocabulary grows by adding agent types and blocks in Python, not by complicating the YAML schema.

**Rationale for Option A over schema-driven (Option B):** The primary user (the developer) knows Python. The number of agent types is small. Hardcoded classes allow arbitrary logic in composition (conditional block selection based on configuration). A schema-driven approach would require building a schema interpreter — essentially a DSL on top of a DSL — with high implementation cost and low near-term payoff. If the number of agent types grows large and follows repetitive patterns, Option B can be layered on later.

**Key interface (sketch):**

```python
class AgentType(ABC):
    def __init__(self, name: str, config: dict): ...

    @abstractmethod
    def declare(self) -> AgentDeclaration:
        """
        Declare this agent's external interface: what quantities it
        exposes, what parameters it requires, what decision variables
        it will create.
        """
        pass

    @abstractmethod
    def build(self, solver: SolverBackend) -> None:
        """
        Instantiate internal blocks, wire them, register everything
        with the solver backend.
        """
        pass

    @abstractmethod
    def resolve(self, path: str) -> SymbolDescriptor:
        """
        Resolve a dotted path (e.g., 'spacecraft_1/sma') to a
        SymbolDescriptor for the corresponding CasADi symbol.
        """
        pass

    @abstractmethod
    def get_decision_variables(self) -> List[DecisionVariableInfo]:
        """
        Return list of all decision variables this agent creates,
        with names, descriptions, and suggested default initial guesses.
        """
        pass
```

**Planned agent types:**
- `Constellation` — N spacecraft with orbital states, dynamics, coverage
- `FlybySatellite` — flyby vehicle with resource budgets, processing chain, observation windows
- `GroundStation` — ground segment with downlink, processing (stretch goal)

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
- All symbols are MX. The NLP is assembled in MX space to allow embedding `ca.Function` call nodes from function-producing blocks.
- The `symbol` field may hold either a raw MX variable (from `solver.add_variable()`) or an MX expression (the result of algebraic operations or a Function call). Both are valid MX objects and both work identically in downstream expressions.
- Metadata is advisory at first — the framework warns on mismatches but does not block. Strictness can increase as the type system matures.
- `semantic_type` is drawn from a small fixed set: `scalar`, `vector`, `trajectory`, `indexed_set`, `time_grid`. New types added only when an existing type genuinely doesn't fit.
- `frame` is relevant for spatial vectors. Non-spatial quantities leave it as `None`.
- `units` is optional and not enforced programmatically. It exists for documentation and debugging.

### 4.2 FunctionDescriptor

The currency exchanged between blocks for reusable computations. Wraps a `ca.Function` object with metadata about its interface, enabling the assembly layer to validate producer-consumer compatibility.

```python
@dataclass
class FunctionDescriptor:
    function: ca.Function      # The CasADi Function object
    name: str                  # Human-readable name (e.g., 'keplerian_dynamics')
    input_names: List[str]     # Ordered list of input argument names
    output_names: List[str]    # Ordered list of output argument names
    input_shapes: List[tuple]  # Shape of each input
    output_shapes: List[tuple] # Shape of each output
    description: str = ''      # What this function computes
```

**Design notes:**
- `ca.Function` objects are created during the build phase of function-producing blocks using SX internals, then wrapped in this descriptor.
- Consuming blocks call `descriptor.function(mx_arg1, mx_arg2, ...)` during their own build phase. The call returns MX expression(s) — nodes in the global graph that CasADi differentiates through automatically.
- The assembly layer validates that a consumer's declared `FunctionDependency` (input/output names) matches the producer's `FunctionDescriptor` before passing it through.
- A single `FunctionDescriptor` can be passed to multiple consuming blocks. The `ca.Function` is defined once; each call site creates a new MX call node referencing the same underlying Function. This is how graph bloat is avoided.

**Lifecycle of a function-producing block:**

1. **Declare:** Block declares `produced_functions: [FunctionProduction(name='keplerian_dynamics', ...)]`.
2. **Build:** Block creates SX symbols, builds the SX expression for the dynamics, wraps it in `ca.Function('keplerian_dynamics', [x_sx, u_sx], [xdot_sx], ['x', 'u'], ['xdot'])`, then returns it as a `FunctionDescriptor` in its `BlockOutputs.functions`.
3. **Consumption:** Any block that declared `required_functions: [FunctionDependency(name='keplerian_dynamics', ...)]` receives this `FunctionDescriptor` in its `build()` call's `functions` argument, and calls `functions['keplerian_dynamics'].function(X_mx, U_mx)` to get MX expressions.

### 4.3 Path Resolution

Agents resolve string paths to `SymbolDescriptor` objects. Paths are hierarchical, using `/` as separator.

**Resolution rules:**
1. The problem compiler splits the path at the first `/` to identify the agent name.
2. The remainder is passed to the agent's `resolve()` method.
3. The agent walks its internal structure to find the referenced quantity.
4. If the path is invalid, the agent raises a descriptive error.

**Example:**
- `main_constellation/spacecraft_1/sma` → compiler finds agent `main_constellation`, calls `resolve('spacecraft_1/sma')`, which returns the SMA descriptor for spacecraft 1.

**Note:** Path resolution returns `SymbolDescriptor` objects only — it is used by the problem compiler to wire cost functions and constraints to agent quantities. `FunctionDescriptor` objects are resolved separately during the block assembly process (within agent types), not through the path system.

---

## 5. Design Decisions Log

| # | Decision | Alternatives Considered | Rationale |
|---|----------|------------------------|-----------|
| 1 | YAML is descriptive ("bill of materials"), not declarative | Declarative YAML with full expression support | Keeps YAML simple; new features are Python, not schema extensions. Inspired by Astrogator paradigm. |
| 2 | Agent types are Python classes with hardcoded structural knowledge (Option A) | Schema-driven agent type definitions (Option B) | Small number of agent types, primary user knows Python, arbitrary logic needed in composition. Option B can layer on later if needed. |
| 3 | Blocks communicate via SymbolDescriptor objects (Approach 2) | Raw CasADi symbols (Approach 1), lazy callables (Approach 3) | Raw symbols lose semantic metadata, causing silent errors. Lazy callables add indirection without near-term benefit. Descriptors are lightweight and provide assembly-time validation. |
| 4 | Decision variables are continuous; mixed-integer deferred | Integrating MINLP solver (e.g., Bonmin) | CasADi handles continuous NLPs natively. Discrete decisions can likely be relaxed or avoided. MINLP adds significant solver complexity. |
| 5 | IPOPT as default solver | SNOPT, custom solvers | IPOPT is free, handles large sparse NLPs well. Other solvers can be swapped in later via solver backend configuration. |
| 6 | Initial guesses: user-provided override agent defaults; warnings printed for every default used | Silent defaults only; require all guesses upfront | Silent defaults risk garbage convergence. Requiring all guesses upfront is hostile for exploration. Warnings balance usability with transparency. |
| 7 | Dynamics models are swappable blocks (different f in ẋ = f(x,u)) | Hardcoded dynamics | Extensibility is a core requirement. The block interface naturally supports this. |
| 8 | Two-phase declare/build pattern for blocks | Single-phase build | Declare phase allows graph inspection and validation before any CasADi symbols exist. Prevents wasted computation on malformed problems. |
| 9 | Solver backend uses `nlpsol`, not `Opti` | CasADi `Opti` interface | `nlpsol` gives explicit control over decision vector layout, bound vectors, and constraint assembly. The solver backend provides the named-variable convenience of `Opti` without the black-box indexing. Closer to the math. |
| 10 | NLP assembled in MX space; SX used inside `ca.Function` objects for reusable computations | Pure SX throughout; pure MX throughout | MX allows embedding `ca.Function` call nodes, keeping the graph compact when a computation is called many times (e.g., dynamics at every collocation point). SX gives the fastest per-element evaluation inside those Functions. Hybrid pattern is CasADi's recommended approach for large problems. |
| 11 | Reusable computations shared via `FunctionDescriptor` (wrapping `ca.Function`) | Blocks rebuild expressions independently; share raw `ca.Function` without metadata | Sharing raw Functions loses interface metadata (argument names, shapes), making validation impossible. Rebuilding expressions independently bloats the graph. `FunctionDescriptor` provides both graph efficiency and assembly-time validation. |
| 12 | Solver backend owns all MX variables; blocks receive symbols, build expressions, return them | Blocks own their own variables; solver collects at the end | Central ownership gives the backend explicit control over the global decision vector layout and index mapping. Blocks don't need to know about global indexing. Clean separation of concerns. |

---

## 6. Phase Plan

### Phase 1 — Foundation and Proof of Concept

**Goal:** Prove the block interface design works end to end with CasADi/IPOPT.

**Tasks:**
- [ ] Implement `SymbolDescriptor` dataclass
- [ ] Implement `FunctionDescriptor` dataclass
- [ ] Implement `PortDeclaration`, `FunctionDependency`, `FunctionProduction`, and `BlockDeclaration` dataclasses
- [ ] Implement `BlockOutputs` dataclass
- [ ] Implement base `Block` class with declare/build pattern (accepting both symbol and function inputs)
- [ ] Implement `SolverBackend` wrapper around CasADi `nlpsol` (variable registration, constraint/cost collection, build, solve, extract)
- [ ] Implement `SolutionResult` dataclass
- [ ] Write 1-2 toy expression blocks (e.g., a quadratic cost, a simple bound constraint)
- [ ] Write 1 toy function-producing block (e.g., a simple dynamics function) and 1 consuming block to verify function passing works
- [ ] Write a test script that manually instantiates blocks, resolves function dependencies, wires them to the solver backend, and solves
- [ ] Verify IPOPT convergence, result extraction by name, and solver statistics

**No YAML, no agent types.** Pure Python wiring of blocks to solver.

### Phase 2 — Block Library for the Flyby Problem

**Goal:** Solve the actual flyby goodput optimization problem.

**Tasks:**
- [ ] Implement `SigmoidGoodput` block
- [ ] Implement `AggregateGoodput` block (weighted sum of per-product goodputs)
- [ ] Implement `TTPComputation` block (assembles TTP from component delays)
- [ ] Implement resource constraint blocks: `PowerBudget`, `ComputeThroughput`, `DownlinkCapacity`
- [ ] Implement flyby geometry / contact window modeling (analytical or simple dynamics)
- [ ] Write Excel data ingestion to load baseline product data
- [ ] Write integration script: load data → build problem → solve → dump results
- [ ] Validate results against hand calculations / expected behavior

**Still no YAML or agent types.** Manual Python assembly, but solving the real problem.

### Phase 3 — Agent Types and Path Resolution

**Goal:** Replace manual block wiring with domain-aware agent types.

**Tasks:**
- [ ] Implement base `AgentType` class with declare/build/resolve interface
- [ ] Implement `FlybySatellite` agent type
- [ ] Implement `Constellation` agent type (if needed for this problem phase)
- [ ] Implement path resolution system
- [ ] Implement `get_decision_variables()` with suggested defaults
- [ ] Refactor Phase 2 integration script to use agent types instead of manual wiring
- [ ] Verify identical results to Phase 2

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

1. ~~**SolverBackend encapsulation:**~~ **RESOLVED.** Backend fully encapsulates `nlpsol`. Blocks interact only through `add_variable()`, `add_constraint()`, `add_cost()`. No direct access to the solver object.

2. **Result packaging:** `SolutionResult` sketch exists (Section 3.1). Needs refinement during Phase 1 implementation — exact fields TBD based on what's actually useful during debugging.

3. **Output file format:** JSON, YAML, CSV, or multiple? JSON is easiest to parse programmatically. CSV is useful for tabular results. YAML is consistent with the input format. Defer decision until Phase 4.

4. **SymbolDescriptor validation strictness:** Start advisory (warn on mismatch) and tighten later? Or enforce from day one? Advisory is more practical for rapid development. **Lean: advisory.**

5. **Time grid management:** For trajectory problems (Phase 5), who owns the time grid — the dynamics block, the agent type, or the solver backend? How do multiple blocks share a common time discretization?

6. **Block registry:** How does the problem compiler look up block types by name? A simple dictionary mapping string names to classes? A plugin/entry-point system? A registry class? **Lean: simple dictionary for Phase 4, reassess if it gets unwieldy.**

7. **GUI/interface for YAML generation:** Deferred, but worth noting as a future usability feature. A tool that presents available agent types, required parameters, and decision variables, then produces a valid YAML.

8. **Function namespace scoping:** When multiple agent instances use the same block type (e.g., two constellations each with Keplerian dynamics), should the produced `ca.Function` be shared (one Function, called from both agents) or duplicated (one per agent instance)? Sharing is more efficient; duplication is simpler and avoids coupling. **Lean: share by default, since `ca.Function` objects are stateless.**

9. **`expand=True` as default:** Should the solver backend default to `expand=True` (converts MX→SX at build time for faster evaluation)? This is a free performance win for problems without callbacks or non-expandable operations, but it's incompatible with `ca.Callback` objects (iteration callbacks). **Lean: default True, disable when callbacks are attached.**

10. **Variable scaling:** CasADi/IPOPT works best when variables are in the 0.01–100 range. Orbital mechanics variables span many orders of magnitude (km, rad/s, kg). Should the solver backend handle scaling automatically (user provides nominal values), or should blocks be responsible for their own scaling? Defer to Phase 2 when working with real problem data.

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
