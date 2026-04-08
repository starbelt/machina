# Layer 3 — Agent Types Interface Specification

**Status:** v1.2 (Phases 3a–3b complete)
**Last Updated:** 2026-04-08
**Author:** Nathan
**Parent Document:** `CLAUDE.md` (v0.7)

---

## 1. Purpose

Layer 3 provides domain-aware composition classes ("agent types") that translate high-level mission descriptions into CasADi NLP components. An agent type knows how to assemble Layer 2 functions into a meaningful subsystem — what quantities to declare, what functions to call, what constraints are inherent to the physics.

Agent types are the bridge between the YAML vocabulary (Layer 5) and the mathematical building blocks (Layer 2). They encode structural domain knowledge: a flyby satellite has products, resource budgets, and TTP chains; a constellation has orbital states and coverage geometry. The framework user extends the system by writing new agent types in Python.

---

## 2. Design Principles

### 2.1 Agents Are Decoupled from the Solver

Agent types never interact with the `SolverBackend` directly. They declare quantities, receive MX symbols, build expressions, and return constraints. The problem compiler (Layer 4) is the sole entity that creates variables/parameters and registers constraints and costs with the solver backend.

**Rationale:** Full decoupling gives the compiler complete visibility into the NLP structure. Every variable, parameter, constraint, and cost flows through the compiler, enabling comprehensive logging, the default-warning system, the visualization tool, and future features like constraint suppression from the YAML.

### 2.2 Two-Phase Lifecycle: Declare → Build

Agent types use a two-phase lifecycle, but the purpose differs from the original Block ABC pattern (which was eliminated in design decision #13):

- **Declare phase:** The agent examines its config and produces a manifest of all quantities it needs — their names, shapes, bounds, defaults, and whether each should be a decision variable, a parameter, or is flexible (determined by the YAML at problem definition time). No MX symbols exist yet. No solver interaction.

- **Build phase:** The compiler has created MX symbols for every declared quantity (as either variables or parameters based on the YAML and defaults) and passes them to the agent. The agent builds internal expressions using Layer 2 functions, populates its resolvable namespace, and returns a list of constraints.

**Rationale:** The declare phase enables the compiler to decide the role of each quantity (variable vs. parameter) based on the YAML. This allows the same agent type to be used in different problem configurations — for example, eccentricity can be a decision variable in one problem and fixed to zero in another, without modifying the agent code. This capability was the motivating argument for reintroducing the two-phase pattern.

### 2.3 Config-Driven Structure

The agent's config dict (from YAML or Python) determines what quantities exist. An agent with no `resources.power` config entry creates no power-related variables, constraints, or namespace paths. An agent with `geometry: fixed` creates no orbital state variables. The config selects a subset of the agent's full capability.

### 2.4 Agents Are Isolated

Agents cannot see each other's namespaces or quantities. Cross-agent references are resolved by the problem compiler, which has access to all agents. Cross-agent constraints (e.g., "flyby downlink ≤ ground station receive capacity") are assembled and registered by the compiler, not by either agent.

### 2.5 Costs Are the Compiler's Job

Agents do not register cost terms. They expose computed quantities (e.g., `aggregate_goodput`) in their namespace. The YAML cost function section references these quantities by path, and the compiler resolves them, applies weights, and registers the assembled cost with the solver backend. This keeps objective function composition at the problem level, not baked into agents.

### 2.6 Structural Constraints Are the Agent's Job

Constraints that are inherent to the agent's physics — resource budget limits, physical feasibility bounds, coupling between internal quantities — are returned by the agent's `build()` method. The compiler registers them with the solver. The agent decides which constraints to create based on its config (e.g., no power constraint if power is unconstrained in this run).

---

## 3. Key Types

### 3.1 QuantityDeclaration

Describes a single quantity the agent needs. Produced during `declare()`, consumed by the compiler.

```python
@dataclass
class QuantityDeclaration:
    path: str
    """
    Path relative to the agent (e.g., 'orbital/sma',
    'products/cmi_conus/allocations/compute'). The compiler
    prepends the agent name to form the full solver-level name
    (e.g., 'flyby_sat/orbital/sma').
    """

    shape: tuple
    """Shape of the quantity, e.g., (1, 1) for scalar, (6, 1) for
    a state vector. Must match the MX symbol the compiler creates."""

    semantic_type: str
    """One of: 'scalar', 'vector', 'trajectory', 'indexed_set',
    'time_grid'. Used for validation at resolution boundaries."""

    default_value: float | np.ndarray
    """Default value. Interpretation depends on assigned role:
    - If assigned as decision variable: used as initial guess.
    - If assigned as parameter: used as parameter value.
    The compiler warns when this default is used instead of a
    user-provided override from the YAML."""

    lb: float | np.ndarray
    """Lower bound. Only meaningful when assigned as a decision
    variable. Scalar broadcasts to all elements."""

    ub: float | np.ndarray
    """Upper bound. Only meaningful when assigned as a decision
    variable. Scalar broadcasts to all elements."""

    description: str
    """Human-readable description for logging and documentation."""

    units: str = None
    """Optional, advisory. For documentation and debugging."""

    frame: str = None
    """Reference frame if applicable ('ECI', 'LVLH', 'body').
    None for non-spatial quantities."""

    role: str = 'flexible'
    """
    Determines who decides whether this quantity is a variable
    or parameter:
    - 'flexible': The YAML decides. If YAML says nothing, the
      compiler uses default_role.
    - 'always_variable': Internal optimization DOF. The compiler
      cannot override this to a parameter (e.g., per-product
      resource allocations).
    - 'always_parameter': Fixed constant. The compiler cannot
      promote this to a decision variable (e.g., physical
      constants used internally).
    """

    default_role: str = 'variable'
    """
    Default role assignment when role='flexible' and the YAML
    does not specify. Either 'variable' or 'parameter'.
    Only meaningful when role='flexible'.

    Examples:
    - SMA: default_role='variable' (likely optimized, but could
      be fixed for a specific study)
    - Baseline T_VAGL: default_role='parameter' (usually fixed,
      but could be freed for sensitivity analysis)
    """
```

### 3.2 ConstraintDeclaration

Describes a constraint returned by `build()` for the compiler to register with the solver.

```python
@dataclass
class ConstraintDeclaration:
    expr: ca.MX
    """MX expression for the constraint. Can be scalar or
    vector-valued."""

    lb: float | np.ndarray
    """Lower bound. Scalar broadcasts. Use -np.inf for no lower
    bound. For equality constraints, set lb == ub."""

    ub: float | np.ndarray
    """Upper bound. Scalar broadcasts. Use np.inf for no upper
    bound. For equality constraints, set lb == ub."""

    name: str
    """Constraint name. The compiler prepends the agent name to
    form the full solver-level constraint name
    (e.g., 'flyby_sat/compute_budget')."""

    description: str
    """Human-readable description for logging and debugging."""
```

### 3.3 AgentType Base Class

```python
class AgentType(ABC):
    def __init__(self, name: str, config: dict):
        """
        Args:
            name: Unique agent name. Used as path prefix for all
                  quantities registered with the solver
                  (e.g., 'flyby_sat').
            config: Agent-specific configuration dict. Structure
                    defined by each subclass. The problem compiler
                    merges YAML overrides into this dict before
                    constructing the agent.
        """
        self.name = name
        self.config = config
        self._namespace: dict[str, SymbolDescriptor] = {}
        self._built = False

    @abstractmethod
    def declare(self) -> list[QuantityDeclaration]:
        """
        Declare all quantities this agent needs.

        Reads self.config to determine which quantities exist.
        For example, if config has no 'resources.power' entry,
        no power-related quantities are declared. If config
        specifies 'geometry: keplerian', orbital element
        quantities are declared.

        No MX symbols are created. No solver interaction.

        Returns:
            List of QuantityDeclarations describing every
            quantity the agent will use during build().
        """
        pass

    @abstractmethod
    def build(self, symbols: dict[str, ca.MX]) -> list[ConstraintDeclaration]:
        """
        Build internal expressions, populate the resolvable
        namespace, and return constraints.

        Args:
            symbols: Maps declared quantity paths to MX symbols.
                     Every path from declare() has an entry —
                     the compiler has created each as either a
                     variable or parameter via the solver backend.
                     Keys are paths relative to this agent
                     (same strings used in QuantityDeclaration.path).

        Returns:
            List of ConstraintDeclarations for the compiler to
            register with the solver backend.

        Post-conditions:
            - self._namespace is populated with all externally
              resolvable quantities (variables, parameters,
              and computed expressions).
            - self._built is True.

        The agent should assert that symbols.keys() matches its
        declared paths as a defensive check.
        """
        pass

    def resolve(self, path: str) -> SymbolDescriptor:
        """
        Look up a path in this agent's namespace.

        The problem compiler strips the agent name prefix before
        calling this method, so path is relative to this agent
        (e.g., 'products/cmi_conus/ttp', not
        'flyby_sat/products/cmi_conus/ttp').

        Args:
            path: Relative path string using '/' as separator.

        Returns:
            SymbolDescriptor for the requested quantity.

        Raises:
            RuntimeError: If the agent has not been built yet.
            KeyError: If the path is not in the namespace.
                      Error message includes available paths.
        """
        if not self._built:
            raise RuntimeError(
                f"Agent '{self.name}' has not been built yet"
            )
        if path not in self._namespace:
            available = list(self._namespace.keys())
            raise KeyError(
                f"Agent '{self.name}': no path '{path}'. "
                f"Available: {available}"
            )
        return self._namespace[path]

    def list_paths(self) -> list[str]:
        """
        Return all resolvable paths. Only meaningful after build().

        Includes decision variables, parameters, and computed
        expressions — everything the agent has exposed in its
        namespace.
        """
        return list(self._namespace.keys())
```

---

## 4. Lifecycle

The full lifecycle of an agent, from construction through solve:

```
1. Problem compiler reads YAML, identifies agent entries.

2. For each agent:
   a. Compiler merges YAML agent config with any overrides.
   b. Compiler constructs AgentType subclass:
      agent = FlybySatellite(name='flyby_sat', config=merged_config)

3. For each agent:
   a. Compiler calls agent.declare() → list[QuantityDeclaration]
   b. For each QuantityDeclaration:
      - Check YAML for overrides (fixed value, initial guess, bounds)
      - Determine role (variable or parameter) based on:
        * role='always_variable' → variable
        * role='always_parameter' → parameter
        * role='flexible' + YAML override → YAML wins
        * role='flexible' + no override → use default_role
      - Create MX symbol via solver backend:
        * Variable: solver.add_variable(full_path, shape, lb, ub, guess)
        * Parameter: solver.add_parameter(full_path, shape)
      - Store the MX symbol keyed by the declaration's path
      - Warn if default_value was used (no YAML override provided)

4. For each agent:
   a. Compiler calls agent.build(symbols) → list[ConstraintDeclaration]
   b. For each ConstraintDeclaration:
      - Compiler calls solver.add_constraint(expr, lb, ub,
            name=f"{agent.name}/{constraint.name}")

5. Compiler assembles cost function:
   a. Read YAML cost section
   b. Resolve source paths through agents (agent.resolve(path))
   c. Build weighted cost expression from resolved SymbolDescriptors
   d. Register with solver: solver.add_cost(cost_expr, name=...)

6. Compiler calls solver.build()

7. Compiler calls solver.solve(p_val=parameter_values)
   - Parameter values collected from QuantityDeclarations that
     were assigned as parameters (using default_value or YAML
     override values)

8. Compiler extracts results from SolutionResult and reports.
```

---

## 5. Variable Naming Convention

All quantities registered with the solver use the pattern:

```
{agent_name}/{internal_path}
```

**Rules:**
- `/` as separator, no leading or trailing slash.
- Agent name is always the first segment.
- Internal segments are lowercase, underscore-separated (snake_case).
- Product-specific paths include the product name as a segment.
- No reserved characters beyond `/`.

**Examples:**
```
flyby_sat/orbital/sma
flyby_sat/orbital/ecc
flyby_sat/products/cmi_conus/allocations/compute
flyby_sat/products/cmi_conus/ttp_components/t_vagl
flyby_sat/resources/compute/limit
```

These names appear in `SolutionResult.x_opt`, the visualization tool, debug output, and YAML path references.

---

## 6. Resolvable Namespace

The agent's `_namespace` is a flat `dict[str, SymbolDescriptor]` populated during `build()`. Keys are `/`-separated path strings relative to the agent (no agent name prefix).

### 6.1 What Goes in the Namespace

Three categories of entries:

**A. Decision variables.** The MX symbol for a quantity assigned as a variable by the compiler. The SymbolDescriptor wraps the MX variable directly.

**B. Parameters.** The MX symbol for a quantity assigned as a parameter. Same wrapping.

**C. Computed expressions.** MX expressions built during `build()` from variables, parameters, and Layer 2 function calls. Examples: per-product TTP (output of `util.ttp_computation`), per-product goodput (output of `cost.sigmoid_goodput`), aggregate goodput (weighted sum), resource usage totals (sum of allocations).

All three categories are valid namespace entries and valid return values from `resolve()`. The `SymbolDescriptor` metadata (semantic_type, frame, units) applies uniformly.

### 6.2 Exposure Policy

**Expose if:** the quantity has a meaningful name and a plausible external consumer (the problem compiler, the YAML, the viz tool, result reporting).

**Do not expose if:** the quantity is a purely internal intermediate computation with no external meaning.

The config determines the namespace structure. If the config defines 30 products, the namespace has 30 sets of product paths. If the config omits power resources, no power paths exist.

### 6.3 Namespace Structure (Illustrative for FlybySatellite)

```
# Orbital state (if geometry != 'fixed')
orbital/sma
orbital/ecc
orbital/inc
orbital/raan
orbital/aop
orbital/ta

# Per-product quantities
products/{product_name}/ttp                       # computed expression
products/{product_name}/goodput                   # computed expression
products/{product_name}/ttp_components/{component} # variable or parameter
products/{product_name}/allocations/{resource}     # variable (always)

# Resource totals
resources/{resource}/usage                        # computed expression (sum)
resources/{resource}/limit                        # parameter

# Aggregate
aggregate_goodput                                 # computed expression
```

---

## 7. Product Data Contract

Product data is subclass-specific, not defined in the base `AgentType`. For `FlybySatellite`, each product entry in the config contains:

- **name** — unique identifier, used in path construction
- **weight** — importance weight in aggregate goodput (fixed)
- **goodput** — function specification: registry key and factory parameters (e.g., `{type: 'sigmoid', k: 0.1, t50: 30.0}`)
- **ttp_baseline** — nominal value for each TTP component (dict of component_name → value)
- **vp_effects** — which value propositions apply to this product, which TTP components they affect, what resource they consume, and what reduction model to use

The exact schema for `vp_effects` and the reduction model specification will be defined during `FlybySatellite` implementation. The base `AgentType` imposes no requirements on config structure.

Product data enters the agent through the config dict. The ingestion pathway (Excel spreadsheet, CSV, Python dict literal) is external to the agent — the agent receives a Python data structure and doesn't care where it came from.

---

## 8. Resource Constraint Pattern

Resource constraints are config-driven and internal to the agent. Each resource in the config (compute, downlink, power, etc.) generates:

1. Per-product allocation decision variables (role='always_variable')
2. A total budget parameter (role='flexible', default_role='parameter')
3. A computed usage expression (sum of allocations)
4. A `ConstraintDeclaration` returned from `build()`: usage ≤ limit

If a resource is absent from the config, none of the above is created. The agent class defines the superset of possible resources; the config selects which ones are active for a given problem instance.

Cross-agent resource constraints (if they arise) are the compiler's responsibility, assembled from resolved paths across multiple agents.

---

## 9. Compiler Validation Rules

The following validation rules are the compiler's responsibility during the lifecycle:

1. **All declared quantities are supplied.** Every path in `declare()` must have a corresponding entry in the `symbols` dict passed to `build()`. Missing entries indicate a compiler bug.

2. **No undeclared symbols.** The `symbols` dict must not contain keys that were not in `declare()`. This catches typos and stale references.

3. **Role enforcement.** The compiler must not assign a parameter role to a quantity with `role='always_variable'`, or a variable role to a quantity with `role='always_parameter'`. Attempting to do so raises an error pointing to the YAML override and the quantity's role constraint.

4. **Shape consistency.** The MX symbol created by the compiler must match the shape declared by the agent.

5. **Default warnings.** For every quantity where no YAML override was provided and the default was used, the compiler emits a warning: `"Using default for flyby_sat/orbital/sma: 42164.0 km (decision variable, initial guess)"` or `"Using default for flyby_sat/products/cmi_conus/ttp_components/t_vagl: 45.0 s (parameter)"`.

6. **Agent name uniqueness.** No two agents may share the same name.

7. **All agent inputs supplied.** After build, the compiler verifies that every declared quantity was provided as an MX symbol. The agent's `build()` method should also perform this check defensively via an assertion.

---

## 10. Relationship to Other Layers

### Layer 1 (Solver Backend)
Agents have no direct interaction with the solver backend. The compiler mediates all solver access. MX symbols flow from the solver through the compiler to the agent; constraints flow from the agent through the compiler to the solver.

### Layer 2 (Function Library)
Agents call Layer 2 functions during `build()`. They look up factories in the function registry, construct `FunctionDescriptor` objects, and call them with MX symbols from the `symbols` dict. The resulting MX expressions are stored in the namespace or used to build constraint expressions.

### Layer 4 (Problem Compiler)
The compiler orchestrates the agent lifecycle (construct → declare → assign roles → create symbols → build → register constraints → assemble costs → solve). It is the sole entity with full visibility into the NLP structure. Detailed compiler specification is deferred to the Layer 4 design document.

### Layer 5 (YAML)
The YAML references agent types by name, provides config overrides, specifies quantity role overrides (fixed vs. free), provides initial guesses, and defines the cost function in terms of agent namespace paths. The agent type's Python class defines what config keys are valid; the YAML supplies values.

---

## 11. Design Decisions Log (Layer 3)

| # | Decision | Alternatives Considered | Rationale |
|---|----------|------------------------|-----------|
| 17 | Two-phase declare/build lifecycle for agents (interface negotiation, not validation) | Single-phase build() only | The declare phase lets the compiler decide variable vs. parameter roles based on the YAML, enabling the same agent to serve different problem configurations without code changes. Different motivation from the original Block ABC declare phase (which was eliminated). |
| 18 | Agents are fully decoupled from the solver backend | Agents call solver.add_variable() and solver.add_constraint() directly | Full decoupling gives the compiler complete visibility. Every NLP component flows through the compiler, enabling logging, the warning system, the viz tool, and future constraint suppression. |
| 19 | Agents return constraints from build(); compiler registers them | Agents register constraints directly with solver | Returning constraints gives the compiler full knowledge of all constraints in the problem, enables logging and optional YAML-driven suppression, and maintains the principle that the compiler is the sole solver interface. |
| 20 | Costs are never registered by agents; always assembled by the compiler from YAML | Agents register their own cost terms | Keeps objective function composition at the problem level. The YAML defines what is optimized, not the agent. Agents expose candidate cost quantities in their namespace. |
| 21 | QuantityDeclaration.role field with 'flexible', 'always_variable', 'always_parameter' options and default_role for the flexible case | All quantities are flexible; no always_variable/always_parameter | Internal optimization DOFs (resource allocations) should not be overridable to parameters — they are structurally required as variables. Physical constants should not be promotable to variables. The role field enforces these constraints while keeping most quantities flexible. |
| 22 | resolve() returns SymbolDescriptor (not raw ca.MX) | Return raw ca.MX for simplicity | SymbolDescriptor metadata (semantic_type, frame, units) enables catching errors like plugging a position vector into a velocity port at assembly time. The metadata cost is minimal and the debugging value is high. |
| 23 | Agent namespace is a flat dict with '/'-separated string keys | Nested tree structure | Flat is simpler to implement, query, and debug. Hierarchical structure is purely a naming convention. The agent populates the dict during build(); resolve() is a single dict lookup. |
| 24 | Static orbital geometry (KOE as decision variables, geometry/link functions from Layer 2) is in scope for Layer 3; dynamics (propagation, collocation) is Phase 5 | Defer all orbital mechanics to Phase 5; include dynamics in Layer 3 | The near-term optimization problem requires orbital state to be nontrivial (resource allocation depends on mission geometry). But orbital state as static decision variables is just algebra and Kepler's equation — no integration, no time-stepping. Kepler's equation is handled as a ca.rootfinder-based Layer 2 function, architecturally transparent to the agent. |

---

## 12. Open Questions (Layer 3)

1. **Universal variable Kepler solver in Layer 2.** The factory will use `ca.rootfinder` to solve the universal Kepler equation inside a `ca.Function`. Needs specification: rootfinder method (Newton vs. IPOPT), convergence tolerance, Stumpff function branch handling via `ca.if_else`, and how IFT-based differentiation through the rootfinder interacts with the outer NLP. See Appendix A for the mathematical formulation.

2. **VP effect reduction models.** The functional relationship between resource allocation and TTP component reduction needs empirical models (from image processing compute benchmarks). The Layer 2 factories for these models will be specified when the data is available. The agent architecture supports arbitrary reduction models via the function registry.

3. ~~**`expand=True` interaction.**~~ **RESOLVED.** `ca.rootfinder` is compatible with `expand=True` in the tested CasADi version — confirmed by `test_expand_true_compatibility` in `tests/test_blocks_phase3b.py`.

4. **Constellation agent type.** Not specified in this document. Will be designed when needed, following the same base class interface. May share orbital state utilities with `FlybySatellite` via composition (shared utility functions or mixins), not inheritance.

5. **Agent-internal parameter value delivery.** When the compiler assigns a quantity as a parameter, it creates the MX symbol and passes it to the agent. But the numeric value (from config default or YAML override) needs to reach the solver at solve time via `p_val`. The compiler must track which parameter symbols map to which numeric values. This is a Layer 4 implementation detail but worth noting as a requirement.

6. **Stumpff function branch smoothness.** The `ca.if_else` implementation of Stumpff functions C(ψ) and S(ψ) must be verified to produce continuous and differentiable transitions at ψ = 0 (the parabolic boundary). Taylor series expansions near ψ = 0 may be needed for numerical stability.

---

## Appendix A — Coverage Optimization Example Problem

### A.1 Purpose

A single-satellite coverage optimization problem that exercises the full agent framework, forces orbital mechanics into the Layer 2 function library, and validates the declare → build → resolve lifecycle with nontrivial geometry. This problem is complex enough to be a meaningful test case but simple enough to have verifiable results.

### A.2 Problem Statement

**Given:** A single satellite and a ground target location (latitude, longitude on Earth's surface).

**Find:** The orbit (defined by modified equinoctial elements) that maximizes total daily coverage time over the ground target, subject to altitude constraints.

**Coverage definition:** The satellite covers the target when the elevation angle from the target to the satellite exceeds a minimum threshold (e.g., 10°).

### A.3 Orbital Element Representation

#### A.3.1 Why Not Classical KOE

Classical Keplerian orbital elements (a, e, i, Ω, ω, ν) have well-known singularities:

- **Circular orbits (e → 0):** Argument of perigee ω is undefined. The perigee direction vanishes, so the angle measured from it is meaningless.
- **Equatorial orbits (i → 0 or π):** RAAN Ω is undefined. The line of nodes vanishes, so the angle measured from it is meaningless.

For an optimizer using gradient-based methods, these singularities cause the Jacobian to become singular, leading to solver failures or nonsensical search directions. Any orbit design problem where the optimizer might explore near-circular or near-equatorial orbits (which is most problems) cannot safely use classical KOE as decision variables.

#### A.3.2 Modified Equinoctial Elements (MEE)

Modified equinoctial elements replace the angular elements with smooth trigonometric combinations that eliminate the circular and equatorial singularities:

| Element | Definition | Description |
|---------|------------|-------------|
| p | a(1 - e²) | Semi-latus rectum [km] |
| f | e·cos(ω + Ω) | x-component of eccentricity vector |
| g | e·sin(ω + Ω) | y-component of eccentricity vector |
| h | tan(i/2)·cos(Ω) | x-component of node vector |
| k | tan(i/2)·sin(Ω) | y-component of node vector |
| L | Ω + ω + ν | True longitude [rad] |

**Properties:**
- Circular orbits: f = g = 0 (smooth, no singularity).
- Equatorial orbits: h = k = 0 (smooth, no singularity).
- Retrograde equatorial orbits (i = π): h and k diverge (tan(π/2)). This is the only remaining singularity, rarely encountered in practice. An alternative MEE definition using cot(i/2) handles the retrograde case at the cost of a prograde singularity.
- Eccentricity is recovered as e = √(f² + g²). Inclination as i = 2·arctan(√(h² + k²)).
- All elements are continuous and differentiable across the circular and equatorial boundaries.

**Decision variables for the coverage problem:** p, f, g, h, k. True longitude L is not a decision variable — it parameterizes position around the orbit and is swept as a set of fixed sample points.

#### A.3.3 Element Transforms (Layer 2 Functions)

| Registry Key | Inputs | Outputs | Notes |
|-------------|--------|---------|-------|
| `transform.koe_to_mee` | (a, e, i, Ω, ω, ν) | (p, f, g, h, k, L) | User-facing convenience. Used for config input parsing and result display. |
| `transform.mee_to_koe` | (p, f, g, h, k, L) | (a, e, i, Ω, ω, ν) | Inverse transform. Used for result reporting in familiar terms. |
| `transform.mee_to_eci` | (p, f, g, h, k, L, μ) | (r_eci, v_eci) | Position and velocity in ECI frame. Direct computation without going through classical KOE. |

### A.4 Universal Variable Kepler Solver

#### A.4.1 Problem

Given a satellite's initial position r₀ and velocity v₀ at time t₀, find the position and velocity at time t₀ + Δt. This requires solving Kepler's equation, which relates elapsed time to position along the orbit.

The classical formulation M = E - e·sin(E) only works for elliptic orbits (e < 1). The hyperbolic equivalent M = e·sinh(H) - H works only for e > 1. The parabolic case (e = 1) uses Barker's equation. If the optimizer is free to vary eccentricity across regime boundaries, a unified formulation is required.

#### A.4.2 Universal Variable Formulation

The universal variable χ (chi) is defined continuously across all orbit types. The universal Kepler equation is:

```
√μ · Δt = χ³ · S(α·χ²) + (r₀·v₀/√μ) · χ² · C(α·χ²) + r₀ · χ
```

where:
- α = 1/a = (2/r₀ - v₀²/μ) — positive for elliptic, zero for parabolic, negative for hyperbolic
- r₀ = |r₀|, the initial radius magnitude
- S(ψ) and C(ψ) are Stumpff functions (see below)

Given χ, the new position and velocity are computed via the Lagrange coefficients:

```
F = 1 - (χ²/r₀) · C(α·χ²)
G = Δt - (χ³/√μ) · S(α·χ²)
r = F · r₀ + G · v₀
|r| = r₀ + (r₀·v₀/√μ) · χ² · C(α·χ²) + (1 - α·r₀) · χ³ · S(α·χ²)  [unused if computing r directly]
```

#### A.4.3 Stumpff Functions

The Stumpff functions C(ψ) and S(ψ) unify the trigonometric (elliptic), linear (parabolic), and hyperbolic regimes:

**C(ψ):**
```
ψ > 0 (elliptic):    C = (1 - cos(√ψ)) / ψ
ψ = 0 (parabolic):   C = 1/2
ψ < 0 (hyperbolic):  C = (cosh(√(-ψ)) - 1) / (-ψ)
```

**S(ψ):**
```
ψ > 0 (elliptic):    S = (√ψ - sin(√ψ)) / (√ψ)³
ψ = 0 (parabolic):   S = 1/6
ψ < 0 (hyperbolic):  S = (sinh(√(-ψ)) - √(-ψ)) / (√(-ψ))³
```

**CasADi implementation:** The branch structure is implemented with `ca.if_else`. Near ψ = 0, both functions approach their parabolic limits smoothly (C → 1/2, S → 1/6), but direct evaluation suffers from catastrophic cancellation. Taylor series expansions should be used in a small neighborhood of ψ = 0:

```
C(ψ) ≈ 1/2 - ψ/24 + ψ²/720 - ...
S(ψ) ≈ 1/6 - ψ/120 + ψ²/5040 - ...
```

The switching threshold (|ψ| < ε) and number of Taylor terms need to be determined empirically for CasADi's AD precision.

#### A.4.4 Layer 2 Function Registry

| Registry Key | Factory params | Function inputs | Outputs | Implementation |
|-------------|----------------|-----------------|---------|----------------|
| `transform.stumpff_cs` | _(none)_ | ψ(1,1) | C(1,1), S(1,1) | `ca.if_else` branches; Taylor near \|ψ\| < 1e-4 |
| `transform.lagrange_coefficients` | `mu` | r₀(3,1), v₀(3,1), χ(1,1), Δt(1,1) | F, G, Ḟ, Ġ (all 1,1) | Calls `stumpff_cs` at SX level; computes α internally |
| `transform.universal_kepler` | `mu`, `method`, `opts` | r₀(3,1), v₀(3,1), Δt(1,1) | χ(1,1) | `ca.rootfinder` (Newton); p-vec = [r₀, σ₀, α, Δt] |
| `transform.propagate_universal` | `mu` | r₀(3,1), v₀(3,1), Δt(1,1) | r(3,1), v(3,1) | SX-level composition of the three above |

**Implementation notes (Phase 3b):**
- All four functions are implemented in `machina/blocks/library/transforms.py` and tested in `tests/test_blocks_phase3b.py` (38 tests).
- `mu` is a factory parameter (baked in at construction time), not a runtime argument — consistent with `mee_to_eci`. This differs from the original interface spec which listed μ as a function input.
- `transform.lagrange_coefficients` computes α = 2/r₀ − ‖v₀‖²/μ internally. The original spec listed α as an input; internal computation keeps the calling interface cleaner.
- `ca.rootfinder` is compatible with `expand=True` in the tested CasADi version (see Open Question 3 above — resolved).
- Math reference document: `universal_kepler_math_v1.md` in project root.
- **Correct Stumpff identity:** `1 − ψ·C(ψ) = cos(√ψ)` for ψ > 0 (and `cosh(√(−ψ))` for ψ < 0). The planning identity `ψ·S(ψ)+C(ψ)=1/2` is incorrect and was removed from tests.

**Note on `transform.universal_kepler`:** This is the first Layer 2 function that uses `ca.rootfinder`. The rootfinder solves for χ such that the universal Kepler equation residual is zero. CasADi differentiates through the rootfinder via the implicit function theorem (IFT), so the outer NLP gets exact gradients without differentiating through the iterative solve. The rootfinder method and options are factory kwargs.

### A.5 Coverage Geometry

#### A.5.1 Elevation Angle

Given the satellite position r_sat in ECI and a ground target position r_target in ECI (fixed, computed from geodetic latitude/longitude), the elevation angle ε as seen from the target is:

```
ρ = r_sat - r_target                    (range vector)
ρ_hat = ρ / |ρ|                          (unit range vector)
n_hat = r_target / |r_target|            (local vertical at target, spherical Earth approx)
ε = arcsin(ρ_hat · n_hat) - π/2 + π/2   (simplified: ε = arcsin(dot(ρ_hat, n_hat)))
```

More precisely, the elevation is the complement of the angle between the range vector and the local horizontal plane:

```
ε = arcsin((r_sat - r_target) · r_target / (|r_sat - r_target| · |r_target|))
```

For an oblate Earth, the local vertical would use the geodetic normal instead of the geocentric radial. Spherical Earth is sufficient for this test problem.

#### A.5.2 Smooth Coverage Metric

The binary coverage condition (ε > ε_min) is approximated with a sigmoid:

```
c(ε) = 1 / (1 + exp(-k · (ε - ε_min)))
```

where k controls steepness. Large k → sharp transition (close to binary), small k → smooth but less accurate. For the optimizer, k should be large enough that the approximation is faithful but small enough that the gradient is well-conditioned. k ≈ 10–50 (with ε in degrees) is a reasonable starting range.

Total coverage metric over one orbit:

```
C_total = (T_orbit / N) · Σᵢ c(εᵢ)
```

where εᵢ is the elevation at the i-th sample point and T_orbit/N converts the count to a time. This is the quantity to maximize.

#### A.5.3 Layer 2 Functions

| Registry Key | Inputs | Outputs | Notes |
|-------------|--------|---------|-------|
| `geometry.elevation_angle` | (r_sat, r_target) | ε | ECI vectors, spherical Earth |
| `geometry.ground_target_eci` | (lat, lon, R_earth) | r_target | Geodetic to ECI (fixed, no rotation — snapshot geometry) |
| `cost.smooth_coverage` | (ε, ε_min, k) | c | Sigmoid approximation of coverage indicator |

### A.6 NLP Formulation

**Decision variables:** p, f, g, h, k (5 modified equinoctial elements, excluding true longitude).

**Parameters:**
- N mean anomaly sample points Lᵢ (equally spaced in true longitude from 0 to 2π), `always_parameter`
- Ground target location (lat, lon), `always_parameter`
- μ (gravitational parameter), `always_parameter`
- ε_min (minimum elevation), `always_parameter`
- k_sigmoid (sigmoid steepness), `always_parameter`
- R_earth, `always_parameter`

**Objective:** Maximize C_total = (T_orbit/N) · Σᵢ sigmoid(k · (εᵢ - ε_min))

Equivalently, minimize -C_total.

**Constraints:**
- Perigee altitude: r_p - R_earth ≥ h_min → p/(1+e) - R_earth ≥ h_min, where e = √(f² + g²)
- Apogee altitude: r_a - R_earth ≤ h_max → p/(1-e) - R_earth ≤ h_max
- Optionally: inclination bounds via h, k (i = 2·arctan(√(h² + k²)))

**Computation graph during build():**

```
For each sample point i = 1..N:
    1. Form full MEE state: (p, f, g, h, k, Lᵢ)
    2. Call transform.mee_to_eci → r_sat_i (ECI position)
    3. Call geometry.elevation_angle(r_sat_i, r_target) → εᵢ
    4. Call cost.smooth_coverage(εᵢ, ε_min, k) → cᵢ

C_total = (T_orbit / N) · Σᵢ cᵢ
```

T_orbit is itself a function of the orbital elements: T = 2π·√(a³/μ), where a = p/(1 - f² - g²).

### A.7 Agent Type: SingleSatCoverage

A minimal agent type for this problem. Not intended for the flyby mission — it exists to validate the framework machinery.

**Config structure:**
```python
config = {
    'geometry': 'keplerian',       # only option for this agent
    'n_sample_points': 72,         # orbit discretization (every 5° in L)
    'ground_target': {
        'lat': 38.9,               # degrees
        'lon': -77.0               # degrees
    },
    'coverage': {
        'min_elevation': 10.0,     # degrees
        'sigmoid_k': 20.0          # steepness
    },
    'altitude_bounds': {
        'perigee_min': 200.0,      # km
        'apogee_max': 40000.0      # km
    },
    'constants': {
        'mu': 398600.4418,         # km³/s²
        'R_earth': 6378.137        # km
    }
}
```

**Declared quantities:**

| Path | Shape | Role | Default Role | Default Value | Description |
|------|-------|------|-------------|---------------|-------------|
| `orbital/p` | (1,1) | flexible | variable | 7000.0 | Semi-latus rectum [km] |
| `orbital/f` | (1,1) | flexible | variable | 0.0 | Eccentricity vector x |
| `orbital/g` | (1,1) | flexible | variable | 0.0 | Eccentricity vector y |
| `orbital/h` | (1,1) | flexible | variable | 0.0 | Node vector x |
| `orbital/k` | (1,1) | flexible | variable | 0.0 | Node vector y |
| `sample_points/L` | (N,1) | always_parameter | — | linspace(0, 2π, N) | True longitude samples |
| `target/position` | (3,1) | always_parameter | — | from lat/lon | Ground target ECI |
| `constants/mu` | (1,1) | always_parameter | — | 398600.4418 | Gravitational parameter |
| `constants/R_earth` | (1,1) | always_parameter | — | 6378.137 | Earth radius |
| `coverage/min_elevation` | (1,1) | always_parameter | — | 10.0 | Min elevation [deg] |
| `coverage/sigmoid_k` | (1,1) | always_parameter | — | 20.0 | Sigmoid steepness |

**Namespace (after build):**

```
orbital/p, orbital/f, orbital/g, orbital/h, orbital/k
orbital/sma              (computed: p / (1 - f² - g²))
orbital/ecc              (computed: √(f² + g²))
orbital/inc              (computed: 2·arctan(√(h² + k²)))
coverage/total           (computed: the objective quantity)
coverage/per_point/{i}   (computed: per-sample-point coverage value)
constraints/perigee_alt  (exposed for debugging)
constraints/apogee_alt   (exposed for debugging)
```

**Constraints returned from build():**
- `perigee_altitude`: p/(1 + √(f² + g²)) - R_earth ≥ h_min
- `apogee_altitude`: p/(1 - √(f² + g²)) - R_earth ≤ h_max

### A.8 Development Phases

#### Phase 3a — Core Transforms and Agent Base ✓ COMPLETE

**Implemented files:**
- `machina/agents/agent_type.py` — `QuantityDeclaration`, `ConstraintDeclaration`, `AgentType` ABC (single file; splitting across `declarations.py` / `base.py` was not necessary given the tight coupling between them).
- `machina/agents/__init__.py` — re-exports public API.
- `machina/blocks/library/transforms.py` — `transform.koe_to_mee`, `transform.mee_to_koe`, `transform.mee_to_eci`.
- `machina/compiler/compiler_stub.py` — `CompilerStub`: Python-driven declare → assign → build → register lifecycle.
- `machina/compiler/__init__.py` — re-exports `CompilerStub`.
- `tests/test_blocks_phase3a.py` — 76 tests (all passing): roundtrip, ECI position vs. PQW reference, vis-viva velocity check, orthogonality, agent lifecycle, compiler stub lifecycle.

**Implementation notes:**
- The MEE→ECI velocity formula was derived from `v = (dr/dL)·(dL/dt)` rather than transcribed from a secondary reference. Several commonly cited sources carry a sign error in vy and a spurious extra `1/s²` on vz. The derivation is verified against vis-viva, r·v=0 for circular orbits, and a PQW-frame reference implementation. See design decision #27 in CLAUDE.md.
- The PQW→ECI reference rotation uses **positive** angles: `R3(Ω) @ R1(i) @ R3(ω)`, where `R3(θ) = [[cosθ,−sinθ,0],[sinθ,cosθ,0],[0,0,1]]` (active rotation convention). Using negative angles is a common transcription error from passive-rotation references.
- Angle wrapping in `mee_to_koe`: `atan2` returns angles in (−π, π]. Roundtrip tests compare via `cos`/`sin` to handle the equivalent representation at ω=270° ↔ −90°.

#### Phase 3b — Universal Kepler Solver ✓ COMPLETE

**Files:**
- `machina/blocks/library/transforms.py` — 4 new factory functions appended
- `tests/test_blocks_phase3b.py` — 38 tests (all passing); also 1 stale Phase 3a test updated
- `universal_kepler_math_v1.md` — math reference (motivation, Stumpff functions, universal Kepler equation, Lagrange coefficients, algorithm, references)

**All tasks completed:**
1. `transform.stumpff_cs` — `ca.if_else` with Taylor (EPS=1e-4) and TINY=1e-32 guards
2. `transform.universal_kepler` — `ca.rootfinder` (Newton); initial guess χ₀ = √μ·Δt/r₀
3. `transform.lagrange_coefficients` and `transform.propagate_universal`
4. Tests: ISS, GEO, polar roundtrips; half-period; forward/backward symmetry
5. IFT gradient correct to rtol=1e-3 vs finite differences
6. `expand=True` **compatible** — test passes

**Note on ordering:** Phase 3b does not block Phase 3c. Recommended sequence: 3c → 3d (or 3b already done, proceed to 3c).

#### Phase 3c — Coverage Geometry and Agent

1. Implement `geometry.ground_target_eci` and `geometry.elevation_angle` factories — `machina/blocks/library/geometry.py`.
2. Implement `cost.smooth_coverage` factory.
3. Implement `SingleSatCoverage` agent type — `machina/agents/single_sat_coverage.py`.
4. Solve coverage optimization: verify that optimizer finds expected orbit (high inclination for high-latitude target, etc.).
5. Validate against STK or analytical expectations.

#### Phase 3d — FlybySatellite Agent

1. Implement `FlybySatellite` agent type — `machina/agents/flyby_satellite.py`.
2. Port Phase 2 examples to use agent framework via compiler stub.
3. Verify identical results to Phase 2 manual wiring.
4. Add orbital state path (geometry: keplerian) with link budget functions.
5. Integration test: flyby goodput optimization with orbital freedom.

#### Phase 3e — Consolidation

1. Refactor shared orbital utilities between `SingleSatCoverage` and `FlybySatellite` into composition utilities (not base class).
2. Update viz tool if needed.
3. Full test suite for Layer 3.
4. Update design documents.

