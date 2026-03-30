# CasADi nlpsol: complete Python API reference for optimization frameworks

CasADi's `nlpsol` interface provides a **declarative, symbolic NLP construction** model where you define decision variables, objectives, and constraints as symbolic expressions, then hand the entire problem to a solver plugin like IPOPT. The framework auto-generates exact sparse Jacobians and Hessians via source-code-transformation AD, making it exceptionally suited for large-scale trajectory optimization. This report covers every API surface you need to build a modular solver backend: construction, solving, result extraction, parameter handling, IPOPT options, dual variables, callbacks, symbolic variable types, Function objects, and structural best practices.

---

## 1. The nlpsol construction and calling API

CasADi formulates a parametric NLP of the form: **minimize f(x, p)** subject to **lbx ≤ x ≤ ubx** and **lbg ≤ g(x, p) ≤ ubg**, where **p** represents fixed parameters. The solver is built in two steps: construct, then call.

### Construction

```python
import casadi as ca

# Step 1: Declare symbolic variables
x = ca.SX.sym('x', nx)   # decision variables
p = ca.SX.sym('p', np)   # parameters (optional)

# Step 2: Build symbolic expressions for objective and constraints
f = objective_expression(x, p)       # scalar
g = ca.vertcat(g1(x, p), g2(x, p))  # column vector (ng × 1)

# Step 3: Assemble the NLP dict
nlp = {'x': x, 'f': f, 'g': g, 'p': p}

# Step 4: Create the solver
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
```

The `nlpsol` signature is `casadi.nlpsol(name: str, plugin: str, nlp: dict, opts: dict)`. The NLP dict requires keys `'x'` (decision vector) and `'f'` (scalar objective). Keys `'g'` (constraints) and `'p'` (parameters) are optional. Available plugins include `'ipopt'`, `'sqpmethod'`, `'snopt'`, `'knitro'`, `'fatrop'`, `'blocksqp'`, and `'alpaqa'`.

### Calling the solver and extracting results

The solver object behaves as a CasADi Function. You call it with keyword arguments and get back a dict-like solution object:

```python
sol = solver(
    x0=initial_guess,        # nx × 1, default: zeros
    lbx=lower_bounds_x,      # nx × 1, default: -inf
    ubx=upper_bounds_x,      # nx × 1, default: +inf
    lbg=lower_bounds_g,      # ng × 1, default: -inf
    ubg=upper_bounds_g,      # ng × 1, default: +inf
    p=parameter_values,      # np × 1
    lam_x0=dual_init_x,     # nx × 1, for warm-starting
    lam_g0=dual_init_g      # ng × 1, for warm-starting
)

# Extract results
x_opt  = sol['x']       # optimal decision variables (DM, nx × 1)
f_opt  = sol['f']       # optimal objective value (DM scalar)
g_opt  = sol['g']       # constraint values at optimum (DM, ng × 1)
lam_x  = sol['lam_x']  # Lagrange multipliers for variable bounds
lam_g  = sol['lam_g']  # Lagrange multipliers for constraints
lam_p  = sol['lam_p']  # sensitivity of objective to parameters
```

Scalars broadcast automatically — `lbg=0, ubg=0` applies equality to all constraints. For NumPy conversion: `x_np = sol['x'].full().flatten()`. Solver statistics are available via `solver.stats()`, which returns `return_status`, `iter_count`, and `t_wall_total` among other fields.

**Equality constraints** are encoded by setting `lbg[i] == ubg[i]`. IPOPT detects these automatically. Double-sided inequalities use finite bounds on both sides: `lbg=-1, ubg=1` gives **-1 ≤ g(x) ≤ 1**.

---

## 2. SX versus MX: choosing the right symbolic type

CasADi offers two symbolic types with fundamentally different graph granularity. **SX** builds expression graphs at the scalar level — every addition and multiplication on every element creates a node. **MX** builds graphs at the matrix level — a matrix multiply is a single node, and crucially, MX nodes can embed calls to CasADi `Function` objects.

For a space mission optimization framework, the recommended hybrid pattern is: **define low-level dynamics and cost functions using SX, then build the NLP structure using MX that calls those SX-based Functions**.

| Aspect | SX | MX |
|---|---|---|
| Graph granularity | Scalar operations | Matrix-valued operations |
| Construction speed | Slow for very large problems | Fast for hierarchical problems |
| Evaluation speed | Fastest per-element | Slight overhead per call node |
| Function embedding | Cannot embed `Function` calls | **Can embed arbitrary Functions** |
| Best for | ODE right-hand sides, cost terms | NLP assembly, integrator embedding |

The `expand` option (`{'expand': True}`) converts an MX-based NLP to SX at construction time, combining MX's ergonomic construction with SX's evaluation speed. This works only when the graph contains no non-expandable operations (like SUNDIALS integrators or B-spline interpolants).

### Building the global decision vector with vertcat

The canonical pattern accumulates variables, bounds, and constraints in Python lists during a construction loop, then concatenates at the end:

```python
w = []; w0 = []; lbw = []; ubw = []
g = []; lbg = []; ubg = []
J = 0
offset = 0; var_index = {}

# Register a variable
def add_var(name, n, lb=-ca.inf, ub=ca.inf, init=0.0):
    nonlocal offset
    sym = ca.MX.sym(name, n)
    w.append(sym)
    var_index[name] = (offset, offset + n)
    offset += n
    lbw += [lb]*n if isinstance(lb, (int,float)) else list(lb)
    ubw += [ub]*n if isinstance(ub, (int,float)) else list(ub)
    w0  += [init]*n if isinstance(init, (int,float)) else list(init)
    return sym

# After construction loop:
nlp = {'x': ca.vertcat(*w), 'f': J, 'g': ca.vertcat(*g)}
```

After solving, slice results back using the stored index map:

```python
def extract(sol, name):
    i, j = var_index[name]
    return sol['x'][i:j].full().flatten()
```

A cleaner alternative uses a CasADi Function for extraction — create symbolic slicing expressions during construction, wrap them in `ca.Function('extract', [w_sym], [x_slice, u_slice])`, then call that function on `sol['x']` after solving.

---

## 3. Parameters: fixed values in the NLP formulation

Parameters are symbolic variables that appear in `f` and `g` but are **not optimized**. They enable re-solving with different problem data without rebuilding the solver — critical for mission design trade studies or MPC-style re-planning.

```python
x = ca.SX.sym('x', 2)
p = ca.SX.sym('p', 3)  # e.g., target orbit elements, thrust magnitude, time-of-flight

f = (x[0] - p[0])**2 + p[2] * x[1]**2
g = x[0] + x[1] - p[1]

nlp = {'x': x, 'p': p, 'f': f, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Solve with one set of parameters
sol1 = solver(x0=[0,0], lbg=0, ubg=0, p=[1.0, 2.0, 0.5])

# Re-solve with different parameters — no rebuild needed
sol2 = solver(x0=[0,0], lbg=0, ubg=0, p=[3.0, 4.0, 1.0])
```

The `sol['lam_p']` output gives the **sensitivity of the optimal objective with respect to parameters** — useful for understanding how mission parameters affect optimality. Enable this explicitly with `{'calc_lam_p': True}` in solver options if needed.

---

## 4. Passing IPOPT options through nlpsol

IPOPT options are passed via the solver options dict using either **dot notation** or a **nested dict**. Both are valid and can be mixed:

```python
# Dot notation
opts = {
    'print_time': False,           # CasADi-level option
    'expand': True,                # CasADi: convert MX→SX
    'ipopt.print_level': 0,        # IPOPT option
    'ipopt.tol': 1e-8,
    'ipopt.max_iter': 1000,
    'ipopt.linear_solver': 'mumps',
}

# Equivalent nested dict
opts = {
    'print_time': False,
    'expand': True,
    'ipopt': {
        'print_level': 0,
        'tol': 1e-8,
        'max_iter': 1000,
        'linear_solver': 'mumps',
    }
}
```

The most important options for a trajectory optimization backend fall into several categories.

**Convergence tolerances**: `ipopt.tol` (default **1e-8**, overall convergence), `ipopt.acceptable_tol` (fallback tolerance), `ipopt.constr_viol_tol` (constraint feasibility), `ipopt.dual_inf_tol` (dual feasibility), `ipopt.compl_inf_tol` (complementarity).

**Iteration and time limits**: `ipopt.max_iter` (default **3000**), `ipopt.max_cpu_time` (seconds).

**Linear solver selection**: `ipopt.linear_solver` accepts `'mumps'` (default, open-source), `'ma27'`, `'ma57'`, `'ma86'`, `'ma97'` (HSL, often faster for large problems), or `'pardiso'`.

**Hessian strategy**: `ipopt.hessian_approximation` is `'exact'` by default (uses CasADi's AD-generated Hessian) or `'limited-memory'` for L-BFGS approximation — useful when the exact Hessian is too expensive or dense.

**Barrier strategy**: `ipopt.mu_strategy` can be `'monotone'` or `'adaptive'` (often faster).

**Warm-starting**: set `ipopt.warm_start_init_point` to `'yes'` along with `warm_start_bound_push`, `warm_start_mult_bound_push`, and `warm_start_slack_bound_push` all to **1e-16**, and `ipopt.mu_init` to a small value matching the previous solution's barrier parameter.

**Debugging derivatives**: `ipopt.derivative_test` set to `'first-order'` or `'second-order'` runs IPOPT's finite-difference derivative checker.

**Fixed variable treatment**: When `lbx[i] == ubx[i]` (fixed variables), set `ipopt.fixed_variable_treatment` to `'make_constraint'` or `'relax_bounds'` to get correct Lagrange multipliers. The default `'make_parameter'` can produce incorrect dual values.

---

## 5. Dual variables and Lagrange multipliers

CasADi uses the Lagrangian convention **L(x, λ) = f(x) + λᵀg(x)**. The solution dict provides three dual variable outputs:

- **`sol['lam_g']`**: multipliers for constraints (ng × 1). A nonzero value indicates the constraint is active. Positive values correspond to the upper bound being active; negative values to the lower bound.
- **`sol['lam_x']`**: multipliers for variable bounds (nx × 1). Nonzero when a variable sits at its bound.
- **`sol['lam_p']`**: sensitivity of the optimal objective to parameters (np × 1). Requires `{'calc_lam_p': True}`.

For **warm-starting**, feed previous duals back in:

```python
sol2 = solver(
    x0=sol['x'],
    lam_x0=sol['lam_x'],
    lam_g0=sol['lam_g'],
    lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=new_params
)
```

Combined with the IPOPT warm-start options described above, this typically reduces iteration counts by **50–80%** for nearby problems — essential for parametric sweeps in mission design.

---

## 6. Structuring large NLPs for sparsity and performance

### Variable ordering creates banded Jacobians

For trajectory optimization, **interleave states and controls per timestep** rather than grouping all states then all controls. This produces a block-banded constraint Jacobian that sparse linear solvers (MUMPS, MA57) can factor efficiently:

```python
# GOOD: interleaved ordering → block-banded Jacobian
for k in range(N):
    Xk = add_var(f'X_{k}', nx, ...)   # state at k
    Uk = add_var(f'U_{k}', nu, ...)   # control at k
    # dynamics constraint references only X_k, U_k, X_{k+1}
    add_constraint(X_{k+1} - F(Xk, Uk))

# BAD: grouped ordering → full-width coupling in Jacobian
X_all = add_var('X', nx*(N+1), ...)
U_all = add_var('U', nu*N, ...)
```

### Automatic Jacobian and Hessian generation

CasADi generates all required derivatives automatically from the symbolic graph. The process is: (1) **sparsity propagation** through the graph using bitvector analysis to identify structural nonzeros, (2) **graph coloring** to find minimal seed directions — symmetric star-coloring for Hessians, unidirectional coloring for Jacobians, choosing forward or reverse mode based on dimensions and a cost heuristic, and (3) **symbolic assembly** of the sparse derivative expressions. IPOPT reports the resulting sparsity at startup (e.g., "Number of nonzeros in Lagrangian Hessian: 4502").

You can also **supply custom Jacobians or Hessians** via the `'grad_f'`, `'jac_g'`, and `'hess_lag'` options to `nlpsol`, passing `casadi.Function` objects with the appropriate signatures. Alternatively, use `ipopt.hessian_approximation: 'limited-memory'` to avoid computing the Hessian entirely.

### Scaling for numerical conditioning

Variables should be in the **0.01–100 range** for good IPOPT convergence. For space missions where positions might be in kilometers (10⁵) and masses in kilograms (10³), scale by nominal values:

```python
x_nom = ca.DM([1e5, 1e5, 1e5, 1e3, 1e3, 1e3, 1e3])  # pos, vel, mass
x_raw = add_var('X_k', 7, lb=-1, ub=1, init=0)
x_physical = x_nom * x_raw  # use x_physical in dynamics expressions
```

IPOPT's `nlp_scaling_method: 'gradient-based'` also auto-scales constraints and the objective using gradient magnitudes at the initial point.

---

## 7. Iteration callbacks for monitoring solver progress

CasADi provides an `iteration_callback` option that invokes a user-defined `Callback` at every IPOPT iteration. The callback receives all current NLP solver outputs — primal variables, objective, constraints, and dual variables.

```python
class SolverMonitor(ca.Callback):
    def __init__(self, name, nx, ng, opts={}):
        ca.Callback.__init__(self)
        self.nx = nx; self.ng = ng
        self.history = {'f': [], 'cv': []}
        self.construct(name, opts)

    def get_n_in(self):  return ca.nlpsol_n_out()
    def get_n_out(self): return 1

    def get_name_in(self, i):  return ca.nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f': return ca.Sparsity.scalar()
        if n in ('x', 'lam_x'): return ca.Sparsity.dense(self.nx)
        if n in ('g', 'lam_g'): return ca.Sparsity.dense(self.ng)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        darg = {ca.nlpsol_out(i): arg[i] for i in range(len(arg))}
        f_val = float(darg['f'])
        cv = float(ca.norm_inf(darg['g']))
        self.history['f'].append(f_val)
        self.history['cv'].append(cv)
        return [0]  # 0 = continue, nonzero = stop early

# Attach to solver
monitor = SolverMonitor('monitor', nx_total, ng_total)
opts['iteration_callback'] = monitor
opts['iteration_callback_ignore_errors'] = True
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
```

The callback receives `x`, `f`, `g`, `lam_x`, `lam_g`, and `lam_p` at each iteration. **Returning a nonzero value terminates the solver early** — useful for timeout logic or convergence monitoring. Note that `expand=True` is incompatible with Callback objects.

---

## 8. CasADi Function objects and embedding them in NLPs

`casadi.Function` wraps symbolic input-output mappings into reusable, callable, differentiable objects. They are the fundamental building block for modular NLP construction.

### Creating and calling Functions

```python
# Define with named I/O (recommended)
x = ca.SX.sym('x', nx); u = ca.SX.sym('u', nu)
xdot = dynamics_expr(x, u)
L = cost_expr(x, u)
f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

# Numeric evaluation
res = f(x=[1.0, 2.0], u=0.5)
print(res['xdot'], res['L'])

# Symbolic evaluation — embeds call node in MX graph
X = ca.MX.sym('X', nx); U = ca.MX.sym('U', nu)
xdot_sym, L_sym = f(X, U)  # returns MX expressions
```

When called with MX arguments, the function call becomes **a node in the MX expression graph**. CasADi propagates AD through these call nodes seamlessly using the chain rule — forward sensitivities propagate through, reverse adjoints propagate back, and sparsity detection sees through the call.

### The hybrid SX/MX pattern for trajectory optimization

This is the **recommended architecture** for a space mission framework:

```python
# Layer 1: SX dynamics function (efficient scalar operations)
x_s = ca.SX.sym('x', 7)  # [pos(3), vel(3), mass]
u_s = ca.SX.sym('u', 3)  # thrust vector
xdot = orbital_dynamics(x_s, u_s)  # SX expression
dyn = ca.Function('dyn', [x_s, u_s], [xdot])

# Layer 2: RK4 integrator as a Function (calls dyn symbolically)
X0 = ca.MX.sym('X0', 7); U = ca.MX.sym('U', 3); DT = ca.MX.sym('DT')
X = X0
for j in range(4):  # 4 RK4 substeps
    k1 = dyn(X, U)
    k2 = dyn(X + DT/8*k1, U)
    k3 = dyn(X + DT/8*k2, U)
    k4 = dyn(X + DT/4*k3, U)
    X = X + DT/24*(k1 + 2*k2 + 2*k3 + k4)
step = ca.Function('step', [X0, U, DT], [X], ['x0', 'u', 'dt'], ['xf'])

# Layer 3: NLP assembly in MX (calls step symbolically per interval)
for k in range(N):
    Xk_end = step(x0=Xk, u=Uk, dt=dt_k)['xf']  # MX expression
    g.append(Xk1 - Xk_end)                        # continuity constraint
```

### map and mapaccum for efficient repeated evaluation

`Function.map(N)` evaluates a function N times independently with horizontally stacked inputs — keeping graph size constant regardless of N:

```python
dyn_mapped = dyn.map(N, 'thread', 4)  # parallel with 4 threads
Xdot_all = dyn_mapped(X_all, U_all)   # (7×N) result
```

`Function.mapaccum('name', N)` handles sequential dependencies (each output feeds the next input), producing a logarithmic-depth graph with `base` option for memory-efficient AD — ideal for single-shooting formulations.

### Inlining behavior

By default, MX graphs retain function calls as opaque nodes. With `expand=True`, expandable calls (SX-based functions) are inlined into a flat SX graph. Per-function control is available via `always_inline=True` and `never_inline=True` options on the Function constructor. Keeping calls as nodes reduces graph construction time and code-generation size; expanding maximizes evaluation speed for small-to-medium problems.

---

## Putting it together: a solver backend wrapper skeleton

Based on the patterns above, here is a complete architecture for a modular mission optimization framework:

```python
import casadi as ca
import numpy as np

class MissionNLP:
    """Modular NLP builder for space mission trajectory optimization."""

    def __init__(self):
        self._w = [];  self._w0 = [];  self._lbw = [];  self._ubw = []
        self._g = [];  self._lbg = [];  self._ubg = []
        self._p = [];  self._p_val = []
        self._J = 0
        self._offset = 0
        self._var_map = {}   # name → (start, end)
        self._solver = None

    def add_variable(self, name, n, lb=-np.inf, ub=np.inf, init=0.0):
        sym = ca.MX.sym(name, n)
        self._w.append(sym)
        self._var_map[name] = (self._offset, self._offset + n)
        self._offset += n
        self._lbw += np.full(n, lb).tolist() if np.isscalar(lb) else list(lb)
        self._ubw += np.full(n, ub).tolist() if np.isscalar(ub) else list(ub)
        self._w0  += np.full(n, init).tolist() if np.isscalar(init) else list(init)
        return sym

    def add_parameter(self, name, n):
        sym = ca.MX.sym(name, n)
        self._p.append(sym)
        return sym

    def add_equality(self, expr):
        n = expr.shape[0]
        self._g.append(expr)
        self._lbg += [0.0] * n
        self._ubg += [0.0] * n

    def add_inequality(self, expr, lb=-np.inf, ub=np.inf):
        n = expr.shape[0]
        self._g.append(expr)
        self._lbg += np.full(n, lb).tolist() if np.isscalar(lb) else list(lb)
        self._ubg += np.full(n, ub).tolist() if np.isscalar(ub) else list(ub)

    def add_cost(self, expr):
        self._J += expr

    def build(self, opts=None):
        w = ca.vertcat(*self._w)
        g = ca.vertcat(*self._g) if self._g else ca.MX(0, 1)
        nlp = {'x': w, 'f': self._J, 'g': g}
        if self._p:
            nlp['p'] = ca.vertcat(*self._p)
        default_opts = {
            'ipopt.tol': 1e-8, 'ipopt.max_iter': 2000,
            'ipopt.linear_solver': 'mumps', 'ipopt.mu_strategy': 'adaptive',
            'ipopt.print_level': 5, 'print_time': True,
        }
        if opts: default_opts.update(opts)
        self._solver = ca.nlpsol('mission_solver', 'ipopt', nlp, default_opts)
        return self._solver

    def solve(self, p_val=None, warm=None):
        kwargs = dict(x0=self._w0, lbx=self._lbw, ubx=self._ubw,
                      lbg=self._lbg, ubg=self._ubg)
        if p_val is not None:
            kwargs['p'] = p_val
        if warm is not None:
            kwargs.update(x0=warm['x'], lam_x0=warm['lam_x'], lam_g0=warm['lam_g'])
        return self._solver(**kwargs)

    def extract(self, sol, name):
        i, j = self._var_map[name]
        return sol['x'][i:j].full().flatten()

    def stats(self):
        return self._solver.stats()
```

This wrapper manages variable registration with automatic index tracking, constraint collection with bound specification, parameter declaration, warm-starting from prior solutions, and clean result extraction by name — exactly the abstraction layer needed to build phase-by-phase mission legs, link them with continuity constraints, and sweep over mission parameters without rebuilding the solver each time.

## Conclusion

The key architectural decisions for a CasADi-based mission optimization backend are: use the **hybrid SX/MX pattern** (SX dynamics wrapped in `ca.Function`, MX NLP assembly), **interleave variables per timestep** for banded Jacobian structure, leverage **parametric NLP formulation** to avoid solver rebuilds during trade studies, and **warm-start aggressively** with prior dual variables when solving sequences of related problems. CasADi's automatic sparse AD eliminates the need to hand-code any derivatives — the symbolic graph provides exact Jacobians and Hessians with sparsity patterns that IPOPT exploits directly. The iteration callback mechanism gives full visibility into solver progress for convergence monitoring and early termination, while the `Function.map` and `mapaccum` constructs keep graph sizes manageable even for hundreds of trajectory segments.