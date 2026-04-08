# Layer 1 — SolverBackend Interface Specification

**Status:** Implemented
**Last Updated:** 2026-04-03

---

## Internal State (initialized in `__init__`)

| Name | Type | Description |
|------|------|-------------|
| `solver_name` | `str` | Solver plugin name (default `'ipopt'`). **Public attribute.** |
| `solver_opts` | `dict` | Options from `__init__`. **Public attribute.** |
| `_w` | `list[ca.MX]` | MX variable symbols |
| `_w0` | `list[float]` | Initial guesses, flat |
| `_lbw` | `list[float]` | Variable lower bounds, flat |
| `_ubw` | `list[float]` | Variable upper bounds, flat |
| `_g` | `list[ca.MX]` | Constraint expressions |
| `_lbg` | `list[float]` | Constraint lower bounds, flat |
| `_ubg` | `list[float]` | Constraint upper bounds, flat |
| `_p` | `list[ca.MX]` | Parameter symbols |
| `_J` | `ca.MX` | Accumulated cost expression (scalar, starts at `ca.MX.zeros(1,1)`) |
| `_offset` | `int` | Current position in global decision vector |
| `_p_offset` | `int` | Current position in global parameter vector |
| `_var_map` | `dict[str, tuple[int, int, tuple[int, int]]]` | Variable name → `(start_index, end_index, (rows, cols))` |
| `_param_map` | `dict[str, tuple[int, int, tuple[int, int]]]` | Parameter name → `(start_index, end_index, (rows, cols))` |
| `_names` | `set[str]` | Shared namespace for collision detection — **variables and parameters only**. Constraint and cost term names are NOT stored here. |
| `_cost_terms` | `list[tuple[str, ca.MX]]` | `(name, expr)` pairs for objective breakdown reporting |
| `_constraint_names` | `list[tuple[str, int]]` | `(name, n_rows)` pairs, parallel to `_g`, for reporting |
| `_solver` | `ca.Function or None` | The `nlpsol` object, `None` until `build()` |
| `_built` | `bool` | `False` until `build()` completes |
| `_solved` | `bool` | `False` until the first `solve()` call completes. Guards `stats()` against returning stale/empty data. |

---

## Public Methods

### `__init__(solver='ipopt', solver_opts=None)`

Initialize the solver backend.

- `solver`: Solver plugin name (`'ipopt'`, `'sqpmethod'`, etc.).
- `solver_opts`: Solver-specific options, passed through to CasADi as-is. Accepts nested dicts (`{'ipopt': {'tol': 1e-8}}`) or flat dot-notation (`{'ipopt.tol': 1e-8}`). Applied on top of hardcoded defaults in `build()`.
- Initializes all accumulators to empty.
- `_J` is initialized as `ca.MX.zeros(1, 1)` (not Python `0`) so that the first `add_cost()` call produces a valid MX expression.

---

### `add_variable(name, n, lb=-inf, ub=inf, initial_guess=0.0) -> ca.MX`

Create a decision variable and register it in the global decision vector.

**Args:**
- `name`: Unique name. Must not already exist in `_names`.
- `n`: Size of the variable. Pass an `int` for an `n × 1` column vector, or a `(rows, cols)` tuple for a matrix. The returned MX symbol has the requested shape; bounds and initial guess broadcast to `rows * cols` elements.
- `lb`: Lower bound. Scalar broadcasts to all elements. Can also be a list/array of length `numel` (column-major order for matrices).
- `ub`: Upper bound. Same broadcast rules as `lb`.
- `initial_guess`: Starting value for solver. Same broadcast rules.

**Returns:** MX symbolic variable with the requested shape.

**Behavior:**
- Raises `ValueError` if `name` already exists in `_names`.
- For `n: int`: creates `ca.MX.sym(name, n)` and appends directly to `_w`.
- For `n: (rows, cols)`: creates `ca.MX.sym(name, rows, cols)` and appends `ca.vec(var)` to `_w`. `ca.vec` flattens column-major so the matrix occupies a contiguous column-vector block in the global decision vector, as `nlpsol` requires. The full matrix symbol is returned to the caller for use in expressions.
- Broadcasts `lb`, `ub`, `initial_guess` to flat lists of length `numel = rows * cols`.
- Array-valued bounds/guesses are also flattened (`np.asarray(...).flatten()`) to handle matrix-shaped inputs.
- Records `_var_map[name] = (offset, offset + numel, (rows, cols))`. For vector variables, `cols = 1`.
- Adds `name` to `_names`, advances `_offset` by `numel`.

---

### `add_parameter(name, n) -> ca.MX`

Create a fixed parameter (not optimized, value set at solve time).

**Args:**
- `name`: Unique name. Must not already exist in `_names`.
- `n`: Size of the parameter. Pass an `int` for an `n × 1` column vector, or a `(rows, cols)` tuple for a matrix. The returned MX symbol has the requested shape.

**Returns:** MX symbolic parameter with the requested shape.

**Behavior:**
- Raises `ValueError` if `name` already exists in `_names`.
- For `n: int`: creates `ca.MX.sym(name, n)` and appends directly to `_p`.
- For `n: (rows, cols)`: creates `ca.MX.sym(name, rows, cols)` and appends `ca.vec(param)` to `_p`. Same column-major flattening rationale as `add_variable`.
- Records `_param_map[name] = (p_offset, p_offset + numel, (rows, cols))`. For vector parameters, `cols = 1`.
- Adds `name` to `_names`, advances `_p_offset` by `numel`.

---

### `add_constraint(expr, lb=-inf, ub=inf, name=None)`

Add a constraint to the problem.

**Args:**
- `expr`: MX expression (can be vector-valued). Write the expression so the bound applies to its value directly (e.g., to enforce `x >= 2`, pass `expr=x, lb=2`).
- `lb`: Lower bound. Scalar broadcasts to match `expr.shape[0]`.
- `ub`: Upper bound. Scalar broadcasts to match `expr.shape[0]`.
- `name`: Optional name for debugging/reporting. Does **not** enter `_names` — constraint names need not be globally unique.

**Behavior:**
- Broadcasts `lb`, `ub` to flat lists of length `expr.shape[0]`.
- Appends `expr` to `_g`, bounds to `_lbg`/`_ubg`.
- Appends `(name, expr.shape[0])` to `_constraint_names`.
- Does **not** modify `_names`.

**Constraint types (by bound values):**
- Equality: `lb == ub` (e.g., `lb=0, ub=0`). IPOPT detects these automatically.
- Double-sided inequality: both bounds finite.
- One-sided: leave one as `+/-inf`.

---

### `add_equality(expr, name=None)`

Convenience wrapper. Calls `add_constraint(expr, lb=0, ub=0, name=name)`.

Write the expression so it equals zero at the desired solution:
e.g., to enforce `x + y == 1`, pass `expr = x + y - 1`.

---

### `add_cost(expr, name=None)`

Add a cost term to the objective.

**Args:**
- `expr`: Scalar MX expression (shape must be `(1, 1)`).
- `name`: Optional name for objective breakdown reporting. Does **not** enter `_names`.

**Behavior:**
- Raises `ValueError` if `expr.shape != (1, 1)`. Reduce vector expressions first (e.g., `ca.sumsqr(v)`).
- Adds `expr` to `_J` (cumulative sum).
- Appends `(name, expr)` to `_cost_terms`.
- Does **not** modify `_names`.

---

### `build(opts=None) -> ca.Function`

Assemble the NLP and create the solver.

**Args:**
- `opts`: Override or extend solver options for this build only.

**Returns:** The `ca.nlpsol` solver Function (also stored as `_solver`).

**Behavior:**
- Raises `RuntimeError` if `_w` is empty (no variables registered).
- Raises `RuntimeError` if `build()` has already been called on this instance.
- If no cost terms were registered, emits `UserWarning` (via `warnings.warn`, `stacklevel=2`) before proceeding. Allows feasibility problems but surfaces the likely mistake.
- Concatenates decision variables: `x = ca.vertcat(*self._w)`.
- Concatenates constraints: `g = ca.vertcat(*self._g) if self._g else ca.MX(0, 1)`.
  - `ca.MX(0, 1)` (empty column vector) is used rather than `ca.MX.zeros(0)` (scalar) to satisfy CasADi's dimension expectations.
- Assembles NLP dict: `{'x': x, 'f': self._J, 'g': g}`.
- Conditionally adds `'p': ca.vertcat(*self._p)` to NLP dict **only if `_p` is non-empty**. If `'p'` were always present, every `solver()` call would require a `p=` argument even for parameter-free problems.
- Merges options with precedence: hardcoded defaults → `solver_opts` (from `__init__`) → `opts` (from `build()`).
- Calls `ca.nlpsol('solver', self.solver_name, nlp, merged_opts)`.
- Sets `_built = True`.

**Default solver options:**

| Option | Default | Level |
|--------|---------|-------|
| `print_time` | `True` | CasADi |
| `ipopt.tol` | `1e-8` | IPOPT |
| `ipopt.max_iter` | `2000` | IPOPT |
| `ipopt.linear_solver` | `'mumps'` | IPOPT |
| `ipopt.mu_strategy` | `'adaptive'` | IPOPT |
| `ipopt.print_level` | `5` | IPOPT |

**Note on `expand`:** The `expand=True` option (converts MX→SX at build time for faster evaluation) is not set by default. It is incompatible with `ca.Callback` objects (iteration callbacks). Enable via `solver_opts` or `build(opts)` when callbacks are not in use.

---

### `solve(p_val=None) -> SolutionResult`

Call the solver and return results.

**Args:**
- `p_val`: Numeric values for all registered parameters, as a flat list or numpy array, in registration order. Must be provided when parameters are registered; omit (or pass `None`) when none are.

**Returns:** `SolutionResult`.

**Behavior:**
- Raises `RuntimeError` if `_built` is `False`.
- If parameters registered: raises `ValueError` if `p_val` is `None` or the wrong length. The length check is done before the solver call to surface a clear error rather than a CasADi dimension mismatch.
- If no parameters registered: `p_val=None` is valid; `p` keyword is omitted from the solver call.
- `p_val` is normalized via `np.asarray(p_val).flatten()` before being passed to the solver. This ensures CasADi receives a consistent 1-D input regardless of whether the caller supplied a flat list, a 2-D array, or a column vector. The normalized array (`p_arr`) is what is actually passed; the original `p_val` is discarded after validation. For matrix parameters, values must be supplied in column-major order to match the `ca.vec` layout used at registration.
- Calls solver with `x0=_w0`, `lbx=_lbw`, `ubx=_ubw`, `lbg=_lbg`, `ubg=_ubg`, and `p=p_arr` (if parameters exist).
- Sets `_solved = True` after the solver call returns (regardless of convergence status).
- Extracts named variable values by slicing `sol['x']` via `_var_map`, converting each slice with `.full().flatten()`. Matrix variables (`cols > 1`) are subsequently reshaped to `(rows, cols)` using `np.reshape(..., order='F')` (Fortran/column-major order), reversing the `ca.vec` flattening applied at registration. Vector variables are returned as 1-D arrays.
- Determines `success` from IPOPT `return_status`: `True` if `'Solve_Succeeded'` or `'Solved_To_Acceptable_Level'`, `False` otherwise.
- Converts `f_opt` with `float(sol['f'])`.
- Converts `g_opt`, `lam_x`, `lam_g` with `.full().flatten()`.
- Converts `lam_p` with `.full().flatten()` if parameters exist, otherwise `None`.
- Constructs and returns `SolutionResult`.

**Future extension (deferred):** Warm-starting support via optional `lam_x0` and `lam_g0` arguments. The `SolutionResult` already stores `lam_x` and `lam_g`, so the round-trip data is available.

---

### `extract(sol, name) -> np.ndarray`

Slice a named variable's values from a raw CasADi solution dict.

Lower-level alternative to `result.x_opt[name]`. Works directly on the CasADi solution dict (e.g., `result.raw_sol`).

**Args:**
- `sol`: Dict containing the `'x'` key (raw CasADi solution or `result.raw_sol`).
- `name`: Variable name as registered with `add_variable`.

**Returns:** 1-D numpy array for vector variables; `(rows, cols)` numpy array for matrix variables (same reshape logic as `solve()`).

---

### `stats() -> dict`

Return solver statistics from the last solve call. Delegates to `self._solver.stats()`.

Key fields for IPOPT: `return_status`, `iter_count`, `t_wall_total`.

**Raises:** `RuntimeError` if `_solved` is `False`. CasADi's `stats()` returns stale or empty data before any solve has run; this guard surfaces that as an explicit error rather than silent incorrect output.

---

## SolutionResult (dataclass)

```python
@dataclass
class SolutionResult:
    success: bool                          # True if solver converged
    x_opt: Dict[str, np.ndarray]           # Variable name → optimal value
    f_opt: float                           # Optimal objective value
    g_opt: np.ndarray                      # Constraint values at optimum
    lam_x: np.ndarray                      # Lagrange multipliers (variable bounds)
    lam_g: np.ndarray                      # Lagrange multipliers (constraints)
    stats: dict                            # Solver statistics
    raw_sol: dict                          # Raw CasADi solution dict
    lam_p: Optional[np.ndarray] = None    # Parameter sensitivities (None if no parameters)

    def __getitem__(self, name: str) -> np.ndarray:
        """Shorthand: result['sma'] → result.x_opt['sma']."""
        return self.x_opt[name]
```

**Field ordering note:** `lam_p` is declared last because it is the only field with a default value (`None`). Python dataclasses require all fields with defaults to follow all fields without defaults.

**`success` definition:** `True` when IPOPT `return_status` is `'Solve_Succeeded'` or `'Solved_To_Acceptable_Level'`. `False` for all other statuses (infeasible, max iterations exceeded, etc.).

**`lam_p` note:** Nonzero values require `{'calc_lam_p': True}` in solver options. Set to `None` (not an empty array) when no parameters were registered, so callers can test `if result.lam_p is not None` without special-casing an empty array.

**`x_opt` shapes:** Vector variables (registered with `int` n) produce 1-D arrays of length `n`. Matrix variables (registered with a `(rows, cols)` tuple) produce `(rows, cols)` 2-D arrays, reshaped from the flat decision vector using column-major order (`order='F'`) to reverse the `ca.vec` flattening applied at registration.

**DM → numpy conversions:** All numeric fields extracted from the CasADi solution dict are converted from `ca.DM` to numpy arrays via `.full().flatten()`, or to `float()` for scalars. Callers never interact with CasADi types after `solve()` returns.

---

## Future Extensions (documented, not implemented)

### Warm-starting

`solve()` will accept optional `lam_x0` and `lam_g0` arguments to warm-start from a prior solution's dual variables. `SolutionResult` already stores `lam_x` and `lam_g`, so the round-trip data is available. IPOPT warm-start options (`warm_start_init_point`, `warm_start_bound_push`, etc.) would be passed via `solver_opts`.

### Iteration callback

CasADi supports an `iteration_callback` option that invokes a `ca.Callback` subclass at every IPOPT iteration, receiving primal variables, objective, constraints, and duals. Useful for convergence monitoring and early termination (return nonzero from `eval()` to stop). Incompatible with `expand=True`. Will be added when needed for convergence debugging on the real flyby problem.
