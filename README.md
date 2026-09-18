# machina

Component-based mission modelling and optimization in Python on [CasADi](https://web.casadi.org/).

You assemble a problem from a library of components (orbits, size/weight/power/cost budgets,
latency timelines, vehicle physics). machina wires them into one symbolic graph and either hands
it to an NLP solver (IPOPT, bonmin) or composes it into an ODE for simulation and code
generation. New capability is a new Python component, never a schema extension.

machina is the shared core of two projects and is consumed by each as a git submodule:

| Consumer | Uses it for |
|---|---|
| `flyby_ml` (GEO flyby computing thesis) | flyby system design under goodput + latency + power + cost; placement scoring; power-sensitivity sweeps; thesis tables |
| `icarus-dynamics` (Mach Works aircraft GNC) | the component graph, params pipeline, integrator and codegen helpers; later trim and gain tuning |

**Design documentation lives in the Obsidian vault**, under `20 - Research/Projects/Machina/`
(Architecture, Solver Backend, Function Library, Component Graph, Compiler and Params, Decision
Log, Roadmap, plus the math notes). This repo carries code, tests, docstrings, and `CLAUDE.md`
with the build commands and the hard rules.

## Status

Revived September 2026. The April 2026 code (solver backend, function library, agent types,
universal-variable Kepler propagation, single-satellite coverage optimization; 369 tests) is
intact under `src/machina/`. The restructure that merges in the icarus-dynamics component graph
and params pipeline is in progress; see the Roadmap note in the vault.

## Install

Requires Python 3.10+.

```bash
python -m pip install -e ".[dev]"
make test
make lint
```

Extras: `viz` (NetworkX + matplotlib for the NLP graph tool), `study` (pandas for sweep tables),
`dev` (everything plus pytest and ruff).

## A first problem

```python
import casadi as ca
from machina.solver import SolverBackend
from machina.blocks import registry

solver = SolverBackend(verbose=False)          # solver="bonmin" for discrete variables
xy = solver.add_variable("xy", 2, lb=-2.0, ub=2.0, initial_guess=0.0)

rosenbrock = registry.get("cost.rosenbrock")(a=1.0, b=100.0)   # a FunctionDescriptor
solver.add_cost(rosenbrock(xy=xy), name="rosenbrock")

solver.build()
result = solver.solve()
print(result.success, result.status, result["xy"], result.f_opt)
```

The backend is plugin-agnostic (IPOPT, bonmin, sqpmethod, fatrop) and works in physical units
throughout: `scale=` on a variable or constraint conditions the problem without changing what you
read back. Parameters carry stored values (`add_parameter(value=...)`, `set_parameter`), bounds
can be edited or variables fixed after `build()`, `solve(warm_start=previous_result)` reuses the
previous duals, and results give named access to shadow prices (`result.constraint("power").multiplier`),
parameter sensitivities and the per-term cost breakdown.

More in `examples/`: `least_squares.py` (matrix parameters), `flyby_goodput.py` (three-product
goodput with a shared compute budget), `kepler_propagation.py` (universal-variable propagation
across orbit regimes), `coverage_optimization.py` (the full agent + compiler stack).

## Package map

| Path | Contents |
|---|---|
| `src/machina/solver/` | `SolverBackend` (CasADi `nlpsol` wrapper, MX), `SolutionResult`, public records, per-plugin options |
| `src/machina/blocks/` | `FunctionDescriptor`, `SymbolDescriptor`, the factory registry, and the factory library (`cost`, `constraint`, `util`, `transforms`, `geometry`) |
| `src/machina/agents/` | `AgentType` declare/build lifecycle and `SingleSatCoverage` |
| `src/machina/compiler/` | `CompilerStub`: declare → assign roles → build → register → solve |
| `src/machina/viz.py` | NLP graph drawing (needs the `viz` extra) |
| `examples/`, `tests/` | runnable examples; pytest suite |

The target layout after the restructure (core `model/`, `params/`, `sim/`, `codegen/`, `study/`,
`report/`; packs `astro/`, `swapc/`, `rigid/`, `aero/`) is described in the vault Architecture note.

## Using machina from another repo

```bash
git submodule add https://github.com/starbelt/machina external/machina
```

- Conda/pip project: `pip install -e external/machina` into the project environment (needs
  `casadi>=3.6`).
- Nix project: put `external/machina/src` on `PYTHONPATH` in the dev shell; machina declares only
  `casadi` and `numpy` as runtime dependencies.

Never put two copies of machina on one `sys.path`; its registries are process-global.

## Licence

MIT. See `LICENSE`.
