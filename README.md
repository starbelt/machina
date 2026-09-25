# machina

Component-based mission modelling and optimization in Python on [CasADi](https://web.casadi.org/).

You assemble a problem from components — today an orbit-coverage component in the `astro` pack;
budgets, latency timelines and vehicle physics are the packs' planned components — and machina
wires them into one symbolic graph and either hands it to an NLP solver (IPOPT, bonmin) or
composes it into an ODE for simulation and code generation. New capability is a new Python
component, never a schema extension.

machina is the shared core of two projects and is consumed by each as a git submodule:

| Consumer | Uses it for |
|---|---|
| `flyby_ml` (GEO flyby computing thesis) | flyby system design under goodput + latency + power + cost; placement scoring; power-sensitivity sweeps; thesis tables |
| `icarus-dynamics` (Mach Works aircraft GNC) | the component graph, params pipeline, integrator and codegen helpers; later trim and gain tuning |

**Design documentation lives in the Obsidian vault**, under `20 - Research/Projects/Machina/`
(Architecture, Solver Backend, Function Library, Component Graph, Compiler and Params, Decision
Log, Roadmap, plus the math notes). This repo carries code, tests and docstrings; `make help`
lists the developer commands.

## Status

Revived September 2026. The solver backend (Phase 1), the component graph and params pipeline
absorbed from icarus-dynamics (Phase 2), and the `Problem` compiler with the `astro` and `swapc`
packs (Phase 3) are in; about 1150 tests run on Python 3.10 and 3.12 in CI, together with every
example and a two-process determinism gate. Next are `study` (sweeps) and `report` (LaTeX and
Markdown tables and equations with provenance); see the Roadmap note in the vault.

## Install

Requires Python 3.10+.

```bash
python -m pip install -e ".[dev]"
make test
make lint
make examples
```

Extras: `viz` (NetworkX + matplotlib for the NLP graph tool), `study` (pandas; reserved for the
planned `study` layer), `dev` (everything plus pytest and ruff). The plotting examples need
matplotlib and three of them (`flyby_goodput.py`, `least_squares.py`, `rosenbrock_registry.py`)
also NetworkX; both come with the `dev` and `viz` extras.

## A first problem

One component from the `astro` pack, one problem, one solve: the orbit that maximises smooth
coverage of Washington DC over a 200–1600 km altitude box, starting from an ISS-like orbit.

```python
import math

from machina.astro import SingleSatCoverage      # importing a pack declares its signals and frames
from machina.compiler import Problem
from machina.model import Scope

R = 6378.137
i, raan = math.radians(51.6), math.radians(240.0)              # ISS-like start that passes over DC
sat = SingleSatCoverage(target_lat_deg=38.9, target_lon_deg=-77.0, n_sample_points=24,
                        perigee_min_km=200.0, apogee_max_km=1600.0)

prob = Problem([Scope("sat", [sat])], verbose=False)           # scopes give the paths sat/p, sat/f, ...
prob.compile(overrides={                                       # roles and values resolve here; the tool never guesses
    "sat/p": {"x0": R + 500.0, "lb": R + 200.0, "ub": R + 1600.0},
    "sat/f": {"x0": 0.01, "lb": -0.3, "ub": 0.3},
    "sat/g": {"x0": 0.0, "lb": -0.3, "ub": 0.3},
    "sat/h": {"x0": math.tan(i / 2) * math.cos(raan), "lb": -1.5, "ub": 1.5},
    "sat/k": {"x0": math.tan(i / 2) * math.sin(raan), "lb": -1.5, "ub": 1.5},
})
prob.add_cost(-prob.expr("sat/coverage_total").symbol, name="neg_coverage")   # the problem picks the objective

res = prob.build().solve()
print(res.status, res.iterations, "iterations")                # Solve_Succeeded 21 iterations
print("coverage", round(-res.f_opt, 4))                        # 0.1639
print("p =", round(res["sat/p"].item(), 3), "km, on the apogee bound")   # 7978.137 = R + 1600
```

The component owns its physics: the modified equinoctial elements `p f g h k` as variables, the
sample grid, target and sigmoid parameters, the produced `coverage_total` signal, the two
altitude constraints in their well-conditioned squared form, and the scaling that makes the
strict solve converge (at unit scale the strict solve fails with `Invalid_Number_Detected`, and
with the April prototype's loosened `acceptable_tol` it stops early at 0.126; see the vault's MEE
Numerics note). The problem decides roles, values and the objective; every number it uses comes
from a declared default with a provenance code, an override, or a params CSV, and a value with
none of those is an error, not a guess. Results are in physical units with named access to
shadow prices (`res.constraint("sat/apogee_altitude").multiplier`), bound multipliers, parameter
sensitivities and the per-term cost breakdown; `solve(warm_start=previous)` reuses the duals.

The examples in `examples/` all run headless (`make examples`). Two build on `Problem`:
`fleet_budget.py` is the file to copy when you write your own component (two payloads in scopes,
a budget that reads both by absolute path, roles and provenance, the shadow price of the cap),
and `coverage_optimization.py` is the problem above with plots, the April 2026 prototype's recipe
for comparison and the derived Keplerian elements. Four wire a `SolverBackend` by hand:
`flyby_goodput.py` is a three-product goodput problem on a shared compute budget (the `swapc`
pack's factories), `least_squares.py` shows matrix parameters, and `rosenbrock.py` and
`rosenbrock_registry.py` write the same objective without and with the factory registry.
`kepler_propagation.py` only calls the `astro` factories, propagating with the universal-variable
solver across orbit regimes. The plotting examples save nothing: they open windows under an
interactive matplotlib backend and skip `plt.show()` under a non-interactive one (`make examples`
sets `MPLBACKEND=Agg`).

## Package map

| Path | Contents | Imported by `import machina`? |
|---|---|---|
| `src/machina/solver/` | `SolverBackend` (CasADi `nlpsol` wrapper, MX), `SolutionResult`, public records, per-plugin options, scaling, warm start | yes |
| `src/machina/model/` | signal registry with frames, `Component`/`Declaration`/`Quantity`/`Constraint`/`Cost`/`Scope`, the `Builder` (SX for simulation, MX for the NLP), `FunctionDescriptor`/`SymbolDescriptor` | yes |
| `src/machina/compiler/` | `Problem`: declare → roles → values → leaves → wire → register → build → solve | yes |
| `src/machina/params/` | `param()` declarations, the provenance-tagged CSV, `machina params sync\|check`, injectable lint contract | yes |
| `src/machina/units.py`, `src/machina/cli.py` | the SI unit allow-list that signals, quantities and params are checked against; the `machina params sync\|check` command line | `units.py` yes, `cli.py` no |
| `src/machina/library/` | the factory registry and the generic factories (`cost.*`, `constraint.linear`, `util.*`), numeric guards | yes (registers the generic factories) |
| `src/machina/sim/`, `src/machina/codegen/` | RK4 step function; dense C export, manifest merge, layout blocks | no (light; import machina.sim / machina.codegen explicitly) |
| `src/machina/astro/` | frames `eci/ecef/lvlh`, MEE/KOE transforms, universal-variable Kepler, elevation geometry, smooth coverage, `SingleSatCoverage` | no — importing it declares its signals and registers `transform.*`, `geometry.*`, `cost.smooth_coverage` |
| `src/machina/swapc/` | goodput and latency factories for the thesis cost function (budget signals and components come with the first consumer) | no — importing it registers `cost.sigmoid_goodput`, `cost.aggregate_goodput`, `util.ttp_computation` |
| `src/machina/rigid/`, `src/machina/aero/` | frames `ned/frd`, rigid-body signals and quaternion kinematics (skeleton); placeholder | no |
| `src/machina/report/` | planned (Phase 4): LaTeX/Markdown tables and equations of a compiled problem, with provenance | no |
| `src/machina/viz.py` | NLP-level graph view used by three examples (needs the `viz` extra); a component-level graph with Mermaid/DOT export is planned | no |
| `examples/`, `tests/`, `scripts/` | runnable examples; pytest suite; the determinism probe behind `make determinism` | — |

`import machina` is the core only: no pack, no plotting library, nothing that needs an orbit to
exist. Packs are imported explicitly by the problem that uses them, which is also what registers
their factories and declares their signals; a test (`tests/test_import_graph.py`) enforces this.

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
