"""
Build a small model and its artifacts from scratch, so two runs can be diffed.

    python scripts/determinism_probe.py --out build/det-a
    python scripts/determinism_probe.py --out build/det-b
    diff -r build/det-a build/det-b

``make determinism`` runs exactly that, in two separate processes -- two
calls in one process would share a graph built once and test only the writer.
icarus-dynamics runs the same gate over artifacts built from machina, and the
failure it exists to catch is an ordering that came from a set: Python
randomises string hashing per process, so a set-derived order changes the
expression graph between runs.

Everything is built from fresh registries, one writer per stage (``ARTIFACTS``
below; later phases append a writer rather than restructure this file): a
three-component damped oscillator (one component produces a matrix with
structural zeros, which is what makes :func:`machina.codegen.dense`
necessary), a small params table through ``sync.apply``, the coverage NLP of
``machina.astro`` compiled through ``Problem``, and a ``MANIFEST.json`` merged
from the per-stage blocks. Nothing is differentiated before it is serialised:
CasADi caches derivative Functions inside the objects that ``serialize()``
writes, so the NLP is taken from ``nlp_expressions()`` before ``build()`` ever
constructs a solver.
"""

import argparse
import json
import sys
from pathlib import Path

import casadi as ca

from machina.astro import SingleSatCoverage
from machina.astro import signals as astro_signals
from machina.codegen import dense, generate_c, layout_block, merge, schema_hash, sha256_of
from machina.compiler import Problem
from machina.model import Aggregation, Builder, Component, Declaration, Scope, SignalRegistry
from machina.params import ParamRegistry, param, sync, table
from machina.sim import build_step_function

__all__ = ["run", "ARTIFACTS"]


# --- stage 1: a plant, exported the way icarus exports one ------------------------------------

def registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.declare_frame("line", doc="A one-dimensional track", family="inertial")
    reg.declare("position", 1, "m", frame="line", doc="Displacement from rest")
    reg.declare("velocity", 1, "m/s", frame="line", doc="Rate of displacement")
    reg.declare("force", 1, "N", frame="line", aggregation=Aggregation.SUM, doc="Net force")
    reg.declare("stiffness", (2, 2), "N/m", doc="Diagonal stiffness, reported for inspection")
    reg.declare("drive", 1, "N", frame="line", doc="Applied force command")
    return reg


class Kinematics(Component):
    def declare(self):
        return Declaration(states=("velocity",), derivatives=("position",))

    def build(self, helpers):
        v = ca.SX.sym("velocity")
        return {"f": ca.Function("kinematics_f", [v], [v], ["velocity"], ["position_dot"])}


class Dynamics(Component):
    def declare(self):
        return Declaration(states=("velocity",), algebraic=("force",),
                           derivatives=("velocity",))

    def build(self, helpers):
        v, force = ca.SX.sym("velocity"), ca.SX.sym("force")
        return {"f": ca.Function("dynamics_f", [v, force], [force / 2.0],
                                 ["velocity", "force"], ["velocity_dot"])}


class SpringDamper(Component):
    def declare(self):
        return Declaration(states=("position", "velocity"), inputs=("drive",),
                           produces=("force", "stiffness"))

    def build(self, helpers):
        x, v, u = ca.SX.sym("position"), ca.SX.sym("velocity"), ca.SX.sym("drive")
        stiffness = ca.diag(ca.vertcat(8.0, 0.0))   # a structural zero, on purpose
        return {"g": ca.Function("spring_damper_g", [x, v, u],
                                 [u - 8.0 * x - 0.4 * v, stiffness],
                                 ["position", "velocity", "drive"], ["force", "stiffness"])}


def write_plant(out: Path) -> dict:
    """Serialised ``f``/``g``, generated C of the dense RK4 step, and the layout block."""
    reg = registry()
    builder = Builder([Kinematics(), Dynamics(), SpringDamper()], registry=reg).declare()
    model = builder.build()
    step = dense(build_step_function(model["f"]), "plant_step")

    plant = out / "plant"
    _write(plant / "f_system.casadi", model["f"].serialize())
    _write(plant / "g_system.casadi", dense(model["g"]).serialize())
    _write(plant / "plant_step.c", generate_c(step))

    layout = layout_block(builder.state_order, reg)
    return {"plant": {
        "state_layout": layout,
        "input_layout": layout_block(builder.input_order, reg),
        "algebraic_layout": layout_block(builder.algebraic_order, reg),
        "schema_hash": schema_hash(layout),
        "files": {name: sha256_of(plant / name)
                  for name in ("f_system.casadi", "g_system.casadi", "plant_step.c")},
    }}


# --- stage 2: a params table ------------------------------------------------------------------

def params_rows():
    reg = ParamRegistry()
    param("SPRING_K", type="f64", unit="N/m", lo=0.0, hi=100.0, mutability="design",
          doc="Spring stiffness", registry=reg)
    param("DAMPER_C", type="f32", unit="N*s/m", lo=0.0, hi=10.0, mutability="design",
          doc="Damping coefficient", registry=reg)
    param("N_STEPS", type="u16", unit="1", lo=1, hi=10000, mutability="fixed",
          doc="Steps per scenario", default=100, registry=reg)
    rows = sync.apply(reg.all(), [])
    filled = {"SPRING_K": ("8.0", "P", "probe"), "DAMPER_C": ("0.4", "E", "probe"),
              "N_STEPS": ("100", "A", "probe")}
    for row in rows:
        row["value"], row["provenance"], row["source"] = filled[row["name"]]
    return rows


def write_params(out: Path) -> dict:
    """The table ``sync`` would write, filled with values and provenance codes."""
    params_csv = out / "params" / "params.csv"
    table.write(params_csv, params_rows())
    return {"params": {
        "names": [row["name"] for row in table.read(params_csv)],
        "sha256": sha256_of(params_csv),
    }}


# --- stage 3: the coverage NLP through Problem ------------------------------------------------

def coverage_problem() -> Problem:
    """The motivating problem of Math/MEE Numerics, compiled but never built.

    A fresh signal registry with the astro pack's frames and signals declared into
    it, one ``SingleSatCoverage`` in a scope (so the paths read ``sat/p`` ...), the
    default sourced values, and ``-coverage_total`` as the objective.
    """
    reg = SignalRegistry()
    astro_signals.declare_into(reg)
    problem = Problem([Scope("sat", [SingleSatCoverage()])], registry=reg, verbose=False)
    problem.compile()
    problem.add_cost(-problem.expr("sat/coverage_total").symbol, name="neg_coverage")
    return problem


def _entry(record) -> dict:
    entry = {"name": record.name, "start": record.slice.start, "stop": record.slice.stop}
    shape = getattr(record, "shape", None)
    if shape is not None:
        entry["shape"] = list(shape)
    return entry


def write_coverage(out: Path) -> dict:
    """The NLP as one serialised Function ``(x, p) -> (f, g)`` plus its names and layout.

    Taken from ``nlp_expressions()`` before ``build()``: no solver is constructed and
    nothing has been differentiated, so the bytes depend only on the graph.
    """
    backend = coverage_problem().backend
    nlp = backend.nlp_expressions()
    fn = ca.Function("coverage_nlp", [nlp["x"], nlp["p"]], [nlp["f"], nlp["g"]],
                     ["x", "p"], ["f", "g"])

    folder = out / "coverage"
    _write(folder / "coverage_nlp.casadi", fn.serialize())
    names = {
        "variables": [_entry(r) for r in backend.variables()],
        "parameters": [_entry(r) for r in backend.parameters()],
        "constraints": [_entry(r) for r in backend.constraints()],
        "costs": [r.name for r in backend.cost_terms()],
    }
    _write(folder / "names.json", json.dumps(names, indent=2, sort_keys=True) + "\n")
    return {"coverage": {
        "n_x": backend.n_x, "n_p": backend.n_p, "n_g": backend.n_g,
        "files": {name: sha256_of(folder / name)
                  for name in ("coverage_nlp.casadi", "names.json")},
    }}


# --- the probe --------------------------------------------------------------------------------

# (stage name, writer). A writer builds its artifacts under ``out`` and returns the block it
# contributes to MANIFEST.json. Append new stages; the manifest is merged in this order.
ARTIFACTS = (
    ("plant", write_plant),
    ("params", write_params),
    ("coverage", write_coverage),
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def run(out: Path) -> None:
    out = Path(out)
    for _stage, writer in ARTIFACTS:
        merge(out / "MANIFEST.json", writer(out))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
