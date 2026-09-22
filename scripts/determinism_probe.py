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

Everything is built from fresh registries: a three-component damped
oscillator (one component produces a matrix with structural zeros, which is
what makes :func:`machina.codegen.dense` necessary), a small params table
through ``sync.apply``, the serialised ``f``/``g`` Functions, generated C of
the dense RK4 step, and a ``MANIFEST.json`` merged from per-stage blocks.
Nothing is differentiated before it is serialised: CasADi caches derivative
Functions inside the objects that ``serialize()`` writes.
"""

import argparse
import sys
from pathlib import Path

import casadi as ca

from machina.codegen import dense, generate_c, layout_block, merge, schema_hash, sha256_of
from machina.model import Aggregation, Builder, Component, Declaration, SignalRegistry
from machina.params import ParamRegistry, param, sync, table
from machina.sim import build_step_function

__all__ = ["run"]


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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def run(out: Path) -> None:
    out = Path(out)
    reg = registry()
    builder = Builder([Kinematics(), Dynamics(), SpringDamper()], registry=reg).declare()
    model = builder.build()
    step = dense(build_step_function(model["f"]), "plant_step")

    plant = out / "plant"
    _write(plant / "f_system.casadi", model["f"].serialize())
    _write(plant / "g_system.casadi", dense(model["g"]).serialize())
    _write(plant / "plant_step.c", generate_c(step))

    params_csv = out / "params" / "params.csv"
    table.write(params_csv, params_rows())

    layout = layout_block(builder.state_order, reg)
    merge(out / "MANIFEST.json", {"plant": {
        "state_layout": layout,
        "input_layout": layout_block(builder.input_order, reg),
        "algebraic_layout": layout_block(builder.algebraic_order, reg),
        "schema_hash": schema_hash(layout),
        "files": {name: sha256_of(plant / name)
                  for name in ("f_system.casadi", "g_system.casadi", "plant_step.c")},
    }})
    merge(out / "MANIFEST.json", {"params": {
        "names": [row["name"] for row in table.read(params_csv)],
        "sha256": sha256_of(params_csv),
    }})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
