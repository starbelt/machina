"""
The same model definition must produce the same bytes, in any process.

icarus-dynamics runs a two-process byte-diff over its exported artifacts in
CI, and machina has to keep that gate passing when the submodule swap lands.
The failure mode is specific: Python randomises string hashing per process,
so iterating a set gives a run-dependent order, which gives a run-dependent
CasADi expression graph, which fails the diff for reasons nobody enjoys
finding. Every ordering in the builder therefore comes from the registry's
declaration order or from the component list the caller wrote.

The subprocess test below is the one that would actually catch a regression:
two interpreters with different ``PYTHONHASHSEED`` values, comparing the
serialised graph. The in-process tests are the fast version.

The full two-process export probe, covering the params table and the
manifest as well as the graph, arrives with ``make determinism`` in Phase 2b.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from synthetic_model import Budget, make_registry, point_mass

from machina.model import Builder

pytestmark = pytest.mark.requires_casadi

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"

PROBE = """
import hashlib, sys
sys.path[:0] = [%r, %r]
from synthetic_model import Budget, make_registry, point_mass
from machina.model import Builder

builder = Builder([*point_mass(), Budget()], registry=make_registry()).declare()
model = builder.build()
blob = b"".join(model[key].serialize().encode() for key in ("f", "g", "h", "J"))
layout = repr([builder.state_order, builder.algebraic_order,
               builder.input_order, builder.quantity_order]).encode()
print(hashlib.sha256(blob + layout).hexdigest())
"""


def digest_in_subprocess(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", PROBE % (str(SRC), str(HERE))],
                         capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip()


def digest_in_process() -> str:
    builder = Builder([*point_mass(), Budget()], registry=make_registry()).declare()
    model = builder.build()
    blob = b"".join(model[key].serialize().encode() for key in ("f", "g", "h", "J"))
    return hashlib.sha256(blob).hexdigest()


class TestTheGraphIsReproducible:

    def test_two_builds_in_one_process_are_byte_identical(self):
        assert digest_in_process() == digest_in_process()

    def test_two_processes_with_different_hash_seeds_agree(self):
        """The real gate. A set iterated anywhere in the builder breaks this."""
        assert digest_in_subprocess("0") == digest_in_subprocess("524287")

    def test_a_third_seed_agrees_too(self):
        assert digest_in_subprocess("1") == digest_in_subprocess("99991")


class TestOrderingComesFromDeclarations:

    def test_the_vector_orders_are_lists_not_sets(self):
        builder = Builder([*point_mass(), Budget()], registry=make_registry()).declare()
        for order in (builder.state_order, builder.algebraic_order,
                      builder.input_order, builder.quantity_order):
            assert isinstance(order, list)

    def test_the_signal_registry_hands_back_declaration_order(self):
        registry = make_registry()
        assert list(registry.all()) == ["pos", "vel", "force", "mass", "thrust_cmd",
                                        "separation"]

    def test_no_loop_in_the_model_layer_iterates_a_set_literal(self):
        """Ordering must never come from a set. Membership tests are fine.

        This reads the source rather than the behaviour because the behaviour
        is only wrong one process in a few hundred.
        """
        import ast

        offenders = []
        for path in sorted((SRC / "machina" / "model").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                iterables = []
                if isinstance(node, ast.For):
                    iterables = [node.iter]
                elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                       ast.GeneratorExp)):
                    iterables = [gen.iter for gen in node.generators]
                for iterable in iterables:
                    if isinstance(iterable, (ast.Set, ast.SetComp)):
                        offenders.append(f"{path.name}:{iterable.lineno}")
        assert offenders == []
