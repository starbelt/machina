"""
The same model definition must produce the same bytes, in any process.

icarus-dynamics runs a two-process byte-diff over its exported artifacts in
CI, and machina has to keep that gate passing when the submodule swap lands.
The failure mode is specific: Python randomises string hashing per process,
so iterating a set gives a run-dependent order, which gives a run-dependent
CasADi expression graph, which fails the diff for reasons nobody enjoys
finding. Every ordering in the builder therefore comes from the registry's
declaration order or from the component list the caller wrote.

The model the subprocess probe builds is chosen so that every ordering the
builder computes reaches the serialised bytes: two scopes (scope order), a
SUM signal with two producers read by a component written before them
(topological order and summation order), quantities (quantity layout), and
constraints and costs from components that run out of declaration order.

The full two-process export probe, covering the params table and the
manifest as well as the graph, is ``make determinism`` (Phase 2b).
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_casadi

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
GUARDED = ("model", "library", "rigid", "params", "sim", "codegen")

MODEL = """
from synthetic_model import (Budget, Drag, Dynamics, Kinematics, MassSource, Thruster,
                             make_registry, point_mass)
from machina.model import Builder, Scope

def build():
    registry = make_registry()
    components = [
        Scope("b", [Dynamics(), Drag(), Budget(), Kinematics(), MassSource(), Thruster()]),
        Scope("a", [Budget(), *point_mass(), Drag()]),
    ]
    builder = Builder(components, registry=registry).declare()
    return builder, builder.build(fixed={"a/cap": 1.0})

def digest():
    import hashlib
    builder, model = build()
    blob = b"".join(model[key].serialize().encode() for key in ("f", "g", "h", "J"))
    layout = repr([builder.state_order, builder.algebraic_order, builder.input_order,
                   builder.quantity_order, builder.order,
                   [p for p, _, _ in builder.constraints()]]).encode()
    return hashlib.sha256(blob + layout).hexdigest()
"""

PROBE = "import sys\nsys.path[:0] = [%r, %r]\n" + MODEL + "\nprint(digest())\n"


def digest_in_subprocess(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", PROBE % (str(SRC), str(HERE))],
                         capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip()


def digest_in_process() -> str:
    namespace = {}
    exec(compile(MODEL, "<model>", "exec"), namespace)   # noqa: S102 - test-local source
    return namespace["digest"]()


class TestTheGraphIsReproducible:

    def test_the_probe_model_exercises_every_ordering(self):
        """If these stop holding, the probe no longer tests what it claims to."""
        namespace = {}
        exec(compile(MODEL, "<model>", "exec"), namespace)   # noqa: S102
        builder, model = namespace["build"]()
        assert builder.state_order[:2] == ["b/pos", "a/pos"]                 # scope order
        order = builder.order
        assert order.index("b/dynamics") > order.index("b/drag")             # topological
        assert order.index("b/dynamics") > order.index("b/thruster")
        assert [p for p, _, _ in builder.constraints()] == ["b/headroom", "a/headroom"]
        assert "h" in model and "J" in model
        assert builder.nq == 4 and model["f"].size1_in(2) == 3     # a/cap is fixed

    def test_two_builds_in_one_process_are_byte_identical(self):
        assert digest_in_process() == digest_in_process()

    def test_two_processes_with_different_hash_seeds_agree(self):
        """The real gate. A set iterated anywhere in the builder breaks this."""
        assert digest_in_subprocess("0") == digest_in_subprocess("524287")

    def test_a_third_pair_of_seeds_agrees_too(self):
        assert digest_in_subprocess("1") == digest_in_subprocess("99991")

    def test_the_subprocess_and_this_process_agree(self):
        assert digest_in_subprocess("7") == digest_in_process()


# --- the static guard ------------------------------------------------------------------------

_SET_CALLS = ("set", "frozenset")
_ORDER_KEEPING_WRAPPERS = ("list", "tuple", "enumerate", "reversed", "iter")


def _is_set_expr(node, set_names) -> bool:
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _SET_CALLS:
        return True
    if isinstance(node, ast.Name) and node.id in set_names:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.Sub,
                                                            ast.BitXor)):
        return _is_set_expr(node.left, set_names) or _is_set_expr(node.right, set_names)
    return False


def _iterates_a_set(node, set_names) -> bool:
    """True when iterating ``node`` would take its order from a set."""
    if _is_set_expr(node, set_names):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _ORDER_KEEPING_WRAPPERS and node.args:
        return _iterates_a_set(node.args[0], set_names)
    return False


def set_iterations(source: str, filename: str = "<source>") -> list:
    """Every loop or comprehension whose order would come from a set.

    Tracks names assigned from set expressions within each function, so
    ``seen = set(xs); for x in seen`` is caught; ``for x in sorted(seen)`` is not.
    Membership tests (``x in seen``) are never iterations, and are fine.
    """
    offenders = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [set()]

        def _function(self, node):
            self.scopes.append(set())
            self.generic_visit(node)
            self.scopes.pop()

        visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

        def visit_Assign(self, node):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if _is_set_expr(node.value, self.scopes[-1]):
                    self.scopes[-1].add(name)
                else:
                    self.scopes[-1].discard(name)
            self.generic_visit(node)

        def _check(self, iterable):
            if _iterates_a_set(iterable, self.scopes[-1]):
                offenders.append(f"{filename}:{iterable.lineno}")

        def visit_For(self, node):
            self._check(node.iter)
            self.generic_visit(node)

        def _comprehension(self, node):
            for generator in node.generators:
                self._check(generator.iter)
            self.generic_visit(node)

        visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    Visitor().visit(ast.parse(source))
    return offenders


class TestNothingIteratesASet:

    def test_no_guarded_package_iterates_a_set(self):
        """Ordering must never come from a set; membership tests are fine.

        This reads the source because the behaviour is only wrong one process
        in a few hundred.
        """
        offenders = []
        for package in GUARDED:
            root = SRC / "machina" / package
            for path in sorted(root.rglob("*.py")) if root.exists() else []:
                offenders += set_iterations(path.read_text(encoding="utf-8"),
                                            str(path.relative_to(SRC)))
        assert offenders == []

    @pytest.mark.parametrize("snippet", [
        "for x in {1, 2}: pass",
        "for x in set(xs): pass",
        "[x for x in frozenset(xs)]",
        "def f(xs):\n    seen = set(xs)\n    for x in seen: pass",
        "def f(a, b):\n    both = set(a) | set(b)\n    return [x for x in both]",
        "for i, x in enumerate(set(xs)): pass",
        "for x in list({y for y in ys}): pass",
    ])
    def test_the_guard_catches_a_realistic_regression(self, snippet):
        assert set_iterations(textwrap.dedent(snippet)) != []

    @pytest.mark.parametrize("snippet", [
        "for x in sorted(set(xs)): pass",
        "def f(xs):\n    seen = set(xs)\n    return [x for x in xs if x in seen]",
        "def f(xs):\n    seen = set(xs)\n    seen = list(xs)\n    for x in seen: pass",
        "for x in xs: pass",
        "d = {k: 1 for k in ks}\nfor k in d: pass",
    ])
    def test_the_guard_leaves_ordered_iteration_alone(self, snippet):
        assert set_iterations(textwrap.dedent(snippet)) == []

    def test_the_vector_orders_are_lists_not_sets(self):
        namespace = {}
        exec(compile(MODEL, "<model>", "exec"), namespace)   # noqa: S102
        builder, _ = namespace["build"]()
        for order in (builder.state_order, builder.algebraic_order,
                      builder.input_order, builder.quantity_order, builder.order):
            assert isinstance(order, list)
