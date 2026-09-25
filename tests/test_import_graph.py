"""
``import machina`` loads the core and nothing else.

The core is the model, the compiler, the solver backend, the params pipeline
and the generic library. A domain pack (``astro``, ``swapc``, ``rigid``,
``aero``) is imported explicitly by the study that uses it, and so are the
optional layers (``viz``, ``report``, ``study``) and their heavy dependencies
(networkx, matplotlib, pandas). April's ``machina.agents`` is gone entirely.

An in-process test cannot check any of this: ``tests/conftest.py`` imports
every pack before the first test runs. So each check starts a fresh
interpreter with this checkout's ``src/`` first on its path.
"""

import ast
import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_casadi

SRC = Path(__file__).resolve().parents[1] / "src"
INIT = "src/machina/__init__.py"

PACKS = ("machina.astro", "machina.swapc", "machina.rigid", "machina.aero")
OPTIONAL_LAYERS = ("machina.viz", "machina.report", "machina.study", "machina.agents")
HEAVY_DEPENDENCIES = ("networkx", "matplotlib", "pandas")

# Loaded by ``import machina``, so the absence checks cannot pass on an import that did nothing.
CORE = ("machina.compiler.problem", "machina.model.builder", "machina.solver.backend",
        "machina.params.values", "machina.library", "machina.library.registry")

HOW_TO_FIND_THE_IMPORT = (
    "`python -X importtime -c \"import machina\"` shows which core module pulls it in")


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports this checkout's ``src/`` first."""
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


@functools.lru_cache(maxsize=1)
def import_machina() -> tuple:
    """``(machina.__file__, sorted module names)`` after a bare ``import machina``."""
    out = run_python("import machina, sys; print(machina.__file__); print(sorted(sys.modules))")
    assert out.returncode == 0, (
        f"`import machina` failed in a fresh interpreter; fix {INIT}:\n{out.stderr}")
    where, modules = out.stdout.splitlines()[:2]
    return where, tuple(ast.literal_eval(modules))


def loaded(names) -> list:
    """The subset of ``names`` that ``import machina`` put in ``sys.modules``."""
    modules = import_machina()[1]
    return [name for name in names if name in modules]


class TestImportingMachinaLoadsTheCoreOnly:

    def test_the_subprocess_imports_this_checkout(self):
        where = Path(import_machina()[0]).resolve()
        assert where.is_relative_to(SRC.resolve()), (
            f"the subprocess imported machina from {where}, not from {SRC}; an installed copy "
            f"is shadowing this checkout, so the checks below would test the wrong tree")

    def test_the_core_is_loaded(self):
        modules = import_machina()[1]
        missing = [name for name in CORE if name not in modules]
        assert not missing, (
            f"`import machina` did not load {missing}. {INIT} must import the core (compiler, "
            f"model, solver, params, library); without it the absence checks prove nothing")

    def test_no_domain_pack_is_loaded(self):
        leaked = loaded(PACKS)
        assert not leaked, (
            f"`import machina` loaded the pack(s) {leaked}. Packs are imported explicitly by the "
            f"study that uses them: remove the import from {INIT}, or from the core module that "
            f"makes it ({HOW_TO_FIND_THE_IMPORT})")

    def test_no_optional_layer_is_loaded(self):
        leaked = loaded(OPTIONAL_LAYERS)
        assert not leaked, (
            f"`import machina` loaded {leaked}. viz, report and study are imported explicitly, "
            f"and machina.agents was deleted: remove the import from {INIT}, or from the core "
            f"module that makes it ({HOW_TO_FIND_THE_IMPORT})")

    def test_no_heavy_dependency_is_loaded(self):
        leaked = loaded(HEAVY_DEPENDENCIES)
        assert not leaked, (
            f"`import machina` loaded {leaked}. Only the optional layers may use them, and "
            f"lazily: remove the import from {INIT}, or from the core module that makes it "
            f"({HOW_TO_FIND_THE_IMPORT})")

    def test_the_april_agents_package_is_gone(self):
        out = run_python(
            "try:\n"
            "    import machina.agents as agents\n"
            "except ModuleNotFoundError as error:\n"
            "    print('ModuleNotFoundError', error.name)\n"
            "else:\n"
            "    print('imported', list(getattr(agents, '__path__', [])))\n")
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "ModuleNotFoundError machina.agents", (
            f"`import machina.agents` succeeded ({out.stdout.strip()}). April's agent package "
            f"was deleted; a leftover directory (an ignored __pycache__ is enough) makes it a "
            f"namespace package. Delete the src/machina/agents directory itself (rm -rf, not "
            f"just git rm)")
