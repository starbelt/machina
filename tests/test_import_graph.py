"""
``import machina`` loads the core and nothing else.

The core is the model, the compiler, the solver backend, the params pipeline
and the generic library. A domain pack (``astro``, ``swapc``, ``rigid``,
``aero``) is imported explicitly by the study that uses it, and so are the
optional layers (``viz``, ``report``, ``study``) and their heavy dependencies
(networkx, matplotlib, pandas). April's ``machina.agents`` and
``machina.blocks`` are gone entirely.

Factories follow the same split: ``import machina`` registers the eight
generic ones, and each pack registers its own when it is imported.

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
OPTIONAL_LAYERS = ("machina.viz", "machina.report", "machina.study", "machina.agents",
                   "machina.blocks")
DELETED = ("machina.agents", "machina.blocks")
HEAVY_DEPENDENCIES = ("networkx", "matplotlib", "pandas")

# Loaded by ``import machina``, so the absence checks cannot pass on an import that did nothing.
CORE = ("machina.compiler.problem", "machina.model.builder", "machina.solver.backend",
        "machina.params.values", "machina.library", "machina.library.registry")

HOW_TO_FIND_THE_IMPORT = (
    "`python -X importtime -c \"import machina\"` shows which core module pulls it in")

# Registered factory names, sorted as ``registry.list_registered()`` returns them.
GENERIC_FACTORIES = (
    "constraint.linear", "cost.least_squares", "cost.quadratic", "cost.rosenbrock",
    "util.rotate_x", "util.rotate_y", "util.rotate_z", "util.sum")
ASTRO_FACTORIES = (
    "cost.smooth_coverage", "geometry.elevation_angle", "geometry.ground_target_eci",
    "transform.koe_to_mee", "transform.lagrange_coefficients", "transform.mee_to_eci",
    "transform.mee_to_koe", "transform.propagate_universal", "transform.stumpff_cs",
    "transform.universal_kepler")
SWAPC_FACTORIES = ("cost.aggregate_goodput", "cost.sigmoid_goodput", "util.ttp_computation")


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


@functools.lru_cache(maxsize=1)
def registered_after_each_import() -> tuple:
    """Registered factory names after ``import machina``, then ``machina.astro``, then
    ``machina.swapc``, all in one fresh interpreter."""
    out = run_python(
        "import machina\n"
        "print(sorted(machina.library.registry.list_registered()))\n"
        "import machina.astro\n"
        "print(sorted(machina.library.registry.list_registered()))\n"
        "import machina.swapc\n"
        "print(sorted(machina.library.registry.list_registered()))\n")
    assert out.returncode == 0, (
        f"importing machina, machina.astro and machina.swapc failed in a fresh "
        f"interpreter:\n{out.stderr}")
    return tuple(tuple(ast.literal_eval(line)) for line in out.stdout.splitlines()[:3])


def added(before, after) -> list:
    """The names in ``after`` that are not in ``before``, in ``after``'s order."""
    return [name for name in after if name not in before]


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
            f"and machina.agents and machina.blocks were deleted: remove the import from {INIT}, "
            f"or from the core module that makes it ({HOW_TO_FIND_THE_IMPORT})")

    def test_no_heavy_dependency_is_loaded(self):
        leaked = loaded(HEAVY_DEPENDENCIES)
        assert not leaked, (
            f"`import machina` loaded {leaked}. Only the optional layers may use them, and "
            f"lazily: remove the import from {INIT}, or from the core module that makes it "
            f"({HOW_TO_FIND_THE_IMPORT})")

    @pytest.mark.parametrize("name", DELETED)
    def test_the_april_packages_are_gone(self, name):
        out = run_python(
            "try:\n"
            f"    import {name} as package\n"
            "except ModuleNotFoundError as error:\n"
            "    print('ModuleNotFoundError', error.name)\n"
            "else:\n"
            "    print('imported', list(getattr(package, '__path__', [])))\n")
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == f"ModuleNotFoundError {name}", (
            f"`import {name}` succeeded ({out.stdout.strip()}). April's package was deleted; a "
            f"leftover directory (an ignored __pycache__ is enough) makes it a namespace package. "
            f"Delete the src/{name.replace('.', '/')} directory itself (rm -rf, not just git rm)")


class TestPacksRegisterTheirFactoriesOnImport:

    def test_import_machina_registers_the_generic_factories_only(self):
        generic = registered_after_each_import()[0]
        assert generic == GENERIC_FACTORIES, (
            f"`import machina` registered {list(generic)}; expected exactly the generic "
            f"factories {list(GENERIC_FACTORIES)}. src/machina/library/__init__.py imports the "
            f"generic factory modules; a domain factory belongs in its pack, which registers it "
            f"when the pack is imported")

    def test_import_machina_astro_adds_its_ten_factories(self):
        generic, astro, _ = registered_after_each_import()
        assert added(generic, astro) == list(ASTRO_FACTORIES), (
            f"`import machina.astro` added {added(generic, astro)}; expected exactly "
            f"{list(ASTRO_FACTORIES)}. src/machina/astro/__init__.py imports the modules that "
            f"register them (transforms, geometry, coverage)")
        assert not added(astro, generic), f"`import machina.astro` removed {added(astro, generic)}"

    def test_import_machina_swapc_adds_its_three_factories(self):
        _, astro, swapc = registered_after_each_import()
        assert added(astro, swapc) == list(SWAPC_FACTORIES), (
            f"`import machina.swapc` added {added(astro, swapc)}; expected exactly "
            f"{list(SWAPC_FACTORIES)}. src/machina/swapc/__init__.py imports the modules that "
            f"register them (goodput, latency)")
        assert not added(swapc, astro), f"`import machina.swapc` removed {added(swapc, astro)}"
