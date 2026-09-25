"""
The export probe behind ``make determinism``, run from the test suite.

``make determinism`` runs ``scripts/determinism_probe.py`` in two separate
processes and diffs the output directories byte for byte; CI runs it too. The
tests here are the same gate in two cheaper forms: the probe body twice in
this process, and twice in subprocesses with different ``PYTHONHASHSEED``
values -- the variable that makes a set-derived ordering change between runs.
"""

import filecmp
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_casadi

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "determinism_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("determinism_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_tree(a: Path, b: Path) -> list:
    """Every file under a and b, compared by bytes. Returns the differences."""
    names_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    names_b = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if names_a != names_b:
        return [f"file lists differ: {names_a} vs {names_b}"]
    return [name for name in names_a if not filecmp.cmp(a / name, b / name, shallow=False)]


class TestTheProbe:

    def test_it_writes_the_expected_artifacts(self, tmp_path):
        load_probe().run(tmp_path)
        written = sorted(p.relative_to(tmp_path).as_posix()
                         for p in tmp_path.rglob("*") if p.is_file())
        assert written == ["MANIFEST.json", "coverage/coverage_nlp.casadi",
                           "coverage/names.json", "params/params.csv",
                           "plant/f_system.casadi", "plant/g_system.casadi",
                           "plant/plant_step.c"]

    def test_every_stage_contributes_its_manifest_block(self, tmp_path):
        probe = load_probe()
        probe.run(tmp_path)
        manifest = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert sorted(manifest) == sorted(stage for stage, _writer in probe.ARTIFACTS)

    def test_the_coverage_nlp_is_a_fresh_function_with_the_declared_layout(self, tmp_path):
        """The astro problem's NLP, serialised before any solver or derivative existed."""
        import casadi as ca
        load_probe().run(tmp_path)
        names = json.loads((tmp_path / "coverage" / "names.json").read_text(encoding="utf-8"))
        assert [v["name"] for v in names["variables"]] == [f"sat/{n}" for n in "pfghk"]
        assert [p["name"] for p in names["parameters"]] == [
            "sat/L", "sat/r_target", "sat/min_elevation", "sat/sigmoid_k"]
        assert [c["name"] for c in names["constraints"]] == [
            "sat/perigee_altitude", "sat/apogee_altitude"]
        assert names["costs"] == ["neg_coverage"]
        fn = ca.Function.deserialize(
            (tmp_path / "coverage" / "coverage_nlp.casadi").read_text(encoding="utf-8"))
        assert fn.name_in() == ["x", "p"] and fn.name_out() == ["f", "g"]
        assert fn.size1_in(0) == 5 and fn.size1_out(1) == 2

    def test_two_runs_in_one_process_are_byte_identical(self, tmp_path):
        probe = load_probe()
        probe.run(tmp_path / "a")
        probe.run(tmp_path / "b")
        assert same_tree(tmp_path / "a", tmp_path / "b") == []

    def test_two_processes_with_different_hash_seeds_are_byte_identical(self, tmp_path):
        """The real gate, as ``make determinism`` runs it."""
        for out, seed in (("a", "0"), ("b", "2718281")):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            subprocess.run([sys.executable, str(PROBE), "--out", str(tmp_path / out)],
                           check=True, env=env, capture_output=True)
        assert same_tree(tmp_path / "a", tmp_path / "b") == []

    def test_the_manifest_carries_no_timestamp(self, tmp_path):
        load_probe().run(tmp_path)
        text = (tmp_path / "MANIFEST.json").read_text(encoding="utf-8")
        assert "time" not in text.lower() and "date" not in text.lower()

    def test_the_params_table_it_writes_passes_the_linter(self, tmp_path):
        from machina.params.contract import get
        load_probe().run(tmp_path)
        text = (tmp_path / "params" / "params.csv").read_text(encoding="utf-8")
        errors = [str(i) for i in get().lint_text(text) if not i.warning]
        assert errors == []
