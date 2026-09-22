"""
``machina params sync | check`` -- the pipeline from a shell.

Each test writes a throwaway declaring module into ``tmp_path`` and runs the
CLI with that directory as the working directory, the way a consumer runs it
from its own repository root. The module name is unique per test, because
Python caches imported modules for the life of the process.
"""

import subprocess
import sys
import textwrap

import pytest

from machina import cli
from machina.params import table

pytestmark = pytest.mark.requires_casadi

DECLARATIONS = """
from machina.params import param

REFRESH_S = param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0,
                  mutability="fixed", doc="Refresh period")
"""


def project(tmp_path, monkeypatch, name):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(DECLARATIONS), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "params.csv"


def run(*argv):
    return cli.main(list(argv))


class TestSync:

    def test_sync_creates_the_table_with_the_value_blank(self, tmp_path, monkeypatch, capsys):
        csv = project(tmp_path, monkeypatch, "decl_sync_a")
        assert run("params", "sync", "--csv", str(csv), "--modules", "decl_sync_a") == 0
        (row,) = table.read(csv)
        assert row["name"] == "REFRESH_S" and row["value"] == ""
        assert "does not guess" in capsys.readouterr().out

    def test_sync_reports_what_it_added(self, tmp_path, monkeypatch, capsys):
        csv = project(tmp_path, monkeypatch, "decl_sync_b")
        run("params", "sync", "--csv", str(csv), "--modules", "decl_sync_b")
        assert "missing from the table" in capsys.readouterr().out


class TestCheck:

    def test_check_fails_with_exit_1_on_a_blank_value(self, tmp_path, monkeypatch, capsys):
        csv = project(tmp_path, monkeypatch, "decl_check_a")
        run("params", "sync", "--csv", str(csv), "--modules", "decl_check_a")
        capsys.readouterr()
        assert run("params", "check", "--csv", str(csv), "--modules", "decl_check_a") == 1
        out = capsys.readouterr().out
        assert "value is empty" in out and "Not fit to publish" in out

    def test_check_passes_once_filled(self, tmp_path, monkeypatch, capsys):
        csv = project(tmp_path, monkeypatch, "decl_check_b")
        run("params", "sync", "--csv", str(csv), "--modules", "decl_check_b")
        rows = table.read(csv)
        rows[0]["value"], rows[0]["provenance"], rows[0]["source"] = "300.0", "D", "PUG 5.1"
        table.write(csv, rows)
        assert run("params", "check", "--csv", str(csv), "--modules", "decl_check_b") == 0
        assert "OK" in capsys.readouterr().out

    def test_allow_unfilled_passes_a_work_in_progress_table(self, tmp_path, monkeypatch):
        csv = project(tmp_path, monkeypatch, "decl_check_c")
        run("params", "sync", "--csv", str(csv), "--modules", "decl_check_c")
        assert run("params", "check", "--csv", str(csv), "--modules", "decl_check_c",
                   "--allow-unfilled") == 0

    def test_check_never_writes(self, tmp_path, monkeypatch):
        csv = project(tmp_path, monkeypatch, "decl_check_d")
        run("params", "sync", "--csv", str(csv), "--modules", "decl_check_d")
        before = csv.read_bytes()
        run("params", "check", "--csv", str(csv), "--modules", "decl_check_d")
        assert csv.read_bytes() == before


class TestArguments:

    def test_a_declaring_module_that_does_not_import_says_so(self, tmp_path, monkeypatch):
        csv = project(tmp_path, monkeypatch, "decl_args_a")
        with pytest.raises(ImportError, match="failed to import"):
            run("params", "sync", "--csv", str(csv), "--modules", "no_such_module_here")

    def test_an_empty_module_list_is_refused_naming_the_file(self, tmp_path, monkeypatch,
                                                             capsys):
        csv = project(tmp_path, monkeypatch, "decl_args_b")
        assert run("params", "sync", "--csv", str(csv), "--modules", " , ") == 1
        out = capsys.readouterr().out
        assert "no declaring modules" in out and str(csv) in out

    def test_csv_and_modules_are_required(self):
        with pytest.raises(SystemExit):
            run("params", "check")

    def test_a_contract_can_be_chosen_on_the_command_line(self, tmp_path, monkeypatch):
        (tmp_path / "tiny_contract.py").write_text(textwrap.dedent("""
            from machina.params.lint import DefaultContract

            class Tiny(DefaultContract):
                MAX_NAME_LEN = 4
                NAME_LENGTH_REASON = "tiny"
        """), encoding="utf-8")
        csv = project(tmp_path, monkeypatch, "decl_args_c")
        from machina.params import ParamDeclarationError
        with pytest.raises(ParamDeclarationError, match="the limit is 4"):
            run("params", "sync", "--csv", str(csv), "--modules", "decl_args_c",
                "--contract", "tiny_contract:Tiny")

    def test_a_malformed_contract_spec_is_refused(self, tmp_path, monkeypatch):
        csv = project(tmp_path, monkeypatch, "decl_args_d")
        with pytest.raises(SystemExit, match="module.path:Name"):
            run("params", "sync", "--csv", str(csv), "--modules", "decl_args_d",
                "--contract", "nocolon")

    def test_python_dash_m_machina_is_the_same_entry_point(self):
        out = subprocess.run([sys.executable, "-m", "machina", "--help"],
                             capture_output=True, text=True)
        assert out.returncode == 0 and "params" in out.stdout
