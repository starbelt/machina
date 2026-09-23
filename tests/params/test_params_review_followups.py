"""
Regressions for the targeted review of Phase 2b.

Each test pins one defect the reviewer reproduced. The first is the one
that mattered: a table with a name listed twice -- what a merge-conflict
resolution that keeps both sides produces -- lost a measured value to sync
without a word.
"""

import os
import subprocess
import sys
import textwrap

import casadi as ca
import numpy as np
import pytest

from machina import cli
from machina.codegen import dense
from machina.params import ParamRegistry, all_params, contract, param, sync, table, values_from
from machina.params import check as check_mod
from machina.params.lint import DefaultContract

pytestmark = pytest.mark.requires_casadi

OK = dict(type="f64", unit="m", lo=0.0, hi=10.0, mutability="design", doc="a")
HEADER = ",".join(DefaultContract.COLUMNS)


def write(path, *rows, header=HEADER, newline="\n"):
    path.write_bytes((newline.join([header, *rows]) + newline).encode("utf-8"))
    return path


class TestDuplicateNames:

    def test_sync_refuses_a_table_that_lists_a_name_twice(self, tmp_path):
        param("A_X", **OK)
        path = write(tmp_path / "p.csv",
                     "A_X,1.5,f64,m,0.0,10.0,,design,M,alice bench log 2026-09-01,a",
                     "A_X,2.5,f64,m,0.0,10.0,,design,E,bob guess,a")
        before = path.read_bytes()
        with pytest.raises(ValueError, match=r"lists \['A_X'\] more than once"):
            sync.sync(path, all_params())
        assert path.read_bytes() == before, "nothing may be written when sync refuses"

    def test_check_reports_it_as_an_error(self, tmp_path):
        param("A_X", **OK)
        path = write(tmp_path / "p.csv",
                     "A_X,1.5,f64,m,0.0,10.0,,design,M,alice,a",
                     "A_X,2.5,f64,m,0.0,10.0,,design,E,bob,a")
        errors, _ = check_mod.check(path, all_params())
        assert any("more than once" in e for e in errors)


class TestValuesFromAgreesWithCheck:

    @pytest.mark.parametrize("row,message", [
        ("B_N,2.7,i32,1,0,100,,design,M,s,n", "not an integer"),
        ("B_N,250,i32,1,0,100,,design,M,s,n", "outside"),
        ("B_N,5,i32,1,0,100,,design,,s,n", "provenance code"),
        ("B_N,inf,i32,1,0,100,,design,M,s,n", "infinite"),
    ])
    def test_it_refuses_what_check_refuses(self, tmp_path, row, message):
        with pytest.raises(ValueError, match=message):
            values_from(write(tmp_path / "p.csv", row))


class TestTheCliLeavesNoStateBehind:

    def test_two_syncs_in_one_process_see_only_their_own_modules(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "leak_mod_a.py").write_text(
            "from machina.params import param\n"
            "A_ONE = param('A_ONE', type='f64', unit='m', lo=0.0, hi=1.0, mutability='design',"
            " doc='a')\n", encoding="utf-8")
        (tmp_path / "leak_mod_b.py").write_text(
            "from machina.params import param\n"
            "B_ONE = param('B_ONE', type='f64', unit='m', lo=0.0, hi=1.0, mutability='design',"
            " doc='b')\n", encoding="utf-8")
        cli.main(["params", "sync", "--csv", "a.csv", "--modules", "leak_mod_a"])
        cli.main(["params", "sync", "--csv", "b.csv", "--modules", "leak_mod_b"])
        assert [r["name"] for r in table.read(tmp_path / "b.csv")] == ["B_ONE"]

    def test_a_contract_chosen_for_one_call_is_gone_afterwards(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "short_names.py").write_text(textwrap.dedent("""
            from machina.params.lint import DefaultContract

            class Short(DefaultContract):
                MAX_NAME_LEN = 32
        """), encoding="utf-8")
        (tmp_path / "leak_mod_c.py").write_text(
            "from machina.params import param\n"
            "C_ONE = param('C_ONE', type='f64', unit='m', lo=0.0, hi=1.0, mutability='design',"
            " doc='c')\n", encoding="utf-8")
        cli.main(["params", "sync", "--csv", "c.csv", "--modules", "leak_mod_c",
                  "--contract", "short_names:Short"])
        assert type(contract.get()) is DefaultContract


class TestTheCliOutput:

    def test_redirected_output_survives_non_cp1252_text(self, tmp_path):
        """A doc with a Greek letter used to crash the gate with UnicodeEncodeError."""
        (tmp_path / "greek_mod.py").write_text(
            "from machina.params import param\n"
            "DV = param('DV', type='f64', unit='m/s', lo=0.0, hi=1.0, mutability='design',"
            " doc='\\u0394v budget \\u2192 total')\n", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        out = subprocess.run([sys.executable, "-m", "machina", "params", "sync", "--csv",
                              "p.csv", "--modules", "greek_mod"],
                             cwd=tmp_path, capture_output=True, env=env)
        assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in out.stderr

    def test_a_malformed_table_is_an_error_naming_the_file_not_a_traceback(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "plain_mod.py").write_text(
            "from machina.params import param\n"
            "P = param('P', type='f64', unit='m', lo=0.0, hi=1.0, mutability='design', doc='p')\n",
            encoding="utf-8")
        write(tmp_path / "p.csv", "P,1.0,f64")
        assert cli.main(["params", "sync", "--csv", "p.csv", "--modules", "plain_mod"]) == 1
        assert "error: p.csv:" in capsys.readouterr().out


class TestSpellingsAndBytes:

    @pytest.mark.parametrize("value", ["1_0", " 7 ", chr(0xFF17), "1.0.0", "0x10"])
    def test_python_only_number_spellings_are_refused(self, value):
        issues = DefaultContract().lint_text(
            f"{HEADER}\nA_X,{value},f64,m,0.0,10.0,,design,M,s,a\n")
        assert any("plain decimal" in str(i) for i in issues if not i.warning)

    @pytest.mark.parametrize("value", ["1e0", "1.0E+00", "+2.5", ".5", "7."])
    def test_ordinary_decimal_spellings_are_still_accepted(self, value):
        issues = DefaultContract().lint_text(
            f"{HEADER}\nA_X,{value},f64,m,0.0,10.0,,design,M,s,a\n")
        assert [str(i) for i in issues if not i.warning] == []

    def test_an_excel_bom_is_named_as_such(self, tmp_path):
        param("A_X", **OK)
        path = tmp_path / "p.csv"
        path.write_bytes((chr(0xFEFF) + HEADER + "\nA_X,1.0,f64,m,0.0,10.0,,design,M,s,a\n")
                         .encode("utf-8"))
        errors, _ = check_mod.check(path, all_params())
        assert any("byte-order mark" in e for e in errors)

    def test_a_bare_cr_inside_a_field_is_one_error_naming_the_row(self, tmp_path):
        """3.12's csv.writer quotes such a field and 3.10's does not, so the emitter had two
        spellings and 3.10's reader refused its own output. It is refused on both sides now."""
        reg = ParamRegistry()
        param("A_X", registry=reg, **OK)
        path = tmp_path / "p.csv"
        path.write_bytes(f"{HEADER}\nA_X,1.0,f64,m,0.0,10.0,,design,M,cr\rhere,a\n".encode())
        errors, warnings = check_mod.check(path, reg.all())
        assert len(errors) == 1 and warnings == []
        assert "bare carriage return or newline" in errors[0] and "A_X" in errors[0]

    @pytest.mark.parametrize("character", ["\r", "\n"])
    def test_dumps_refuses_a_field_that_holds_a_line_break(self, character):
        reg = ParamRegistry()
        param("A_X", registry=reg, **OK)
        rows = sync.apply(reg.all(), [])
        rows[0]["value"], rows[0]["provenance"] = "1.0", "M"
        rows[0]["source"] = f"cr{character}here"
        with pytest.raises(ValueError, match="bare carriage return or newline"):
            table.dumps(rows)


class TestDenseKeepsMx:

    def test_dense_works_on_a_function_with_mx_only_nodes(self):
        x = ca.MX.sym("x", 2)
        A = ca.vertcat(ca.horzcat(x[0], 1), ca.horzcat(0, 2))
        f = ca.Function("fm", [x], [ca.solve(A, ca.DM([1.0, 2.0]))], ["x"], ["y"])
        densified = dense(f)
        np.testing.assert_allclose(np.array(densified([2.0, 0.0])).ravel(),
                                   np.array(f([2.0, 0.0])).ravel())
