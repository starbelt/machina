"""
``machina.study.sweep.SweepTable``: the column view, the CSV and pandas.

The CSV is what the determinism probe byte-diffs across two processes and two
interpreters, so its spelling is pinned here exactly: ``repr(float(x))`` for
floats (``nan``, ``inf``), ``True``/``False``, an empty cell for ``None``,
``\\n`` line ends, and the csv module's minimal quoting for a string with a
comma or a quote. ``t_wall`` differs from run to run and is left out unless
asked for.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from machina.compiler import Problem
from machina.model import (
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    Role,
    SignalRegistry,
)
from machina.study.sweep import STATUS_COLUMNS, SweepTable, sweep

pytestmark = pytest.mark.requires_casadi

SRC = Path(__file__).resolve().parents[2] / "src"


def small() -> SweepTable:
    """Two rows covering every cell spelling the CSV pins."""
    return SweepTable.from_rows(("limit",), [
        {"limit": 100.0, "success": True, "objective": -1.5, "iterations": 7, "error": None,
         "note": 'a, "quoted" note', "third": 1.0 / 3.0, "t_wall": 0.25},
        {"limit": np.int64(150), "success": np.bool_(False), "objective": math.nan,
         "iterations": None, "error": "ValueError: bad", "note": "",
         "third": -math.inf, "t_wall": None},
    ])


EXPECTED = (
    "limit,success,objective,iterations,error,note,third\n"
    '100.0,True,-1.5,7,,"a, ""quoted"" note",0.3333333333333333\n'
    "150,False,nan,,ValueError: bad,,-inf\n"
)


def fleet_sweep() -> SweepTable:
    """A real sweep: the power cap of a one-variable budget, with an infeasible point."""

    class Budget(Component):
        def declare(self):
            return Declaration(
                quantities=(Quantity("duty", lb=0.0, ub=1.0, default=0.1, provenance="A",
                                     source="fixture"),
                            Quantity("limit", role=Role.PARAMETER, default=150.0,
                                     provenance="A", source="fixture")),
                constraints=(Constraint("margin", lb=0.0),),
                costs=(Cost("neg_rate"),))

        def build(self, helpers):
            duty, limit = ca.SX.sym("duty"), ca.SX.sym("limit")
            return {"h": ca.Function("h", [duty, limit], [limit - 100.0 * duty],
                                     ["duty", "limit"], ["margin"]),
                    "J": ca.Function("J", [duty], [-50.0 * duty + duty ** 2], ["duty"],
                                     ["neg_rate"])}

    problem = Problem([Budget()], registry=SignalRegistry(), verbose=False).compile().build()
    return sweep(problem, {"limit": [50.0, -10.0, 80.0]}, collect=["duty", "cost:neg_rate"])


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports this checkout's ``src/`` first."""
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


class TestTheCsvIsByteStable:

    def test_the_spelling_of_every_cell_is_pinned(self, tmp_path):
        small().to_csv(tmp_path / "table.csv")
        assert (tmp_path / "table.csv").read_bytes() == EXPECTED.encode("utf-8")

    def test_writing_twice_gives_the_same_bytes(self, tmp_path):
        small().to_csv(tmp_path / "one.csv")
        small().to_csv(tmp_path / "two.csv")
        assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()

    def test_two_runs_of_one_sweep_write_the_same_bytes(self, tmp_path):
        fleet_sweep().to_csv(tmp_path / "one.csv")
        fleet_sweep().to_csv(tmp_path / "two.csv")
        one = (tmp_path / "one.csv").read_bytes()
        assert one == (tmp_path / "two.csv").read_bytes()
        assert b"\r" not in one
        header = one.decode("utf-8").splitlines()[0]
        assert header == ",".join(("limit",) + STATUS_COLUMNS + ("duty", "cost:neg_rate"))

    def test_a_carriage_return_is_refused_before_it_reaches_a_file(self):
        with pytest.raises(ValueError, match="carriage return"):
            SweepTable.from_rows(("x",), [{"x": 1.0, "note": "a\rb"}])

    def test_a_carriage_return_in_a_hand_made_table_is_refused_by_to_csv(self, tmp_path):
        table = SweepTable(axes=("x",), columns=("x", "note"), rows=({"x": 1, "note": "a\rb"},))
        with pytest.raises(ValueError, match="carriage return"):
            table.to_csv(tmp_path / "table.csv")

    # Python 3.12 quotes a bare '\r' in a header ('a,"b\rc"'), 3.10 does not ('a,b\rc');
    # 3.10 raises on a NUL, 3.12 writes it. Each is refused, with the row and column named.

    def test_a_carriage_return_in_a_column_name_is_refused(self):
        with pytest.raises(ValueError, match=r"row 0, column name: 'b\\rc'.*carriage return"):
            SweepTable.from_rows(("a",), [{"a": 1, "b\rc": 2}])

    def test_a_carriage_return_in_an_axis_name_is_refused(self):
        with pytest.raises(ValueError, match="axis name.*carriage return"):
            SweepTable.from_rows(("a\rb",), [{"a\rb": 1}])

    def test_a_nul_in_a_cell_is_refused_naming_the_row_and_column(self):
        with pytest.raises(ValueError, match="row 1, column 'note'.*NUL"):
            SweepTable.from_rows(("x",), [{"x": 1, "note": "ok"}, {"x": 2, "note": "a\x00b"}])

    def test_a_nul_in_a_column_name_is_refused(self):
        with pytest.raises(ValueError, match="row 0, column name.*NUL"):
            SweepTable.from_rows(("a",), [{"a": 1, "b\x00c": 2}])

    @pytest.mark.parametrize("columns, row, match", [
        (("x", "b\rc"), {"x": 1, "b\rc": 2}, "column name.*carriage return"),
        (("x", "b\x00c"), {"x": 1, "b\x00c": 2}, "column name.*NUL"),
        (("x", "note"), {"x": 1, "note": "a\x00b"}, "row 0, column 'note'.*NUL"),
    ])
    def test_a_hand_made_table_is_refused_before_the_file_is_opened(self, tmp_path, columns,
                                                                    row, match):
        table = SweepTable(axes=("x",), columns=columns, rows=(row,))
        with pytest.raises(ValueError, match=match):
            table.to_csv(tmp_path / "table.csv")
        assert not (tmp_path / "table.csv").exists()

    def test_none_and_an_empty_string_write_the_same_cell(self, tmp_path):
        SweepTable.from_rows(("x",), [{"x": None, "note": ""}]).to_csv(tmp_path / "table.csv")
        assert (tmp_path / "table.csv").read_bytes() == b"x,note\n,\n"


class TestTiming:

    def test_t_wall_is_left_out_by_default(self, tmp_path):
        table = small()
        assert "t_wall" not in table.columns
        assert "t_wall" not in table.as_columns()
        table.to_csv(tmp_path / "table.csv")
        assert "t_wall" not in (tmp_path / "table.csv").read_text(encoding="utf-8")

    def test_include_timing_appends_it_last(self, tmp_path):
        table = small()
        table.to_csv(tmp_path / "table.csv", include_timing=True)
        lines = (tmp_path / "table.csv").read_text(encoding="utf-8").splitlines()
        assert lines[0].endswith(",third,t_wall")
        assert lines[1].endswith(",0.25") and lines[2].endswith(",-inf,")
        assert list(table.as_columns(include_timing=True))[-1] == "t_wall"

    def test_t_wall_is_readable_by_name(self):
        assert small()["t_wall"] == [0.25, None]

    def test_a_sweep_times_every_solved_row(self):
        table = fleet_sweep()
        assert all(isinstance(value, float) and value >= 0.0 for value in table["t_wall"])


class TestTheColumnView:

    def test_as_columns_is_one_list_per_column_in_order(self):
        table = small()
        columns = table.as_columns()
        assert list(columns) == list(table.columns)
        assert all(len(values) == len(table) for values in columns.values())
        assert columns["limit"] == [100.0, 150]
        assert columns["success"] == [True, False]

    def test_numpy_scalars_become_python_scalars(self):
        row = small().rows[1]
        assert type(row["limit"]) is int and type(row["success"]) is bool

    def test_indexing_by_column_name(self):
        table = small()
        assert table["iterations"] == [7, None]
        assert len(table) == 2

    def test_an_unknown_column_lists_the_columns(self):
        with pytest.raises(KeyError, match="'objective'"):
            small()["objctive"]

    def test_a_row_index_is_not_a_column(self):
        with pytest.raises(TypeError, match=r"table\.rows\[i\]"):
            small()[0]

    def test_from_rows_puts_the_axes_first(self):
        table = SweepTable.from_rows(("b", "a"), [{"z": 1, "a": 2, "b": 3}])
        assert table.axes == ("b", "a")
        assert table.columns == ("b", "a", "z")

    def test_from_rows_refuses_a_row_missing_a_column(self):
        with pytest.raises(ValueError, match=r"row 1 has no \['note'\]"):
            SweepTable.from_rows(("x",), [{"x": 1, "note": "a"}, {"x": 2}])

    def test_from_rows_refuses_a_value_that_is_not_a_scalar(self):
        with pytest.raises(TypeError, match="not a scalar"):
            SweepTable.from_rows(("x",), [{"x": [1.0, 2.0]}])

    def test_from_rows_refuses_a_bare_axis_name(self):
        with pytest.raises(TypeError, match=r"axes=\('x',\)"):
            SweepTable.from_rows("x", [{"x": 1}])

    def test_results_must_match_the_rows(self):
        with pytest.raises(ValueError, match="one result"):
            SweepTable(axes=("x",), columns=("x",), rows=({"x": 1},), results=(None, None))


class TestPandas:

    def test_to_pandas_round_trips_the_columns(self):
        pytest.importorskip("pandas")
        table = small()
        frame = table.to_pandas()
        assert list(frame.columns) == list(table.columns)
        assert frame["limit"].tolist() == [100.0, 150.0]
        assert frame["note"].tolist() == ['a, "quoted" note', ""]
        assert math.isnan(frame["objective"].tolist()[1])
        assert list(table.to_pandas(include_timing=True).columns)[-1] == "t_wall"

    def test_without_pandas_the_error_names_the_study_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ImportError, match=r"machina\[study\]"):
            small().to_pandas()

    def test_importing_the_sweep_module_does_not_import_pandas(self):
        out = run_python("import machina.study.sweep, sys; print('pandas' in sys.modules)")
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False", (
            "`import machina.study.sweep` imported pandas; SweepTable.to_pandas imports it on "
            "call, and nothing in src/machina/study may import it at module level")


class TestFeedingPareto:

    def test_columns_feed_pareto(self):
        try:
            from machina.study.pareto import Objective, nondominated
        except ImportError as exc:
            pytest.skip(f"machina.study.pareto is not importable yet: {exc}")
        table = SweepTable.from_rows(("x",), [
            {"x": 0, "success": True, "objective": 3.0},
            {"x": 1, "success": True, "objective": 1.0},
            {"x": 2, "success": False, "objective": math.nan},
            {"x": 3, "success": True, "objective": 1.0},
        ])
        front = nondominated(table, [Objective("objective")])
        assert front.tolist() == [False, True, False, True]

    def test_a_real_sweep_feeds_pareto_with_failed_rows_excluded(self):
        try:
            from machina.study.pareto import Objective, excluded, nondominated
        except ImportError as exc:
            pytest.skip(f"machina.study.pareto is not importable yet: {exc}")
        table = fleet_sweep()
        objectives = [Objective("objective"), Objective("duty", sense="max")]
        assert nondominated(table, objectives, feasible="success").tolist() == [False, False,
                                                                                True]
        assert [row for row, _ in excluded(table, objectives, feasible="success")] == [1]
