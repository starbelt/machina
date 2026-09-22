"""
``sync`` writes and ``check`` never does -- plus the column-ownership rules
between them.

Ported from icarus-dynamics ``tests/test_sync_check.py``, with machina's
provenance column added. The property under test throughout: a hand-set
value, provenance code or source is never touched by a tool. That is what
makes the table safe to hand-edit even though the tool also writes to it.
"""

import pytest

from machina.params import all_params, param, sync, table
from machina.params import check as check_mod
from machina.params import registry as param_registry

pytestmark = pytest.mark.requires_casadi

OK = dict(type="f32", unit="1", lo=0.0, hi=10.0, mutability="flight", doc="A gain")


def csv_in(tmp_path):
    return tmp_path / "params.csv"


def declare_one(**over):
    param("KP_ROLL", **{**OK, **over})
    return all_params()


def filled(path, decls, value="2.5", provenance="M", source="bench"):
    sync.sync(path, decls)
    rows = table.read(path)
    for row in rows:
        row["value"], row["provenance"], row["source"] = value, provenance, source
    table.write(path, rows)


class TestSync:

    def test_sync_adds_a_row_with_every_human_column_blank(self, tmp_path):
        """The tool never guesses."""
        path = csv_in(tmp_path)
        sync.sync(path, declare_one())
        (row,) = table.read(path)
        assert row["name"] == "KP_ROLL"
        assert (row["value"], row["provenance"], row["source"]) == ("", "", "")
        assert (row["min"], row["max"]) == ("0.0", "10.0")

    def test_sync_is_idempotent(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        sync.sync(path, decls)
        first = path.read_bytes()
        sync.sync(path, decls)
        assert path.read_bytes() == first

    def test_a_hand_set_value_survives_a_bounds_change(self, tmp_path):
        """The load-bearing one. Widening a range in Python must not revert a tune."""
        path = csv_in(tmp_path)
        filled(path, declare_one(), value="2.5", provenance="M", source="bench run 7")
        param_registry.clear()
        sync.sync(path, declare_one(hi=20.0))
        (row,) = table.read(path)
        assert (row["value"], row["provenance"], row["source"]) == ("2.5", "M", "bench run 7")
        assert row["max"] == "20.0", "the declaration still owns the bound"

    def test_the_tool_owned_columns_follow_the_declaration(self, tmp_path):
        path = csv_in(tmp_path)
        sync.sync(path, declare_one())
        param_registry.clear()
        sync.sync(path, declare_one(doc="A renamed gain", unit="s"))
        (row,) = table.read(path)
        assert (row["comment"], row["unit"]) == ("A renamed gain", "s")

    def test_sync_reports_an_orphan_but_does_not_delete_it(self, tmp_path):
        path = csv_in(tmp_path)
        sync.sync(path, declare_one())
        param_registry.clear()
        diff = sync.sync(path, {})
        assert diff.orphaned == ["KP_ROLL"]
        assert len(table.read(path)) == 1, "sync must not delete without --prune"
        assert any("--prune" in line for line in diff.render())

    def test_prune_removes_it_and_says_so(self, tmp_path):
        path = csv_in(tmp_path)
        sync.sync(path, declare_one())
        param_registry.clear()
        diff = sync.sync(path, {}, prune=True)
        assert table.read(path) == []
        assert any("REMOVED" in line for line in diff.render(pruned=True))

    def test_rows_are_written_sorted(self, tmp_path):
        path = csv_in(tmp_path)
        param("ZZ_LAST", **OK)
        param("AA_FIRST", **OK)
        sync.sync(path, all_params())
        assert [r["name"] for r in table.read(path)] == ["AA_FIRST", "ZZ_LAST"]

    def test_an_empty_diff_is_falsy(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        sync.sync(path, decls)
        assert not sync.plan(decls, table.read(path))

    def test_apply_is_pure(self, tmp_path):
        decls = declare_one()
        rows = []
        sync.apply(decls, rows)
        assert rows == []


class TestCheck:

    def test_check_passes_on_a_filled_table(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls)
        errors, _ = check_mod.check(path, decls)
        assert errors == []

    def test_check_fails_on_an_empty_value(self, tmp_path):
        """The unfilled state is representable on a branch and must not survive CI."""
        path = csv_in(tmp_path)
        decls = declare_one()
        sync.sync(path, decls)
        errors, _ = check_mod.check(path, decls)
        assert any("value is empty" in e for e in errors)

    def test_allow_unfilled_accepts_the_blank(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        sync.sync(path, decls)
        errors, _ = check_mod.check(path, decls, allow_unfilled=True)
        assert errors == []

    def test_a_filled_value_without_provenance_fails(self, tmp_path):
        """machina's addition (Decision Log #47): every number says where it came from."""
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls, provenance="")
        errors, _ = check_mod.check(path, decls)
        assert any("provenance code" in e for e in errors)

    def test_an_unknown_provenance_code_fails(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls, provenance="X")
        errors, _ = check_mod.check(path, decls)
        assert any("provenance 'X'" in e for e in errors)

    def test_check_catches_a_hand_edited_tool_owned_column(self, tmp_path):
        """Editing max in the CSV looks like it works and would be reverted by
        the next sync. Caught instead, with a pointer to where to change it."""
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls)
        rows = table.read(path)
        rows[0]["max"] = "999.0"
        table.write(path, rows)
        errors, _ = check_mod.check(path, decls)
        assert any("owned by the declaration" in e for e in errors)

    def test_check_catches_a_declaration_with_no_row(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls)
        param("KD_ROLL", **OK)
        errors, _ = check_mod.check(path, all_params())
        assert any("missing from the table" in e for e in errors)

    def test_check_never_writes(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        sync.sync(path, decls)  # leaves an empty value, so check will fail
        before = path.read_bytes()
        check_mod.check(path, decls)
        assert path.read_bytes() == before

    def test_check_reports_quantization_as_a_warning_not_an_error(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls, value="0.1")
        errors, warnings = check_mod.check(path, decls)
        assert errors == []
        assert any("on the f32 wire" in w for w in warnings)

    def test_an_exact_f32_value_raises_no_warning(self, tmp_path):
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls, value="2.5")
        _, warnings = check_mod.check(path, decls)
        assert warnings == []

    def test_check_catches_a_trailing_newline(self, tmp_path):
        """One encoding per table, which is what makes the determinism diff mean something."""
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls)
        path.write_bytes(path.read_bytes() + b"\n")
        errors, _ = check_mod.check(path, decls)
        assert any("canonical" in e for e in errors)

    def test_check_catches_crlf_line_ends(self, tmp_path):
        """A Windows checkout with autocrlf would otherwise pass unnoticed."""
        path = csv_in(tmp_path)
        decls = declare_one()
        filled(path, decls)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        errors, _ = check_mod.check(path, decls)
        assert any("line endings" in e for e in errors)

    def test_the_written_table_uses_lf_on_every_platform(self, tmp_path):
        path = csv_in(tmp_path)
        sync.sync(path, declare_one())
        assert b"\r\n" not in path.read_bytes()

    def test_a_missing_file_says_to_run_sync(self, tmp_path):
        errors, _ = check_mod.check(csv_in(tmp_path), declare_one())
        assert any("machina params sync" in e for e in errors)

    def test_a_wrong_header_is_one_clear_error(self, tmp_path):
        path = csv_in(tmp_path)
        path.write_text("name,value\nKP_ROLL,1\n", encoding="utf-8")
        errors, _ = check_mod.check(path, declare_one())
        assert len(errors) == 1 and "header must be exactly" in errors[0]
