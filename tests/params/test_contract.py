"""
The params format is a contract chosen at run time.

This is what lets icarus-dynamics keep its MAVLink table -- ten columns, no
provenance, uppercase names of at most 16 characters -- while machina's own
default is eleven columns with provenance. ``IcarusLikeContract`` below
imitates icarus's rules closely enough to prove every tool follows the
active contract rather than a hardcoded one.
"""

import re

import pytest

from machina.params import ParamDeclarationError, all_params, contract, param, sync, table
from machina.params import check as check_mod
from machina.params.contract import LintIssue
from machina.params.lint import DefaultContract

pytestmark = pytest.mark.requires_casadi


class IcarusLikeContract:
    COLUMNS = ("name", "value", "type", "unit", "min", "max", "default", "mutability",
               "source", "comment")
    NAME_RULE = "[A-Z][A-Z0-9_]*"
    NAME_RE = re.compile(NAME_RULE)
    MAX_NAME_LEN = 16
    NAME_LENGTH_REASON = "MAVLink param_id holds 16 and a longer name corrupts the frame"
    NUMERIC_TYPES = ("f32", "u8")
    INT_TYPES = ("u8",)
    MUTABILITIES = ("flight", "disarmed", "boot")
    PROVENANCE_CODES = ()
    NON_SI_UNITS = DefaultContract.NON_SI_UNITS
    UNIT_RE = DefaultContract.UNIT_RE

    def lint_text(self, text, *, allow_unfilled=False):
        issues = []
        for line in text.splitlines()[1:]:
            name, value = line.split(",")[:2]
            if not value and not allow_unfilled:
                issues.append(LintIssue("value is empty", name=name))
        return issues


OK = dict(type="f32", unit="1", lo=0.0, hi=10.0, mutability="flight", doc="A gain")


class TestSelection:

    def test_the_default_is_machinas_own(self):
        assert isinstance(contract.get(), DefaultContract)

    def test_use_switches_it(self):
        contract.use(IcarusLikeContract())
        assert isinstance(contract.get(), IcarusLikeContract)

    def test_reset_returns_to_the_default(self):
        contract.use(IcarusLikeContract())
        contract.reset()
        assert isinstance(contract.get(), DefaultContract)

    def test_using_is_scoped_to_a_block(self):
        with contract.using(IcarusLikeContract()):
            assert isinstance(contract.get(), IcarusLikeContract)
        assert isinstance(contract.get(), DefaultContract)

    def test_something_that_is_not_a_contract_is_refused_by_name(self):
        with pytest.raises(TypeError, match="missing"):
            contract.use(object())

    def test_a_contract_without_name_and_value_columns_is_refused(self):
        class Broken(IcarusLikeContract):
            COLUMNS = ("key", "number")
        with pytest.raises(TypeError, match="'name' and 'value'"):
            contract.use(Broken())

    def test_importing_the_package_reads_no_contract(self):
        """icarus's eager import of its contract is what failed its whole suite
        at collection outside the Nix shell."""
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, machina.params; from machina.params import contract; "
             "print(contract._active is None, 'machina.params.lint' in sys.modules)"],
            capture_output=True, text=True, check=True)
        assert out.stdout.split() == ["True", "False"]


class TestEveryToolFollowsTheActiveContract:

    def test_declarations_follow_its_name_rule(self):
        with contract.using(IcarusLikeContract()):
            with pytest.raises(ParamDeclarationError, match=r"\[A-Z\]"):
                param("kp_roll", **OK)

    def test_declarations_follow_its_length_limit_and_reason(self):
        with contract.using(IcarusLikeContract()):
            with pytest.raises(ParamDeclarationError, match="corrupts the frame"):
                param("KP_ROLL_RATE_LIMIT", **OK)

    def test_declarations_follow_its_types(self):
        with contract.using(IcarusLikeContract()):
            with pytest.raises(ParamDeclarationError, match="type 'f64'"):
                param("KP_ROLL", **{**OK, "type": "f64"})

    def test_the_table_uses_its_columns(self, tmp_path):
        path = tmp_path / "params.csv"
        with contract.using(IcarusLikeContract()):
            param("KP_ROLL", **OK)
            sync.sync(path, all_params())
            header = path.read_text(encoding="utf-8").splitlines()[0]
            assert header == ",".join(IcarusLikeContract.COLUMNS)
            assert table.human_owned() == ("value", "source")

    def test_sync_keeps_its_human_columns_untouched(self, tmp_path):
        path = tmp_path / "params.csv"
        with contract.using(IcarusLikeContract()):
            param("KP_ROLL", **OK)
            sync.sync(path, all_params())
            rows = table.read(path)
            rows[0]["value"], rows[0]["source"] = "2.5", "manual"
            table.write(path, rows)
            sync.sync(path, all_params())
            (row,) = table.read(path)
            assert (row["value"], row["source"]) == ("2.5", "manual")

    def test_check_uses_its_linter(self, tmp_path):
        path = tmp_path / "params.csv"
        with contract.using(IcarusLikeContract()):
            param("KP_ROLL", **OK)
            sync.sync(path, all_params())
            errors, _ = check_mod.check(path, all_params())
            assert any("value is empty" in e for e in errors)

    def test_a_provenance_value_is_refused_by_a_format_without_that_column(self):
        with contract.using(IcarusLikeContract()):
            param("KP_ROLL", **OK)
            with pytest.raises(ValueError, match="not human-owned columns"):
                table.row_from_decl(all_params()["KP_ROLL"], provenance="M")
