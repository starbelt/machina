"""
The committed fixture table is the one ``make check-params`` checks.

It is the only params CSV in the repository -- machina has no parameters of
its own until the packs arrive -- and it doubles as a worked example of the
pipeline. These tests keep it honest: it passes ``check``, it is already in
canonical form (so ``sync`` would change nothing), and its bytes use LF even
on a Windows checkout, which ``.gitattributes`` guarantees.
"""

import shutil
from pathlib import Path

import pytest

from machina import cli
from machina.params import values_from

pytestmark = pytest.mark.requires_casadi

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def in_fixtures(monkeypatch):
    monkeypatch.chdir(FIXTURES)


class TestTheFixtureTable:

    def test_it_passes_check(self, monkeypatch, capsys):
        in_fixtures(monkeypatch)
        code = cli.main(["params", "check", "--csv", "params.csv",
                         "--modules", "fixture_params"])
        assert code == 0, capsys.readouterr().out

    def test_it_is_already_canonical_so_sync_would_change_nothing(self, monkeypatch, tmp_path):
        in_fixtures(monkeypatch)
        copy = tmp_path / "params.csv"
        shutil.copyfile(FIXTURES / "params.csv", copy)
        cli.main(["params", "sync", "--csv", str(copy), "--modules", "fixture_params"])
        assert copy.read_bytes() == (FIXTURES / "params.csv").read_bytes()

    def test_its_line_endings_are_lf(self):
        assert b"\r\n" not in (FIXTURES / "params.csv").read_bytes()

    def test_every_value_carries_provenance_and_an_honest_source(self):
        for value in values_from(FIXTURES / "params.csv").values():
            assert value.provenance in ("D", "P", "E", "A", "M")
            assert value.source.startswith("fixture:")
