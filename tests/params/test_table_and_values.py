"""
The canonical CSV encoding, and reading numbers back out with provenance.

``values_from`` is the seam Phase 3's compiler reads through: it hands back
each filled value with its provenance code and source, and refuses a blank
rather than letting a compile fall back to a default nobody chose.
"""

import math

import pytest

from machina.params import ParamValue, all_params, param, sync, table, values_from

pytestmark = pytest.mark.requires_casadi

OK = dict(unit="1", lo=0.0, hi=10.0, mutability="design", doc="A number")


class TestNumberSpelling:

    def test_a_float_type_always_carries_a_decimal_point(self):
        assert table.format_number(0, is_integer=False) == "0.0"

    def test_an_integer_type_never_does(self):
        assert table.format_number(3.0, is_integer=True) == "3"

    def test_floats_use_the_shortest_exact_spelling(self):
        assert table.format_number(0.1, is_integer=False) == "0.1"
        assert table.format_number(1e-5, is_integer=False) == "1e-05"

    def test_an_absent_default_is_blank(self):
        assert table.format_number(None, is_integer=False) == ""

    def test_an_infinite_bound_round_trips(self):
        assert float(table.format_number(math.inf, is_integer=False)) == math.inf


class TestCanonicalForm:

    def test_dumps_then_loads_is_lossless(self, tmp_path):
        param("ALPHA", type="f64", **OK)
        param("BETA", type="u8", unit="1", lo=0, hi=9, mutability="fixed", doc="Count",
              default=3)
        rows = sync.apply(all_params(), [])
        assert table.loads(table.dumps(rows)) == rows

    def test_dumps_sorts_by_name_whatever_the_input_order(self):
        param("ZULU", type="f64", **OK)
        param("ALPHA", type="f64", **OK)
        text = table.dumps(sync.apply(all_params(), []))
        names = [line.split(",")[0] for line in text.splitlines()[1:]]
        assert names == ["ALPHA", "ZULU"]

    def test_a_comment_with_a_comma_is_quoted_and_survives(self, tmp_path):
        param("ALPHA", type="f64", unit="1", lo=0.0, hi=1.0, mutability="design",
              doc="Gain, dimensionless")
        path = tmp_path / "p.csv"
        table.write(path, sync.apply(all_params(), []))
        assert table.read(path)[0]["comment"] == "Gain, dimensionless"

    def test_loads_refuses_a_row_with_the_wrong_field_count(self):
        header = ",".join(table.columns())
        with pytest.raises(ValueError, match="line 2 has 2 fields"):
            table.loads(f"{header}\nALPHA,1\n")

    def test_a_misspelt_human_column_is_refused_not_dropped(self):
        param("ALPHA", type="f64", **OK)
        with pytest.raises(ValueError, match="provenence"):
            table.row_from_decl(all_params()["ALPHA"], provenence="M")


def written(tmp_path, values):
    """Declare, sync, and fill in ``{name: (value, provenance, source)}``."""
    path = tmp_path / "params.csv"
    sync.sync(path, all_params())
    rows = table.read(path)
    for row in rows:
        if row["name"] in values:
            row["value"], row["provenance"], row["source"] = values[row["name"]]
    table.write(path, rows)
    return path


class TestValuesFrom:

    def test_it_returns_the_value_with_its_provenance_and_source(self, tmp_path):
        param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0, mutability="fixed",
              doc="Refresh period")
        path = written(tmp_path, {"REFRESH_S": ("300.0", "D", "GOES-R PUG, Table 5.1")})
        value = values_from(path)["REFRESH_S"]
        assert isinstance(value, ParamValue)
        assert (value.value, value.provenance, value.source) == (300.0, "D",
                                                                 "GOES-R PUG, Table 5.1")
        assert (value.unit, value.lo, value.hi) == ("s", 0.0, 3600.0)
        assert float(value) == 300.0

    def test_an_integer_type_comes_back_as_an_int(self, tmp_path):
        param("N_SATS", type="u8", unit="1", lo=1, hi=32, mutability="design", doc="Count")
        value = values_from(written(tmp_path, {"N_SATS": ("4", "A", "trade")}))["N_SATS"]
        assert value.value == 4 and isinstance(value.value, int)

    def test_a_blank_value_is_refused_naming_it(self, tmp_path):
        param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0, mutability="fixed",
              doc="Refresh period")
        param("LATENCY_S", type="f64", unit="s", lo=0.0, hi=60.0, mutability="fixed",
              doc="Latency")
        path = written(tmp_path, {"REFRESH_S": ("300.0", "D", "doc")})
        with pytest.raises(ValueError, match=r"\['LATENCY_S'\] have no value"):
            values_from(path)

    def test_allow_unfilled_leaves_the_blank_out(self, tmp_path):
        param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0, mutability="fixed",
              doc="Refresh period")
        param("LATENCY_S", type="f64", unit="s", lo=0.0, hi=60.0, mutability="fixed",
              doc="Latency")
        path = written(tmp_path, {"REFRESH_S": ("300.0", "D", "doc")})
        assert list(values_from(path, allow_unfilled=True)) == ["REFRESH_S"]

    def test_a_missing_file_says_to_run_sync(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="machina params sync"):
            values_from(tmp_path / "nowhere.csv")

    def test_a_non_numeric_value_points_at_check(self, tmp_path):
        param("REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0, mutability="fixed",
              doc="Refresh period")
        path = written(tmp_path, {"REFRESH_S": ("five", "D", "doc")})
        with pytest.raises(ValueError, match="machina params check"):
            values_from(path)
