"""
``DefaultContract.lint_text`` -- the rules a published params table obeys.

This linter is a reimplementation: icarus's comes from a private repository
that is not distributable. The two substrings icarus's own tests pin -- "value
is empty" and "on the f32 wire" -- are kept, so the two read alike.
"""

import pytest

from machina.params.lint import DefaultContract

pytestmark = pytest.mark.requires_casadi

HEADER = ",".join(DefaultContract.COLUMNS)
GOOD = "KP_ROLL,2.5,f32,1,0.0,10.0,,flight,M,bench,A gain"


def lint(*rows, allow_unfilled=False, header=HEADER):
    text = "\n".join([header, *rows]) + "\n"
    return DefaultContract().lint_text(text, allow_unfilled=allow_unfilled)


def errors(*rows, **kw):
    return [str(i) for i in lint(*rows, **kw) if not i.warning]


def warnings(*rows, **kw):
    return [str(i) for i in lint(*rows, **kw) if i.warning]


def row(**over):
    fields = dict(zip(DefaultContract.COLUMNS, GOOD.split(",")))
    fields.update(over)
    return ",".join(fields[c] for c in DefaultContract.COLUMNS)


class TestTheFormat:

    def test_eleven_columns_with_provenance_beside_value(self):
        assert DefaultContract.COLUMNS == ("name", "value", "type", "unit", "min", "max",
                                           "default", "mutability", "provenance", "source",
                                           "comment")

    def test_a_good_row_is_clean(self):
        assert lint(GOOD) == []

    def test_the_header_must_be_exact(self):
        (issue,) = lint(GOOD, header="name,value,type")
        assert "header must be exactly" in str(issue) and issue.line == 1

    def test_an_empty_table_names_the_header_it_needs(self):
        (issue,) = DefaultContract().lint_text("")
        assert "name,value,type" in str(issue)

    def test_a_short_row_is_an_error_not_a_silent_truncation(self):
        assert any("has 3 fields" in e for e in errors("KP_ROLL,2.5,f32"))

    def test_rows_must_be_sorted_by_name(self):
        assert any("not sorted" in e for e in errors(row(name="ZZ"), row(name="AA")))

    def test_a_duplicate_name_is_reported_once(self):
        found = [e for e in errors(row(), row()) if "appears 2 times" in e]
        assert len(found) == 1

    def test_issues_carry_the_line_and_name(self):
        (issue,) = [i for i in lint(row(unit="deg"))]
        assert issue.line == 2 and issue.name == "KP_ROLL"
        assert str(issue).startswith("line 2, KP_ROLL:")


class TestValues:

    def test_an_empty_value_is_an_error(self):
        assert any("value is empty" in e for e in errors(row(value="", provenance="")))

    def test_allow_unfilled_accepts_it(self):
        assert errors(row(value="", provenance=""), allow_unfilled=True) == []

    def test_a_filled_value_needs_a_provenance_code(self):
        assert any("provenance code" in e for e in errors(row(provenance="")))

    @pytest.mark.parametrize("code", ["D", "P", "E", "A", "M"])
    def test_every_provenance_code_is_accepted(self, code):
        assert errors(row(provenance=code)) == []

    def test_an_unknown_provenance_code_is_refused(self):
        assert any("provenance 'Q'" in e for e in errors(row(provenance="Q")))

    def test_a_value_out_of_range_is_refused(self):
        assert any("outside" in e for e in errors(row(value="11.0")))

    def test_a_value_that_is_not_a_number_is_refused(self):
        assert any("not a number" in e for e in errors(row(value="two")))

    def test_nan_is_refused(self):
        assert any("NaN" in e for e in errors(row(value="nan")))

    def test_f32_quantization_is_a_warning_with_the_loss(self):
        found = warnings(row(value="0.1"))
        assert len(found) == 1 and "on the f32 wire" in found[0] and "loss" in found[0]

    def test_an_exactly_representable_f32_value_is_quiet(self):
        assert warnings(row(value="0.25")) == []

    def test_f64_values_are_never_warned(self):
        assert warnings(row(value="0.1", type="f64")) == []

    def test_an_integer_type_refuses_a_fraction(self):
        assert any("not an integer" in e
                   for e in errors(row(type="u8", value="2.5", min="0", max="10")))

    def test_an_integer_type_refuses_a_value_that_does_not_fit(self):
        assert any("does not fit in u8" in e
                   for e in errors(row(type="u8", value="300", min="0", max="255")))

    def test_an_integer_bound_that_does_not_fit_is_refused(self):
        assert any("does not fit in i8" in e
                   for e in errors(row(type="i8", value="0", min="-200", max="10")))

    def test_an_infinite_float_bound_is_allowed(self):
        assert errors(row(max="inf")) == []

    def test_an_infinite_value_is_not(self):
        assert any("infinite" in e for e in errors(row(value="inf", max="inf")))


class TestDeclarationColumns:

    def test_a_non_si_unit_is_refused(self):
        assert any("not SI" in e for e in errors(row(unit="deg")))

    def test_a_missing_unit_is_refused(self):
        assert any("unit is mandatory" in e for e in errors(row(unit="")))

    def test_an_unknown_type_is_refused(self):
        assert any("type 'float'" in e for e in errors(row(type="float")))

    def test_an_unknown_mutability_is_refused(self):
        assert any("mutability 'sometimes'" in e for e in errors(row(mutability="sometimes")))

    def test_inverted_bounds_are_refused(self):
        assert any("greater than max" in e for e in errors(row(min="5.0", max="1.0",
                                                                value="2.0")))

    def test_a_default_out_of_range_is_refused(self):
        assert any("default" in e and "outside" in e for e in errors(row(default="50.0")))

    def test_an_empty_comment_is_refused(self):
        assert any("comment is empty" in e for e in errors(row(comment="")))

    def test_a_bad_name_is_refused(self):
        assert any("name must match" in e for e in errors(row(name="1KP")))

    def test_a_name_over_64_characters_is_refused(self):
        assert any("the limit is 64" in e for e in errors(row(name="A" * 65)))
