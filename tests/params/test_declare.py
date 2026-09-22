"""
``param()`` must refuse a declaration that cannot become a legal row.

Ported from icarus-dynamics ``tests/test_declare.py``. The linter is the
authority on the format, but hearing about a mistake here -- in Python, with a
traceback pointing at the declaration -- is worth having as well. These are
the same rules, checked earlier, under machina's default contract; the
icarus-specific rules (uppercase names, 16 characters for MAVLink) are
exercised through an injected contract in ``test_contract.py``.
"""

import math

import casadi as ca
import pytest

from machina.params import (
    DuplicateParam,
    ParamDeclarationError,
    ParamRegistry,
    StaticParam,
    all_params,
    param,
)

pytestmark = pytest.mark.requires_casadi

OK = dict(type="f32", unit="1", lo=0.0, hi=1.0, mutability="flight", doc="A param")


class TestWhatADeclarationReturns:

    def test_a_symbolic_param_is_a_casadi_symbol_named_after_it(self):
        symbol = param("KP_ROLL", **OK)
        assert isinstance(symbol, ca.MX)
        assert symbol.name() == "KP_ROLL"

    def test_a_static_param_is_not(self):
        assert isinstance(param("ENABLE_TECS", kind="static", **OK), StaticParam)

    def test_a_static_param_refuses_to_enter_the_graph(self):
        """Misuse fails at model-build time with a message naming the fix,
        rather than silently producing a wrong expression graph."""
        static = param("ENABLE_TECS", kind="static", **OK)
        with pytest.raises(ParamDeclarationError, match="kind='symbolic'"):
            _ = static * 2
        with pytest.raises(ParamDeclarationError):
            _ = 2 + static
        with pytest.raises(ParamDeclarationError):
            _ = float(static)
        with pytest.raises(ParamDeclarationError):
            _ = -static

    def test_a_static_param_can_still_be_tested_for_truth_and_printed(self):
        static = param("ENABLE_TECS", kind="static", **OK)
        assert static and "ENABLE_TECS" in repr(static)

    def test_the_declaration_records_where_it_came_from(self):
        """So an error message can say which file to go and fix."""
        param("KP_ROLL", **OK)
        assert all_params()["KP_ROLL"].declared_in == __name__

    def test_the_declaration_carries_everything_but_the_value(self):
        param("KP_ROLL", default=0.2, **OK)
        decl = all_params()["KP_ROLL"]
        assert (decl.type, decl.unit, decl.lo, decl.hi, decl.mutability, decl.kind,
                decl.doc, decl.default) == ("f32", "1", 0.0, 1.0, "flight", "symbolic",
                                            "A param", 0.2)

    def test_declaration_order_is_kept(self):
        for name in ("ZULU", "ALPHA", "MIKE"):
            param(name, **OK)
        assert list(all_params()) == ["ZULU", "ALPHA", "MIKE"]

    def test_a_private_registry_keeps_the_default_one_clean(self):
        private = ParamRegistry()
        param("KP_ROLL", registry=private, **OK)
        assert list(private.all()) == ["KP_ROLL"]
        assert "KP_ROLL" not in all_params()


class TestWhatADeclarationRefuses:

    @pytest.mark.parametrize("name", ["1KP", "KP-ROLL", "", "kp roll", "_leading"])
    def test_bad_names_are_rejected(self, name):
        with pytest.raises(ParamDeclarationError, match="must match"):
            param(name, **OK)

    def test_mixed_case_names_are_fine_under_the_default_contract(self):
        """The uppercase rule is MAVLink's (Decision Log #50), not machina's."""
        assert param("conusRefresh_s", **OK) is not None

    def test_a_name_over_the_limit_is_rejected_and_says_why(self):
        with pytest.raises(ParamDeclarationError, match="the limit is 64"):
            param("A" * 65, **OK)

    def test_a_missing_unit_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="unit is mandatory"):
            param("KP_ROLL", **{**OK, "unit": ""})

    def test_a_non_si_unit_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="not SI"):
            param("AOA_MAX", **{**OK, "unit": "deg"})

    def test_an_unknown_type_lists_the_known_ones(self):
        with pytest.raises(ParamDeclarationError, match="f32 f64 i8"):
            param("KP_ROLL", **{**OK, "type": "float"})

    def test_a_bad_mutability_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="mutability"):
            param("KP_ROLL", **{**OK, "mutability": "whenever"})

    def test_a_bad_kind_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="kind"):
            param("KP_ROLL", kind="sort-of", **OK)

    def test_a_missing_doc_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="doc is mandatory"):
            param("KP_ROLL", **{**OK, "doc": ""})

    def test_inverted_bounds_are_rejected(self):
        with pytest.raises(ParamDeclarationError, match="greater than"):
            param("KP_ROLL", **{**OK, "lo": 1.0, "hi": 0.0})

    def test_a_default_outside_the_bounds_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="outside"):
            param("KP_ROLL", default=5.0, **OK)

    def test_a_fractional_default_on_an_integer_type_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="not an integer"):
            param("N_CHAN", type="u8", unit="1", lo=0, hi=16, mutability="boot",
                  doc="Channels", default=2.5)

    def test_a_bound_must_be_a_number(self):
        with pytest.raises(ParamDeclarationError, match="must be a number"):
            param("KP_ROLL", **{**OK, "hi": "1.0"})

    def test_a_boolean_is_not_a_number_here(self):
        with pytest.raises(ParamDeclarationError, match="must be a number"):
            param("KP_ROLL", **{**OK, "hi": True})

    def test_a_nan_bound_is_rejected(self):
        with pytest.raises(ParamDeclarationError, match="NaN"):
            param("KP_ROLL", **{**OK, "hi": math.nan})

    def test_an_infinite_bound_is_fine_on_a_float_type(self):
        assert param("RANGE_M", **{**OK, "unit": "m", "hi": math.inf}) is not None

    def test_an_infinite_bound_is_refused_on_an_integer_type(self):
        with pytest.raises(ParamDeclarationError, match="not an integer"):
            param("N_CHAN", type="u8", unit="1", lo=0, hi=math.inf, mutability="boot",
                  doc="Channels")

    def test_quantization_is_not_a_declaration_error(self):
        """0.03 is inexact on the f32 wire and that is fine. The linter reports
        the loss; refusing the declaration would ban most numbers anyone writes."""
        assert param("STICK_DB", **{**OK, "hi": 0.25, "default": 0.03}) is not None

    def test_a_duplicate_name_is_rejected_naming_both_places(self):
        param("KP_ROLL", **OK)
        with pytest.raises(DuplicateParam, match="declared twice"):
            param("KP_ROLL", **OK)

    def test_the_duplicate_error_mentions_two_copies_on_the_path(self):
        param("KP_ROLL", **OK)
        with pytest.raises(DuplicateParam, match="two copies of machina"):
            param("KP_ROLL", **OK)
