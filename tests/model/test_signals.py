"""
The signal registry: what it refuses, and what order it hands things back in.

A declaration is the only thing a wrong shape can disagree with, so the tests
here are mostly about refusals. The ordering tests matter as much: the
registry walk is the vector layout ABI.
"""

import pytest

from machina.model import signals as sig
from machina.model.errors import SignalError

pytestmark = pytest.mark.requires_casadi


def fresh():
    reg = sig.SignalRegistry()
    reg.declare_frame("world", doc="Flat 2-D inertial", family="inertial")
    return reg


class TestDeclaration:

    def test_a_declaration_carries_shape_unit_frame_and_aggregation(self):
        reg = fresh()
        declared = reg.declare("force", (3, 1), "N", frame="world",
                               aggregation=sig.Aggregation.SUM, doc="Applied force")
        assert declared.name == "force"
        assert declared.shape == (3, 1)
        assert declared.unit == "N"
        assert declared.frame == "world"
        assert declared.aggregation is sig.Aggregation.SUM
        assert declared.doc == "Applied force"
        assert declared.size == 3

    def test_an_int_shape_means_a_column(self):
        assert fresh().declare("vel", 3, "m/s").shape == (3, 1)

    def test_a_matrix_signal_reports_its_element_count(self):
        assert fresh().declare("inertia", (3, 3), "kg*m^2").size == 9

    def test_aggregation_defaults_to_unique(self):
        assert fresh().declare("mass", 1, "kg").aggregation is sig.Aggregation.UNIQUE

    def test_frame_defaults_to_none_which_the_registry_always_declares(self):
        reg = sig.SignalRegistry()
        assert reg.declare("mass", 1, "kg").frame == "none"
        assert reg.frame("none").name == "none"


class TestRefusals:

    def test_a_unit_is_mandatory(self):
        with pytest.raises(SignalError, match="unit is mandatory"):
            fresh().declare("mass", 1, "")

    def test_a_non_si_unit_is_refused_and_says_where_the_conversion_belongs(self):
        with pytest.raises(SignalError, match="is not SI"):
            fresh().declare("inclination", 1, "deg")

    @pytest.mark.parametrize("unit", ["deg/s", "rpm/s", "ft/s", "lbf*ft", "nmi/hr", "1/min",
                                      "degC", "kg*ft^2", "RPM", "Deg", "hrs", "kWh", "mAh",
                                      "AU", "rev/s", "psia", "gal", "h", "C", "F"])
    def test_a_non_si_symbol_inside_a_compound_unit_is_refused(self, unit):
        with pytest.raises(SignalError, match="is not SI"):
            fresh().declare("rate", 1, unit)

    @pytest.mark.parametrize("unit", ["m/s^2", "kg*m^2", "N*m", "rad/s", "1/s", "km",
                                      "km^3/s^2", "W", "1", "s^-1", "W/(m^2*K)", "J/(kg*K)",
                                      "Pa*s", "N*m/rad", "hPa", "GHz", "mm"])
    def test_si_and_si_prefixed_units_are_accepted(self, unit):
        assert fresh().declare("ok", 1, unit).unit == unit

    def test_a_malformed_unit_is_refused(self):
        with pytest.raises(SignalError, match="not a well-formed unit"):
            fresh().declare("weird", 1, "kg m")

    def test_declaring_twice_names_the_collision(self):
        reg = fresh()
        reg.declare("mass", 1, "kg")
        with pytest.raises(SignalError, match="declared twice"):
            reg.declare("mass", 1, "kg")

    def test_an_unknown_signal_says_a_typo_is_not_a_new_signal(self):
        with pytest.raises(SignalError, match="not a new signal"):
            fresh().get("body_linear_velocty")

    def test_an_unknown_signal_points_at_quantities_for_private_values(self):
        with pytest.raises(SignalError, match="Quantity, not a signal"):
            fresh().get("gain")

    def test_a_name_that_is_not_an_identifier_is_refused(self):
        # Component build() functions name their CasADi arguments after signals.
        with pytest.raises(SignalError, match="must be a Python identifier"):
            fresh().declare("sat/mass", 1, "kg")

    def test_a_zero_dimension_is_refused(self):
        with pytest.raises(SignalError, match="must be positive"):
            fresh().declare("empty", 0, "m")

    def test_a_shape_that_is_not_a_pair_is_refused(self):
        with pytest.raises(SignalError, match="int or a \\(rows, cols\\) pair"):
            fresh().declare("odd", "three", "m")

    def test_an_unknown_frame_lists_the_declared_ones(self):
        with pytest.raises(SignalError, match=r"Declared frames: \['none', 'world'\]"):
            fresh().declare("r", 3, "m", frame="eci")

    def test_aggregation_must_be_an_enum_member(self):
        with pytest.raises(SignalError, match="must be an Aggregation member"):
            fresh().declare("force", 3, "N", aggregation="sum")


class TestFrames:

    def test_a_frame_registry_is_extensible_where_an_enum_would_not_be(self):
        reg = sig.SignalRegistry()
        reg.declare_frame("eci", doc="Earth-centred inertial", family="inertial")
        reg.declare_frame("ecef", doc="Earth-centred Earth-fixed", family="earth")
        assert [f.name for f in reg.frames().values()] == ["none", "eci", "ecef"]
        assert reg.frame("eci").family == "inertial"

    def test_a_frame_cannot_be_declared_twice(self):
        with pytest.raises(SignalError, match="already declared"):
            fresh().declare_frame("world")

    def test_a_frame_name_must_be_an_identifier(self):
        with pytest.raises(SignalError, match="must be a Python identifier"):
            fresh().declare_frame("body frame")


class TestOrderIsTheAbi:

    def test_all_returns_declaration_order(self):
        reg = fresh()
        for name in ("zulu", "alpha", "mike"):
            reg.declare(name, 1, "m")
        assert list(reg.all()) == ["zulu", "alpha", "mike"]

    def test_all_is_a_copy_so_callers_cannot_reorder_the_abi(self):
        reg = fresh()
        reg.declare("mass", 1, "kg")
        reg.all().clear()
        assert list(reg.all()) == ["mass"]

    def test_has_answers_without_raising(self):
        reg = fresh()
        reg.declare("mass", 1, "kg")
        assert reg.has("mass") and not reg.has("typo")


class TestIsolation:

    def test_clear_empties_the_signals_and_keeps_the_none_frame(self):
        reg = fresh()
        reg.declare("mass", 1, "kg")
        reg.clear()
        assert reg.all() == {}
        assert list(reg.frames()) == ["none"]

    def test_restore_puts_back_both_signals_and_frames(self):
        reg = fresh()
        reg.declare("mass", 1, "kg")
        state = reg.snapshot()
        reg.clear()
        reg.declare("other", 1, "m")
        reg.restore(state)
        assert list(reg.all()) == ["mass"]
        assert list(reg.frames()) == ["none", "world"]

    def test_restore_brings_back_exactly_the_snapshot(self):
        """A signal declared after the snapshot is gone; one declared before is back."""
        reg = fresh()
        reg.declare("kept", 1, "kg")
        state = reg.snapshot()
        reg.declare("added_later", 1, "kg")
        reg.restore(state)
        assert not reg.has("added_later")
        reg.declare("added_later", 1, "kg")          # free again
        with pytest.raises(SignalError, match="declared twice"):
            reg.declare("kept", 1, "kg")             # still taken

    def test_a_snapshot_is_not_affected_by_later_declarations(self):
        reg = fresh()
        state = reg.snapshot()
        reg.declare("later", 1, "kg")
        reg.restore(state)
        reg.restore(state)
        assert not reg.has("later")

    def test_the_default_registry_is_the_one_the_module_shortcuts_write_to(self):
        assert sig.declare.__self__ is sig.DEFAULT
        assert sig.get.__self__ is sig.DEFAULT
        assert sig.all_signals() == sig.DEFAULT.all()
