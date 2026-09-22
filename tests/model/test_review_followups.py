"""
Regressions for the second, targeted review of Phase 2a.

Two focused reviewers checked the role-per-path rewrite: one hunting false
positives in the declare pass, one in wire/build and the declaration
validators. Each test pins one thing they demonstrated.
"""

import casadi as ca
import numpy as np
import pytest
from synthetic_model import make_registry, point_mass

from machina.model import (
    Aggregation,
    Builder,
    Component,
    Cost,
    Declaration,
    ModelError,
    Quantity,
    Scope,
    SignalRegistry,
)

pytestmark = pytest.mark.requires_casadi


def scalar_registry():
    reg = SignalRegistry()
    for name in ("x", "u", "y", "z"):
        reg.declare(name, 1, "1")
    reg.declare("F", 1, "1", aggregation=Aggregation.SUM)
    return reg


def G(label, const=0.0, **groups):
    """A scalar component: every output is const plus the sum of what it reads."""
    class Generic(Component):
        def declare(self):
            return Declaration(**groups)

        def build(self, helpers):
            dec = self.declare()
            args = [ca.SX.sym(local) for local in dec.consumed()]
            total = sum(args, ca.SX(const))
            names = list(dec.consumed())
            out = {}
            for group, key in (("derivatives", "f"), ("produces", "g")):
                produced = getattr(dec, group)
                if produced:
                    out[key] = ca.Function(f"{self.name}_{key}", args, [total] * len(produced),
                                           names, list(produced))
            return out

    Generic.__name__ = f"G{label}"
    return Generic()


def declared(components, registry=None):
    return Builder(components, registry=registry or scalar_registry()).declare()


class TestScopedInputReferences:

    def test_a_relative_scoped_input_binds_to_the_scope_that_exists(self):
        """'a/u' read from scope b used to bind to 'b/a/u', a scope that does not exist."""
        builder = declared([Scope("a", [G("r", inputs=("u",), derivatives=("x",))]),
                            Scope("b", [G("r", inputs=(("u", "a/u"),), derivatives=("x",))])])
        assert builder.resolved("b/gr")["u"] == "a/u"
        assert builder.input_order == ["a/u"]

    def test_a_relative_scoped_input_is_checked_only_at_the_path_it_binds_to(self):
        """A root scope 'inner' producing u does not enclose 'a/inner'."""
        builder = declared([
            Scope("inner", [G("p", produces=("u",))]),
            Scope("a", [Scope("inner", [G("q", derivatives=("x",))]),
                        G("r", inputs=(("u", "inner/u"),), derivatives=("z",))]),
        ])
        assert builder.resolved("a/gr")["u"] == "a/inner/u"


class TestSignalsBesideInputs:

    def test_an_unproduced_signal_one_scope_below_an_input_is_refused(self):
        """It used to read as a silent zero while the input sat one scope up."""
        with pytest.raises(ModelError, match=r"'a/F' is an input .* \('F', '/a/F'\)"):
            declared([Scope("a", [G("ext", inputs=("F",), derivatives=("x",)),
                                  Scope("inner", [G("r", algebraic=("F",), produces=("z",))])])])

    def test_the_unique_version_gets_the_same_explanation(self):
        with pytest.raises(ModelError, match="is an input"):
            declared([Scope("a", [G("ext", inputs=("u",), derivatives=("x",)),
                                  Scope("inner", [G("r", algebraic=("u",), produces=("z",))])])])

    def test_reading_the_input_explicitly_works(self):
        builder = declared([Scope("a", [
            G("ext", inputs=("F",), derivatives=("x",)),
            Scope("inner", [G("r", inputs=(("F", "/a/F"),), produces=("z",))])])])
        assert builder.input_order == ["a/F"]


class TestMessagesPointAtTheFix:

    def test_an_unproduced_explicit_path_names_where_the_signal_is_produced(self):
        """The old hint said 'read it by its bare name', which gave a silent zero."""
        with pytest.raises(ModelError, match=r"produced at \['a/inner/F'\].*'/a/inner/F'"):
            declared([Scope("a", [Scope("inner", [G("p", 3.0, produces=("F",))])]),
                      G("r", algebraic=(("F", "/a/F"),), derivatives=("x",))])

    def test_a_pass_through_loop_suggests_the_absolute_path_of_the_outer_value(self):
        with pytest.raises(ModelError, match=r"\('F', '/F'\)"):
            declared([G("root", 1.0, produces=("F",)),
                      Scope("a", [G("pass", algebraic=("F",), produces=("F",)),
                                  G("use", algebraic=("F",), derivatives=("x",))])])

    def test_the_pass_through_written_that_way_works(self):
        builder = declared([G("root", 1.0, produces=("F",)),
                            Scope("a", [G("pass", 0.5, algebraic=(("F_parent", "/F"),),
                                          produces=("F",)),
                                        G("use", algebraic=("F",), derivatives=("x",))])])
        model = builder.build()
        where = builder.layout(builder.state_order)["a/x"]
        assert np.array(model["f"]([0.0], [])).ravel()[where][0] == pytest.approx(1.5)


class TestWireAndBuildValues:

    def test_a_sparse_function_argument_is_refused(self):
        """A diag-pattern argument silently dropped the off-diagonal entries: 5 instead of 10."""
        class DiagArg(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("A", shape=(2, 2)),), costs=(Cost("c"),))

            def build(self, helpers):
                A = ca.SX.sym("A", ca.Sparsity.diag(2))
                return {"J": ca.Function("J", [A], [ca.sum1(ca.sum2(A))], ["A"], ["c"])}

        builder = Builder([DiagArg()], registry=make_registry()).declare()
        with pytest.raises(ModelError, match="sparse pattern"):
            builder.wire({"A": ca.DM([[1, 2], [3, 4]])})

    @pytest.mark.parametrize("value,message", [
        (np.nan, "NaN"), (np.inf, "not finite"), ("2.0", "must be numeric"),
        (True, "must be numeric"), (np.bool_(True), "must be numeric"),
    ])
    def test_a_fixed_value_follows_the_quantity_rules(self, value, message):
        with pytest.raises(ModelError, match=message):
            Builder(point_mass(), registry=make_registry()).declare().build(
                fixed={"dry_mass": value})

    def test_a_scalar_is_not_broadcast_into_a_vector_quantity(self):
        class Vec(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("k", shape=3),), produces=("mass",))

            def build(self, helpers):
                k = ca.SX.sym("k", 3)
                return {"g": ca.Function("vec_g", [k], [ca.sum1(k)], ["k"], ["mass"])}

        builder = Builder([Vec()], registry=make_registry()).declare()
        with pytest.raises(ModelError, match="nothing is broadcast"):
            builder.build(fixed={"k": 1.0})
        model = builder.build(fixed={"k": [1.0, 2.0, 3.0]})
        assert float(model["g"]([], [])) == pytest.approx(6.0)

    def test_python_lists_and_numbers_are_valid_leaves(self):
        """They used to pass the shape check and then crash inside CasADi."""
        builder = Builder(point_mass(), registry=make_registry()).declare()
        wired = builder.wire({"pos": [0.0, 0.0], "vel": [1.0, 2.0], "thrust_cmd": 0.5,
                              "dry_mass": 2.0})
        np.testing.assert_allclose(np.array(ca.evalf(wired.xdot["vel"])).ravel(), [0.5, 0.0])

    def test_a_matrix_quantity_accepts_a_nested_list_leaf(self):
        class Mat(Component):
            def declare(self):
                return Declaration(quantities=(Quantity("M", shape=(2, 2)),), produces=("mass",))

            def build(self, helpers):
                m = ca.SX.sym("M", 2, 2)
                return {"g": ca.Function("mat_g", [m], [m[1, 0]], ["M"], ["mass"])}

        builder = Builder([Mat()], registry=make_registry()).declare()
        wired = builder.wire({"M": [[1.0, 2.0], [3.0, 4.0]]})
        assert float(ca.evalf(wired.values["mass"])) == 3.0


class TestDeclarationValues:

    @pytest.mark.parametrize("weight", [ca.DM(np.inf), ca.DM(np.nan)])
    def test_a_non_finite_dm_weight_is_refused(self, weight):
        with pytest.raises(ModelError, match="not finite"):
            Cost("c", weight=weight)

    @pytest.mark.parametrize("weight", [np.array(2.0), ca.DM(2.0), np.array([2.0])])
    def test_a_size_one_array_or_dm_weight_is_accepted(self, weight):
        assert Cost("c", weight=weight).weight is weight

    def test_a_list_weight_is_refused_as_ambiguous(self):
        with pytest.raises(ModelError, match="weight"):
            Cost("c", weight=[1.0])

    @pytest.mark.parametrize("shape,ub", [(4, np.ones((2, 2))), ((2, 2), np.ones((4, 1)))])
    def test_bounds_follow_the_backend_shape_rule(self, shape, ub):
        """These used to pass here and fail at compile time naming a solver variable."""
        with pytest.raises(ModelError, match="expected a scalar, an array of shape"):
            Quantity("q", shape=shape, ub=ub)

    def test_a_flat_column_major_bound_is_accepted_for_a_matrix(self):
        assert Quantity("q", shape=(2, 2), ub=[1.0, 2.0, 3.0, 4.0]).size == 4

    def test_a_row_for_a_column_bound_is_accepted_as_the_backend_does(self):
        assert Quantity("q", shape=3, ub=np.ones((1, 3))).size == 3

    @pytest.mark.parametrize("label", ["lb", "default"])
    def test_a_numpy_bool_bound_or_default_is_refused(self, label):
        with pytest.raises(ModelError, match="must be numeric"):
            Quantity("q", **{label: np.True_})

    def test_a_dm_bound_is_accepted(self):
        assert Quantity("q", shape=2, lb=ca.DM([0.0, 1.0]), ub=ca.DM([2.0, 3.0])).size == 2
