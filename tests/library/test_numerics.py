"""
The numeric guards, and the rule that there is only one copy of them.

``TINY`` is not a magic number someone picked; it is the reason the coverage
problem has a finite Jacobian at zero eccentricity. It lived in three files
under two spellings until Phase 2, which is how a guard quietly becomes a
guard with two values.
"""

import re
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from machina.library.numerics import EPS, TINY, safe_divide, safe_norm, safe_sqrt

pytestmark = pytest.mark.requires_casadi

REPO = Path(__file__).resolve().parents[2]
HOME = REPO / "src" / "machina" / "library" / "numerics.py"


def evaluate(expr, sym, value):
    return float(ca.Function("f", [sym], [expr])(value))


class TestTheGuards:

    def test_tiny_and_eps_are_the_documented_values(self):
        assert (TINY, EPS) == (1e-32, 1e-4)

    def test_safe_sqrt_is_finite_and_tiny_at_zero(self):
        x = ca.SX.sym("x")
        # abs=0: pytest.approx's default absolute tolerance (1e-12) would accept 0.0,
        # which is exactly what an unguarded sqrt returns.
        assert evaluate(safe_sqrt(x), x, 0.0) == pytest.approx(1e-16, rel=1e-9, abs=0)

    def test_safe_sqrt_is_the_ordinary_sqrt_away_from_zero(self):
        x = ca.SX.sym("x")
        np.testing.assert_allclose(evaluate(safe_sqrt(x), x, 4.0), 2.0)

    def test_the_derivative_of_safe_sqrt_is_finite_at_zero(self):
        """This is the whole point: sqrt' is unbounded at zero, so the
        argument is floored rather than the result."""
        x = ca.SX.sym("x")
        slope = evaluate(ca.jacobian(safe_sqrt(x), x), x, 0.0)
        assert np.isfinite(slope)

    def test_safe_norm_is_defined_at_the_origin(self):
        v = ca.SX.sym("v", 3)
        assert evaluate(safe_norm(v), v, [0.0, 0.0, 0.0]) == pytest.approx(1e-16, rel=1e-9, abs=0)

    def test_safe_norm_is_the_ordinary_norm_elsewhere(self):
        v = ca.SX.sym("v", 3)
        np.testing.assert_allclose(evaluate(safe_norm(v), v, [3.0, 4.0, 0.0]), 5.0)

    def test_the_jacobian_of_safe_norm_at_the_origin_is_finite(self):
        v = ca.SX.sym("v", 3)
        jac = ca.Function("j", [v], [ca.jacobian(safe_norm(v), v)])([0.0, 0.0, 0.0])
        assert np.all(np.isfinite(np.array(jac)))

    def test_safe_divide_floors_the_denominator(self):
        d = ca.SX.sym("d")
        assert evaluate(safe_divide(1.0, d), d, 0.0) == pytest.approx(1e32)

    def test_safe_divide_is_ordinary_division_elsewhere(self):
        d = ca.SX.sym("d")
        np.testing.assert_allclose(evaluate(safe_divide(1.0, d), d, 4.0), 0.25)


class TestThereIsOnlyOneCopy:

    def test_the_guard_value_is_written_down_in_exactly_one_file(self):
        """Three copies under two spellings is how a guard gets two values."""
        pattern = re.compile(r"1e-32")
        hits = []
        for top in ("src", "examples"):
            for path in sorted((REPO / top).rglob("*.py")):
                if path == HOME:
                    continue
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
        assert not hits, (
            "import TINY from machina.library.numerics instead of redefining it:\n"
            + "\n".join(hits))

    def test_the_stumpff_threshold_is_written_down_in_exactly_one_file(self):
        pattern = re.compile(r"\bEPS\s*=\s*1e-4\b")
        hits = []
        for path in sorted((REPO / "src").rglob("*.py")):
            if path == HOME:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{lineno}")
        assert not hits, "\n".join(hits)

    def test_the_orbital_factories_now_import_the_shared_guard(self):
        from machina.astro import geometry, transforms
        assert geometry.TINY is TINY
        assert transforms.TINY is TINY and transforms.EPS is EPS

    @pytest.mark.parametrize("psi", [1e-6, -1e-6, 2e-4, 1.0, -1.0, 25.0])
    def test_centralising_it_did_not_move_any_numbers(self, psi):
        """The Stumpff functions are the sensitive consumer. psi = +-1e-6 exercises the
        EPS Taylor branch, the others the closed forms that TINY guards."""
        import machina.astro  # noqa: F401
        from machina.library import registry
        stumpff = registry.get("transform.stumpff_cs")()
        c, s = (float(v) for v in stumpff.function(psi))
        if psi > 0:
            root = np.sqrt(psi)
            want = ((1 - np.cos(root)) / psi, (root - np.sin(root)) / root ** 3)
        else:
            root = np.sqrt(-psi)
            want = ((np.cosh(root) - 1) / -psi, (np.sinh(root) - root) / root ** 3)
        np.testing.assert_allclose([c, s], want, rtol=1e-9)
