"""
The numeric guards, in one place.

These are load-bearing numerics, not style. CasADi differentiates every
branch of an expression, including the one an ``if_else`` does not select,
so a denominator that is only zero on the unselected branch still produces a
NaN in the gradient. The optimizer then reports an ``Invalid_Number_Detected``
that has nothing to do with the model.

``TINY`` guards two things:

* every ``sqrt`` of a quantity that can be exactly zero -- the norm of a
  vector at the origin, the eccentricity of a circular orbit;
* every denominator inside a branch that may not be selected.

Until Phase 2 the constant was copied into three files under two spellings.
The value has not changed; only its address has. Vault: Design/Function
Library, Decision Log #30.
"""

import casadi as ca

__all__ = ["TINY", "EPS", "safe_sqrt", "safe_norm", "safe_divide"]

TINY = 1e-32
"""Denominator and square-root guard. Small enough to be negligible against
any physical quantity in SI, large enough that its square root (1e-16) and
its reciprocal (1e32) are both far from the edges of double precision."""

EPS = 1e-4
"""Threshold below which a series expansion replaces a cancelling formula.
Four Taylor terms are accurate to double precision for arguments under 0.1;
1e-4 leaves three orders of margin. Used by the Stumpff functions."""


def safe_sqrt(x):
    """``sqrt(x)`` with the argument floored at :data:`TINY`.

    The derivative of ``sqrt`` is unbounded at zero, so guarding the argument
    rather than the result is what keeps the Jacobian finite.
    """
    return ca.sqrt(ca.fmax(x, TINY))


def safe_norm(v):
    """The 2-norm of ``v``, defined and differentiable at the origin."""
    return safe_sqrt(ca.dot(v, v))


def safe_divide(numerator, denominator):
    """``numerator / denominator`` with the denominator floored at :data:`TINY`.

    Only correct for a denominator that is non-negative by construction --
    a norm, a squared quantity. For a denominator that can change sign, guard
    the expression that makes it vanish instead.
    """
    return numerator / ca.fmax(denominator, TINY)
