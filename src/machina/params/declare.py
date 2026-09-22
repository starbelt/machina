"""
``param()`` -- the one way a parameter comes into existence.

    CONUS_REFRESH_S = param("CONUS_REFRESH_S", type="f64", unit="s", lo=0.0, hi=3600.0,
                            mutability="fixed", doc="ABI Mode 6A CONUS refresh period")

Adopted from icarus-dynamics ``params/declare.py``. The declaration owns
everything except the number. ``sync`` writes a row into the CSV with the
value blank; a human fills it in, with a provenance code and a source; CI's
``check`` refuses the blank. A plausible default nobody chose is how a number
reaches a result unexamined.

The rules are the active contract's (:mod:`machina.params.contract`), so a
mistake surfaces here, in Python, with a traceback pointing at the
declaration, rather than three stages later in CI. The linter remains the
authority; this is an earlier place to hear about it.

What does not belong here: a discretisation step, a table dimension, an
array capacity. Those are build-time constants, not parameters. Write them as
module constants.

``kind="symbolic"`` returns an ``MX`` symbol named after the parameter, for
hand-wired expressions (icarus's control law). ``kind="static"`` returns a
:class:`StaticParam` that refuses arithmetic, because a feature flag cannot
honestly be symbolic: ``if_else`` evaluates both branches. machina's compiler
reaches parameters differently -- ``Quantity(param="NAME")`` links a
component's leaf to a declaration by name -- and the two coexist.
"""

import inspect

import casadi as ca

from machina.params import contract
from machina.params.registry import DEFAULT, ParamDecl

__all__ = ["param", "StaticParam", "ParamDeclarationError", "KINDS"]

KINDS = ("symbolic", "static")


class ParamDeclarationError(Exception):
    """A declaration that must not reach the CSV."""


class StaticParam:
    """A param that is read by hand-written code, never by the expression graph.

    It exists as an object rather than as ``None`` so that misuse fails loudly
    and specifically.
    """

    __slots__ = ("decl",)

    def __init__(self, decl: ParamDecl):
        self.decl = decl

    def __repr__(self) -> str:
        return f"<StaticParam {self.decl.name} ({self.decl.type})>"

    def _reject(self, *_args, **_kwargs):
        raise ParamDeclarationError(
            f"{self.decl.name} is declared kind='static' and cannot enter the symbolic graph. "
            f"If it is genuinely part of a CasADi expression, declare it kind='symbolic'."
        )

    __add__ = __radd__ = _reject
    __sub__ = __rsub__ = _reject
    __mul__ = __rmul__ = _reject
    __matmul__ = __rmatmul__ = _reject
    __truediv__ = __rtruediv__ = _reject
    __pow__ = __rpow__ = _reject
    __neg__ = __pos__ = __abs__ = _reject
    __float__ = __int__ = _reject
    __array__ = _reject


def _fail(name, message) -> ParamDeclarationError:
    return ParamDeclarationError(f"{name}: {message}")


def _check_bound(name, label, value, type_name, rules) -> None:
    """Shape only. Float32 quantization is deliberately not checked here -- the
    linter reports it with the actual loss, because almost no ordinary decimal
    is exactly representable and banning them would ban most numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(name, f"{label} must be a number, got {value!r}")
    if value != value:
        raise _fail(name, f"{label} is NaN")
    if type_name in getattr(rules, "INT_TYPES", ()) and not float(value).is_integer():
        raise _fail(name, f"{label} {value!r} is not an integer, but the type is {type_name}")


def param(name: str, *, type: str, unit: str, lo, hi, mutability: str, doc: str,  # noqa: A002
          kind: str = "symbolic", default=None, registry=None):
    """Declare a parameter. Returns an MX symbol, or a StaticParam for kind='static'.

    ``type`` matches the CSV column name; renaming the argument would be worse
    than shadowing the builtin.
    """
    rules = contract.get()
    if not isinstance(name, str) or not rules.NAME_RE.fullmatch(name):
        raise _fail(name, f"name must match {getattr(rules, 'NAME_RULE', rules.NAME_RE.pattern)}")
    if len(name) > rules.MAX_NAME_LEN:
        reason = getattr(rules, "NAME_LENGTH_REASON", "")
        raise _fail(name, f"name is {len(name)} characters; the limit is {rules.MAX_NAME_LEN}"
                          + (f" ({reason})" if reason else ""))
    if type not in rules.NUMERIC_TYPES:
        raise _fail(name, f"type {type!r} is not one of {' '.join(rules.NUMERIC_TYPES)}")
    if not unit:
        raise _fail(name, 'unit is mandatory (use "1" for dimensionless)')
    if not rules.UNIT_RE.match(unit):
        raise _fail(name, f"unit {unit!r} has unexpected characters")
    unit_is_si = getattr(rules, "is_si", None)
    if (not unit_is_si(unit)) if unit_is_si else (unit in rules.NON_SI_UNITS):
        raise _fail(name, f"unit {unit!r} is not SI; convert at the boundary that produced "
                          f"the number, and keep display units in plot labels")
    if mutability not in rules.MUTABILITIES:
        raise _fail(name, f"mutability must be one of {' | '.join(rules.MUTABILITIES)}")
    if kind not in KINDS:
        raise _fail(name, f"kind must be one of {' | '.join(KINDS)}")
    if not doc:
        raise _fail(name, "doc is mandatory; a parameter nobody can explain is one nobody "
                          "can review")

    _check_bound(name, "lo", lo, type, rules)
    _check_bound(name, "hi", hi, type, rules)
    if lo > hi:
        raise _fail(name, f"lo {lo} is greater than hi {hi}")
    if default is not None:
        _check_bound(name, "default", default, type, rules)
        if not lo <= default <= hi:
            raise _fail(name, f"default {default} is outside [{lo}, {hi}]")

    # One frame up is the declaring module. inspect.stack() would resolve source context for
    # every frame on the stack to answer the same question.
    caller = inspect.currentframe().f_back
    declared_in = caller.f_globals.get("__name__", "<unknown>") if caller else "<unknown>"

    target = DEFAULT if registry is None else registry
    decl = target.register(ParamDecl(
        name=name, type=type, unit=unit, lo=lo, hi=hi, mutability=mutability,
        kind=kind, doc=doc, default=default, declared_in=declared_in,
    ))
    return ca.MX.sym(name) if kind == "symbolic" else StaticParam(decl)
