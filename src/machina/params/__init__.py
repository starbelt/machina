"""
machina.params -- parameters declared in code, valued by hand, checked in CI.

Adopted from icarus-dynamics' params pipeline (Decision Log #36, rule I6):

1. **Declared in code** with :func:`param` -- type, unit, bounds, mutability,
   documentation. Everything except the number.
2. **Valued by hand** in a CSV. ``machina params sync`` adds a row with the
   value blank; a person fills in ``value``, ``provenance`` (D P E A M) and
   ``source``. The tool never guesses a number.
3. **Checked in CI** by ``machina params check``, which never writes and
   refuses a blank value.

The CSV format is an injectable contract (:mod:`machina.params.contract`);
:class:`machina.params.lint.DefaultContract` is the default.

Importing this package declares nothing and reads no contract: icarus's
eager import of its contract is what made its test suite fail at collection
outside its Nix shell.
"""

from machina.params import contract
from machina.params.contract import LintIssue, use, using
from machina.params.declare import KINDS, ParamDeclarationError, StaticParam, param
from machina.params.registry import (
    DEFAULT,
    DuplicateParam,
    ParamDecl,
    ParamRegistry,
    all_params,
)
from machina.params.values import ParamValue, values_from

__all__ = [
    "param", "StaticParam", "ParamDeclarationError", "KINDS",
    "ParamDecl", "ParamRegistry", "DuplicateParam", "DEFAULT", "all_params",
    "ParamValue", "values_from",
    "contract", "use", "using", "LintIssue",
]
