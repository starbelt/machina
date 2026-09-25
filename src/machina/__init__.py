"""
machina -- component-based mission modelling and optimization on CasADi.

``import machina`` loads the core only: the signal registry, components and
the builder (:mod:`machina.model`), the ``Problem`` compiler
(:mod:`machina.compiler`), the solver backend (:mod:`machina.solver`), the
params pipeline (:mod:`machina.params`) and the generic library
(:mod:`machina.library`).

It never imports a domain pack (``machina.astro``, ``machina.swapc``,
``machina.rigid``, ``machina.aero``), nor ``machina.viz``, ``machina.report``
or ``machina.study``, nor networkx, matplotlib or pandas. A pack declares its
signals and registers its factories when it is imported, so a study imports
the packs it uses, explicitly. ``tests/test_import_graph.py`` enforces this.
"""

from machina import library, params  # noqa: F401  (generic factories; the params pipeline)
from machina.compiler import Problem
from machina.model import (
    Aggregation,
    Builder,
    Component,
    Constraint,
    Cost,
    Declaration,
    Frame,
    FunctionDescriptor,
    ModelError,
    Quantity,
    Role,
    Scope,
    Signal,
    SignalError,
    SignalRegistry,
    SymbolDescriptor,
)
from machina.solver import SolutionResult, SolverBackend, SolverError

__all__ = [
    "Problem",
    "Aggregation", "Builder", "Component", "Constraint", "Cost", "Declaration", "Frame",
    "FunctionDescriptor", "ModelError", "Quantity", "Role", "Scope", "Signal", "SignalError",
    "SignalRegistry", "SymbolDescriptor",
    "SolutionResult", "SolverBackend", "SolverError",
]
