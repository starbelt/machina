"""
machina.blocks — Layer 2 block library.

Public API:
    FunctionDescriptor  — thin wrapper around ca.Function with validated calling
    SymbolDescriptor    — metadata wrapper for MX variables/expressions
    registry            — module with register(), get(), list_registered(), list_by_domain()

Usage:
    from machina.blocks import registry, SymbolDescriptor, FunctionDescriptor

    descriptor = registry.get('cost.quadratic')(n=1)
    cost_expr  = descriptor(x=x_mx)
"""

from machina.model.descriptor import FunctionDescriptor, SymbolDescriptor

# Import library package to trigger all @register decorators.
from . import (
    library,  # noqa: F401
    registry,
)

__all__ = ['FunctionDescriptor', 'SymbolDescriptor', 'registry']
