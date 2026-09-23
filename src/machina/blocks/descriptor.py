"""Transitional re-export of :mod:`machina.model.descriptor`; Phase 3b deletes ``blocks/``."""

from machina.model.descriptor import (
    _VALID_SEMANTIC_TYPES,  # noqa: F401  -- machina/agents/agent_type.py imports it from here
    FunctionDescriptor,
    SymbolDescriptor,
)

__all__ = ["FunctionDescriptor", "SymbolDescriptor"]
