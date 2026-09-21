"""
machina.library -- generic, domain-free building blocks.

Phase 2 puts the numeric guards here. The factory registry and the generic
cost and constraint factories move in from ``machina.blocks`` in Phase 3,
when the domain factories split out into the packs.
"""

from machina.library.numerics import EPS, TINY, safe_divide, safe_norm, safe_sqrt

__all__ = ["TINY", "EPS", "safe_sqrt", "safe_norm", "safe_divide"]
