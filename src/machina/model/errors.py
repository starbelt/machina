"""
Errors raised by the model layer.

They live in their own module so that ``signals`` and ``builder`` can both
raise them without importing each other. Every message names the file or the
declaration to edit; that is the rule the whole layer is written to (adopted
from icarus-dynamics, rule I7 in the vault Decision Log).
"""


class SignalError(Exception):
    """A signal declaration or use that must not reach a built model."""


class ModelError(Exception):
    """A model that must not be built. Every message names what to change."""
