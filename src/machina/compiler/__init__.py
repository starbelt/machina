"""
machina.compiler — Layer 4 problem compiler.

Public API
----------
Problem      -- declared components in, a built and solvable NLP out
CompilerStub -- minimal Python-driven declare → assign → build → register lifecycle
"""

from .compiler_stub import CompilerStub
from .problem import Problem

__all__ = ['CompilerStub', 'Problem']
