"""
Which modules declare parameters -- as an explicit list, on purpose.

Adopted from icarus-dynamics ``params/modules.py``, with one change: icarus
hardcodes its list, and machina is a library, so the consumer passes its own.

The obvious alternative is to walk a package and import everything. It is
rejected because of what it does when it goes wrong: a module that fails to
import, or that sits where the walker does not look, contributes no
declarations -- and ``sync`` then sees rows with no declaration and offers to
prune them. A typo would quietly propose deleting numbers in use. An explicit
list makes adding a declaring module a one-line diff a reviewer can see.
"""

import importlib

from machina.params.registry import DEFAULT

__all__ = ["load_all", "parse_module_list"]


def load_all(modules, *, registry=None) -> dict:
    """Import every declaring module, then return the registry's declarations.

    Call this before any tool reads the registry. Otherwise the answer is
    whatever happened to be imported already, which is a subtly wrong answer
    rather than an obviously wrong one.
    """
    modules = list(modules)
    if not modules:
        raise ValueError(
            "no declaring modules given. Pass the modules that call param(), e.g. "
            "--modules mypkg.params.power,mypkg.params.timing; an empty list would make every "
            "row in the table look orphaned."
        )
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"declaring module {name!r} failed to import ({exc}). Fix it rather than drop "
                f"it from the list: a missing module makes its rows look orphaned."
            ) from exc
    return (DEFAULT if registry is None else registry).all()


def parse_module_list(text: str) -> list:
    """``"a.b, c.d"`` to ``["a.b", "c.d"]``, order kept, blanks dropped."""
    return [part.strip() for part in text.split(",") if part.strip()]
