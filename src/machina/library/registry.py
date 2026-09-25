"""
Function registry: factories by name.

Usage:
    from machina.library import registry

    # Register a factory (typically done via decorator in library modules):
    @registry.register('cost.my_function')
    def make_my_function(*, param: float) -> FunctionDescriptor:
        ...

    # Look up and call a factory:
    descriptor = registry.get('cost.my_function')(param=1.0)

    # Enumerate the registry:
    registry.list_registered()
    registry.list_by_domain('cost')

The registry is process-global. Tests isolate themselves with
``snapshot()`` / ``restore()``; ``clear()`` exists for the same purpose.
Factory modules register at import time -- ``machina.library`` the generic
ones, each pack its own when it is imported -- so clearing does not bring
their factories back on a later import (Python caches modules): restore a
snapshot.
"""

from collections.abc import Callable

_registry: dict[str, Callable] = {}
# name -> module that registered it, so a duplicate error can name both sides.
_registered_in: dict[str, str] = {}


def register(name: str):
    """
    Decorator that registers a factory function under the given name.

    Raises ValueError if the name is already registered (prevents silent
    overwrites from accidental double-import or name collision).
    """
    def decorator(factory: Callable) -> Callable:
        module = getattr(factory, '__module__', '<unknown>')
        if name in _registry:
            raise ValueError(
                f"Registry: '{name}' is already registered (by "
                f"{_registered_in.get(name, '<unknown>')}; now again by {module}). "
                "Each factory name must be unique. If two copies of machina "
                "are on sys.path, remove one."
            )
        _registry[name] = factory
        _registered_in[name] = module
        return factory
    return decorator


def get(name: str) -> Callable:
    """
    Return the factory registered under *name*.

    Raises KeyError listing all available names if *name* is not found.
    """
    if name not in _registry:
        available = ', '.join(sorted(_registry.keys()))
        raise KeyError(
            f"Registry: '{name}' not found. "
            f"Available factories: [{available}]"
        )
    return _registry[name]


def list_registered() -> list[str]:
    """Return a sorted list of all registered factory names."""
    return sorted(_registry.keys())


def list_by_domain(domain: str) -> list[str]:
    """
    Return registered names under a domain prefix.

    Example: list_by_domain('cost') returns all names starting with 'cost.'.
    """
    prefix = domain + '.'
    return sorted(k for k in _registry if k.startswith(prefix))


def registered_in(name: str) -> str:
    """Return the module that registered *name*."""
    get(name)
    return _registered_in[name]


def snapshot() -> tuple[dict, dict]:
    """Return a copy of the registry state, for ``restore()``. Test isolation only."""
    return dict(_registry), dict(_registered_in)


def restore(state: tuple[dict, dict]) -> None:
    """Replace the registry state with a ``snapshot()``. Test isolation only."""
    factories, modules = state
    _registry.clear()
    _registry.update(factories)
    _registered_in.clear()
    _registered_in.update(modules)


def clear() -> None:
    """Empty the registry. Test isolation only; see the module docstring."""
    _registry.clear()
    _registered_in.clear()
