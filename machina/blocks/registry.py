"""
Function registry for the block library.

Usage:
    from machina.blocks import registry

    # Register a factory (typically done via decorator in library modules):
    @registry.register('cost.my_function')
    def make_my_function(*, param: float) -> FunctionDescriptor:
        ...

    # Look up and call a factory:
    descriptor = registry.get('cost.my_function')(param=1.0)

    # Enumerate the registry:
    registry.list_registered()
    registry.list_by_domain('cost')
"""

from typing import Callable

_registry: dict[str, Callable] = {}


def register(name: str):
    """
    Decorator that registers a factory function under the given name.

    Raises ValueError if the name is already registered (prevents silent
    overwrites from accidental double-import or name collision).
    """
    def decorator(factory: Callable) -> Callable:
        if name in _registry:
            raise ValueError(
                f"Registry: '{name}' is already registered. "
                "Each factory name must be unique."
            )
        _registry[name] = factory
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
