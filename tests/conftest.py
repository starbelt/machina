"""Pytest configuration and shared fixtures."""

import importlib.util

import pytest

try:
    import casadi
    HAS_CASADI = True
except ImportError:
    casadi = None
    HAS_CASADI = False

HAS_BONMIN = HAS_CASADI and casadi.has_nlpsol('bonmin')
HAS_NETWORKX = importlib.util.find_spec('networkx') is not None

# marker -> (available, reason when it is not)
_REQUIREMENTS = {
    'requires_casadi':   (HAS_CASADI,   "CasADi not installed"),
    'requires_bonmin':   (HAS_BONMIN,   "bonmin nlpsol plugin not available in this CasADi build"),
    'requires_networkx': (HAS_NETWORKX, "networkx not installed (viz extra)"),
}


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose ``requires_*`` marker names something missing."""
    for item in items:
        for marker, (available, reason) in _REQUIREMENTS.items():
            if not available and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


def pytest_report_header(config):
    """Show which optional solver plugins this run can exercise."""
    if not HAS_CASADI:
        return "casadi: not installed"
    plugins = {name: casadi.has_nlpsol(name)
               for name in ('ipopt', 'bonmin', 'sqpmethod', 'fatrop')}
    return f"casadi {casadi.__version__} nlpsol plugins: {plugins}"


def _import_packs():
    """
    Packs declare signals and register factories on import; they must be in the
    baseline both fixtures restore to. Declaration order is the layout ABI: rigid (2a)
    before astro (3a) before swapc (3b). Append here, never insert; the isort splits
    keep the linter from re-sorting them.
    """
    import machina.library  # noqa: F401  (registers the generic factories)
    import machina.rigid  # noqa: F401  (declares ned/frd and the rigid-body signals)
    # isort: split
    import machina.astro  # noqa: F401  (declares eci/ecef/lvlh, coverage_total; 10 factories)
    # isort: split
    import machina.swapc  # noqa: F401  (registers the goodput and latency factories)


@pytest.fixture(autouse=True)
def _isolated_function_registry():
    """
    The function registry is process-global. Snapshot it around every test so a
    test that registers a factory cannot leak it into the next one. Packs register
    their factories at import time, so they are imported first: a pack imported for
    the first time inside a test would otherwise lose its factories to the restore.
    """
    if not HAS_CASADI:
        yield
        return
    _import_packs()
    from machina.library import registry
    state = registry.snapshot()
    yield
    registry.restore(state)


@pytest.fixture(autouse=True)
def _isolated_signal_registry():
    """
    The default signal registry is process-global too, and packs declare into it
    at import time. Packs are imported here first so their declarations are part
    of the baseline every test restores to: a pack imported for the first time
    inside a test would otherwise have its signals removed by the restore, and
    Python's module cache would stop them ever coming back.
    """
    if not HAS_CASADI:
        yield
        return
    _import_packs()
    from machina.model import signals
    state = signals.DEFAULT.snapshot()
    yield
    signals.DEFAULT.restore(state)


@pytest.fixture(autouse=True)
def _isolated_params():
    """
    The default param registry is process-global, and the active params
    contract is module state. Every test starts from the default contract and
    leaves the registry as it found it.
    """
    if not HAS_CASADI:
        yield
        return
    from machina.params import contract, registry
    state = registry.snapshot()
    contract.reset()
    yield
    contract.reset()
    registry.restore(state)
