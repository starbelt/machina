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


@pytest.fixture(autouse=True)
def _isolated_function_registry():
    """
    The function registry is process-global. Snapshot it around every test so a
    test that registers a factory cannot leak it into the next one.
    """
    if not HAS_CASADI:
        yield
        return
    from machina.blocks import registry
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
    import machina.rigid  # noqa: F401  (declares ned/frd and the rigid-body signals)
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
