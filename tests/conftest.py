"""Pytest configuration and shared fixtures."""

import pytest

try:
    import casadi  # noqa: F401
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.requires_casadi
    when CasADi is not installed."""
    if HAS_CASADI:
        return
    skip_marker = pytest.mark.skip(reason="CasADi not installed")
    for item in items:
        if "requires_casadi" in item.keywords:
            item.add_marker(skip_marker)
