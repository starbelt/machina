"""
The function registry is process-global; tests must not leak registrations.

``conftest.py`` snapshots and restores it around every test. The two
``test_leak_*`` tests below only pass in both orders if that fixture works.
"""

import pytest

import machina.library  # noqa: F401  (registers the generic factories)
from machina.library import registry

pytestmark = pytest.mark.requires_casadi


def _register_probe():
    @registry.register('test.isolation_probe')
    def make_probe():
        return None
    return make_probe


class TestSnapshotRestore:

    def test_restore_removes_later_registrations(self):
        state = registry.snapshot()
        _register_probe()
        assert 'test.isolation_probe' in registry.list_registered()
        registry.restore(state)
        assert 'test.isolation_probe' not in registry.list_registered()

    def test_snapshot_is_a_copy(self):
        state = registry.snapshot()
        _register_probe()
        assert 'test.isolation_probe' not in state[0]

    def test_clear_then_restore(self):
        state = registry.snapshot()
        registry.clear()
        assert registry.list_registered() == []
        registry.restore(state)
        assert 'cost.rosenbrock' in registry.list_registered()

    def test_leak_first(self):
        _register_probe()       # would break test_leak_second without the fixture

    def test_leak_second(self):
        _register_probe()


class TestProvenance:

    def test_registered_in_names_the_module(self):
        assert registry.registered_in('cost.rosenbrock') == 'machina.library.cost'

    def test_duplicate_error_names_both_sources(self):
        _register_probe()
        with pytest.raises(ValueError, match="test_registry_isolation") as err:
            _register_probe()
        assert "sys.path" in str(err.value)
