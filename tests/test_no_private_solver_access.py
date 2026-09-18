"""
Guard for the Phase 1 exit criterion: nothing outside ``src/machina/solver/``
reads the solver backend's internal state.

The April 2026 backend exposed its bookkeeping as private lists and tuple maps
(``_var_map``, ``_lbw``, ...), and the viz tool, the compiler stub and the tests
all reached into them. Those names no longer exist; this test keeps them from
coming back. Use the public accessors instead (``variables()``,
``constraints()``, ``slice_of()``, ``bounds()``, ``is_built``, ...).
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RETIRED = (
    '_var_map', '_param_map', '_lbw', '_ubw', '_w0', '_lbg', '_ubg',
    '_cost_terms', '_constraint_names', '_p_offset',
)
PATTERN = re.compile(r'\.(' + '|'.join(RETIRED) + r')\b')


def _python_files():
    for top in ('src', 'tests', 'examples'):
        for path in sorted((REPO / top).rglob('*.py')):
            if path == Path(__file__).resolve():
                continue
            yield path


def test_retired_backend_privates_are_not_referenced():
    hits = []
    for path in _python_files():
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not hits, "Backend internals referenced; use the public accessors:\n" + "\n".join(hits)


def test_nothing_outside_solver_reaches_into_a_backend():
    """``<something>.solver._x`` / ``self._solver._x`` is a reach into the backend."""
    reach = re.compile(r'(\.solver|\._solver)\._[a-z]')
    hits = []
    for path in _python_files():
        if (REPO / 'src' / 'machina' / 'solver') in path.parents:
            continue
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if reach.search(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not hits, "\n".join(hits)
