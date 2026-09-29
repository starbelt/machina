"""
machina.study -- design studies over tables of evaluated configurations.

A study fills a table with one row per configuration (from a sweep, a replay
evaluator, or a CSV of operating points) and reads trade-offs off it. Today the
package holds the Pareto layer, :mod:`machina.study.pareto`: non-dominated
sets, ideal/nadir normalization, the exact weighted-sum switch points between
two objectives, and the epsilon-constraint pick; and the evaluator,
:mod:`machina.study.evaluate`, which tabulates a model's signals at given leaf
values without a solve.

``import machina`` never imports this package; import it explicitly
(``tests/test_import_graph.py`` enforces the split). It depends on numpy only.
pandas is optional (the ``study`` extra in pyproject.toml): a DataFrame is
accepted wherever a table is, and is recognised without importing pandas.
"""

from machina.study.evaluate import Evaluator
from machina.study.pareto import (
    Normalized,
    Objective,
    Segment,
    WeightedSweep,
    epsilon_constraint,
    excluded,
    nondominated,
    normalize,
    weighted_sweep,
)

__all__ = [
    "Evaluator",
    "Objective", "nondominated", "excluded", "normalize", "Normalized",
    "weighted_sweep", "WeightedSweep", "Segment", "epsilon_constraint",
]
