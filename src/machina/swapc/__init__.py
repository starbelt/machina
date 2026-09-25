"""
machina.swapc -- SWaP-C budgets for flyby computing.

The thesis pack: compute, power, link and cost budgets, and the goodput
objective that scores a flyby's delivered data products by their timeliness.

In Phase 3 it holds **factories only**: ``cost.sigmoid_goodput`` and
``cost.aggregate_goodput`` (:mod:`~machina.swapc.goodput`) and
``util.ttp_computation`` (:mod:`~machina.swapc.latency`). The budget signals
and the ``Budget``/``Timeline`` components wait for their first consumer,
because a budget's currency unit needs a decision first (vault Decision
Log #74).

Importing this package registers its factories. ``import machina`` never
imports it.
"""

from machina.swapc import goodput, latency  # noqa: F401

__all__ = ["goodput", "latency"]
