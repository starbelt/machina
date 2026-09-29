"""
machina.swapc -- SWaP-C budgets for flyby computing.

The thesis pack: compute, power, link and cost budgets, and the goodput
objective that scores a flyby's delivered data products by their timeliness.

Factories: ``cost.sigmoid_goodput``, ``cost.aggregate_goodput`` and
``cost.loglogistic_goodput`` (:mod:`~machina.swapc.goodput`),
``util.ttp_computation`` (:mod:`~machina.swapc.latency`), and
``util.scene_compute_power``, ``util.solar_array_power`` and
``util.battery_energy`` (:mod:`~machina.swapc.power`).

Signals (:mod:`~machina.swapc.signals`): ``power_load`` (W, SUM) and
``battery_energy`` (J). Components (:mod:`~machina.swapc.components`):
``ComputeLoad``, the orbit-average load of one processor, and ``PowerBudget``,
the solar array and battery that carry the total load through the longest
eclipse. The power budget is SI throughout (W, J, m^2, s), so it did not wait.

Still parked: the cost ``Budget`` and the ``Timeline`` component. A cost
budget needs a currency unit, which the SI allow-list in
:mod:`machina.units` cannot express, and that is a decision first (vault
Decision Log #74).

Importing this package declares its signals, then registers its factories.
``import machina`` never imports it.
"""

from machina.swapc import signals  # noqa: F401  (declares power_load, battery_energy)

# isort: split
from machina.swapc import goodput, latency, power  # noqa: F401
from machina.swapc.components import ComputeLoad, PowerBudget

__all__ = ["goodput", "latency", "power", "signals", "ComputeLoad", "PowerBudget"]
