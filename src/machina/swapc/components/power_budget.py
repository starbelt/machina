"""
Power budget of a flyby compute spacecraft: compute loads, a solar array sized
for the orbit-average load, and the battery for the longest eclipse.

``ComputeLoad`` turns the energy of one processed scene into the orbit-average
load E_scene / T_refresh and produces it as ``power_load``. ``PowerBudget``
reads ``power_load``, adds its own front-end, bus and communications loads,
and requires the array to cover the total:

    power_margin = P_gen(array_area) - (power_load + front + bus + comm) >= 0,

with P_gen the SMAD orbit-average array power (``util.solar_array_power``). It
produces ``battery_energy`` for the total load over ``eclipse_duration``
(``util.battery_energy``).

Neither component declares a cost. Sizing the array is the problem's choice:
add ``array_area`` as the objective and the margin is active at the optimum,
so the array is the smallest that closes the balance. Its multiplier is the
area one more watt of load costs, and ``result.sensitivity`` gives the same
for any load held as a parameter.

Loads are summed per instance path. Two producers of ``power_load`` in one
scope share the path and add (the signal is ``SUM``). Two ``ComputeLoad`` s
cannot share a scope -- their quantities would both sit at ``E_scene`` -- so
each goes in its own ``Scope``, and ``PowerBudget(loads=...)`` names every
path it pays for.
"""

import casadi as ca

from machina.model import Component, Constraint, Declaration, ModelError, Quantity, Role
from machina.swapc.power import (
    make_battery_energy,
    make_scene_compute_power,
    make_solar_array_power,
)

__all__ = ["ComputeLoad", "PowerBudget"]

# PowerBudget's quantities in util.solar_array_power's argument order
# (area, flux, efficiency, degradation, T_eclipse, T_orbit, X_e, X_d).
_ARRAY_ARGS = ("array_area", "solar_flux", "cell_efficiency", "degradation", "eclipse_duration",
               "orbit_period", "X_e", "X_d")


class ComputeLoad(Component):
    """One processor's orbit-average load: ``power_load = E_scene / T_refresh``.

    ``E_scene`` defaults to a parameter, so a study sweeps it (one frontier
    point per model, E_scene read off an exported artifact) or pins it; it can
    be compiled as a variable too. ``T_refresh`` is the scene cadence and is
    fixed by the observing instrument.
    """

    def declare(self) -> Declaration:
        return Declaration(
            quantities=(
                Quantity(
                    "E_scene", unit="J", doc="Energy the processor spends on one scene",
                    role=Role.FLEXIBLE, default_role=Role.PARAMETER, default=10.0, lb=0.0,
                    provenance="E",
                    source="order of magnitude for an edge accelerator on one CONUS scene; "
                           "replace with a measured row"),
                Quantity(
                    "T_refresh", unit="s", doc="Scene cadence: time between processed scenes",
                    role=Role.FIXED, default=300.0,
                    provenance="D", source="scene cadence: ABI CONUS refresh, 5 min (Mode 6)"),
            ),
            produces=("power_load",),
        )

    def build(self, helpers: dict) -> dict:
        scene_power = make_scene_compute_power()
        E_scene = ca.SX.sym("E_scene")
        T_refresh = ca.SX.sym("T_refresh")
        load = scene_power.function(E_scene, T_refresh)
        return {"g": ca.Function(f"{self.name}_g", [E_scene, T_refresh], [load],
                                 ["E_scene", "T_refresh"], ["power_load"])}


class PowerBudget(Component):
    """Solar array and battery for the total electrical load of one spacecraft.

    Parameters
    ----------
    name:
        Instance suffix. ``None`` gives the default name, ``power_budget``.
    loads:
        References to the ``power_load`` signals this budget pays for, summed.
        The default, ``("power_load",)``, reads the bare name: the loads
        produced in this component's scope, else in the nearest enclosing scope
        that produces one, else zero (the signal is ``SUM``). A load inside a
        child scope is not visible by the bare name -- list it by path, e.g.
        ``loads=("cpu/power_load", "gpu/power_load")``. ``()`` sizes the
        spacecraft for its own front-end, bus and communications loads alone.

    The budget reads ``power_load`` and never produces it: a component that
    reads and produces one path is an algebraic loop, and the builder refuses
    it. Its own three loads are quantities, added to the signal inside
    ``build()``.
    """

    def __init__(self, name: str = None, *, loads=("power_load",)):
        super().__init__(name)
        if isinstance(loads, str) or not isinstance(loads, (tuple, list)):
            raise ModelError(
                f"PowerBudget(loads={loads!r}) must be a tuple of power_load references, e.g. "
                f"loads=('a/power_load', 'b/power_load'); a one-element tuple needs the "
                f"trailing comma.")
        loads = tuple(loads)
        for index, reference in enumerate(loads):
            if not isinstance(reference, str) or \
                    reference.rpartition("/")[2] != "power_load":
                raise ModelError(
                    f"PowerBudget(loads=...): entry {index} is {reference!r}; every entry is a "
                    f"path to a power_load signal, such as 'power_load', 'cpu/power_load' or "
                    f"'/cpu/power_load'.")
            if reference in loads[:index]:
                raise ModelError(
                    f"PowerBudget(loads=...) lists {reference!r} twice; the load would be "
                    f"counted twice. Remove the repeat.")
        self.loads = loads

    def _load_refs(self) -> tuple:
        """``(local_name, reference)`` per load: ``load_0``, ``load_1``, ..."""
        return tuple((f"load_{index}", reference) for index, reference in enumerate(self.loads))

    def declare(self) -> Declaration:
        return Declaration(
            algebraic=self._load_refs(),
            # Order is the decision-vector layout: the one variable first.
            quantities=(
                Quantity(
                    "array_area", unit="m^2", doc="Solar array area",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=1.0, lb=0.0, ub=10.0,
                    provenance="A",
                    source="initial guess; the 10 m^2 bound is a small-spacecraft deployable "
                           "array, widen it with compile(overrides=...)"),
                Quantity(
                    "front_power", unit="W",
                    doc="Payload front-end load: the electronics ahead of the processor",
                    role=Role.FLEXIBLE, default_role=Role.PARAMETER, default=5.0, lb=0.0,
                    provenance="A", source="placeholder until the payload is chosen"),
                Quantity(
                    "bus_power", unit="W",
                    doc="Spacecraft bus load: attitude control, thermal, command and data "
                        "handling, power conditioning",
                    role=Role.FLEXIBLE, default_role=Role.PARAMETER, default=20.0, lb=0.0,
                    provenance="A", source="placeholder for a small-satellite bus"),
                Quantity(
                    "comm_power", unit="W",
                    doc="Communications load: transmitter and receiver",
                    role=Role.FLEXIBLE, default_role=Role.PARAMETER, default=10.0, lb=0.0,
                    provenance="A", source="placeholder until the link budget is closed"),
                Quantity(
                    "solar_flux", unit="W/m^2", doc="Solar flux at the array",
                    role=Role.FIXED, default=1361.0,
                    provenance="D", source="solar constant at 1 AU (total solar irradiance)"),
                Quantity(
                    "cell_efficiency", unit="1", doc="Array conversion efficiency, beginning of "
                                                     "life",
                    role=Role.FIXED, default=0.30,
                    provenance="A", source="triple-junction GaAs cells"),
                Quantity(
                    "degradation", unit="1", doc="Lifetime degradation factor L_d, end of life",
                    role=Role.FIXED, default=0.85,
                    provenance="A", source="end-of-life factor for a multi-year GEO mission"),
                Quantity(
                    "X_e", unit="1",
                    doc="Array-to-load path efficiency in eclipse, through the battery",
                    role=Role.FIXED, default=0.65,
                    provenance="D", source="SMAD, direct energy transfer, eclipse path"),
                Quantity(
                    "X_d", unit="1", doc="Array-to-load path efficiency in daylight",
                    role=Role.FIXED, default=0.85,
                    provenance="D", source="SMAD, direct energy transfer, daylight path"),
                Quantity(
                    "eclipse_duration", unit="s", doc="Eclipse the battery carries the load "
                                                      "through",
                    role=Role.FLEXIBLE, default_role=Role.PARAMETER, default=4320.0, lb=0.0,
                    provenance="P", source="maximum GEO eclipse, 72 min, at equinox"),
                Quantity(
                    "orbit_period", unit="s", doc="Orbit period",
                    role=Role.FIXED, default=86164.0,
                    provenance="P", source="sidereal day: the GEO period"),
                Quantity(
                    "depth_of_discharge", unit="1", doc="Usable fraction of battery capacity",
                    role=Role.FIXED, default=0.8,
                    provenance="A", source="Li-ion at GEO: about 90 eclipses a year"),
                Quantity(
                    "battery_efficiency", unit="1", doc="Battery-to-load efficiency",
                    role=Role.FIXED, default=0.9,
                    provenance="A", source="battery discharge and regulation losses"),
            ),
            produces=("battery_energy",),
            constraints=(
                Constraint(
                    "power_margin", lb=0.0,
                    doc="Orbit-average array power minus the total load [W], >= 0"),
            ),
        )

    def build(self, helpers: dict) -> dict:
        solar_array = make_solar_array_power()
        battery = make_battery_energy()

        # One dense scalar per local name, named as declared: the builder binds by name.
        consumed = self.declare().consumed()
        sym = {name: ca.SX.sym(name) for name in consumed}

        total = sym["front_power"] + sym["bus_power"] + sym["comm_power"]
        for local, _ in self._load_refs():
            total = total + sym[local]
        generated = solar_array.function(*[sym[name] for name in _ARRAY_ARGS])
        energy = battery.function(total, sym["eclipse_duration"], sym["depth_of_discharge"],
                                  sym["battery_efficiency"])

        args = [sym[name] for name in consumed]
        return {
            "g": ca.Function(f"{self.name}_g", args, [energy], list(consumed),
                             ["battery_energy"]),
            "h": ca.Function(f"{self.name}_h", args, [generated - total], list(consumed),
                             ["power_margin"]),
        }
