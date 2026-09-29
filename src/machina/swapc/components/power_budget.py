"""
Power budget of a flyby compute spacecraft: compute loads, a solar array sized
for the cycle-average load, and the battery for the longest eclipse.

``ComputeLoad`` turns the energy of one processed scene into the orbit-average
load E_scene / T_refresh and produces it as ``power_load``. ``PowerBudget``
reads ``power_load``, adds its own front-end, bus and communications loads,
and requires the array to cover the total:

    power_margin = P_gen(array_area) - (power_load + front + bus + comm) >= 0,

with P_gen the SMAD cycle-average array power (``util.solar_array_power``),

    P_gen = S cos(theta) eta I_d L_d A T_d / (T_e / X_e + T_d / X_d),
    T_d   = T_cycle - T_e.

It produces ``battery_energy`` for the total load over ``eclipse_duration``
(``util.battery_energy``).

Sizing assumption: the array is sun-normal (``cos_incidence = 1``) through the
longest eclipse, the 72-minute GEO eclipse at equinox, recurring once per solar
day. At equinox the Sun is in the equatorial plane, so a one-axis (north-south)
tracked GEO array is sun-normal then. At solstice the same array sees
cos 23.44 deg = 0.917 and no eclipse; at these defaults that case needs about
2 % more area, so size both (``cos_incidence = 0.917``, ``eclipse_duration =
0``) and take the larger.

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

# PowerBudget's quantities in util.solar_array_power's argument order after the area and the
# flux (efficiency, inherent_degradation, degradation, T_eclipse, T_cycle, X_e, X_d). The flux
# on the array, solar_flux * cos_incidence, is formed in build().
_ARRAY_ARGS = ("cell_efficiency", "inherent_degradation", "degradation", "eclipse_duration",
               "cycle_period", "X_e", "X_d")


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
        **Every load path must be listed**: the budget pays for exactly these
        paths, and a ``power_load`` produced anywhere else in the model is not
        in the total. The default, ``("power_load",)``, reads the bare name:
        the loads produced in this component's scope, else in the nearest
        enclosing scope that produces one, else zero (the signal is ``SUM``).
        A load inside a child scope is not visible by the bare name -- list it
        by path, e.g. ``loads=("cpu/power_load", "gpu/power_load")`` for two
        sibling scopes. A bare name that would read zero while a child scope
        produces ``power_load`` is refused when the model is declared, naming
        those paths; so are two spellings of one path (``"a/power_load"`` and
        ``"/a/power_load"``), which would count the load twice. ``()`` sizes
        the spacecraft for its own front-end, bus and communications loads
        alone.

    The budget reads ``power_load`` and never produces it: a component that
    reads and produces one path is an algebraic loop, and the builder refuses
    it. Its own three loads are quantities, added to the signal inside
    ``build()``.

    The array is sized sun-normal through the equinox eclipse (the module
    docstring's assumption). ``cell_efficiency`` is the bare-cell figure;
    ``inherent_degradation`` carries SMAD's array-level losses and
    ``degradation`` the lifetime loss, and the three multiply.
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

    def check_wiring(self, path: str, resolved: dict, producers: dict) -> None:
        """Refuse a bare load name that reads zero while a child scope produces loads.

        Called by the Builder at ``declare()``. A bare ``power_load`` looks in
        this budget's scope and outward, never into child scopes; when nothing
        there produces it, it reads zero, and a load in a child scope would
        drop out of the total without a word.
        """
        scope = path.rpartition("/")[0]
        prefix = f"{scope}/" if scope else ""
        for local, reference in self._load_refs():
            if "/" in reference or resolved[local] in producers:
                continue
            below = [p for p in producers
                     if p.rpartition("/")[2] == "power_load" and p.startswith(prefix)
                     and "/" in p[len(prefix):]]
            if not below:
                continue
            relative = tuple(p[len(prefix):] for p in below)
            where = f"scope {scope!r}" if scope else "the root scope"
            raise ModelError(
                f"{path}: loads entry {reference!r} is a bare name, which looks in {where} "
                f"and outward, never into child scopes. Nothing there produces "
                f"{resolved[local]!r}, so it would read zero, while power_load is produced "
                f"below at {below} and would be left out of the budget. List every load path: "
                f"PowerBudget(loads={relative!r}) (relative to the budget's scope; "
                f"'/{below[0]}' is the absolute form)."
            )

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
                    "solar_flux", unit="W/m^2", doc="Solar flux at the spacecraft",
                    role=Role.FIXED, default=1361.0,
                    provenance="D", source="solar constant at 1 AU (total solar irradiance)"),
                Quantity(
                    "cos_incidence", unit="1",
                    doc="Cosine of the Sun's incidence angle on the array, multiplying the "
                        "flux; 1 is sun-normal. A one-axis (north-south) tracked GEO array at "
                        "solstice sees cos 23.44 deg = 0.917",
                    role=Role.FIXED, default=1.0,
                    provenance="A",
                    source="sun-normal array: a one-axis tracked GEO array at equinox, when "
                           "the longest eclipse occurs"),
                Quantity(
                    "cell_efficiency", unit="1",
                    doc="Solar cell conversion efficiency, beginning of life: the bare-cell "
                        "figure, before inherent and lifetime degradation",
                    role=Role.FIXED, default=0.30,
                    provenance="A", source="triple-junction GaAs cell efficiency (bare cell)"),
                Quantity(
                    "inherent_degradation", unit="1",
                    doc="Inherent degradation I_d: array-level losses the cell efficiency "
                        "leaves out (design and assembly, temperature, shadowing)",
                    role=Role.FIXED, default=0.77,
                    provenance="D", source="SMAD nominal inherent degradation"),
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
                    "cycle_period", unit="s",
                    doc="Eclipse recurrence period: one eclipse and one daylight pass per "
                        "cycle. One solar day at GEO; the orbit period in LEO",
                    role=Role.FIXED, default=86400.0,
                    provenance="P",
                    source="GEO eclipse recurrence period: one solar day (86400 s), since the "
                           "Sun sets on a GEO spacecraft once per day"),
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
                    doc="Cycle-average array power minus the total load [W], >= 0"),
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
        flux = sym["solar_flux"] * sym["cos_incidence"]
        generated = solar_array.function(sym["array_area"], flux,
                                         *[sym[name] for name in _ARRAY_ARGS])
        energy = battery.function(total, sym["eclipse_duration"], sym["depth_of_discharge"],
                                  sym["battery_efficiency"])

        args = [sym[name] for name in consumed]
        return {
            "g": ca.Function(f"{self.name}_g", args, [energy], list(consumed),
                             ["battery_energy"]),
            "h": ca.Function(f"{self.name}_h", args, [generated - total], list(consumed),
                             ["power_margin"]),
        }
