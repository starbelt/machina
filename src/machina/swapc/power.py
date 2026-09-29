"""
Power factories: ``util.scene_compute_power`` turns the energy of one processed
scene into an orbit-average load, ``util.solar_array_power`` is the
average power a solar array delivers to the loads across one eclipse cycle
(an orbit in LEO, a solar day at GEO), and ``util.battery_energy`` is the
stored energy that carries a load through that eclipse.

Everything is SI: joules, seconds, watts, square metres; efficiencies and
factors are dimensionless. None of the three takes a factory parameter, so the
inputs stay runtime quantities a Problem can make variables, parameters or
constants.

The registered names carry the ``util.`` prefix: registered names are a stable
ABI. Importing this module registers the three factories;
``import machina.swapc`` imports it.
"""

import casadi as ca

from machina.library.numerics import safe_divide
from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor

__all__ = ["make_scene_compute_power", "make_solar_array_power", "make_battery_energy"]


@register('util.scene_compute_power')
def make_scene_compute_power() -> FunctionDescriptor:
    """
    Orbit-average compute power: P = E_scene / T_refresh.

    The processor spends E_scene joules on each scene and a new scene arrives
    every T_refresh seconds, so the average electrical load is the energy per
    scene over the scene cadence. Idle power between scenes is not in it: fold
    it into E_scene (idle power times T_refresh) or carry it as a separate load.

    Function interface
    ------------------
    Inputs
        E_scene   : (1, 1)  -- energy per processed scene [J], >= 0.
        T_refresh : (1, 1)  -- scene cadence [s], > 0.
    Output
        power : (1, 1)  -- orbit-average load [W].

    Numerics
    --------
    The division is ``safe_divide``: T_refresh is floored at TINY, so a zero
    cadence gives a huge finite power instead of inf or NaN in the gradient.
    """
    E_scene = ca.SX.sym('E_scene')
    T_refresh = ca.SX.sym('T_refresh')
    power = safe_divide(E_scene, T_refresh)
    f = ca.Function('scene_compute_power', [E_scene, T_refresh], [power],
                    ['E_scene', 'T_refresh'], ['power'])
    return FunctionDescriptor(f, description='Compute power = E_scene / T_refresh')


@register('util.solar_array_power')
def make_solar_array_power() -> FunctionDescriptor:
    """
    Cycle-average power a solar array delivers to the loads (SMAD sizing).

        P_bar = S * eta * I_d * L_d * A * T_d / (T_e / X_e + T_d / X_d),
        T_d   = T_cycle - T_e.

    Derivation: the array generates only in daylight, S eta I_d L_d A watts
    for T_d seconds. The loads draw P_bar for the whole cycle: in daylight
    straight from the array through a path of efficiency X_d, in eclipse from
    the battery, which the array recharged in daylight, through a path of
    efficiency X_e. Energy balance over one cycle,

        S eta I_d L_d A T_d = P_bar (T_e / X_e + T_d / X_d),

    solved for P_bar. This is SMAD's array-sizing relation with the eclipse and
    daylight loads equal. eta is the bare-cell efficiency; I_d is SMAD's
    inherent degradation, the array-level losses a cell figure leaves out
    (design and assembly, temperature, shadowing; nominal 0.77); L_d is the
    lifetime degradation to end of life. S is the flux the array actually
    sees: the solar flux times the cosine of the incidence angle, so the
    caller folds any pointing loss into it (``PowerBudget`` passes
    ``solar_flux * cos_incidence``).

    T_cycle is the period over which one eclipse and one daylight pass recur:
    the orbit period in LEO, one solar day (86400 s) at GEO, where the Sun
    sets on the spacecraft once per day.

    Function interface
    ------------------
    Inputs
        area                 : (1, 1)  -- array area A [m^2].
        flux                 : (1, 1)  -- flux on the array S [W/m^2]: 1361 at 1 AU times
                                          the cosine of the incidence angle.
        efficiency           : (1, 1)  -- bare-cell conversion efficiency eta [1].
        inherent_degradation : (1, 1)  -- SMAD inherent degradation I_d [1], nominal 0.77.
        degradation          : (1, 1)  -- lifetime degradation factor L_d [1], end of life.
        T_eclipse            : (1, 1)  -- eclipse duration T_e [s], 0 <= T_e <= T_cycle.
        T_cycle              : (1, 1)  -- eclipse recurrence period [s].
        X_e                  : (1, 1)  -- array-to-load path efficiency in eclipse [1].
        X_d                  : (1, 1)  -- array-to-load path efficiency in daylight [1].
    Output
        power : (1, 1)  -- cycle-average power available to loads [W].

    Numerics
    --------
    T_d is floored at 0: an eclipse as long as the cycle leaves no daylight and
    no generation, rather than a negative one. Each of the three divisions is a
    ``safe_divide``, correct here because T_e, T_d, X_e and X_d are non-negative
    for physical inputs; P_bar is linear in A, so an area-sizing solve is an LP.
    """
    area = ca.SX.sym('area')
    flux = ca.SX.sym('flux')
    efficiency = ca.SX.sym('efficiency')
    inherent_degradation = ca.SX.sym('inherent_degradation')
    degradation = ca.SX.sym('degradation')
    T_eclipse = ca.SX.sym('T_eclipse')
    T_cycle = ca.SX.sym('T_cycle')
    X_e = ca.SX.sym('X_e')
    X_d = ca.SX.sym('X_d')

    T_day = ca.fmax(T_cycle - T_eclipse, 0.0)
    path_time = safe_divide(T_eclipse, X_e) + safe_divide(T_day, X_d)
    power = safe_divide(
        flux * efficiency * inherent_degradation * degradation * area * T_day, path_time)
    f = ca.Function(
        'solar_array_power',
        [area, flux, efficiency, inherent_degradation, degradation, T_eclipse, T_cycle, X_e,
         X_d], [power],
        ['area', 'flux', 'efficiency', 'inherent_degradation', 'degradation', 'T_eclipse',
         'T_cycle', 'X_e', 'X_d'],
        ['power'],
    )
    return FunctionDescriptor(f, description='Cycle-average solar array power (SMAD)')


@register('util.battery_energy')
def make_battery_energy() -> FunctionDescriptor:
    """
    Battery energy that carries a load through an eclipse: E = P T_e / (DoD eta).

    The load P draws P T_e joules during the eclipse. Only the fraction DoD of
    the stored energy may be used (depth of discharge, a cycle-life limit), and
    the battery-to-load path loses a factor eta, so the nameplate energy is
    P T_e / (DoD eta). This is SMAD's C_r = P_e T_e / (DoD N n) with one
    battery (N = 1).

    Function interface
    ------------------
    Inputs
        power              : (1, 1)  -- eclipse load P [W].
        T_eclipse          : (1, 1)  -- eclipse duration T_e [s].
        depth_of_discharge : (1, 1)  -- usable fraction of capacity, (0, 1].
        efficiency         : (1, 1)  -- battery-to-load efficiency eta, (0, 1].
    Output
        energy : (1, 1)  -- battery nameplate energy [J]. Divide by 3600 for Wh,
                            in display code only.

    Numerics
    --------
    ``safe_divide``: the product DoD eta is floored at TINY.
    """
    power = ca.SX.sym('power')
    T_eclipse = ca.SX.sym('T_eclipse')
    depth_of_discharge = ca.SX.sym('depth_of_discharge')
    efficiency = ca.SX.sym('efficiency')
    energy = safe_divide(power * T_eclipse, depth_of_discharge * efficiency)
    f = ca.Function(
        'battery_energy',
        [power, T_eclipse, depth_of_discharge, efficiency], [energy],
        ['power', 'T_eclipse', 'depth_of_discharge', 'efficiency'], ['energy'],
    )
    return FunctionDescriptor(f, description='Battery energy = P T_e / (DoD eta)')
