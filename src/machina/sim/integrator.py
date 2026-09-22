"""
Fixed-step RK4. Deliberately the least interesting file in the package.

Adopted verbatim in substance from icarus-dynamics ``sim/integrator.py``
(Decision Log #49: ODE-generic, so it lives in ``machina.sim``).

FIXED STEP IS A REQUIREMENT, NOT A SIMPLIFICATION. A replay chain rests on a
scenario producing byte-identical results run to run. An adaptive integrator
chooses its step from a local error estimate, which makes the trajectory a
function of floating-point comparisons -- reproducible on one machine and one
build, which is exactly the guarantee not wanted. Anything that needs
adaptive stepping is analysis and belongs behind a different entry point.

The builder substitutes algebraic signals out, so what arrives here is a plain
ODE ``f(x, u) -> xdot`` and this file has nothing to know about the model.
"""

from collections.abc import Callable

import casadi as ca
import numpy as np

__all__ = ["rk4_step", "build_step_function", "Derivative"]

Derivative = Callable[[np.ndarray, np.ndarray], np.ndarray]


def rk4_step(f: Derivative, x, u, dt: float):
    """One classical Runge-Kutta step. ``u`` is held constant across the step.

    Zero-order hold on ``u`` is the honest model of a digital controller: it
    computes one command per tick and the actuators hold it until the next.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_step_function(f: ca.Function, *, normalise: slice = None,
                        name: str = "plant_step") -> ca.Function:
    """The same RK4, as a symbolic ``step(x, u, dt) -> x_next``.

    Exporting the stepper rather than the derivative puts the integration
    scheme inside the artifact: a consumer calls one function and cannot
    diverge from it. (icarus learned this when a harness re-implemented RK4
    and silently omitted the quaternion renormalisation.)

    ``normalise`` is the state slice holding a unit quaternion. RK4 works in
    the embedding space and does not preserve the norm; the constraint is
    reapplied once per step, outside the stages.

    ``f`` must have the two-input ``(x, u)`` signature. A model with free
    quantities builds ``f_system(x, u, q)``; fix them first with
    ``Builder.build(fixed=...)``, because a step function whose physics
    depends on an unbound decision variable is not an artifact anyone can run.
    """
    if f.n_in() != 2:
        raise ValueError(
            f"build_step_function needs f(x, u); {f.name()} takes {f.n_in()} inputs "
            f"{[f.name_in(i) for i in range(f.n_in())]}. If the third is the quantity "
            f"vector q, bind it with Builder.build(fixed={{...}}) before exporting."
        )
    if f.size2_in(0) != 1 or f.size2_in(1) != 1:
        raise ValueError(f"{f.name()}: x and u must be column vectors.")
    if normalise is not None:
        if normalise.step not in (None, 1) or normalise.start is None or normalise.stop is None:
            raise ValueError("normalise must be a contiguous slice with explicit start and stop.")
        if not 0 <= normalise.start < normalise.stop <= f.size1_in(0):
            raise ValueError(
                f"normalise {normalise} is outside the state vector of length {f.size1_in(0)}.")

    x = ca.SX.sym("x", f.size1_in(0))
    u = ca.SX.sym("u", f.size1_in(1))
    dt = ca.SX.sym("dt")

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if normalise is not None:
        q = x_next[normalise.start:normalise.stop]
        x_next = ca.vertcat(
            x_next[0:normalise.start],
            q / ca.sqrt(ca.dot(q, q)),
            x_next[normalise.stop:],
        )

    return ca.Function(name, [x, u, dt], [x_next], ["x", "u", "dt"], ["x_next"])
