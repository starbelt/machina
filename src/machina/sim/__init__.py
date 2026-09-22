"""
machina.sim -- fixed-step integration of the ODE the builder composes.

``Builder.build()`` substitutes algebraic signals out and returns a plain
``f_system(x, u)``; this package integrates it (Decision Log #49).
"""

from machina.sim.integrator import Derivative, build_step_function, rk4_step

__all__ = ["rk4_step", "build_step_function", "Derivative"]
