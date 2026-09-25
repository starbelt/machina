"""
Coverage factory: ``cost.smooth_coverage``, the sigmoid that maps an elevation
angle to a coverage indicator in ``(0, 1)``.

The registered name keeps its ``cost.`` prefix: registered names are a stable
ABI. Importing this module registers the factory; ``import machina.astro``
imports it.
"""

import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register('cost.smooth_coverage')
def make_smooth_coverage() -> FunctionDescriptor:
    """
    Smooth sigmoid coverage indicator: c(e) = 1 / (1 + exp(-k*(e - e_min))).

    Returns ~0 when elevation << min_elevation, ~1 when >> min_elevation.
    Exactly 0.5 when elevation == min_elevation (the soft threshold).

    Unlike cost.sigmoid_goodput (where k and t50 are factory parameters baked
    into the function), all three inputs here are function arguments.  This
    allows the component to pass MX parameter symbols for min_elevation and k,
    which is needed in the SingleSatCoverage component where these quantities
    are runtime parameters, not compile-time constants.

    Function interface
    ------------------
    Inputs
        elevation     : (1,1)  -- elevation angle [rad]
        min_elevation : (1,1)  -- threshold elevation [rad]; coverage = 0.5 here
        k             : (1,1)  -- sigmoid steepness [1/rad]; larger = sharper
                                   transition near min_elevation.
                                   Typical: 20-50 [1/rad] (~2-5 deg transition width)
    Output
        coverage : (1,1)  -- smooth coverage value in (0, 1)

    Sign convention
    ---------------
    Coverage is a value to MAXIMIZE.  Negate when adding to the solver:

        solver.add_cost(-coverage_expr)

    Usage example
    -------------
        smooth_cov = registry.get('cost.smooth_coverage')()

        # With numeric inputs
        e_rad = ca.DM(math.radians(20.0))
        e_min_rad = ca.DM(math.radians(10.0))
        k_val = ca.DM(20.0)   # [1/rad]
        cov = smooth_cov(elevation=e_rad, min_elevation=e_min_rad, k=k_val)

        # With MX symbols (typical component usage)
        elev_sym = ca.MX.sym('elev', 1, 1)
        e_min_param = solver.add_parameter('e_min', 1)
        k_param     = solver.add_parameter('k', 1)
        cov_expr = smooth_cov(elevation=elev_sym,
                              min_elevation=e_min_param,
                              k=k_param)
    """
    elevation     = ca.SX.sym('elevation')
    min_elevation = ca.SX.sym('min_elevation')
    k             = ca.SX.sym('k')

    coverage = 1.0 / (1.0 + ca.exp(-k * (elevation - min_elevation)))

    fn = ca.Function(
        'smooth_coverage',
        [elevation, min_elevation, k], [coverage],
        ['elevation', 'min_elevation', 'k'], ['coverage'],
    )
    return FunctionDescriptor(
        fn,
        description='Sigmoid coverage indicator c(elevation, min_elevation, k)',
    )
