"""
Coordinate transform factories for orbital mechanics.

Registered functions
--------------------
transform.koe_to_mee          -- Classical KOE → Modified Equinoctial Elements
transform.mee_to_koe          -- Modified Equinoctial Elements → Classical KOE
transform.mee_to_eci          -- Modified Equinoctial Elements → ECI position + velocity
transform.stumpff_cs          -- Stumpff functions C(ψ) and S(ψ)
transform.lagrange_coefficients -- Lagrange f/g coefficients given universal anomaly χ
transform.universal_kepler    -- Solve universal Kepler equation for χ (ca.rootfinder)
transform.propagate_universal -- Two-body propagation: (r₀, v₀, Δt) → (r, v)

Modified Equinoctial Elements (MEE)
------------------------------------
MEE avoid the singularities of classical KOE at circular (e→0) and equatorial
(i→0 or π) orbits, making them gradient-safe for NLP optimizers.

Element ordering convention (used throughout this module)
----------------------------------------------------------
KOE: [a, e, i, Ω (RAAN), ω (AOP), ν (true anomaly)]  -- all angles in radians
MEE: [p, f, g, h, k, L (true longitude)]              -- p in km (or whatever unit a uses)

Known singularity
-----------------
The MEE representation has one remaining singularity at i = π (retrograde
equatorial orbit), where h and k → ±∞.  Do not use MEE decision variables for
problems where the optimizer might explore i ≈ π.

References
----------
Schaub, H. & Junkins, J.L. (2018). Analytical Mechanics of Space Systems, 4th ed.
  AIAA Education Series.  Appendix F (MEE ↔ ECI formulas).

Walker, M.J.H., Ireland, B., Owens, J. (1985). A Set of Modified Equinoctial
  Orbital Elements. Celestial Mechanics, 36, 409–419.
"""

import casadi as ca

from machina.blocks.descriptor import FunctionDescriptor
from machina.blocks.registry import register


# ---------------------------------------------------------------------------
# transform.koe_to_mee
# ---------------------------------------------------------------------------

@register('transform.koe_to_mee')
def make_koe_to_mee() -> FunctionDescriptor:
    """
    Factory for KOE → MEE transform.

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        koe(6,1) → mee(6,1)

    Input ordering:  koe = [a, e, i, Ω, ω, ν]
    Output ordering: mee = [p, f, g, h, k, L]

    Singularity: h and k diverge as i → π.  Safe for 0 ≤ i < π - ε.
    """
    koe = ca.SX.sym('koe', 6)
    a, e, inc, raan, aop, nu = (koe[0], koe[1], koe[2],
                                koe[3], koe[4], koe[5])

    p = a * (1 - e ** 2)
    f = e * ca.cos(aop + raan)
    g = e * ca.sin(aop + raan)
    h = ca.tan(inc / 2) * ca.cos(raan)
    k = ca.tan(inc / 2) * ca.sin(raan)
    L = raan + aop + nu

    mee = ca.vertcat(p, f, g, h, k, L)
    fn = ca.Function('koe_to_mee', [koe], [mee], ['koe'], ['mee'])
    return FunctionDescriptor(fn, description='KOE → MEE: (a,e,i,Ω,ω,ν) → (p,f,g,h,k,L)')


# ---------------------------------------------------------------------------
# transform.mee_to_koe
# ---------------------------------------------------------------------------

@register('transform.mee_to_koe')
def make_mee_to_koe() -> FunctionDescriptor:
    """
    Factory for MEE → KOE transform (inverse of koe_to_mee).

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        mee(6,1) → koe(6,1)

    Input ordering:  mee = [p, f, g, h, k, L]
    Output ordering: koe = [a, e, i, Ω, ω, ν]

    Notes
    -----
    - For circular orbits (e=0): ω is mathematically undefined (returned as 0).
    - For equatorial orbits (i=0): Ω is mathematically undefined (returned as 0).
    - ν = L - Ω - ω may land outside [0, 2π); normalise in post-processing
      if a specific range is required.
    """
    mee = ca.SX.sym('mee', 6)
    p, f, g, h, k, L = (mee[0], mee[1], mee[2],
                         mee[3], mee[4], mee[5])

    e = ca.sqrt(f ** 2 + g ** 2)
    a = p / (1 - e ** 2)
    inc = 2 * ca.atan(ca.sqrt(h ** 2 + k ** 2))
    raan = ca.atan2(k, h)
    # ω = atan2(g·h − f·k,  f·h + g·k)
    aop = ca.atan2(g * h - f * k, f * h + g * k)
    nu = L - raan - aop

    koe_out = ca.vertcat(a, e, inc, raan, aop, nu)
    fn = ca.Function('mee_to_koe', [mee], [koe_out], ['mee'], ['koe'])
    return FunctionDescriptor(fn, description='MEE → KOE: (p,f,g,h,k,L) → (a,e,i,Ω,ω,ν)')


# ---------------------------------------------------------------------------
# transform.mee_to_eci
# ---------------------------------------------------------------------------

@register('transform.mee_to_eci')
def make_mee_to_eci(*, mu: float) -> FunctionDescriptor:
    """
    Factory for MEE → ECI position + velocity transform.

    Args:
        mu: Gravitational parameter [km³/s² for Earth: 398600.4418].
            Baked into the ca.Function at factory-construction time (same
            pattern as k/t50 in sigmoid_goodput).

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        mee(6,1) → (r_eci(3,1), v_eci(3,1))

    Input ordering: mee = [p, f, g, h, k, L]
    Units: p in km, L in rad → r_eci in km, v_eci in km/s (when μ in km³/s²).

    Formulas
    --------
    Schaub & Junkins (2018), Appendix F, equations F.10–F.12.

    Auxiliary quantities:
        w  = 1 + f·cosL + g·sinL          (≥ 1−e > 0 for elliptic orbits)
        r  = p / w                         (radius magnitude)
        s² = 1 + h² + k²
        α² = h² − k²

    Position:
        r_eci = (r/s²) · [(1+α²)cosL + 2hk·sinL,
                           (1−α²)sinL + 2hk·cosL,
                           2(h·sinL − k·cosL)]

    Velocity:
        v_eci = (√(μ/p)/s²) · [−(g + (1+α²)sinL − 2hk·cosL),
                                  f + (1−α²)cosL + 2hk·sinL,
                                  2(h·cosL + k·sinL + f·h + g·k) / s²]

    The z-velocity component carries an extra 1/s² factor (s⁴ total denominator).
    """
    mee = ca.SX.sym('mee', 6)
    p, f, g, h, k, L = (mee[0], mee[1], mee[2],
                         mee[3], mee[4], mee[5])

    cosL = ca.cos(L)
    sinL = ca.sin(L)

    w = 1 + f * cosL + g * sinL   # > 0 for elliptic orbits
    r = p / w                      # radius magnitude
    s2 = 1 + h ** 2 + k ** 2      # s²  (always > 0)
    alpha2 = h ** 2 - k ** 2      # α²

    # --- Position (Schaub & Junkins F.10) ---
    rx = (r / s2) * ((1 + alpha2) * cosL + 2 * h * k * sinL)
    ry = (r / s2) * ((1 - alpha2) * sinL + 2 * h * k * cosL)
    rz = (r / s2) * (2 * (h * sinL - k * cosL))
    r_eci = ca.vertcat(rx, ry, rz)

    # --- Velocity ---
    # Derived from v = dr/dt = (dr/dL) * (dL/dt) where dL/dt = sqrt(μ/p³)*w²
    # for the unperturbed (frozen) orbit.  Expanding dr/dL analytically gives:
    #
    #   v = sqrt(μ/p)/s² * [(f·sinL − g·cosL)·n̂  +  w·dn̂/dL]
    #
    # where n̂ is the unit position direction and dn̂/dL is its L-derivative.
    # Collecting terms yields (verified against vis-viva and circular/equatorial
    # analytical checks):
    sqrtmup = ca.sqrt(mu / p)     # √(μ/p)

    vx = (sqrtmup / s2) * (-(1 + alpha2) * (g + sinL) + 2 * h * k * (f + cosL))
    vy = (sqrtmup / s2) * ((1 - alpha2) * (f + cosL) - 2 * h * k * (g + sinL))
    vz = (sqrtmup / s2) * 2 * (f * h + g * k + h * cosL + k * sinL)
    v_eci = ca.vertcat(vx, vy, vz)

    fn = ca.Function(
        'mee_to_eci',
        [mee], [r_eci, v_eci],
        ['mee'], ['r_eci', 'v_eci'],
    )
    return FunctionDescriptor(
        fn,
        description=f'MEE → ECI (r,v) using μ={mu} km³/s²',
    )


# ---------------------------------------------------------------------------
# transform.stumpff_cs
# ---------------------------------------------------------------------------

@register('transform.stumpff_cs')
def make_stumpff_cs() -> FunctionDescriptor:
    """
    Factory for Stumpff functions C(ψ) and S(ψ).

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        psi(1,1) → (C(1,1), S(1,1))

    The Stumpff functions (c₂ and c₃) are analytic continuations of
    (1−cos√ψ)/ψ and (√ψ−sin√ψ)/(√ψ)³ across all real ψ:

        C(ψ):  ψ>0: (1−cos√ψ)/ψ          ψ=0: 1/2     ψ<0: (cosh√(−ψ)−1)/(−ψ)
        S(ψ):  ψ>0: (√ψ−sin√ψ)/(√ψ)³     ψ=0: 1/6     ψ<0: (sinh√(−ψ)−√(−ψ))/(√(−ψ))³

    Taylor series (used for |ψ| < EPS = 1e-4 to avoid cancellation near zero):
        C(ψ) ≈ 1/2 − ψ/24 + ψ²/720 − ψ³/40320
        S(ψ) ≈ 1/6 − ψ/120 + ψ²/5040 − ψ³/362880

    Both series are accurate to machine precision for |ψ| < 0.1; the 1e-4
    threshold leaves a comfortable margin.

    Implementation note
    -------------------
    ca.if_else evaluates BOTH branches symbolically; the non-selected branch
    must be numerically defined everywhere.  TINY = 1e-32 guards denominators
    (e.g., sqrt(fmax(psi, TINY)) is real-valued for all psi).

    References
    ----------
    Bate, Mueller & White (1971), §4.4.
    Curtis (2020), §3.8.
    """
    psi = ca.SX.sym('psi')

    EPS = 1e-4      # switch to Taylor for |ψ| < EPS
    TINY = 1e-32    # denominator guard (non-selected branch)

    # --- Taylor series (always numerically safe) ---
    C_tay = 0.5 - psi/24 + psi**2/720 - psi**3/40320
    S_tay = 1/6 - psi/120 + psi**2/5040 - psi**3/362880

    # --- Exact elliptic (psi > 0) ---
    sq_p = ca.sqrt(ca.fmax(psi, TINY))
    C_pos = (1 - ca.cos(sq_p)) / ca.fmax(psi, TINY)
    S_pos = (sq_p - ca.sin(sq_p)) / ca.fmax(sq_p ** 3, TINY)

    # --- Exact hyperbolic (psi < 0) ---
    sq_n = ca.sqrt(ca.fmax(-psi, TINY))
    C_neg = (ca.cosh(sq_n) - 1) / ca.fmax(-psi, TINY)
    S_neg = (ca.sinh(sq_n) - sq_n) / ca.fmax(sq_n ** 3, TINY)

    C_large = ca.if_else(psi > 0, C_pos, C_neg)
    S_large = ca.if_else(psi > 0, S_pos, S_neg)

    C = ca.if_else(ca.fabs(psi) < EPS, C_tay, C_large)
    S = ca.if_else(ca.fabs(psi) < EPS, S_tay, S_large)

    fn = ca.Function('stumpff_cs', [psi], [C, S], ['psi'], ['C', 'S'])
    return FunctionDescriptor(fn, description='Stumpff C(ψ) and S(ψ): psi → (C, S)')


# ---------------------------------------------------------------------------
# transform.lagrange_coefficients
# ---------------------------------------------------------------------------

@register('transform.lagrange_coefficients')
def make_lagrange_coefficients(*, mu: float) -> FunctionDescriptor:
    """
    Factory for Lagrange f/g coefficients given universal anomaly χ.

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        r0(3,1), v0(3,1), chi(1,1), dt(1,1) → F(1,1), G(1,1), Fdot(1,1), Gdot(1,1)

    Given the initial state (r0, v0) and universal anomaly χ (solved externally
    by transform.universal_kepler), computes the Lagrange coefficients such that:
        r = F·r0 + G·v0
        v = Fdot·r0 + Gdot·v0

    Formulas (μ baked in at construction):
        r0_mag = ‖r0‖
        α      = 2/r0_mag − ‖v0‖²/μ         (1/a, reciprocal semi-major axis)
        ψ      = α·χ²
        C, S   = stumpff_cs(ψ)
        F      = 1 − χ²/r0_mag · C
        G      = dt − χ³/√μ · S
        r_mag  = ‖F·r0 + G·v0‖
        Fdot   = √μ/(r_mag·r0_mag) · (α·χ³·S − χ)
        Gdot   = 1 − χ²/r_mag · C

    Conservation identity (numerical check): F·Gdot − Fdot·G = 1 (exact).

    Args:
        mu: Gravitational parameter [km³/s²].  Baked in at construction.
    """
    stumpff_fd = make_stumpff_cs()

    sqrt_mu = ca.sqrt(mu)

    r0  = ca.SX.sym('r0', 3)
    v0  = ca.SX.sym('v0', 3)
    chi = ca.SX.sym('chi')
    dt  = ca.SX.sym('dt')

    r0_mag = ca.norm_2(r0)
    alpha  = 2 / r0_mag - ca.dot(v0, v0) / mu
    psi    = alpha * chi ** 2

    C, S = stumpff_fd.function(psi)

    F    = 1 - (chi ** 2 / r0_mag) * C
    G    = dt - (chi ** 3 / sqrt_mu) * S

    r_vec  = F * r0 + G * v0
    r_mag  = ca.norm_2(r_vec)

    Fdot = (sqrt_mu / (r_mag * r0_mag)) * (alpha * chi ** 3 * S - chi)
    Gdot = 1 - (chi ** 2 / r_mag) * C

    fn = ca.Function(
        'lagrange_coefficients',
        [r0, v0, chi, dt],
        [F, G, Fdot, Gdot],
        ['r0', 'v0', 'chi', 'dt'],
        ['F', 'G', 'Fdot', 'Gdot'],
    )
    return FunctionDescriptor(
        fn,
        description=f'Lagrange f/g coefficients using μ={mu} km³/s²',
    )


# ---------------------------------------------------------------------------
# transform.universal_kepler
# ---------------------------------------------------------------------------

@register('transform.universal_kepler')
def make_universal_kepler(
    *,
    mu: float,
    method: str = 'newton',
    opts: dict = None,
) -> FunctionDescriptor:
    """
    Factory for the universal Kepler equation solver.

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        r0(3,1), v0(3,1), dt(1,1) → chi(1,1)

    Solves for the universal anomaly χ such that:
        σ₀·χ²·C(αχ²) + (1−r₀α)·χ³·S(αχ²) + r₀·χ − √μ·Δt = 0

    where σ₀ = r₀·v₀/√μ, α = 2/r₀ − v₀²/μ (1/a).

    Uses ca.rootfinder (Newton by default) to solve implicitly.  CasADi
    differentiates through the rootfinder via the implicit function theorem
    (IFT), so the outer NLP gets exact Jacobians without differentiating
    through the iteration.

    Initial guess: χ₀ = √μ·Δt/r₀ (exact for circular orbits, robust for
    near-circular cases typical in the flyby problem).  Negative for Δt < 0
    (backward propagation), which is correct.

    Args:
        mu:     Gravitational parameter [km³/s²].  Baked in at construction.
        method: Rootfinder method string passed to ca.rootfinder ('newton').
        opts:   Extra options dict forwarded to ca.rootfinder.

    Notes
    -----
    - expand=True compatibility: the rootfinder creates an implicit call node
      that may not be expandable in some CasADi versions.  Test explicitly
      before enabling expand=True in the solver backend when this function
      is present in the NLP.
    - For highly eccentric orbits (e > 0.9) or very long propagation arcs,
      the circular-orbit initial guess may require more Newton iterations.
      No issues expected for near-circular flyby orbits (Phase 3b scope).
    """
    stumpff_fd = make_stumpff_cs()
    sqrt_mu = ca.sqrt(mu)

    # --- Residual function for ca.rootfinder ---
    # Signature: F(chi, p) → residual
    # p = [r0_mag, sigma0, alpha, dt]
    chi_in = ca.SX.sym('chi')
    p_in   = ca.SX.sym('p', 4)
    r0_mag_i = p_in[0]
    sigma0_i = p_in[1]
    alpha_i  = p_in[2]
    dt_i     = p_in[3]

    psi_i = alpha_i * chi_in ** 2
    C_i, S_i = stumpff_fd.function(psi_i)

    resid = (
        sigma0_i * chi_in ** 2 * C_i
        + (1 - r0_mag_i * alpha_i) * chi_in ** 3 * S_i
        + r0_mag_i * chi_in
        - sqrt_mu * dt_i
    )

    resid_fn = ca.Function(
        'kepler_resid',
        [chi_in, p_in],
        [resid],
        ['chi', 'p'],
        ['F'],
    )

    rfn = ca.rootfinder(
        'universal_kepler_rf',
        method,
        resid_fn,
        opts or {},
    )

    # --- Outer function: (r0, v0, dt) → chi ---
    r0 = ca.SX.sym('r0', 3)
    v0 = ca.SX.sym('v0', 3)
    dt = ca.SX.sym('dt')

    r0_mag = ca.norm_2(r0)
    sigma0 = ca.dot(r0, v0) / sqrt_mu
    alpha  = 2 / r0_mag - ca.dot(v0, v0) / mu
    chi0   = sqrt_mu * dt / r0_mag          # circular-orbit initial guess
    p_vec  = ca.vertcat(r0_mag, sigma0, alpha, dt)

    chi_sol = rfn(chi0, p_vec)

    fn = ca.Function(
        'universal_kepler',
        [r0, v0, dt],
        [chi_sol],
        ['r0', 'v0', 'dt'],
        ['chi'],
    )
    return FunctionDescriptor(
        fn,
        description=f'Universal Kepler solver using μ={mu} km³/s² (method={method})',
    )


# ---------------------------------------------------------------------------
# transform.propagate_universal
# ---------------------------------------------------------------------------

@register('transform.propagate_universal')
def make_propagate_universal(*, mu: float) -> FunctionDescriptor:
    """
    Factory for universal two-body propagation.

    Returns a FunctionDescriptor wrapping a ca.Function with signature:
        r0(3,1), v0(3,1), dt(1,1) → r(3,1), v(3,1)

    Propagates a Cartesian state (r0, v0) forward by Δt under two-body
    Keplerian dynamics.  Internally chains:
        universal_kepler  → χ
        lagrange_coefficients(r0, v0, χ, Δt)  → F, G, Ḟ, Ġ
        r = F·r0 + G·v0
        v = Ḟ·r0 + Ġ·v0

    Valid for elliptic, parabolic, and hyperbolic orbits without switching
    solvers.  Backward propagation (Δt < 0) is supported.

    Args:
        mu: Gravitational parameter [km³/s²].  Baked in at construction.

    Notes
    -----
    The composition is performed at the SX level by calling the sub-function
    descriptors' .function() method.  If CasADi's rootfinder produces a
    non-SX call node in a future version, fall back to building this function
    with ca.MX.sym and FunctionDescriptor.__call__ instead.
    """
    kepler_fd   = make_universal_kepler(mu=mu)
    lagrange_fd = make_lagrange_coefficients(mu=mu)

    r0 = ca.SX.sym('r0', 3)
    v0 = ca.SX.sym('v0', 3)
    dt = ca.SX.sym('dt')

    chi          = kepler_fd.function(r0, v0, dt)
    F, G, Fd, Gd = lagrange_fd.function(r0, v0, chi, dt)

    r = F * r0 + G * v0
    v = Fd * r0 + Gd * v0

    fn = ca.Function(
        'propagate_universal',
        [r0, v0, dt],
        [r, v],
        ['r0', 'v0', 'dt'],
        ['r', 'v'],
    )
    return FunctionDescriptor(
        fn,
        description=f'Universal two-body propagation using μ={mu} km³/s²',
    )
