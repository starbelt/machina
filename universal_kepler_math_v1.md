# Universal Variable Kepler Propagation — Math Reference

**Document:** `universal_kepler_math_v1.md`
**Project:** machina-v2 — Flyby Satellite Mission Optimizer
**Phase:** 3b
**Status:** Reference for `transform.stumpff_cs`, `transform.universal_kepler`,
`transform.lagrange_coefficients`, `transform.propagate_universal`
**Last Updated:** 2026-04-07

---

## 1. Motivation: Why Universal Variables?

Classical Kepler propagation splits into three separate formulations depending on orbit type:

| Orbit type | Condition | Equation |
|---|---|---|
| Elliptic | 0 ≤ e < 1 | Kepler's equation: M = E − e sin E |
| Parabolic | e = 1 | Barker's equation: M = D + D³/3 |
| Hyperbolic | e > 1 | Hyperbolic Kepler: M = e sinh H − H |

These are **discontinuous across e = 0 and e = 1**. A gradient-based NLP that allows the optimizer to vary eccentricity will encounter Jacobian discontinuities or singularities when crossing these boundaries, causing IPOPT to fail or converge to incorrect solutions.

**Universal variables** (Battin 1964, Goodyear 1965) provide a single, smooth formulation valid for all orbit types including degenerate cases. The key idea is to introduce a path variable χ (chi) that unifies eccentric anomaly-like quantities across all regimes:

- Elliptic orbit: χ = √a · ΔE where ΔE is the eccentric anomaly change
- Parabolic orbit: χ = D (parabolic anomaly)
- Hyperbolic orbit: χ = √(−a) · ΔH where ΔH is the hyperbolic anomaly change

The unifying ingredient is the **Stumpff functions** C(ψ) and S(ψ), which are analytic continuations of trigonometric/hyperbolic expressions across all real ψ = αχ².

---

## 2. Stumpff Functions

### 2.1 Definition via Taylor Series

The Stumpff functions are defined by absolutely convergent power series valid for all ψ ∈ ℝ:

```
C(ψ) = Σ_{n=0}^∞  (−ψ)ⁿ / (2n+2)!   =   1/2! − ψ/4! + ψ²/6! − ψ³/8! + ...

S(ψ) = Σ_{n=0}^∞  (−ψ)ⁿ / (2n+3)!   =   1/3! − ψ/5! + ψ²/7! − ψ³/9! + ...
```

Numerically: C(ψ) ≈ 1/2 − ψ/24 + ψ²/720 − ψ³/40320 + ...
             S(ψ) ≈ 1/6 − ψ/120 + ψ²/5040 − ψ³/362880 + ...

### 2.2 Closed-Form Expressions by Regime

For computational efficiency away from ψ = 0, closed-form expressions exist:

**ψ > 0 (elliptic regime, α > 0):**
```
C(ψ) = (1 − cos √ψ) / ψ
S(ψ) = (√ψ − sin √ψ) / (√ψ)³
```

**ψ = 0 (parabolic limit, α = 0):**
```
C(0) = 1/2
S(0) = 1/6
```

**ψ < 0 (hyperbolic regime, α < 0):**
```
C(ψ) = (cosh √(−ψ) − 1) / (−ψ)
S(ψ) = (sinh √(−ψ) − √(−ψ)) / (√(−ψ))³
```

The connection to familiar trigonometry: for ψ > 0, let θ = √ψ. Then C(ψ) = (1−cos θ)/θ² = sin²(θ/2)/(θ/2)² · (1/4) — essentially a sinc-like function. For ψ < 0, let φ = √(−ψ). Then C(ψ) = (cosh φ − 1)/φ² — the hyperbolic analogue.

### 2.3 Useful Identities

```
ψ·S(ψ) + C(ψ) = 1/2               (from summing the two series)
2·C(ψ)² = C(ψ/2) + ...            (doubling formula; rarely needed)
dC/dψ = (1/2 − C(ψ)) / ψ          (ψ ≠ 0)
dS/dψ = (C(ψ) − 3·S(ψ)) / (2ψ)   (ψ ≠ 0)
```

The derivative formulas are used in deriving Newton's method convergence for the Kepler equation.

### 2.4 CasADi Implementation Notes

The branch structure uses `ca.if_else`. Since `ca.if_else` evaluates **both branches symbolically**, both must be numerically defined everywhere — including in the non-selected branch. The issue is that the ψ > 0 formula involves `√ψ` (undefined for ψ < 0) and the ψ < 0 formula involves `√(−ψ)` (undefined for ψ > 0). Denominators also go to zero as ψ → 0.

**Fix:** Guard with `ca.fmax(|ψ|, TINY)` in all denominators, where `TINY = 1e-32`. This has no effect on the selected branch but prevents NaN in the symbolic graph.

The Taylor series is used for |ψ| < ε = 1e-4. Four terms are sufficient for double precision (error < 10⁻¹⁶ for |ψ| < 0.1):

```python
C = ca.if_else(ca.fabs(psi) < EPS,
        0.5 - psi/24 + psi**2/720 - psi**3/40320,      # Taylor
        ca.if_else(psi > 0,
            (1 - ca.cos(ca.sqrt(ca.fmax(psi,TINY)))) / ca.fmax(psi,TINY),   # elliptic
            (ca.cosh(ca.sqrt(ca.fmax(-psi,TINY))) - 1) / ca.fmax(-psi,TINY) # hyperbolic
        ))
```

---

## 3. Universal Kepler Equation

### 3.1 Setup

**Given:** Initial Cartesian state (r₀, v₀) in ECI at time t₀, propagation interval Δt, gravitational parameter μ.

**Goal:** Find χ (the universal anomaly) such that the two-body orbital equations are satisfied at t = t₀ + Δt.

**Auxiliary scalar quantities:**
```
r₀    = ‖r₀‖                         # initial radius magnitude (km)
σ₀    = (r₀ · v₀) / √μ              # = r₀ · ṙ₀ / √μ  (radial velocity, km^(1/2))
α     = 2/r₀ − ‖v₀‖²/μ              # reciprocal semi-major axis 1/a (km⁻¹)
```

α classification:
- **α > 0** → elliptic orbit (a = 1/α > 0, bounded)
- **α = 0** → parabolic orbit (escape at exactly v_escape)
- **α < 0** → hyperbolic orbit (a = 1/α < 0, unbounded)

### 3.2 The Equation

Find χ ∈ ℝ such that F(χ) = 0:

```
F(χ) = σ₀ · χ² · C(αχ²)  +  (1 − r₀α) · χ³ · S(αχ²)  +  r₀ · χ  −  √μ · Δt
```

This equation is equivalent to Kepler's equation in each regime when χ is expressed in the regime-specific anomaly.

### 3.3 Newton's Method and the Radius Identity

Newton's method requires F'(χ). Using the identities dC/dψ = (1/2−C)/ψ and dS/dψ = (C−3S)/(2ψ):

```
F'(χ) = r(χ)
```

where r(χ) is the **radius at the propagated point**:
```
r(χ) = σ₀ · χ · [2C(αχ²)] ... (simplification) = F·r₀ + G·‖v₀‖²...
```

This remarkable result means the Newton step is simply:
```
Δχ = −F(χ) / r(χ)
```

Since r(χ) > 0 for any physical orbit, Newton's method is guaranteed to make progress when started near the root. Convergence is typically quadratic.

### 3.4 Initial Guess

For nearly circular orbits (the common case in satellite design):
```
χ₀ = √μ · Δt / r₀
```

This is exact for a circular orbit (χ = v_circular · Δt = √(μ/r₀) · Δt). For moderately eccentric orbits (e ≲ 0.7), it converges within 5–10 iterations. For highly eccentric or hyperbolic orbits, more sophisticated initial guesses exist (Goodyear, Danby) — deferred to Phase 3d if needed.

For backward propagation (Δt < 0), χ₀ is negative, which is correct: χ tracks signed arc length.

### 3.5 CasADi rootfinder Implementation

`ca.rootfinder` wraps an implicit solver (Newton by default) around the residual function and handles differentiation via the **implicit function theorem (IFT)**:

```
dχ/dp = −[∂F/∂χ]⁻¹ · [∂F/∂p]  =  −[r(χ)]⁻¹ · [∂F/∂p]
```

This provides exact first derivatives of the converged χ with respect to all parameters (r₀, v₀, Δt) **without differentiating through the iteration**. IPOPT gets correct Jacobians and Hessians automatically. This is the key advantage of `ca.rootfinder` over a hand-rolled Newton loop.

The residual function has the form `F(chi, p)` where `p = [r₀_mag, σ₀, α, Δt]` (precomputed scalars). The outer function computes p from r₀, v₀, Δt and calls the rootfinder:

```
outer: (r₀, v₀, Δt) → χ
  1. compute r₀_mag, σ₀, α
  2. compute χ₀ = √μ · Δt / r₀_mag
  3. χ = rootfinder(χ₀, [r₀_mag, σ₀, α, Δt])
```

---

## 4. Lagrange Coefficients

### 4.1 The f and g Functions

Given the converged χ, the propagated state is a **linear combination** of r₀ and v₀:

```
r(t) = f · r₀  +  g · v₀          (position vector)
v(t) = ḟ · r₀  +  ġ · v₀         (velocity vector)
```

The scalar coefficients (using ψ = α·χ², C = C(ψ), S = S(ψ)):

```
f    = 1 − (χ²/r₀) · C
g    = Δt − (χ³/√μ) · S
```

To compute ḟ and ġ, we need the final radius r = ‖f·r₀ + g·v₀‖:

```
ḟ   = (√μ / (r · r₀)) · (α·χ³·S − χ)
ġ   = 1 − (χ²/r) · C
```

**Computation order:** f, g → r (vector) → r (scalar) → ḟ, ġ.
There is no circular dependency: ḟ and ġ require r (final), not r₀.

### 4.2 Conservation Identity (Numerical Check)

```
f · ġ  −  ḟ · g  =  1            (exact, for any χ)
```

This identity follows from conservation of angular momentum and provides a cheap sanity check on any numerical propagation result. Deviation from 1 indicates integration error or sign convention issues.

### 4.3 Special Cases

| Δt | χ | f | g | ḟ | ġ |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 0 (via limit) | 1 |
| T/2 (circular) | χ = √a·π | −1 | T/2 | 0 | −1 |
| T (circular) | χ = 2π√a | 1 | 0 | 0 | 1 |

The Δt = 0 case (χ = 0): C(0) = 1/2, S(0) = 1/6, so f = 1 − 0 = 1, g = 0 − 0 = 0; identity propagation. ḟ requires the limit as χ → 0, which is 0.

---

## 5. Full Propagation Algorithm

**Inputs:** r₀ ∈ ℝ³, v₀ ∈ ℝ³, Δt ∈ ℝ, μ (baked in)
**Outputs:** r ∈ ℝ³, v ∈ ℝ³

```
Step 1 — Scalars:
    r₀    ← ‖r₀‖
    σ₀    ← (r₀ · v₀) / √μ
    α     ← 2/r₀ − ‖v₀‖² / μ

Step 2 — Universal anomaly (rootfinder):
    χ₀    ← √μ · Δt / r₀          (initial guess)
    χ     ← Newton solve of F(χ; r₀, σ₀, α, Δt) = 0

Step 3 — Stumpff functions:
    ψ     ← α · χ²
    C, S  ← stumpff_cs(ψ)

Step 4 — Lagrange position coefficients:
    f     ← 1 − χ²/r₀ · C
    g     ← Δt − χ³/√μ · S

Step 5 — Final position:
    r_vec ← f·r₀ + g·v₀
    r     ← ‖r_vec‖

Step 6 — Lagrange velocity coefficients:
    ḟ    ← √μ/(r · r₀) · (α·χ³·S − χ)
    ġ    ← 1 − χ²/r · C

Step 7 — Final velocity:
    v_vec ← ḟ·r₀ + ġ·v₀

return r_vec, v_vec
```

---

## 6. Operational Limits and Singularities

| Condition | Behavior | Mitigation |
|---|---|---|
| r₀ → 0 | σ₀ and α undefined (0/0) | Not physical; r₀ > 0 for any orbit above Earth |
| Δt = 0 | χ = 0; f=1, g=0, ḟ=0, ġ=1 (identity) | Well-defined via Taylor limit; test explicitly |
| α = 0 (parabolic, e=1) | C and S via Taylor; Newton may be slower | Taylor branch handles ψ=0; acceptable convergence |
| Very large Δt (many orbits) | χ large; initial guess error grows | Add orbit-count logic if needed (not needed for Phase 3b) |
| Δt < 0 (backward) | χ < 0; χ₀ < 0 | Well-defined; F is an odd function of χ for σ₀=0 |
| e → ∞ (very hyperbolic) | sinh(√(−ψ)) overflows for large ψ | Restrict to |e| ≲ 5 in practice; not a NLP concern |

---

## 7. References

1. **Battin, R.H. (1964).** *Astronautical Guidance.* McGraw-Hill. — Original universal variable formulation. Chapter 3.

2. **Goodyear, W.H. (1965).** "Completely General Closed-Form Solution for Coordinates and Partial Derivatives of the Two-Body Problem." *The Astronomical Journal*, 70, 189–192. — First complete treatment including partial derivatives (used here via IFT instead).

3. **Bate, R.R., Mueller, D.D., & White, J.E. (1971).** *Fundamentals of Astrodynamics.* Dover Publications. — §4.4 (universal variables), §4.4-1 through §4.4-3. **Recommended first read.** Freely available; worked numerical examples.

4. **Schaub, H. & Junkins, J.L. (2018).** *Analytical Mechanics of Space Systems*, 4th ed. AIAA Education Series. — Also referenced in this project for MEE→ECI formulas.

5. **Curtis, H.D. (2020).** *Orbital Mechanics for Engineering Students*, 4th ed. Elsevier. — §3.7 (Lagrange coefficients), §3.8 (universal variables). **Best source for worked numerical test cases.** Example 3.7 gives a fully worked propagation with numerical values.

6. **Stumpff, K. (1956).** "Neue Formeln und Hilfstafeln zur Ephemeridenrechnung." *Astronomische Nachrichten*, 283, 1. — Original c-function (Stumpff function) definitions.

7. **Danby, J.M.A. (1992).** *Fundamentals of Celestial Mechanics*, 2nd ed. Willmann-Bell. — §6.9–6.11. Includes Laguerre's method for robust initial guesses in high-eccentricity cases.

**Recommended sequence:** Read Bate §4.4 first (intuition + algorithm), then Curtis §3.7–3.8 (worked examples for testing), then Danby §6.9 if robust initial guesses are needed.
