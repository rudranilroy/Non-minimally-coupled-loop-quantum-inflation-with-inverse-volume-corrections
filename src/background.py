"""
background.py
=============
Solves the FRW background equations for non-minimally coupled scalar field
inflation with loop-quantum inverse volume (IV) corrections.

Units: M_Pl = 1  (reduced Planck mass).

Background action:
    S ⊃ ∫d⁴x √(-g) [ F(φ)/2 · R  -  D(a)/2 · (∂φ)²  -  V(φ) ]
  F(φ) = 1 + ξ φ²  (NMC)
  D(a) = 1 - α (a_*/a)^σ  (IV correction)

Equations of motion (independent variable: N = ln a):
------------------------------------------------------
  φ'' + [3 + (H'/H)] φ' + (V'/H²)/D - ξ R φ / (H² D) = 0
  where  R = 6 H²(2 + H'/H)

These lead (after eliminating the mixed second-derivative terms) to:

  ψ' = (B̃ F + 3 ξ φ Ã) / M  −  (H'/H) ψ
  H'/H = (Ã D − 2 ξ φ B̃) / (2M)

  Ã = −(D + 2ξ) ψ² + 2 ξ φ ψ
  B̃ = −(3D + D') ψ + 12 ξ φ  −  V'/(H²)
  M  = D F + 6 ξ² φ²
  D' = dD/dN = α σ (a_*/a)^σ

  H² = V / (3F − D ψ²/2 + 6 ξ φ ψ)     [Friedmann constraint]

with  ψ = dφ/dN.

The end of inflation is defined by  ε₁ = −H'/H = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .model import F, dF_dphi, D_IV, dD_dN, potential


# ---------------------------------------------------------------------------
# Slow-roll initial conditions
# ---------------------------------------------------------------------------

def slow_roll_psi(phi, H, xi, D_val, V, Vp):
    """
    Slow-roll approximation for  ψ₀ = (dφ/dN)_SR.

    From  (3D + D') ψ ≈ −V'/(H²) + 12 ξ φ  (keeping D' ≈ 0 in early SR):
    Also including NMC coupling to Ricci scalar (at SR level: R ≈ 12 H²):
      ψ_SR ≈ (12 ξ φ − V'/(H²)) / (3 D + 12 ξ)
    """
    denom = 3.0 * D_val + 12.0 * xi
    if abs(denom) < 1e-30:
        # fallback to minimal-coupling SR
        return -Vp / (3.0 * H ** 2)
    return (12.0 * xi * phi - Vp / H ** 2) / denom


def friedmann_H(phi, psi, a, xi, pot_params, iv_params):
    """
    Compute H from the Friedmann constraint given φ and ψ = dφ/dN.

    Returns H (positive root).
    """
    V, _ = potential(phi, pot_params)
    alpha = iv_params['alpha']
    sigma = iv_params['sigma']
    a_star = iv_params['a_star']

    D = float(D_IV(a, alpha, sigma, a_star))
    Fval = F(phi, xi)

    denom = 3.0 * Fval - 0.5 * D * psi ** 2 + 6.0 * xi * phi * psi
    if denom <= 0.0:
        raise ValueError(
            f"Friedmann denominator ≤ 0: denom={denom:.3e}, "
            f"φ={phi:.3f}, ψ={psi:.3f}"
        )
    return np.sqrt(V / denom)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def _rhs(N, y, xi, pot_params, iv_params):
    """
    ODE system:  dy/dN = f(N, y)
    y = [φ, ψ, ln H]
    """
    phi, psi, lnH = y
    H = np.exp(lnH)
    a = np.exp(N)

    alpha = iv_params['alpha']
    sigma = iv_params['sigma']
    a_star = iv_params['a_star']

    V, Vp = potential(phi, pot_params)
    Fval = F(phi, xi)
    D = float(D_IV(a, alpha, sigma, a_star))
    Dp = float(dD_dN(a, alpha, sigma, a_star))   # dD/dN

    H2 = H ** 2

    # Tilde quantities (scaled by H²)
    A_tilde = -(D + 2.0 * xi) * psi ** 2 + 2.0 * xi * phi * psi
    B_tilde = -(3.0 * D + Dp) * psi + 12.0 * xi * phi - Vp / H2
    M = D * Fval + 6.0 * xi ** 2 * phi ** 2

    if M < 1e-30:
        M = 1e-30   # avoid division by zero

    HpH = (A_tilde * D - 2.0 * xi * phi * B_tilde) / (2.0 * M)
    psip = (B_tilde * Fval + 3.0 * xi * phi * A_tilde) / M - HpH * psi

    return [psi, psip, HpH]


# ---------------------------------------------------------------------------
# End-of-inflation detector
# ---------------------------------------------------------------------------

def _end_of_inflation(N, y, xi, pot_params, iv_params):
    """Event: ε₁ = −H'/H = 1  (end of inflation)."""
    phi, psi, lnH = y
    H = np.exp(lnH)
    a = np.exp(N)

    alpha = iv_params['alpha']
    sigma = iv_params['sigma']
    a_star = iv_params['a_star']

    V, Vp = potential(phi, pot_params)
    Fval = F(phi, xi)
    D = float(D_IV(a, alpha, sigma, a_star))
    Dp = float(dD_dN(a, alpha, sigma, a_star))

    H2 = H ** 2
    A_tilde = -(D + 2.0 * xi) * psi ** 2 + 2.0 * xi * phi * psi
    B_tilde = -(3.0 * D + Dp) * psi + 12.0 * xi * phi - Vp / H2
    M = D * Fval + 6.0 * xi ** 2 * phi ** 2
    if M < 1e-30:
        M = 1e-30
    HpH = (A_tilde * D - 2.0 * xi * phi * B_tilde) / (2.0 * M)
    epsilon1 = -HpH
    return epsilon1 - 1.0   # zero when ε₁ = 1


_end_of_inflation.terminal = True
_end_of_inflation.direction = 1.0   # only trigger when crossing upward


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def max_phi0_for_xi(xi, safety=0.90):
    """
    For non-minimal coupling ξ < 0, the effective Planck mass
    F(φ) = 1 + ξ φ² changes sign at φ = 1/√|ξ|.
    Return the maximum physically allowed initial field value.
    For ξ ≥ 0 there is no upper bound (returns np.inf).
    """
    if xi >= 0:
        return np.inf
    return safety / np.sqrt(-xi)


def solve_background(phi0, xi, pot_params, iv_params,
                     N_max=120.0, rtol=1e-8, atol=1e-10):
    """
    Solve the background inflationary evolution.

    Parameters
    ----------
    phi0 : float
        Initial field value (in Planck units).  φ > 0 assumed; slow-roll
        initial conditions are set automatically.
    xi : float
        Non-minimal coupling constant.
    pot_params : dict
        Potential parameters (see model.py).
    iv_params : dict
        IV correction parameters: {'alpha', 'sigma', 'a_star'}.
    N_max : float
        Maximum number of e-folds to integrate.
    rtol, atol : float
        ODE solver tolerances.

    Returns
    -------
    sol : ODE solution object with attributes:
        .t   → N (e-folds array, starting from 0)
        .y   → [φ(N), ψ(N), ln H(N)]
    Also computes derived quantities:
        .phi, .psi, .H, .a, .V, .Vp
        .epsilon1, .epsilon2  (slow-roll parameters)
        .N_total              (total e-folds of inflation)
    """
    a0 = 1e-3   # initial scale factor (in IV correction units, starts small)
    N0 = np.log(a0)

    alpha = iv_params['alpha']
    sigma = iv_params['sigma']
    a_star = iv_params['a_star']

    V0, Vp0 = potential(phi0, pot_params)
    D0 = float(D_IV(a0, alpha, sigma, a_star))
    Fval0 = F(phi0, xi)

    # First, compute H₀ in slow-roll (ψ ≈ 0) as seed
    denom0 = 3.0 * Fval0
    if denom0 <= 0.0:
        raise ValueError(
            f"Cannot start integration: F(φ₀) ≤ 0 with ξ={xi}, φ₀={phi0}"
        )
    H0_sr = np.sqrt(V0 / denom0)

    # Slow-roll ψ₀
    psi0 = slow_roll_psi(phi0, H0_sr, xi, D0, V0, Vp0)

    # Recompute H₀ with the SR ψ₀
    H0 = friedmann_H(phi0, psi0, a0, xi, pot_params, iv_params)
    lnH0 = np.log(H0)

    y0 = [phi0, psi0, lnH0]

    # Integrate from N0 to N0 + N_max
    sol = solve_ivp(
        _rhs,
        [N0, N0 + N_max],
        y0,
        args=(xi, pot_params, iv_params),
        events=_end_of_inflation,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=0.05,
    )

    # Shift N so that inflation ends at N=0 (count from start of integration)
    N_arr = sol.t - N0

    phi_arr = sol.y[0]
    psi_arr = sol.y[1]
    H_arr = np.exp(sol.y[2])
    a_arr = np.exp(sol.t)

    # Compute ε₁ = −d(lnH)/dN
    eps1 = np.zeros_like(N_arr)
    for i in range(len(N_arr)):
        a_i = a_arr[i]
        phi_i = phi_arr[i]
        psi_i = psi_arr[i]
        lnH_i = sol.y[2, i]
        H_i = H_arr[i]

        V_i, Vp_i = potential(phi_i, pot_params)
        Fi = F(phi_i, xi)
        Di = float(D_IV(a_i, alpha, sigma, a_star))
        Dpi = float(dD_dN(a_i, alpha, sigma, a_star))

        H2_i = H_i ** 2
        A_t = -(Di + 2.0 * xi) * psi_i ** 2 + 2.0 * xi * phi_i * psi_i
        B_t = -(3.0 * Di + Dpi) * psi_i + 12.0 * xi * phi_i - Vp_i / H2_i
        M_i = Di * Fi + 6.0 * xi ** 2 * phi_i ** 2
        if M_i < 1e-30:
            M_i = 1e-30
        HpH = (A_t * Di - 2.0 * xi * phi_i * B_t) / (2.0 * M_i)
        eps1[i] = -HpH

    # ε₂ = dε₁/dN / ε₁
    eps2 = np.gradient(eps1, N_arr) / np.where(eps1 > 1e-10, eps1, 1e-10)

    # Attach derived arrays to solution object
    sol.N_arr = N_arr
    sol.phi = phi_arr
    sol.psi = psi_arr
    sol.H = H_arr
    sol.a = a_arr
    sol.epsilon1 = eps1
    sol.epsilon2 = eps2
    sol.N_total = N_arr[-1] if len(N_arr) > 0 else 0.0

    return sol
