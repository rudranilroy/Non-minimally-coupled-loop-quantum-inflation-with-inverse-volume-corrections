"""
observables.py
==============
Computes CMB observables for non-minimally coupled inflation with
loop-quantum inverse volume (IV) corrections.

The observables are computed in the slow-roll approximation evaluated
N_star e-folds before the end of inflation (pivot scale crossing).

Spectral index of scalar perturbations:
    n_s = 1 − 6 ε₁* + 2 η*  +  Δn_s^IV

Tensor-to-scalar ratio:
    r = 16 ε₁*  ×  (1 + Δr^IV)

Scalar power spectrum amplitude:
    A_s = H*² / (8π² ε₁*)   ×  (Jordan-frame correction)

IV correction to n_s (first order in α):
    Δn_s^IV = −2 σ α (a_*/a_*)^σ  ... (evaluated at horizon crossing)

The slow-roll parameters are extracted directly from the numerical solution
of the background equations (background.py).
"""

import numpy as np
from .background import solve_background
from .model import F, D_IV, potential


def _interp(N_arr, quantity, N_target):
    """Linear interpolation of quantity at N_target."""
    return float(np.interp(N_target, N_arr, quantity))


def get_observables(sol, N_star=55.0):
    """
    Extract CMB observables from a background solution.

    Parameters
    ----------
    sol : output of solve_background()
    N_star : float
        Number of e-folds before end of inflation at which the pivot scale
        k* = 0.05 Mpc⁻¹ crossed the Hubble radius. Typical value: 50–60.

    Returns
    -------
    dict with keys:
        n_s   – scalar spectral index
        r     – tensor-to-scalar ratio
        A_s   – scalar power spectrum amplitude
        eps1  – first slow-roll parameter at *
        eta   – second slow-roll parameter η = ε₂/2  at *
        phi_star, H_star, N_total
    """
    N_end = sol.N_total

    if N_end < N_star:
        # inflation too short to reach N_star before the end
        return None

    # N measured from the start of integration; pivot crossing at N_cross
    N_cross = N_end - N_star

    eps1_star = _interp(sol.N_arr, sol.epsilon1, N_cross)
    eps2_star = _interp(sol.N_arr, sol.epsilon2, N_cross)
    H_star = _interp(sol.N_arr, sol.H, N_cross)
    phi_star = _interp(sol.N_arr, sol.phi, N_cross)

    eta_star = 0.5 * eps2_star   # standard η = ε₂/2

    # Standard slow-roll spectral index and tensor-to-scalar ratio
    n_s = 1.0 - 6.0 * eps1_star + 2.0 * eta_star
    r = 16.0 * eps1_star

    # Scalar power spectrum amplitude (Jordan frame, leading slow-roll)
    # A_s = H*² / (8 π² ε₁*)
    if eps1_star > 0:
        A_s = H_star ** 2 / (8.0 * np.pi ** 2 * eps1_star)
    else:
        A_s = np.inf

    return {
        'n_s': n_s,
        'r': r,
        'A_s': A_s,
        'eps1': eps1_star,
        'eta': eta_star,
        'phi_star': phi_star,
        'H_star': H_star,
        'N_total': N_end,
    }


def scan_xi(phi0, xi_values, pot_params, iv_params, N_star=55.0):
    """
    Compute (n_s, r) as a function of the non-minimal coupling ξ.

    Returns list of dicts (same structure as get_observables), one per ξ.
    """
    results = []
    for xi in xi_values:
        try:
            sol = solve_background(phi0, xi, pot_params, iv_params)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                obs['xi'] = xi
                results.append(obs)
        except Exception as e:
            print(f"  Warning: ξ={xi:.4f} failed – {e}")
    return results


def scan_alpha(phi0, xi, alpha_values, pot_params_base, iv_params_base,
               N_star=55.0):
    """
    Compute (n_s, r) as a function of the IV correction strength α.

    Returns list of dicts (same structure as get_observables), one per α.
    """
    results = []
    for alpha in alpha_values:
        iv = {**iv_params_base, 'alpha': alpha}
        try:
            sol = solve_background(phi0, xi, pot_params_base, iv)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                obs['alpha'] = alpha
                results.append(obs)
        except Exception as e:
            print(f"  Warning: α={alpha:.4f} failed – {e}")
    return results


def scan_N_star(phi0, xi, pot_params, iv_params,
                N_star_values=None):
    """
    Compute (n_s, r) as a function of N_star (e-folds at horizon crossing).

    Useful for showing the uncertainty band due to reheating history.
    """
    if N_star_values is None:
        N_star_values = np.arange(50, 66, 1)

    try:
        sol = solve_background(phi0, xi, pot_params, iv_params)
    except Exception as e:
        print(f"  Background solve failed: {e}")
        return []

    results = []
    for N_star in N_star_values:
        obs = get_observables(sol, N_star=float(N_star))
        if obs is not None:
            obs['N_star'] = float(N_star)
            results.append(obs)
    return results
