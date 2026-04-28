"""
model.py
========
Defines the physical model for non-minimally coupled loop quantum inflation
with inverse volume corrections.

Physics setup (natural units: reduced Planck mass M_Pl = 1):
------------------------------------------------------------
Action:
    S = ∫d⁴x √(-g) [ F(φ)/2 · R  -  D(a)/2 · (∂φ)²  -  V(φ) ]

Non-minimal coupling (Jordan frame):
    F(φ) = 1 + ξ φ²

Inverse volume (IV) correction:
    D(a) = 1 - α (a_*/a)^σ    (for a > a_*)
           clamped to [ε_floor, 1]

Inflationary potential (selectable):
    'quadratic'   : V = ½ m² φ²
    'quartic'     : V = ¼ λ φ⁴
    'natural'     : V = Λ⁴ [1 + cos(φ/f)]
    'starobinsky' : V = (3/4) m² [1 - exp(-√(2/3) φ)]²
"""

import numpy as np


# ---------------------------------------------------------------------------
# Non-minimal coupling
# ---------------------------------------------------------------------------

def F(phi, xi):
    """Effective Planck mass squared: F(φ) = 1 + ξ φ²."""
    return 1.0 + xi * phi ** 2


def dF_dphi(phi, xi):
    """dF/dφ = 2 ξ φ."""
    return 2.0 * xi * phi


def d2F_dphi2(xi):
    """d²F/dφ² = 2 ξ (constant)."""
    return 2.0 * xi


# ---------------------------------------------------------------------------
# Inverse volume correction
# ---------------------------------------------------------------------------

def D_IV(a, alpha, sigma, a_star, eps_floor=1e-6):
    """
    Inverse volume correction factor.

    D(a) = 1 - α (a_*/a)^σ

    Clamped to [eps_floor, 1] so that D stays physical.
    For α = 0 (GR limit) returns 1 everywhere.
    """
    if alpha == 0.0:
        return np.ones_like(np.asarray(a, dtype=float))
    ratio = (a_star / np.asarray(a, dtype=float)) ** sigma
    return np.clip(1.0 - alpha * ratio, eps_floor, 1.0)


def dD_dN(a, alpha, sigma, a_star):
    """
    dD/dN  where N = ln a  (e-folds).
    dD/dN = dD/da · a = α σ (a_*/a)^σ
    """
    if alpha == 0.0:
        return np.zeros_like(np.asarray(a, dtype=float))
    return alpha * sigma * (a_star / np.asarray(a, dtype=float)) ** sigma


# ---------------------------------------------------------------------------
# Inflationary potentials
# ---------------------------------------------------------------------------

def potential(phi, params):
    """
    Return (V, V') for the selected potential type.

    params dict keys: 'type', and type-specific keys:
        quadratic   → 'm'
        quartic     → 'lam'
        natural     → 'Lambda', 'f'
        starobinsky → 'm'
    """
    ptype = params.get('type', 'quadratic')

    if ptype == 'quadratic':
        m = params['m']
        V = 0.5 * m ** 2 * phi ** 2
        Vp = m ** 2 * phi

    elif ptype == 'quartic':
        lam = params['lam']
        V = 0.25 * lam * phi ** 4
        Vp = lam * phi ** 3

    elif ptype == 'natural':
        Lam = params['Lambda']
        f = params['f']
        V = Lam ** 4 * (1.0 + np.cos(phi / f))
        Vp = -Lam ** 4 * np.sin(phi / f) / f

    elif ptype == 'starobinsky':
        m = params['m']
        ex = np.exp(-np.sqrt(2.0 / 3.0) * phi)
        V = 0.75 * m ** 2 * (1.0 - ex) ** 2
        Vp = 0.75 * m ** 2 * 2.0 * (1.0 - ex) * np.sqrt(2.0 / 3.0) * ex

    else:
        raise ValueError(f"Unknown potential type: {ptype!r}")

    return V, Vp


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_POTENTIAL = {
    'type': 'quadratic',
    'm': 6.0e-6,       # inflaton mass in Planck units (gives right CMB amplitude)
}

DEFAULT_IV = {
    'alpha': 0.0,      # IV correction strength  (0 → GR)
    'sigma': 3,        # correction power index
    'a_star': 1e-3,    # reference scale in Planck units
}
