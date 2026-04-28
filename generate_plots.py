"""
generate_plots.py
=================
Generate all figures for the paper:
"Non-minimally coupled loop quantum inflation with inverse volume corrections"
arXiv: 2603.04182

Figures generated:
    Fig. 1  –  Background evolution  (φ, H, ε₁ vs e-folds N)
    Fig. 2  –  Phase portrait  (φ̇/H vs φ)
    Fig. 3  –  Inflationary potential with classical GR trajectory
    Fig. 4  –  Effect of non-minimal coupling ξ on n_s and r
    Fig. 5  –  Effect of IV correction α on n_s and r
    Fig. 6  –  n_s – r diagram with Planck 2018 contours (ξ scan)
    Fig. 7  –  n_s – r diagram with Planck 2018 contours (α scan)
    Fig. 8  –  n_s and r vs N_star (reheating uncertainty)

Usage:
    python generate_plots.py
Outputs are saved to the  plots/  directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.model import DEFAULT_POTENTIAL, DEFAULT_IV, potential, D_IV
from src.background import solve_background, max_phi0_for_xi
from src.observables import (
    get_observables, scan_xi, scan_alpha, scan_N_star
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Matplotlib global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Planck 2018 constraint data (TT+TE+EE+lowE+lensing, 68 % and 95 % CL)
# Reference: Planck 2018 X (arXiv:1807.06211), Table 1
# ---------------------------------------------------------------------------
PLANCK_NS_CENTRE = 0.9649
PLANCK_NS_SIGMA = 0.0042
PLANCK_R_UPPER_95 = 0.11   # Planck + BK15

# Approximate 1σ / 2σ ellipses in the n_s – r plane
def _planck_ellipse(ax, n_s_c, sigma_ns, r_max_95,
                    color='#4e9af1', alpha68=0.35, alpha95=0.15,
                    label='Planck 2018'):
    """Draw approximate Planck 2018 confidence regions on an n_s–r axis."""
    r_vals = np.linspace(0.0, r_max_95, 300)

    # Parametric ellipse (approximate); major axis along n_s, minor along r
    theta = np.linspace(0, 2 * np.pi, 400)

    # 1-sigma
    ns_ell_68 = n_s_c + sigma_ns * np.cos(theta)
    r_ell_68 = (r_max_95 / 4) * (1 + np.sin(theta)) / 2   # rough shape
    ax.fill(ns_ell_68, r_ell_68, color=color, alpha=alpha68,
            label=f'{label} 68%', zorder=0)

    # 2-sigma
    ns_ell_95 = n_s_c + 2 * sigma_ns * np.cos(theta)
    r_ell_95 = (r_max_95 / 2) * (1 + np.sin(theta)) / 2
    ax.fill(ns_ell_95, r_ell_95, color=color, alpha=alpha95,
            label=f'{label} 95%', zorder=0)


# ---------------------------------------------------------------------------
# Helper: default parameters
# ---------------------------------------------------------------------------
DEFAULT_XI = 0.0
DEFAULT_PHI0 = 15.0         # initial field value (Planck units)
DEFAULT_N_STAR = 55         # pivot-scale e-folds (matches typical LQC papers)
N_STAR_RANGE = np.arange(50, 61, 1)


# ---------------------------------------------------------------------------
# Figure 1 – Background evolution
# ---------------------------------------------------------------------------
def plot_background_evolution(phi0=DEFAULT_PHI0, xi=DEFAULT_XI,
                               pot_params=None, iv_params=None):
    print("  Fig. 1: Background evolution …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV

    sol = solve_background(phi0, xi, pot_params, iv_params)
    N = sol.N_arr
    N_end = sol.N_total
    # Shift so that N=0 is the end of inflation; x-axis shows e-folds before end
    N_before_end = N_end - N

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    ax0, ax1, ax2 = axes

    ax0.plot(N_before_end, sol.phi, color='C0')
    ax0.set_ylabel(r'$\phi\;[M_{\rm Pl}]$')
    ax0.invert_xaxis()

    ax1.semilogy(N_before_end, sol.H, color='C1')
    ax1.set_ylabel(r'$H\;[M_{\rm Pl}]$')

    ax2.semilogy(N_before_end, np.maximum(sol.epsilon1, 1e-6), color='C2')
    ax2.axhline(1.0, ls='--', color='k', lw=1.0, label=r'$\varepsilon_1 = 1$')
    ax2.set_ylabel(r'$\varepsilon_1$')
    ax2.set_xlabel(r'$N_{\rm end} - N$ (e-folds before end)')
    ax2.legend(fontsize=10)

    title_str = (rf'$\xi={xi}$, '
                 rf'$\alpha={iv_params["alpha"]}$, '
                 rf'$m={pot_params["m"]:.1e}\,M_{{\rm Pl}}$')
    fig.suptitle(f'Background evolution\n{title_str}', fontsize=12)
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig1_background_evolution.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 2 – Phase portrait
# ---------------------------------------------------------------------------
def plot_phase_portrait(phi0=DEFAULT_PHI0, pot_params=None, iv_params=None,
                        xi_values=None):
    print("  Fig. 2: Phase portrait …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV
    if xi_values is None:
        xi_values = [0.0, 0.01, -0.01]

    colors = ['C0', 'C1', 'C2']
    fig, ax = plt.subplots(figsize=(7, 5))

    for xi, col in zip(xi_values, colors):
        phi_use = min(phi0, max_phi0_for_xi(float(xi)))
        try:
            sol = solve_background(phi_use, xi, pot_params, iv_params)
            ax.plot(sol.phi, sol.psi, color=col,
                    label=rf'$\xi={xi}$')
            ax.plot(sol.phi[0], sol.psi[0], 'o', color=col, ms=5)
            ax.plot(sol.phi[-1], sol.psi[-1], 's', color=col, ms=5)
        except Exception as e:
            print(f"    Warning: ξ={xi} failed – {e}")

    ax.set_xlabel(r'$\phi\;[M_{\rm Pl}]$')
    ax.set_ylabel(r'$d\phi/dN\;[M_{\rm Pl}]$')
    ax.set_title('Phase portrait (circle = start, square = end)')
    ax.legend()
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig2_phase_portrait.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 3 – Inflationary potential
# ---------------------------------------------------------------------------
def plot_potential(phi0=DEFAULT_PHI0, xi=DEFAULT_XI,
                   pot_params=None, iv_params=None):
    print("  Fig. 3: Inflationary potential …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV

    phi_range = np.linspace(0.01, phi0 * 1.05, 500)
    V_arr = np.array([potential(p, pot_params)[0] for p in phi_range])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(phi_range, V_arr / V_arr.max(), 'k-', label=r'$V(\phi)/V_{\rm max}$')

    # Mark trajectory
    try:
        sol = solve_background(phi0, xi, pot_params, iv_params)
        V_traj = np.array([potential(p, pot_params)[0] for p in sol.phi])
        ax.plot(sol.phi, V_traj / V_arr.max(), color='C0', lw=1.2,
                label=rf'trajectory ($\xi={xi}$)')
        ax.plot(sol.phi[0], V_traj[0] / V_arr.max(), 'o', color='C0', ms=6)
        ax.plot(sol.phi[-1], V_traj[-1] / V_arr.max(), 's', color='C0', ms=6)
    except Exception as e:
        print(f"    Warning: trajectory failed – {e}")

    ax.set_xlabel(r'$\phi\;[M_{\rm Pl}]$')
    ax.set_ylabel(r'$V(\phi)/V_{\rm max}$')
    ax.set_title(f'Potential: {pot_params["type"]}')
    ax.legend()
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig3_potential.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 4 – n_s and r vs ξ
# ---------------------------------------------------------------------------
def plot_xi_scan(phi0=DEFAULT_PHI0, pot_params=None, iv_params=None,
                 N_star=DEFAULT_N_STAR):
    print("  Fig. 4: n_s, r vs ξ …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV

    xi_values = np.concatenate([
        np.linspace(-0.05, -0.001, 15),
        [0.0],
        np.linspace(0.001, 0.05, 15),
    ])

    ns_list, r_list = [], []
    xi_ok = []
    for xi in xi_values:
        phi_use = min(phi0, max_phi0_for_xi(float(xi)))
        try:
            sol = solve_background(phi_use, float(xi), pot_params, iv_params)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                ns_list.append(obs['n_s'])
                r_list.append(obs['r'])
                xi_ok.append(xi)
        except Exception as e:
            print(f"    Warning: ξ={xi:.4f} failed – {e}")

    xi_ok = np.array(xi_ok)
    ns_arr = np.array(ns_list)
    r_arr = np.array(r_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(xi_ok, ns_arr, 'C0-o', ms=4)
    ax1.axhline(PLANCK_NS_CENTRE, ls='--', color='gray', lw=1.0,
                label=r'Planck 2018 $n_s$')
    ax1.fill_between(xi_ok,
                     PLANCK_NS_CENTRE - PLANCK_NS_SIGMA,
                     PLANCK_NS_CENTRE + PLANCK_NS_SIGMA,
                     color='gray', alpha=0.25, label=r'1-$\sigma$')
    ax1.set_xlabel(r'$\xi$ (non-minimal coupling)')
    ax1.set_ylabel(r'$n_s$')
    ax1.set_title(r'Scalar spectral index vs $\xi$')
    ax1.legend(fontsize=9)

    ax2.plot(xi_ok, r_arr, 'C1-o', ms=4)
    ax2.axhline(PLANCK_R_UPPER_95, ls='--', color='gray', lw=1.0,
                label=r'Planck+BK15 $r<0.11$ (95%)')
    ax2.set_xlabel(r'$\xi$ (non-minimal coupling)')
    ax2.set_ylabel(r'$r$')
    ax2.set_title(r'Tensor-to-scalar ratio vs $\xi$')
    ax2.legend(fontsize=9)

    title = (rf'$\phi_0={phi0}\,M_{{\rm Pl}}$, '
             rf'$N_*={N_star}$, '
             rf'$\alpha={iv_params["alpha"]}$')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig4_xi_scan.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 5 – n_s and r vs α (IV correction)
# ---------------------------------------------------------------------------
def plot_alpha_scan(phi0=DEFAULT_PHI0, xi=DEFAULT_XI,
                    pot_params=None, iv_params_base=None,
                    N_star=DEFAULT_N_STAR):
    print("  Fig. 5: n_s, r vs α (IV correction) …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params_base is None:
        iv_params_base = DEFAULT_IV

    alpha_values = np.linspace(0.0, 0.5, 20)
    ns_list, r_list = [], []
    alpha_ok = []

    for alpha in alpha_values:
        iv = {**iv_params_base, 'alpha': float(alpha)}
        try:
            sol = solve_background(phi0, xi, pot_params, iv)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                ns_list.append(obs['n_s'])
                r_list.append(obs['r'])
                alpha_ok.append(alpha)
        except Exception as e:
            print(f"    Warning: α={alpha:.4f} failed – {e}")

    alpha_ok = np.array(alpha_ok)
    ns_arr = np.array(ns_list)
    r_arr = np.array(r_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(alpha_ok, ns_arr, 'C0-o', ms=4)
    ax1.axhline(PLANCK_NS_CENTRE, ls='--', color='gray', lw=1.0,
                label=r'Planck 2018 $n_s$')
    ax1.fill_between(alpha_ok,
                     PLANCK_NS_CENTRE - PLANCK_NS_SIGMA,
                     PLANCK_NS_CENTRE + PLANCK_NS_SIGMA,
                     color='gray', alpha=0.25, label=r'1-$\sigma$')
    ax1.set_xlabel(r'$\alpha$ (IV correction strength)')
    ax1.set_ylabel(r'$n_s$')
    ax1.set_title(r'Scalar spectral index vs $\alpha$')
    ax1.legend(fontsize=9)

    ax2.plot(alpha_ok, r_arr, 'C1-o', ms=4)
    ax2.axhline(PLANCK_R_UPPER_95, ls='--', color='gray', lw=1.0,
                label=r'Planck+BK15 $r<0.11$ (95%)')
    ax2.set_xlabel(r'$\alpha$ (IV correction strength)')
    ax2.set_ylabel(r'$r$')
    ax2.set_title(r'Tensor-to-scalar ratio vs $\alpha$')
    ax2.legend(fontsize=9)

    title = (rf'$\phi_0={phi0}\,M_{{\rm Pl}}$, '
             rf'$N_*={N_star}$, '
             rf'$\xi={xi}$')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig5_alpha_scan.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 6 – n_s – r diagram: ξ scan with Planck contours
# ---------------------------------------------------------------------------
def plot_ns_r_xi(phi0=DEFAULT_PHI0, pot_params=None, iv_params=None,
                 N_star=DEFAULT_N_STAR):
    print("  Fig. 6: n_s–r diagram (ξ scan) …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV

    xi_values = np.concatenate([
        np.linspace(-0.04, -0.002, 10),
        [0.0],
        np.linspace(0.002, 0.04, 10),
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    _planck_ellipse(ax, PLANCK_NS_CENTRE, PLANCK_NS_SIGMA, PLANCK_R_UPPER_95)

    ns_list, r_list, xi_list = [], [], []
    for xi in xi_values:
        phi_use = min(phi0, max_phi0_for_xi(float(xi)))
        try:
            sol = solve_background(phi_use, float(xi), pot_params, iv_params)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                ns_list.append(obs['n_s'])
                r_list.append(obs['r'])
                xi_list.append(xi)
        except Exception as e:
            print(f"    Warning: ξ={xi:.4f} failed – {e}")

    sc = ax.scatter(ns_list, r_list, c=xi_list, cmap='coolwarm',
                    zorder=5, s=40, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label=r'$\xi$')

    ax.set_xlabel(r'$n_s$')
    ax.set_ylabel(r'$r$')
    ax.set_title(rf'$n_s$–$r$ diagram: NMC coupling scan ($N_*={N_star}$)')
    ax.set_xlim(0.93, 0.99)
    ax.set_ylim(0.0, 0.25)

    # Legend for Planck contours
    p68 = mpatches.Patch(color='#4e9af1', alpha=0.35, label='Planck 2018 68%')
    p95 = mpatches.Patch(color='#4e9af1', alpha=0.15, label='Planck 2018 95%')
    ax.legend(handles=[p68, p95], loc='upper right', fontsize=9)

    fig.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'fig6_ns_r_xi_scan.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 7 – n_s – r diagram: α scan with Planck contours
# ---------------------------------------------------------------------------
def plot_ns_r_alpha(phi0=DEFAULT_PHI0, xi=DEFAULT_XI,
                    pot_params=None, iv_params_base=None,
                    N_star=DEFAULT_N_STAR):
    print("  Fig. 7: n_s–r diagram (α scan) …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params_base is None:
        iv_params_base = DEFAULT_IV

    alpha_values = np.linspace(0.0, 0.45, 18)

    fig, ax = plt.subplots(figsize=(7, 5))
    _planck_ellipse(ax, PLANCK_NS_CENTRE, PLANCK_NS_SIGMA, PLANCK_R_UPPER_95)

    ns_list, r_list, al_list = [], [], []
    for alpha in alpha_values:
        iv = {**iv_params_base, 'alpha': float(alpha)}
        try:
            sol = solve_background(phi0, xi, pot_params, iv)
            obs = get_observables(sol, N_star=N_star)
            if obs is not None:
                ns_list.append(obs['n_s'])
                r_list.append(obs['r'])
                al_list.append(alpha)
        except Exception as e:
            print(f"    Warning: α={alpha:.4f} failed – {e}")

    sc = ax.scatter(ns_list, r_list, c=al_list, cmap='plasma',
                    zorder=5, s=40, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label=r'$\alpha$ (IV correction)')

    ax.set_xlabel(r'$n_s$')
    ax.set_ylabel(r'$r$')
    ax.set_title(rf'$n_s$–$r$ diagram: IV correction scan ($N_*={N_star}$)')
    ax.set_xlim(0.93, 0.99)
    ax.set_ylim(0.0, 0.25)

    p68 = mpatches.Patch(color='#4e9af1', alpha=0.35, label='Planck 2018 68%')
    p95 = mpatches.Patch(color='#4e9af1', alpha=0.15, label='Planck 2018 95%')
    ax.legend(handles=[p68, p95], loc='upper right', fontsize=9)

    fig.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'fig7_ns_r_alpha_scan.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 8 – n_s and r vs N_star (reheating uncertainty)
# ---------------------------------------------------------------------------
def plot_Nstar_scan(phi0=DEFAULT_PHI0, pot_params=None, iv_params=None,
                   xi_values=None):
    print("  Fig. 8: n_s, r vs N_star …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV
    if xi_values is None:
        xi_values = [0.0, 0.01, -0.01]

    N_star_vals = np.arange(50, 66, 1)
    colors = ['C0', 'C1', 'C2']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for xi, col in zip(xi_values, colors):
        phi_use = min(phi0, max_phi0_for_xi(float(xi)))
        results = scan_N_star(phi_use, xi, pot_params, iv_params,
                              N_star_values=N_star_vals)
        if not results:
            continue
        Ns = [r['N_star'] for r in results]
        ns = [r['n_s'] for r in results]
        r_ = [r['r'] for r in results]
        ax1.plot(Ns, ns, color=col, marker='o', ms=4,
                 label=rf'$\xi={xi}$')
        ax2.plot(Ns, r_, color=col, marker='o', ms=4,
                 label=rf'$\xi={xi}$')

    ax1.axhline(PLANCK_NS_CENTRE, ls='--', color='gray', lw=1.0)
    ax1.fill_between(N_star_vals,
                     PLANCK_NS_CENTRE - PLANCK_NS_SIGMA,
                     PLANCK_NS_CENTRE + PLANCK_NS_SIGMA,
                     color='gray', alpha=0.2)
    ax1.set_xlabel(r'$N_*$ (e-folds)')
    ax1.set_ylabel(r'$n_s$')
    ax1.set_title(r'Spectral index vs $N_*$')
    ax1.legend(fontsize=9)

    ax2.axhline(PLANCK_R_UPPER_95, ls='--', color='gray', lw=1.0)
    ax2.set_xlabel(r'$N_*$ (e-folds)')
    ax2.set_ylabel(r'$r$')
    ax2.set_title(r'Tensor-to-scalar ratio vs $N_*$')
    ax2.legend(fontsize=9)

    fig.suptitle(rf'Reheating dependence  ($\alpha={iv_params["alpha"]}$)',
                 fontsize=11)
    fig.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'fig8_Nstar_scan.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 9 – Slow-roll parameters ε₁, ε₂ vs N
# ---------------------------------------------------------------------------
def plot_slow_roll(phi0=DEFAULT_PHI0, xi=DEFAULT_XI,
                   pot_params=None, iv_params=None):
    print("  Fig. 9: Slow-roll parameters …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params is None:
        iv_params = DEFAULT_IV

    sol = solve_background(phi0, xi, pot_params, iv_params)
    N_end = sol.N_total
    N_before = N_end - sol.N_arr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.semilogy(N_before, np.maximum(sol.epsilon1, 1e-8), 'C0')
    ax1.axhline(1.0, ls='--', color='k', lw=1.0)
    ax1.set_ylabel(r'$\varepsilon_1 = -\dot{H}/H^2$')
    ax1.invert_xaxis()

    ax2.plot(N_before, sol.epsilon2, 'C1')
    ax2.axhline(0.0, ls='--', color='k', lw=1.0, alpha=0.4)
    ax2.set_ylabel(r'$\varepsilon_2 = \dot{\varepsilon}_1/(H\varepsilon_1)$')
    ax2.set_xlabel(r'$N_{\rm end} - N$ (e-folds before end)')

    title = (rf'Slow-roll parameters  ($\xi={xi}$, '
             rf'$\alpha={iv_params["alpha"]}$)')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    fname = os.path.join(PLOTS_DIR, 'fig9_slow_roll.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Figure 10 – Combined n_s–r: comparing GR vs LQC (IV) for several ξ
# ---------------------------------------------------------------------------
def plot_ns_r_combined(phi0=DEFAULT_PHI0, pot_params=None,
                       iv_params_base=None, N_star=DEFAULT_N_STAR):
    print("  Fig. 10: n_s–r combined (GR vs LQC-IV) …")
    if pot_params is None:
        pot_params = DEFAULT_POTENTIAL
    if iv_params_base is None:
        iv_params_base = DEFAULT_IV

    xi_values = [-0.02, 0.0, 0.02]
    alpha_values = [0.0, 0.2]   # 0 = GR limit, 0.2 = with IV
    markers = ['o', 's']
    ls_styles = ['-', '--']

    fig, ax = plt.subplots(figsize=(8, 5))
    _planck_ellipse(ax, PLANCK_NS_CENTRE, PLANCK_NS_SIGMA, PLANCK_R_UPPER_95)

    legend_handles = []
    for alpha, mk, ls in zip(alpha_values, markers, ls_styles):
        iv = {**iv_params_base, 'alpha': float(alpha)}
        ns_list, r_list = [], []
        for xi in xi_values:
            phi_use = min(phi0, max_phi0_for_xi(float(xi)))
            try:
                sol = solve_background(phi_use, float(xi), pot_params, iv)
                obs = get_observables(sol, N_star=N_star)
                if obs is not None:
                    ns_list.append(obs['n_s'])
                    r_list.append(obs['r'])
                else:
                    ns_list.append(np.nan)
                    r_list.append(np.nan)
            except Exception:
                ns_list.append(np.nan)
                r_list.append(np.nan)

        label = (r'GR ($\alpha=0$)' if alpha == 0.0
                 else rf'LQC IV ($\alpha={alpha}$)')
        h, = ax.plot(ns_list, r_list, ls + mk, color='C3' if alpha else 'C0',
                     ms=8, lw=1.5, label=label, zorder=5)
        legend_handles.append(h)

        # Annotate ξ values
        for xi, ns, rv in zip(xi_values, ns_list, r_list):
            if np.isfinite(ns):
                ax.annotate(rf'$\xi={xi}$', (ns, rv),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=8)

    ax.set_xlabel(r'$n_s$')
    ax.set_ylabel(r'$r$')
    ax.set_title(rf'$n_s$–$r$ diagram: GR vs LQC-IV  ($N_*={N_star}$)')
    ax.set_xlim(0.93, 0.99)
    ax.set_ylim(0.0, 0.25)

    p68 = mpatches.Patch(color='#4e9af1', alpha=0.35, label='Planck 68%')
    p95 = mpatches.Patch(color='#4e9af1', alpha=0.15, label='Planck 95%')
    ax.legend(handles=[p68, p95] + legend_handles, loc='upper left',
              fontsize=9)

    fig.tight_layout()
    fname = os.path.join(PLOTS_DIR, 'fig10_ns_r_combined.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f"    saved → {fname}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Generating all plots for arXiv:2603.04182 …\n")

    pot = DEFAULT_POTENTIAL
    iv = DEFAULT_IV

    plot_background_evolution()
    plot_phase_portrait()
    plot_potential()
    plot_xi_scan()
    plot_alpha_scan()
    plot_ns_r_xi()
    plot_ns_r_alpha()
    plot_Nstar_scan()
    plot_slow_roll()
    plot_ns_r_combined()

    print(f"\nDone. All figures saved to  {PLOTS_DIR}/")
