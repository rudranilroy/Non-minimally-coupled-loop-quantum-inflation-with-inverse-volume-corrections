# Non-minimally coupled loop quantum inflation with inverse volume corrections

Companion code repository for the paper:

> **"Non-minimally coupled loop quantum inflation with inverse volume corrections"**
> arXiv: [2603.04182](https://arxiv.org/abs/2603.04182)

## Physics overview

The code implements the inflationary cosmology described by the Jordan-frame action

$$S = \int d^4x\,\sqrt{-g}\left[\frac{F(\phi)}{2}R \;-\; \frac{D(a)}{2}(\partial\phi)^2 \;-\; V(\phi)\right]$$

with:

| Symbol | Description | Default |
|--------|-------------|---------|
| $F(\phi) = 1 + \xi\phi^2$ | Non-minimal coupling (NMC) to gravity | $\xi = 0$ (GR limit) |
| $D(a) = 1 - \alpha(a_*/a)^\sigma$ | Loop-quantum inverse-volume (IV) correction | $\alpha = 0$ (GR limit) |
| $V(\phi) = \tfrac{1}{2}m^2\phi^2$ | Quadratic inflationary potential | $m = 6\times10^{-6}\,M_{\rm Pl}$ |

Units: reduced Planck mass $M_{\rm Pl} = 1$.

## Equations of motion

The background evolution is governed by the modified Friedmann and
Klein–Gordon equations derived from the action above in a flat FRW metric.
The code solves the full coupled system as a first-order ODE in e-folds
$N = \ln a$ (see `src/background.py` for details).

CMB observables (spectral index $n_s$, tensor-to-scalar ratio $r$,
amplitude $A_s$) are extracted via the slow-roll approximation evaluated
$N_*$ e-folds before the end of inflation.

## Figures generated

| File | Description |
|------|-------------|
| `fig1_background_evolution.pdf` | φ(N), H(N), ε₁(N) – background evolution |
| `fig2_phase_portrait.pdf` | Phase portrait dφ/dN vs φ for several ξ values |
| `fig3_potential.pdf` | Inflationary potential with classical trajectory |
| `fig4_xi_scan.pdf` | n_s and r as functions of ξ |
| `fig5_alpha_scan.pdf` | n_s and r as functions of IV correction strength α |
| `fig6_ns_r_xi_scan.pdf` | n_s–r diagram: ξ scan with Planck 2018 contours |
| `fig7_ns_r_alpha_scan.pdf` | n_s–r diagram: α scan with Planck 2018 contours |
| `fig8_Nstar_scan.pdf` | n_s and r vs N_* (reheating uncertainty) |
| `fig9_slow_roll.pdf` | Slow-roll parameters ε₁, ε₂ vs e-folds |
| `fig10_ns_r_combined.pdf` | n_s–r: GR vs LQC-IV comparison for several ξ |

## Repository structure

```
.
├── requirements.txt          # Python dependencies
├── generate_plots.py         # Main script – generates all figures
├── src/
│   ├── model.py              # Potential, NMC function F(φ), IV correction D(a)
│   ├── background.py         # Background ODE solver (FRW + NMC + IV)
│   └── observables.py        # CMB observables (n_s, r, A_s) via slow-roll
└── plots/                    # Output figures (created on first run)
```

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate all plots

```bash
python generate_plots.py
```

All figures are saved to the `plots/` directory as PDF files.

### 3. Use the modules interactively

```python
from src.model import DEFAULT_POTENTIAL, DEFAULT_IV
from src.background import solve_background
from src.observables import get_observables

# GR baseline (ξ=0, α=0)
sol = solve_background(phi0=15.0, xi=0.0,
                       pot_params=DEFAULT_POTENTIAL,
                       iv_params=DEFAULT_IV)
obs = get_observables(sol, N_star=55)
print(f"n_s = {obs['n_s']:.4f},  r = {obs['r']:.4f}")

# With NMC
sol_nmc = solve_background(phi0=15.0, xi=0.02,
                            pot_params=DEFAULT_POTENTIAL,
                            iv_params=DEFAULT_IV)
obs_nmc = get_observables(sol_nmc, N_star=55)
print(f"n_s = {obs_nmc['n_s']:.4f},  r = {obs_nmc['r']:.4f}")

# With IV corrections
iv_lqc = {'alpha': 0.2, 'sigma': 3, 'a_star': 1e-3}
sol_iv = solve_background(phi0=15.0, xi=0.0,
                           pot_params=DEFAULT_POTENTIAL,
                           iv_params=iv_lqc)
obs_iv = get_observables(sol_iv, N_star=55)
print(f"n_s = {obs_iv['n_s']:.4f},  r = {obs_iv['r']:.4f}")
```

## Parameters

### Model parameters (`src/model.py`)

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `xi` | ξ | Non-minimal coupling constant (ξ=0: minimal coupling) |
| `alpha` | α | IV correction strength (α=0: standard GR/LQC) |
| `sigma` | σ | IV correction power index (default: 3) |
| `a_star` | a_* | IV reference scale in Planck units (default: 10⁻³) |
| `m` | m | Inflaton mass (default: 6×10⁻⁶ M_Pl) |

### Observational reference

Planck 2018 TT+TE+EE+lowE constraints (arXiv:1807.06211):
- $n_s = 0.9649 \pm 0.0042$ (68% CL)
- $r < 0.11$ (95% CL, Planck + BICEP2/Keck Array)

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7
