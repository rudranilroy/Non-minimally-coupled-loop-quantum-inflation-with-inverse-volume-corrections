# [Non-minimally coupled loop quantum inflation with inverse volume corrections](https://arxiv.org/abs/2603.04182)
Rudranil Roy [:id:](https://orcid.org/0000-0003-3114-419X) $^a$, Giovanni Otalora [:id:](http://orcid.org/0000-0001-6753-0565) $^a$, Joel Saavedra [:id:](https://orcid.org/0000-0002-1430-3008) $^b$, Salvatore Capozziello [:id:](https://orcid.org/0000-0003-4886-2024) $^{c,d,e}$

$^a$ Departamento de Física, Facultad de Ciencias, Universidad de Tarapacá, Casilla 7-D, Arica, Chile

$^b$ Instituto de Física, Pontificia Universidad Católica de Valparaíso, Casilla 4950, Valparaíso, Chile

$^c$ Dipartimento di Fisica “E. Pancini”, Università degli Studi di Napoli “Federico II”, Complesso Universitario di Monte Sant’Angelo, Edificio G, Via Cinthia, I-80126, Napoli, Italy

$^d$ Istituto Nazionale di Fisica Nucleare (INFN), Sezione di Napoli, Via Cinthia 9, I-80126, Napoli, Italy

$^e$ Scuola Superiore Meridionale, Via Mezzocannone 4, I-80134, Napoli, Italy

---

Welcome! This repository contains the analytical derivations, numerical data, and plotting scripts accompanying the research paper "[Non-minimally coupled loop quantum inflation with inverse-volume corrections](https://arxiv.org/abs/2603.04182)".  The codebase provided here contains the complete computational framework used to investigate slow-roll inflation driven by a scalar field non-minimally coupled to gravity within the effective framework of Loop Quantum Cosmology (LQC). Using the included Mathematica notebooks and Python scripts, you can fully reproduce the theoretical predictions and background dynamics for two physically motivated models: a Higgs-like quartic potential and string-inspired fractional monomial potentials.  Additionally, the repository includes the computational tools necessary to evaluate the phase-space probability of achieving a sufficiently long inflationary phase, demonstrating how non-minimal coupling reshapes the favorable initial conditions following a quantum bounce.


## [Algebraic Computations](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Algebraic%20Computations)
This folder contains the Mathematica notebooks used to perform the core algebraic computations and generate the phase-space probability plots presented in the paper. The code files are categorized based on the specific inflationary potential being evaluated: the Higgs-like quartic potential ($V\propto\phi^4$) and the string-inspired fractional monomial potentials ($V\propto\phi^{1/3}$ and $V\propto\phi^{2/3}$).

**What the codes do:**

- **Spacetime Geometry:** Systematically derives the background geometrical quantities, including Christoffel symbols, the Riemann and Ricci tensors, and the Ricci scalar.

- **Equations of Motion:** Computes both the classical and the effective Loop Quantum Cosmology (LQC) equations of motion for a scalar field non-minimally coupled to gravity, incorporating inverse-volume corrections.

- **Slow-Roll Approximation:** Applies slow-roll conditions to evaluate the Hubble parameter and the inflaton's evolutionary dynamics.

- **Probability of Inflation:** Calculates the canonical Liouville measure over the effective phase space to determine the fraction of post-bounce trajectories that lead to sufficient inflation.

- **Visualization:** Ultimately outputs the evaluated probability expressions and generates the 3D surface plots (finalplot and finalplotlqg) to map the typicality of inflation across the non-minimal coupling ($\xi$) and e-folding ($N$) parameter space.


## [Numerical Computations and Plots](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots)
The folder contains the Python and Mathematica scripts used to evaluate the model's numerical predictions, constrain the parameter space, and generate the figures presented in the paper. To ensure clarity, the code is organized into five distinct subfolders based on the specific analysis performed:

- [**Inverse Volume Correction :**](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots/Inverse%20Volume%20Correction) Contains Python code dedicated to evaluating and plotting the behavior of the LQC inverse-volume correction factor, $D_l(q)$, across the quantum-to-classical transition.

- [**Slow-Roll Parameters :**](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots/Slow-Roll%20Parameters) Includes both Python and Mathematica scripts used to track and plot the dynamical evolution of the modified slow-roll parameters, verifying the consistency and natural termination of the inflationary regime.

- [**alpha_s - n_s :**](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots/alpha_s%20-%20n_s) Scripts that compute and plot the theoretical predictions for the running of the spectral index ($\alpha_s$) against the scalar spectral index ($n_s$). The outputs are superimposed on the $1\sigma$ and $2\sigma$ confidence contours from the Planck 2018 (P-LB-BK18) and ACT DR6 (P-ACT-LB-BK18) datasets.

- [**n_s - r :**](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots/n_s%20-%20r) Codes that generate plots of the theoretical tensor-to-scalar ratio ($r$) versus the scalar spectral index ($n_s$), displayed alongside the same combined observational data contours to highlight the model's viability.

- [**parameter :**](https://github.com/rudranilroy/arXiv-2603.04182/tree/main/Numerical%20Computations%20and%20Plots/parameter) Contains the scripts used to filter and visualize the phenomenologically viable parameter space. It extracts and plots the allowed regions for the non-minimal coupling strength ($\xi$) and the number of e-folds ($N$) that strictly satisfy the combined observational bounds at various confidence levels.
