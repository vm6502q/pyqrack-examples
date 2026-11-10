# Created neutrinos + ghost coexistence gas in CLASS (Strano model)

By (Anthropic) Claude and Dan Strano.

## Model

**Background.** A negative-energy ghost gas is held at a coexistence temperature T_c, with ρ_g = −G and p_g = ρ_g/3.

- It feeds a *global* creation rate Q = 4HG into pairs of the three neutrino mass eigenstates.
- The pairs come out in number ratio m_i², because of the gravitational mass-monopole coupling g_i = α m_i / M_P.
- They are born thermal (Fermi–Dirac in E) at T_c, at rest in the ghost frame.

In steady state each species has a fixed physical spectrum, f₀ = C·Φ(p/T_c) with Φ(x) = ∫ₓ^∞ φ(y) dy/y. As a result:

- the dark sector (ghost + created) is **exactly Λ** in the background, for any T_c and masses;
- G is fixed by closure, so T_c and m_lightest are the only free parameters, besides the mass ordering.

**Perturbations** (synchronous gauge).

*Created species.* Each has a full Boltzmann hierarchy on a log momentum grid. A source-relaxation term creates new particles unperturbed on the CDM-comoving slicing, which is what "global" creation means here. The dipole is covariant (Q u_ghost), so the created gas receives exactly the momentum aQθ_ghost.

*Ghost.* The ghost is a fluid with w = c_s² = 1/3, no shear, and δQ = 0.

*Quadrature corrections.* At every step two factors are applied: the discrete momentum exchange and the h′ coupling are normalized to their exact values, so that the total stress-energy is conserved at quadrature precision.

**Files.**

| File | What it is |
|---|---|
| `created_nu_class.patch` | Against CLASS v3.4.1, commit 814ce59. Touches `perturbations.c/h` and `input.c`. |
| `run_grid.py` | Runs the T_c × m_lightest grid against ΛCDM with the *same* relic neutrino masses, Planck 2018 best-fit parameters held fixed. |
| `planck_chi2.py` | Planck 2018 low-ℓ TT + low-ℓ EE + plik-lite χ² from CLASS C_ℓ files, through Cobaya. |
| `setup_and_run.sh` | Builds everything and runs the grid. Takes about 5 min per point on 2 cores. |
| `slicing_test.py`, `slicing_results.json` | Creation-slicing robustness test, and its results. |
| `cobaya_fit.py`, `cobaya_results.json` | Cobaya refit by template iteration, and its results. |
| `grid_results.json`, `grid_results_dof.json` | The results below (T_c grid; self-consistent T_c). |
| `tc_selfconsistent.py` | Standalone solve of the self-consistent T_c. |

**New CLASS parameters.**

| Parameter | Default | Meaning |
|---|---|---|
| `created_neutrinos` | — | `yes` turns the model on. |
| `created_Tc_K` | 60 | Coexistence temperature T_c, in K. |
| `created_q_per_decade` | 6 | Momentum-grid resolution. |
| `created_l_max` | 8 | Hierarchy truncation. |
| `created_z_on` | 100 | The grid resolves the spectrum from this redshift on. |
| `created_E_max` | 30 | Top of the momentum grid, in units of T_c. |
| `created_k_max` | 0 = all | Skip the new species above this k, in Mpc⁻¹. The grid uses 0.05. |
| `created_slicing` | 0 | The hypersurfaces that carry the uniform creation rate. **0** = CDM-comoving (δQ = 0 in synchronous gauge). **1** = ghost-comoving (δQ/Q = (H′/H)θ_g/k²). **2** = coexistence-pinned: ghost held at T_c locally (δρ_g = δp_g = 0), so creation follows local expansion (δQ/Q = (θ_g + h′/2)/(3aH)). |
| `created_ghost_dof` | 0 = off | g > 0: **solve T_c** so that the ghost gas is a blackbody with g massless bosonic degrees of freedom, (g/2)ρ_γ(T_c/T_cmb)⁴, equal to the |ρ_ghost| that closure (exact Λ) requires. The lowest root is used, and `created_Tc_K` is then ignored. |

The masses are taken from `m_ncdm`. Normal ordering uses Δm²₂₁ = 7.42×10⁻⁵ and Δm²₃₁ = 2.51×10⁻³ eV². Species carrying less than 10⁻⁴ of the heaviest one's creation share are dropped.

## Checks

- **Steady-state identity:** 3(J_ρ + J_p) = I_E holds to 4×10⁻⁵.
- **Momentum injection on the grid:** 3.000, against an exact value of 3.
- **Dark-sector conservation:** energy to about 10⁻³ at k = 3×10⁻⁴ to 3×10⁻² Mpc⁻¹, from finite differences of the perturbation output. Momentum to about 10⁻³ of the pressure-gradient term, evaluated exactly inside the derivative routine.
- **Convergence at low ℓ:** the following changes move TT by under 0.1% and C_L^φφ by under 0.05% absolute:
  - q per decade 6 → 10;
  - l_max 8 → 12;
  - z_on 100 → 300.
- **k-cutoff 0.05 vs. none:** C_ℓ agree to 1.3×10⁻⁴ (relative to ΛCDM); Δχ² agrees to 0.015.
- **Pipeline:** Planck χ² for ΛCDM reproduces the standard values (low-ℓ TT 23.5, low-ℓ EE 396.1).

## Result

Cosmological parameters are held fixed at the Planck 2018 best fit. Δχ² = χ²(model) − χ²(ΛCDM with the same masses).

| T_c | m_lightest (eV) | Δχ² total | low-ℓ TT | low-ℓ EE | plik-lite | TT ℓ=2 | TT ℓ=10 | A_φφ (8–400) | P(k=0.2) |
|---|---|---|---|---|---|---|---|---|---|
| 45 K | 0 | −1.31 | −0.94 | +0.03 | −0.40 | −30% | −6.3% | +0.09% | 0 |
| 60 K | 0 | −1.16 | −0.88 | +0.04 | −0.32 | −28% | −5.7% | +0.07% | 0 |
| 75 K | 0 | −1.06 | −0.84 | +0.04 | −0.26 | −26% | −5.2% | +0.05% | 0 |
| 45 K | 0.01 | −1.33 | −0.94 | +0.03 | −0.42 | −30% | −6.3% | +0.09% | 0 |
| 60 K | 0.01 | −1.16 | −0.88 | +0.04 | −0.33 | −28% | −5.7% | +0.07% | 0 |
| 75 K | 0.01 | −1.06 | −0.84 | +0.04 | −0.27 | −25% | −5.2% | +0.05% | 0 |
| 45 K | 0.03 | −1.46 | −0.93 | −0.01 | −0.53 | −30% | −6.3% | +0.09% | 0 |
| 60 K | 0.03 | −1.30 | −0.87 | −0.01 | −0.42 | −28% | −5.7% | +0.07% | 0 |
| 75 K | 0.03 | −1.17 | −0.83 | 0.00 | −0.34 | −26% | −5.2% | +0.06% | 0 |

**Reading.**

- **Small scales are untouched.** The created gas is hot and the ghost doesn't cluster. Lensing amplitude changes by under 0.1%, P(k) by under 10⁻⁴, and BAO and supernovae are identical to ΛCDM, since the background is exactly Λ.
- **The signature is at the largest angular scales.** The two large dark components (ghost ≈ −4 ρ_Λ, created gas ≈ +5 ρ_Λ) perturb differently near the horizon. That changes the late ISW effect and suppresses the low multipoles: about −28% at the quadrupole and −6% at ℓ = 10.
- **Planck fit.** The observed Planck quadrupole is low, so the data mildly prefer this, by Δχ² ≈ −1.1 to −1.5. That is within cosmic variance: the model is **allowed, not evidenced**.
- **m_lightest barely matters.** Creation goes as m², so the heaviest eigenstate (m₃ ≥ 50 meV) dominates.
- **Lower T_c means a stronger effect.**

## Self-consistent T_c (no free temperature)

A pure global state seen from inside is thermal, at a temperature set by how many modes share the energy (canonical
typicality). Requiring the ghost gas to be its own blackbody at T_c therefore fixes T_c from Λ. `tc_selfconsistent.py` is a
standalone check of the same solve.

| g (ghost dof) | m_lightest (eV) | T_c (solved) | Δχ² total | low-ℓ TT | low-ℓ EE | plik-lite | TT ℓ=2 | TT ℓ=10 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 47.55 K | −1.29 | −0.92 | +0.03 | −0.39 | −29.5% | −6.2% |
| 1 | 0.03 | 47.54 K | −1.43 | −0.91 | −0.01 | −0.51 | −29.3% | −6.2% |
| 2 | 0 | 39.64 K | −1.37 | −0.96 | +0.04 | −0.45 | −30.1% | −6.5% |
| 2 | 0.03 | 39.63 K | −1.55 | −0.96 | −0.01 | −0.59 | −29.9% | −6.5% |

The natural choice for a single scalar monopole mode is g = 1. It gives T_c = 47.5 K with no continuous parameter beyond
ΛCDM's. Only the lightest neutrino mass and the mass ordering remain, and the mass barely matters. In every case the lensing
amplitude changes by less than 0.1% and P(k) by less than 10⁻⁴. `grid_results_dof.json` holds these rows.

**Assumption:** the ghost gas is fully thermalized, with zero chemical potential.

## Cobaya best fit (g = 1, normal ordering, m_lightest = 0)

All six ΛCDM parameters plus A_planck are refit with Cobaya (bobyqa), against Planck 2018 low-ℓ TT + low-ℓ EE + plik-lite.

| | χ²_min | H0 | n_s | ln(10¹⁰A_s) |
|---|---|---|---|---|
| ΛCDM (same neutrino masses) | 1002.445 | 67.38 | 0.9649 | 3.0436 |
| Model, T_c = 47.55 K (solved, g = 1) | 1001.090 | 67.36 | 0.9644 | 3.0438 |

**Δχ²_min = −1.36, with zero added parameters.** The ΛCDM minimizer's own convergence is good to about 0.07: ΛCDM evaluated at
the model's best fit gives 1002.378. All best-fit shifts are tiny, for example ΔH0 = −0.02 and Δn_s = −0.0005, so the
large-angle suppression doesn't trade off against the standard parameters.

**Method.** One full-model call takes about 5 minutes, so the model is minimized by template iteration (`cobaya_fit.py`):

- At each step the model is approximated as ΛCDM(θ) × R_ℓ, where R_ℓ is the ratio computed with the full patched CLASS at
  the current best fit.
- The iteration converged at the first step: the second iteration reproduced the first exactly.
- **Validation.** At the final point, model − ΛCDM is −1.323 from the full Boltzmann calculation and −1.288 from the
  template, so they agree to 0.04. Each difference is taken within one pipeline, because the CLASS-binary pipeline and
  Cobaya's classy differ by a constant precision offset of about 1.5 in plik-lite. That offset cancels in model − ΛCDM.

`cobaya_results.json` holds the full numbers. To reproduce:

    PYTHONPATH=<class>/python/build/lib.* python3 cobaya_fit.py lcdm  --packages packages
    PYTHONPATH=<class>/python/build/lib.* python3 cobaya_fit.py model --packages packages --class-bin <class>/class --dof 1

Build the patched classy first with `cd <class> && make libclass.a && cd python && python3 setup.py build`. Then symlink
`<class>/external` into the build directory.

## Creation-slicing test

"Uniform creation" needs a choice of which hypersurfaces count as "everywhere at once". The test below takes g = 1, uses the
Cobaya ΛCDM best fit, and compares against ΛCDM at the same point (`slicing_test.py`, `slicing_results.json`).

| slicing | Δχ² | low-ℓ TT | plik-lite | TT ℓ=2 | ℓ=3 | ℓ=6 | ℓ=10 | ℓ=30 | φφ L=10 | φφ L=100 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0, CDM-comoving | −1.27 | −0.91 | −0.37 | −29.5% | −23.5% | −12.7% | −6.2% | −0.4% | +6.8% | +0.25% |
| 1, ghost-comoving | −1.25 | −0.90 | −0.37 | −28.3% | −22.8% | −12.5% | −6.1% | −0.4% | +6.6% | +0.24% |
| 2, coexistence-pinned | **+33.9** | +27.1 | +6.8 | **+81%** | +80% | +64% | +41% | +5.3% | −25% | −5.6% |

**Reading.**

- **Global creation is robust.** For either global slicing (0 and 1), the quadrupole suppression agrees to about 1 point
  (−29.5% vs −28.3%), and Δχ² agrees to 0.02. The large-angle signature therefore belongs to the model, not to the
  gauge choice.
- **Locally pinned creation is excluded.** If creation follows the local expansion so that the ghost is pinned at T_c
  locally, the low multipoles are *enhanced* instead, and Planck rejects that by Δχ² ≈ +34. Low-ℓ TT is insensitive to the
  standard parameters, so a refit cannot rescue it.
- **This is the differential rule-out.** Creation that responds to local volume is a local channel. The data exclude
  that, and they allow the global, no-information creation the model argues for.

Both new slicings conserve the total (ghost + created) energy to about 10⁻³ at k = 3×10⁻⁴ to 3×10⁻² Mpc⁻¹.

## Caveats

1. **Refit done.** The parameter refit is now done (see the section above). As expected, the parameters barely move.
2. **Slicing.** This is now tested; see the section above. Global slicings agree with each other, and the locally pinned variant is excluded.
3. **Ghost microphysics.** The ghost is treated as a w = c_s² = 1/3 fluid with no shear. A free-streaming ghost gas would differ at the largest scales.
4. **Missing likelihoods.** Planck lensing, DESI and Pantheon+ weren't available in the sandbox. DESI and supernovae are unaffected because the background is exact Λ. The lensing change is below 0.1%, against Planck's roughly 2.5% precision on the lensing amplitude.
5. **Scope of the checks.** Low-ℓ TT is cosmic-variance limited, and the low-quadrupole anomaly has well-known look-elsewhere issues.

## Run

    bash setup_and_run.sh                         # default grid
    bash setup_and_run.sh --Tc 30,40,50 --m1 0     # or any other grid
    bash setup_and_run.sh --ghost-dof 1 --m1 0     # self-consistent T_c
