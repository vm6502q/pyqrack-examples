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
| `grid_results.json` | The results below. |

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

## Caveats

1. **Fixed parameters.** ΛCDM parameters weren't refit. Because the change sits at ℓ ≲ 20 (late ISW), refitting should move them little. A Cobaya minimization with a classy build of this patch would settle it; it costs about 5 min per evaluation.
2. **Slicing.** "Global creation" means a uniform rate on CDM-comoving (synchronous) slices. That is a physical choice of cosmic time. A "local" version, where creation tracks local expansion and the ghost is pinned at T_c locally, is not implemented and could change the large-angle signal.
3. **Ghost microphysics.** The ghost is treated as a w = c_s² = 1/3 fluid with no shear. A free-streaming ghost gas would differ at the largest scales.
4. **Missing likelihoods.** Planck lensing, DESI and Pantheon+ weren't available in the sandbox. DESI and supernovae are unaffected because the background is exact Λ. The lensing change is below 0.1%, against Planck's roughly 2.5% precision on the lensing amplitude.
5. **Scope of the checks.** Low-ℓ TT is cosmic-variance limited, and the low-quadrupole anomaly has well-known look-elsewhere issues.

## Run

    bash setup_and_run.sh                         # default grid
    bash setup_and_run.sh --Tc 30,40,50 --m1 0     # or any other grid
