# Fitting the Strano black-hole emission law against cosmological data

**Model.** A fraction `f_pbh` of the dark matter starts out as primordial black holes, all of one mass.
Each one radiates P = E_P/(2 r_s) per Planck time into invisible scalar/graviton radiation. So:

- M(t) = M₀ √(1 − t/t_evap)
- t_evap = 2G²M₀²/(c⁵ℓ_P)
- the energy lost becomes dark radiation.

The patched CLASS (`pbh_class.patch`, against CLASS v3.4.1, commit 814ce59) follows this law exactly:

- The black-hole density is the CDM-like density times √(1 − t/t_evap). It is computed analytically, so the integration isn't stiff.
- Dark radiation is sourced at Γ(t) = 1/(2(t_evap − t)).
- The same time-dependent rate is used everywhere in the perturbations that CLASS's exponential `dcdm` used a constant Γ.

To turn it on, set `pbh_evaporation_law = yes` and `tau_dcdm = t_evap` in seconds.

**Question answered.** For each evaporation time t_evap, the fit finds the best `f_pbh ≥ 0` together with all the
ΛCDM and nuisance parameters, then reports Δχ² = χ²(PBH) − χ²(ΛCDM).

- Because `f_pbh = 0` *is* ΛCDM, Δχ² ≤ 0 always.
- With 2 extra parameters, Δχ² ≳ −6 means no meaningful improvement.

**Data** (default `--data all`):

- Planck 2018: TT/TE/EE lite, low-ℓ TT, low-ℓ EE, lensing
- DESI DR2 BAO
- Pantheon+ supernovae

`--data planck` runs on CMB only.

## Run

    bash setup_and_run.sh 2        # 2 profile points in parallel; OMP threads split automatically

Or step by step, inside an environment where the patched `classy` is importable:

    python3 pbh_fit.py install
    python3 pbh_fit.py lcdm
    python3 pbh_fit.py profile --jobs 2
    python3 pbh_fit.py report
    python3 pbh_fit.py free        # optional: t_evap free as well

Add `--test` to any step to check the pipeline with a single likelihood evaluation, which takes seconds.

## Checks done before delivery

- **Background:** ρ_PBH a³ follows √(1 − t/t_evap) to 4–5 digits for t_evap = 5 Gyr and 30 Gyr, and the
  exponential `dcdm` mode is unchanged.
- **Energy:** the energy in dark radiation matches the integrated black-hole mass loss to 0.1% in the
  evaporate-early case. The residual comes from the √(pbh_cutoff) cap, with a default cutoff of 1e-6.
- **CMB spectra:** no NaNs. Evaporation before recombination with f = 0.1 distorts the spectra by about 10%,
  as expected for that much extra radiation.
- **Likelihoods:** the Planck CMB likelihoods give the standard χ² at a Planck-like point:
  - lite TTTEEE: 583.9
  - low-ℓ TT: 23.5
  - low-ℓ EE: 396.0
- **End-to-end runs (Planck CMB only, in the sandbox):**
  - ΛCDM best fit: χ² = 1002.59, H0 = 67.39.
  - One profile point at t_evap = 10 Gyr: best f_pbh = 0.0000, Δχ² = −0.01.
  - Profiles start from the ΛCDM best fit with f_pbh = 0, so each profile point can only match or improve on ΛCDM.
- **Not checked:** the DESI, Pantheon+ and Planck-lensing downloads were blocked here. They install normally with
  `cobaya-install`.

## Reading the result

- **t_evap before recombination (≲ 10¹³ s):** f_pbh acts as extra radiation (ΔN_eff). Expect a tight upper limit
  on f and Δχ² ≈ 0.
- **t_evap of a few to tens of Gyr:** dark matter turns into dark radiation at late times. This is the only window
  where the fit could pull toward a higher H0 or a lower S8, and it is where to look for a negative Δχ².
- **Mass:** the black-hole mass that evaporates at each t_evap is printed in the report.
  For example, t_evap = 13.8 Gyr corresponds to about 4×10²² kg.

## Added channels

### Rest-mass channel (`meprofile`, `me`)

The rest-mass excitation is modeled as the electron mass at recombination relative to today, using CLASS's
`varying_me` with a step at z = 50. ω_b and ω_cdm are fitted freely, so this stands in for a near-uniform shift
of all rest masses relative to the Planck mass. The free fit (`me`) starts from the best `meprofile` point.
The report flags the case where the profile still beats the free fit.

### Dilaton channel (`bdprofile`)

The trace (spin-0) part of the rank-2 potential, coupled to rest mass, is a Brans–Dicke scalar with matter coupling
α² = 1/(2ω_BD + 3). This is the continuous version of the rest-mass channel: rest masses relative to the Planck
mass drift over cosmic history instead of jumping at one redshift. G_eff for matter today is pinned to the local
G (`normalize_G_NR`).

- **Code:** hi_class (Zumalacárregui, Bellini, Sawicki, Lesgourgues & Ferreira 2017; Bellini et al. 2020),
  wrapped for Cobaya by `hiclass_theory.py`.
- **Grid:** default `--bdgrid` is 0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2.
- **Comparison:** Δχ² is taken against hi_class's own α² = 0 run, which is ω_BD = 10⁸. A code-to-code offset of
  about 0.1 in χ² against patched CLASS cancels this way.
- **Scale:** Cassini allows α² ≲ 1.2×10⁻⁵ in the solar system unless the field is screened. Cosmology on its own
  constrains α² much more weakly.

### Not fitted, and why

- **False-vacuum dark energy:** w = −1 exactly, which is ΛCDM.
- **Oscillating-scalar dark matter:** needs a dedicated code such as axionCAMB, and the clock searches are the
  sharper test.
- **The axion's cosmic birefringence:** needs the Planck EB likelihood, which Cobaya doesn't ship.

### Melting false vacuum (`ivacprofile`, `ivac`), which replaces Λ

A false-vacuum energy ρ_V converts into scalar gas (dark radiation) at the rate Q = ξHρ_V. Since early times that
gives:

- ρ_V ∝ a^(−ξ);
- ρ_gas = ξ/(4 − ξ)·ρ_V exactly;
- the vacuum plus gas together behave exactly like constant-w dark energy, with w = −1 + ξ/3.

**Implementation.** It runs on CLASS's built-in dark-energy fluid, in PPF form, with Ω_Λ = 0. The PPF treatment is
exact at ξ = 0 and treats the dark sector as smooth on small scales.

**Checks.**

- At ξ = 0 the χ² is identical to ΛCDM (1002.70 on Planck).
- The gas-to-vacuum ratio and w match the analytic values to machine precision.
- On Planck alone, ξ = 0.1 gives Δχ² = +0.08 and H0 = 66.6, with the gas today at 13.6 K.

**Output.** The report also lists today's gas temperature, for one bosonic degree of freedom.

**Default grid.** ξ = 0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45.

### Ice-water coexistence (`iceprofile`, `ice`), which replaces Λ

This is Dan Strano's picture: the scalar gas sits at a fixed coexistence temperature T_c.

- **Physics.** With T_c fixed, ρ_gas stays constant, and the vacuum supplies exactly the latent heat
  Q = 4Hρ_gas that expansion removes.
- **Result.** ρ_V = ρ_V0 (1 − 4r ln a), with r = ρ_gas/ρ_V0, and the dark sector has
  1 + w(a) = (4r/3)/(1 + r − 4r ln a). Today w₀ ≈ −1 + 4r/3, and w relaxes logarithmically back toward −1 in the
  past (thawing).
- **Parameters.** One: r, which is equivalent to T_c.
- **Implementation.** CPL, with w₀ = −1 + (4r/3)/(1 + r) and w_a = −(16r²/3)/(1 + r)². For r = 0.05 this
  matches the exact w(a) to within 0.004 out to z = 3.
- **Consistency requirement.** Coexistence must have held since z ≳ 15. Otherwise the gas was hotter earlier and
  would add radiation at recombination.
- **Checks.** At r = 0 the χ² is identical to ΛCDM (1002.70 on Planck).
- **Report.** Δχ² is taken relative to this channel's own r = 0 run, which removes the ±2 minimizer offset seen in
  the ivac run. T_c is printed for one bosonic degree of freedom.
