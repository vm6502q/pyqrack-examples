# Planck 2018 lensing test — laptop kit

By (Anthropic) Claude and Dan Strano.

## Run it

Put these in one folder, along with `created_nu_class.patch`, `planck_chi2.py` and `cobaya_results.json`:

- `run_lensing.sh`
- `lensing_test.py`
- `cobaya_fit.py`

Run:
    fix_class_public.sh

Then:

    bash run_lensing.sh quick     # tens of minutes: LCDM and the model at the existing best fits, + lensing
    bash run_lensing.sh refit     # hours: re-minimize both with lensing in the fit (template iteration)

**What the script sets up.** It builds the patched CLASS (binary + Python wrapper) and installs Cobaya with these Planck
2018 likelihoods:

- low-ℓ TT
- low-ℓ EE
- plik-lite
- **lensing (native)**

**What `quick` reports.** For both lensing likelihoods (the native T+P one and the CMB-marginalized one), it prints χ² per
likelihood and Δχ² for:

- the model at the ΛCDM best fit (the pure model effect);
- the model at its own best fit.

It also prints the model/ΛCDM ratio of C_L^φφ in the nine Planck lensing bands.

**Check this first.** In the log, the line "`classy` module loaded successfully from …/class_public/build/lib.*/classy"
must point at the *patched* build. Cobaya will otherwise quietly load any classy already installed on the system, and
that one has no `created_*` parameters, so it would fail or silently compute plain ΛCDM.

## Verified here (without the lensing data, which I couldn't download)

The full patched model, run through Cobaya with no template, gives:

| | low-ℓ TT | low-ℓ EE | plik-lite | total |
|---|---|---|---|---|
| ΛCDM at ΛCDM best fit | 23.459 | 396.169 | 582.816 | 1002.445 |
| model at ΛCDM best fit | 22.548 | 396.172 | 582.484 | 1001.204 (Δ −1.241) |
| model at model best fit | 22.621 | 396.166 | 582.306 | 1001.092 (**Δ −1.353**) |

That matches the template-iteration result (−1.355). The predicted lensing change, at the same cosmological parameters:

| L band | 8–40 | 41–84 | 85–129 | 130–174 | 175–219 | ≥ 220 |
|---|---|---|---|---|---|---|
| ΔC_L^φφ / C_L^φφ | +5.8% | +0.8% | +0.25% | +0.11% | +0.06% | < 0.03% |

**What to expect.** Essentially all of the lensing signal is in the first Planck band, L = 8–40. That bin's error is
several percent, so the lensing likelihood should move Δχ² by a fraction of a unit. The sign depends on whether Planck's
first bin sits high or low relative to ΛCDM, so it is a genuine test.

## Notes

- **Speed.** One model evaluation took 3.5 min on 2 sandbox cores. ΛCDM takes about 12 s.
- **`--marged`.** This uses the CMB-marginalized lensing likelihood on its own: lensing only, no TT/EE.
- **Refit validation.** `cobaya_fit.py --lensing` validates the template against the full model without the lensing
  term, because the file pipeline has no φφ likelihood. The template's φφ ratio comes from the same full-model run.
