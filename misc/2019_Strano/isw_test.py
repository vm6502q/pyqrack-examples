# ISW-galaxy cross-correlation test of the created-neutrino (Strano) model.  By Dan Strano and (Anthropic) Claude.
#
# What it does: runs the patched CLASS for LCDM and for the model at the SAME parameters (the Cobaya LCDM best fit),
# computes the CMB-temperature x galaxy-density cross spectra C_l^{Tg} for the redshift bins of published ISW analyses
# (plus T x CMB-lensing for Planck's kappa tracer), and turns them into the ISW amplitude A that each analysis would
# have measured IF the model were true (A = 1 is LCDM by construction). Then compares with the measured A's.
#
#   A_pred   = sum_l w_l (C_model/C_lcdm)  with inverse-variance weights w_l ~ (2l+1) C_lcdm^2 / (C^TT C^gg + C^Tg^2)
#   A_bias   = A_pred / sqrt(C^gg_model / C^gg_lcdm)   (galaxy bias is fit from the auto-spectrum with LCDM in these
#              papers, so a model change in clustering is partly absorbed into b; this is that correction)
#   dchi2    = [(A_pred - A_obs)^2 - (1 - A_obs)^2] / sigma^2     (model minus LCDM, per measurement)
#
# CLASS has C_l^{Td} switched off, so the galaxy cross spectra are computed here with the Limber approximation from
# CLASS's own transfer functions (phi + psi and d_m on a grid of redshifts). Limber is crude at l < 10, but the SAME
# calculation is used for both models, and only the model/LCDM ratio enters A. The CMB-lensing tracer uses CLASS's
# full C_l^{T phi}.
# Approximations (stated, not hidden): top-hat bins with uniform dN/dz inside each photo-z bin; an approximate NVSS
# dN/dz (gamma distribution, mean z ~ 1); Gaussian cosmic variance without shot noise; l = 4..100; galaxy bias 1 in
# both models (it cancels in A_pred). The measurements are not independent (same CMB sky), so they are reported
# separately and NOT summed.
# Usage: python3 isw_test.py CLASS_BINARY [--dof 1] [--m1 0] [--reuse]
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cobaya_fit as F                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("class_bin")
ap.add_argument("--dof", type=float, default=1.0)
ap.add_argument("--m1", type=float, default=0.0)
ap.add_argument("--reuse", action="store_true", help="reuse existing CLASS outputs")
a = ap.parse_args()
best = json.load(open(os.path.join(F.OUT, "results.json")))[f"lcdm_m1_{a.m1:g}"]["best"]
OUTD = os.path.join(F.OUT, "isw")
os.makedirs(OUTD, exist_ok=True)

# Photo-z bins of the published analyses (z_lo, z_hi), by survey
SURVEYS = {
    "2MPZ":      [(0.0, 0.105), (0.105, 0.195), (0.195, 0.3)],
    "WIxSC":     [(0.0, 0.09), (0.09, 0.21), (0.21, 0.3), (0.3, 0.6)],
    "SDSS DR12": [(0.0, 0.1), (0.1, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0)],
    "DR6 QSO":   [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0)],
    "DESI LS":   [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8)],
}
BINS = sorted({b for v in SURVEYS.values() for b in v}, key=lambda b: (0.5*(b[0]+b[1]), b[1]-b[0]))


ZPK = np.unique(np.r_[np.linspace(0.0, 1.0, 51), np.linspace(1.04, 3.0, 35), np.linspace(3.2, 5.0, 10)])[:100]


def ini(path, root, created):
    lines = [f"ln10^{{10}}A_s = {best['logA']:.8g}", f"n_s = {best['n_s']:.8g}", f"100*theta_s = {best['theta_s_100']:.8g}",
             f"omega_b = {best['omega_b']:.8g}", f"omega_cdm = {best['omega_cdm']:.8g}", f"tau_reio = {best['tau_reio']:.8g}"]
    lines += [f"{k} = {v}" for k, v in F.nu_args(a.m1).items()]
    lines += ["output = tCl, lCl, mTk", "lensing = no", "l_max_scalars = 200", "P_k_max_1/Mpc = 3",
              "extra_metric_transfer_functions = yes", "write_background = yes",
              "z_pk = " + ",".join(f"{z:.4f}" for z in ZPK), f"root = {root}_"]
    if created:
        lines += ["created_neutrinos = yes", f"created_ghost_dof = {a.dof}", "created_q_per_decade = 6",
                  "created_l_max = 8", "created_k_max = 0.05", "created_z_on = 100"]
    open(path, "w").write("\n".join(lines) + "\n")


def table(path):
    hdr = [l for l in open(path) if l.startswith("#")][-1]
    names = [h.split(":", 1)[1].strip() for h in hdr.strip("# \n").split("  ") if ":" in h]
    d = np.loadtxt(path)
    return {n: d[:, i] for i, n in enumerate(names)}


def run(tag, created):
    root = os.path.join(OUTD, tag)
    if not (a.reuse and os.path.exists(root + "_00_cl.dat")):
        ini(root + ".ini", root, created)
        p = subprocess.run([a.class_bin, root + ".ini"], capture_output=True, text=True)
        if p.returncode != 0 or "Error" in p.stdout:
            raise RuntimeError(p.stdout[-800:] + p.stderr[-400:])
    bg = table(root + "_00_background.dat")
    tks = [table(root + f"_00_z{i+1}_tk.dat") for i in range(len(ZPK))]
    return {"cl": table(root + "_00_cl.dat"), "bg": bg, "tk": tks}


print(f"Running CLASS (LCDM, then model; the model run takes minutes)...", flush=True)
L, M = run("isw_lcdm", False), run("isw_model", True)
print("  done", flush=True)

from scipy.interpolate import CubicSpline, interp1d
TCMB = 2.7255e6                                         # muK
ELL = np.arange(4, 101)


class Limber:
    """Limber C_l^{Tg} (ISW x galaxy density) and C_l^{gg} from CLASS transfer functions (CLASS conventions:
    ISW source d(phi+psi)/d(conformal time); galaxies trace the synchronous-gauge matter density d_m)."""
    def __init__(self, R):
        bg = R["bg"]
        zb = bg["z"][::-1]
        self.chi = interp1d(zb, bg["comov. dist."][::-1])
        self.eta = interp1d(zb, bg["conf. time [Mpc]"][::-1])
        self.H = interp1d(zb, bg["H [1/Mpc]"][::-1])
        h = bg["H [1/Mpc]"][-1]*299792.458/100
        k = R["tk"][0]["k (h/Mpc)"]*h
        S = np.array([t["phi"] + t["psi"] for t in R["tk"]])          # [z, k]
        D = np.array([t["d_m"] for t in R["tk"]])
        eta = self.eta(ZPK)
        dS = CubicSpline(eta[::-1], S[::-1], axis=0)(eta, 1)           # d(phi+psi)/d eta (eta falls as z rises)
        As, ns = np.exp(best["logA"])*1e-10, best["n_s"]
        P = 2*np.pi**2/k**3*As*(k/0.05)**(ns - 1)
        lk = np.log(k)
        self.PSd = [interp1d(lk, P*dS[i]*D[i], bounds_error=False, fill_value=0.0) for i in range(len(ZPK))]
        self.Pdd = [interp1d(lk, P*D[i]*D[i], bounds_error=False, fill_value=0.0) for i in range(len(ZPK))]

    def spectra(self, nz):
        """nz: normalized dN/dz on ZPK. Returns C_l^{Tg} [muK], C_l^{gg} for ELL."""
        z = ZPK[1:]                                                    # skip z = 0 (chi = 0)
        chi, Hz, n = self.chi(z), self.H(z), nz[1:]
        Tg, gg = np.zeros(len(ELL)), np.zeros(len(ELL))
        for j, l in enumerate(ELL):
            lk = np.log((l + 0.5)/chi)
            fT = np.array([self.PSd[i + 1](lk[i]) for i in range(len(z))])
            fg = np.array([self.Pdd[i + 1](lk[i]) for i in range(len(z))])
            Tg[j] = TCMB*np.trapezoid(n*fT/chi**2, z)
            gg[j] = np.trapezoid(n**2*Hz*fg/chi**2, z)
        return Tg, gg


LL, LM = Limber(L), Limber(M)


def tophat(lo, hi):
    zf = np.linspace(0, 5, 5001)
    w = ((zf >= lo) & (zf <= hi)).astype(float)
    w = np.convolve(w, np.ones(21)/21, mode="same")                    # soften the edges slightly
    n = np.interp(ZPK, zf, w)
    return n/np.trapezoid(n, ZPK)


def amp_from(nz):
    Tl, gl = LL.spectra(nz)
    Tm, gm = LM.spectra(nz)
    TT = L["cl"]["TT"][np.searchsorted(L["cl"]["l"], ELL)]
    TT = TT*2*np.pi/(ELL*(ELL + 1))*TCMB**2                           # cl.dat stores l(l+1)C_l/2pi, dimensionless
    w = (2*ELL + 1)*Tl**2/(TT*gl + Tl**2)
    A = np.sum(w*Tm/Tl)/np.sum(w)
    rgg = np.sum(w*gm/gl)/np.sum(w)
    return A, A/np.sqrt(rgg), np.sum(w), Tm/Tl


per_bin = {b: amp_from(tophat(*b)) for b in BINS}
nv_n = ZPK**1.3*np.exp(-1.3*ZPK/0.8)
nv = amp_from(nv_n/np.trapezoid(nv_n, ZPK))
s = (L["cl"]["l"] >= 8) & (L["cl"]["l"] <= 100)
lk_, TPl, TPm, PPl, PPm = L["cl"]["l"][s], L["cl"]["TPhi"][s], M["cl"]["TPhi"][s], L["cl"]["phiphi"][s], M["cl"]["phiphi"][s]
wk = (2*lk_ + 1)*TPl**2/(L["cl"]["TT"][s]*PPl + TPl**2)
kp = (np.sum(wk*TPm/TPl)/np.sum(wk), np.sum(wk*TPm/TPl)/np.sum(wk)/np.sqrt(np.sum(wk*PPm/PPl)/np.sum(wk)), np.sum(wk))

print("\nPredicted ISW amplitude if the model is true (LCDM = 1), per tracer, l = 4..100:")
print(f"   {'tracer':12s} {'bins':>5s} {'A_pred':>8s} {'A_bias-corr':>12s}")
surv_A = {}
for name, bins in SURVEYS.items():
    w = np.array([per_bin[b][2] for b in bins])
    A = np.sum(w*np.array([per_bin[b][0] for b in bins]))/w.sum()
    Ab = np.sum(w*np.array([per_bin[b][1] for b in bins]))/w.sum()
    surv_A[name] = (A, Ab)
    print(f"   {name:12s} {len(bins):5d} {A:8.4f} {Ab:12.4f}")
surv_A["NVSS"] = nv[:2]
surv_A["CMB lensing"] = kp[:2]
print(f"   {'NVSS':12s} {'1':>5s} {nv[0]:8.4f} {nv[1]:12.4f}   (approximate dN/dz)")
print(f"   {'CMB lensing':12s} {'-':>5s} {kp[0]:8.4f} {kp[1]:12.4f}   (CLASS T x phi, full calculation, l = 8..100)")
print("   per bin (z range: A_pred):", "  ".join(f"{lo:g}-{hi:g}: {per_bin[(lo, hi)][0]:.4f}" for lo, hi in BINS))
r = per_bin[(0.3, 0.6)][3]
print("   scale dependence, WIxSC 0.3-0.6 bin, C_l^Tg model/LCDM at l = 4, 10, 30, 100:",
      " ".join(f"{r[l - 4]:.4f}" for l in (4, 10, 30, 100)))

# Published measurements (A relative to the authors' fiducial LCDM)
MEAS = [
    ("Planck 2015 XXI, all tracers incl. kappa", 1.00, 0.25,
     {"NVSS": 0.36, "SDSS DR12": 0.56, "CMB lensing": 0.33, "WIxSC": 0.53}),
    ("Planck 2015 XXI, NVSS + WISE + SDSS", 0.89, 0.31, {"NVSS": 0.36, "SDSS DR12": 0.56, "WIxSC": 0.53}),
    ("Stolzner et al. 2018, five catalogues (binned)", 1.51, 0.30,
     {"2MPZ": 1.07, "WIxSC": 0.56, "SDSS DR12": 0.57, "DR6 QSO": 1.13, "NVSS": 0.57}),
    ("Hang et al. 2021, DESI Legacy Survey", 0.984, 0.349, {"DESI LS": 0.349}),
]
print("\nComparison with measurements (dchi2 = model - LCDM; measurements share the CMB sky, so NOT additive):")
print(f"   {'measurement':48s} {'A_obs':>12s} {'A_model':>8s} {'dchi2':>7s} {'A_bias':>7s} {'dchi2':>7s}")
out = {"per_bin": {f"{lo}-{hi}": per_bin[(lo, hi)][:2] for lo, hi in BINS}, "tracers": surv_A, "measurements": []}
for name, Aobs, sig, comp in MEAS:
    w = {k: 1/s**2 for k, s in comp.items()}                       # weight tracers by their own errors
    Am = sum(w[k]*surv_A[k][0] for k in w)/sum(w.values())
    Ab = sum(w[k]*surv_A[k][1] for k in w)/sum(w.values())
    d = lambda A: ((A - Aobs)**2 - (1 - Aobs)**2)/sig**2
    print(f"   {name:48s} {Aobs:5.2f} ± {sig:4.2f} {Am:8.4f} {d(Am):+7.3f} {Ab:7.4f} {d(Ab):+7.3f}")
    out["measurements"].append({"name": name, "A_obs": Aobs, "sigma": sig, "A_model": Am, "dchi2": d(Am),
                                "A_model_bias": Ab, "dchi2_bias": d(Ab)})
json.dump(out, open(os.path.join(F.OUT, "isw_results.json"), "w"), indent=1, default=float)
print(f"\nSaved {os.path.join(F.OUT, 'isw_results.json')}")
