# "Dark-matter ice melting in a bath of dark-energy water": a quantitative, charitable version of Dan Strano's 2019
# heuristic, fit to Planck 2018 CMB distance priors + DESI DR2 BAO + Pantheon+ supernovae, against LCDM and w0waCDM.
# Heuristic: Dan Strano (2019). Quantification and fit: (Anthropic) Claude, 2026.
#
# MODEL (background):
#   Dark energy "water" rho_x with its own (microscopic) equation of state w_x  (-1 < w_x < -1/3, a free parameter).
#   Dark matter "ice" rho_c melts into it at rate Q:
#       d rho_c/dt = -3 H rho_c - Q,      d rho_x/dt = -3 H (1 + w_x) rho_x + Q
#   Phase coexistence: while there is enough ice, melting exactly offsets the water's dilution, so rho_x stays constant
#   (the "temperature" of the bath is pinned, as at a melting point):
#       Q = 3 H (1 + w_x) rho_x * F,      F = 1 / (1 + (r_star / r)^n),   r = rho_c / rho_x
#   F -> 1 while r >> r_star (coexistence: looks exactly like Lambda); F -> 0 once the ice runs low (the bath cools,
#   rho_x dilutes as a^{-3(1+w_x)}).  r_star is the "melting point" ice/water ratio; n the sharpness (fixed; varied
#   as a check).  w_x -> -1 or r_star -> 0 recovers LCDM.  Extra parameters vs LCDM: w_x, r_star (same count as w0wa).
#   Early universe: r is huge, F = 1 and rho_x is constant, so pre-recombination physics is that of LCDM, and the
#   Planck distance priors apply.
#
# DATA:
#   Planck 2018 TT,TE,EE+lowE distance priors (R, l_A, omega_b) with correlations: Chen, Huang & Wang 2019
#     (arXiv:1808.05724, LCDM table).  DESI DR2 BAO (CobayaSampler/bao_data, desi_bao_dr2, ALL_GCcomb).
#   Pantheon+ (PantheonPlusSH0ES/DataRelease, STAT+SYS covariance, zHD > 0.01, absolute magnitude marginalized).
# LIMITS: background only (no perturbations: ISW, lensing and growth are not in this fit); radiation with N_eff = 3.044
#   two massless + one 0.06 eV neutrino (smooth interpolation); z_* from Hu-Sugiyama; r_d from Aubourg et al. 2015.
import os, sys, json
import numpy as np
from scipy.integrate import solve_ivp, quad, cumulative_trapezoid
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
c_km = 299792.458
NEFF, TCMB = 3.044, 2.7255
og = 2.4728e-5*(TCMB/2.7255)**4                              # omega_gamma = Omega_gamma h^2
ONU = 0.06/93.14                                             # one massive neutrino, sum m_nu = 0.06 eV (Planck baseline)
orad = og*(1 + 0.2271*NEFF*2/3)                              # photons + two massless neutrino species
onu_rel = og*0.2271*NEFF/3                                   # the massive species while relativistic


def onu(a):
    """massive-neutrino density, smooth relativistic -> non-relativistic interpolation (approximation)."""
    return np.sqrt((onu_rel*a**-4)**2 + (ONU*a**-3)**2)
N_SHARP = float(os.environ.get("NSHARP", 4))

# ---------------- data ----------------
bm = [l.split() for l in open(os.path.join(D, "bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt"))
      if not l.startswith("#")]
BAO_Z, BAO_V, BAO_Q = np.array([float(r[0]) for r in bm]), np.array([float(r[1]) for r in bm]), [r[2] for r in bm]
BAO_ICOV = np.linalg.inv(np.loadtxt(os.path.join(D, "bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt")))

P = os.path.join(D, "pantheon/Pantheon+_Data/4_DISTANCES_AND_COVAR")
hdr = open(os.path.join(P, "Pantheon+SH0ES.dat")).readline().split()
sn = np.genfromtxt(os.path.join(P, "Pantheon+SH0ES.dat"), names=hdr, skip_header=1, dtype=None, encoding=None)
cov_flat = np.loadtxt(os.path.join(P, "Pantheon+SH0ES_STAT+SYS.cov"))
nsn = int(cov_flat[0])
SN_COV = cov_flat[1:].reshape(nsn, nsn)
keep = sn["zHD"] > 0.01
SN_ZHD, SN_ZHEL, SN_M = sn["zHD"][keep], sn["zHEL"][keep], sn["m_b_corr"][keep]
SN_ICOV = np.linalg.inv(SN_COV[np.ix_(keep, keep)])
SN_ONE = np.ones(keep.sum())
SN_A = SN_ONE @ SN_ICOV @ SN_ONE

CMB_MEAN = np.array([1.7502, 301.471, 0.02236])
CMB_SIG = np.array([0.0046, 0.0895, 0.00015])
CMB_CORR = np.array([[1.0, 0.46, -0.66], [0.46, 1.0, -0.33], [-0.66, -0.33, 1.0]])
CMB_ICOV = np.linalg.inv(CMB_CORR*np.outer(CMB_SIG, CMB_SIG))


# ---------------- background ----------------
def zstar(ob, om):
    g1 = 0.0783*ob**-0.238/(1 + 39.5*ob**0.763)
    g2 = 0.560/(1 + 21.1*ob**1.81)
    return 1048*(1 + 0.00124*ob**-0.738)*(1 + g1*om**g2)


def zdrag(ob, om):
    b1 = 0.313*om**-0.419*(1 + 0.607*om**0.674)
    b2 = 0.238*om**0.223
    return 1291*om**0.251/(1 + 0.659*om**0.828)*(1 + b1*ob**b2)


class Cosmo:
    """Flat background. kind: 'lcdm' | 'w0wa' | 'ice'.  Densities in omega = Omega h^2 units."""
    def __init__(self, kind, h, ob, oc, p1=0.0, p2=0.0):
        self.h, self.ob, self.oc0 = h, ob, oc
        ox0 = h**2 - ob - oc - orad - onu(1.0)               # flatness today
        self.ox0 = ox0
        lna = np.linspace(0.0, np.log(1/1.5e3) - 1.0, 4000)   # today -> z ~ 4000 (backwards)
        if kind == "lcdm":
            oc_a, ox_a = oc*np.exp(-3*lna), ox0*np.ones_like(lna)
        elif kind == "w0wa":
            w0, wa = p1, p2
            a = np.exp(lna)
            ox_a = ox0*a**(-3*(1 + w0 + wa))*np.exp(-3*wa*(1 - a))
            oc_a = oc*np.exp(-3*lna)
        else:                                                # melting ice
            wx, rs = p1, p2
            def rhs(x, y):
                c_, x_ = y
                r = max(c_, 1e-30)/x_
                F = 1.0/(1.0 + (rs/r)**N_SHARP) if rs > 0 else 1.0
                q = 3*(1 + wx)*x_*F                          # Q / H
                return [-3*c_ - q, -3*(1 + wx)*x_ + q]
            s = solve_ivp(rhs, (lna[0], lna[-1]), [oc, ox0], t_eval=lna, rtol=1e-9, atol=1e-14, method="LSODA")
            oc_a, ox_a = s.y
        a = np.exp(lna)
        self.lna, self.oc_a, self.ox_a = lna, oc_a, ox_a
        E2 = (ob*a**-3 + oc_a + orad*a**-4 + onu(a) + ox_a)
        self.zg = 1/a - 1
        self.Hg = 100*np.sqrt(E2)                            # km/s/Mpc
        # comoving distance on the grid (z increasing)
        zz, HH = self.zg, self.Hg
        self.Dc = cumulative_trapezoid(c_km/HH, zz, initial=0.0)
        self.valid = np.all(np.isfinite(E2)) and np.all(E2 > 0) and np.all(oc_a > -1e-12)

    def H(self, z):
        return np.interp(z, self.zg, self.Hg)

    def DM(self, z):
        return np.interp(z, self.zg, self.Dc)

    def H_early(self, z):                                    # beyond the grid: matter + radiation + (constant) x
        oc_hi = self.oc_a[-1]*((1 + z)/(1 + self.zg[-1]))**3
        return 100*np.sqrt(self.ob*(1 + z)**3 + oc_hi + orad*(1 + z)**4 + onu(1/(1 + z)) + self.ox_a[-1])

    def DM_star(self, zs):
        zmax = self.zg[-1]
        tail = quad(lambda z: c_km/self.H_early(z), zmax, zs, limit=200)[0]
        return self.Dc[-1] + tail

    def rs(self, zx):
        R = lambda z: 3*self.ob/(4*og)/(1 + z)
        f = lambda z: c_km/np.sqrt(3*(1 + R(z)))/self.H_early(z) if z > self.zg[-1] else \
            c_km/np.sqrt(3*(1 + R(z)))/self.H(z)
        return quad(f, zx, 1e7, limit=400, points=[self.zg[-1]])[0]

    def om_early(self):                                      # matter density relevant at recombination
        return self.ob + ONU + self.oc_a[-1]*((1 + self.zg[-1]))**-3   # early omega_c scaled to a=1, + massive nu


def chi2_parts(C):
    ob, h = C.ob, C.h
    # CMB distance priors (early matter density = baryons + the early-time CDM normalization)
    om_e = C.om_early()
    zs = zstar(ob, om_e)
    Dst = C.DM_star(zs)
    R = np.sqrt(om_e)*100*Dst/c_km
    lA = np.pi*Dst/C.rs(zs)
    d = np.array([R, lA, ob]) - CMB_MEAN
    cmb = d @ CMB_ICOV @ d
    # BAO
    # r_d: Aubourg et al. 2015 (eq. 16) fit to CAMB, accurate to ~0.02% (the Eisenstein-Hu z_d formula biases r_d high)
    rd = 55.154*np.exp(-72.3*(ONU + 0.0006)**2)/((om_e - ONU)**0.25351*ob**0.12807)
    th = []
    for z, q in zip(BAO_Z, BAO_Q):
        DMz, DHz = C.DM(z), c_km/C.H(z)
        th.append({"DM_over_rs": DMz/rd, "DH_over_rs": DHz/rd, "DV_over_rs": (z*DMz**2*DHz)**(1/3)/rd}[q])
    d = np.array(th) - BAO_V
    bao = d @ BAO_ICOV @ d
    # SN, absolute magnitude marginalized analytically
    mu = 5*np.log10((1 + SN_ZHEL)*C.DM(SN_ZHD)) + 25
    r = SN_M - mu
    B = SN_ONE @ SN_ICOV @ r
    snc = r @ SN_ICOV @ r - B**2/SN_A + np.log(SN_A/(2*np.pi))
    return {"CMB": cmb, "BAO": bao, "SN": snc, "rd": rd, "R": R, "lA": lA}


def total(kind, x):
    h, ob, oc = x[:3]
    p = list(x[3:]) + [0.0]*(2 - len(x[3:]))
    if not (0.5 < h < 0.9 and 0.018 < ob < 0.027 and 0.08 < oc < 0.16):
        return 1e10
    if kind == "ice" and not (-1.0 < p[0] < -1/3 + 1e-9 and 0.0 <= p[1] < 50):
        return 1e10
    if kind == "w0wa" and not (-3 < p[0] < 0.5 and -5 < p[1] < 3 and p[0] + p[1] < 0):
        return 1e10
    try:
        C = Cosmo(kind, h, ob, oc, *p)
        if not C.valid:
            return 1e10
        cp = chi2_parts(C)
    except Exception:
        return 1e10
    return cp["CMB"] + cp["BAO"] + cp["SN"]


def fit(kind, starts):
    best = None
    for x0 in starts:
        r = minimize(lambda x: total(kind, x), x0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-5, "maxiter": 20000, "maxfev": 20000, "adaptive": True})
        r = minimize(lambda x: total(kind, x), r.x, method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-6, "maxiter": 20000, "maxfev": 20000, "adaptive": True})
        if best is None or r.fun < best.fun:
            best = r
    return best


if __name__ == "__main__":
    base = [0.68, 0.02236, 0.120]
    out = {}
    L = fit("lcdm", [base])
    W = fit("w0wa", [base + [-0.8, -0.6], base + [-1.0, 0.0], base + [-0.6, -1.2]])
    starts_ice = [base + [wx, rs] for wx in (-0.9, -0.7, -0.5) for rs in (0.1, 0.4, 1.0, 3.0)]
    I = fit("ice", starts_ice)
    names = {"lcdm": ["h", "omega_b", "omega_c"], "w0wa": ["h", "omega_b", "omega_c", "w0", "wa"],
             "ice": ["h", "omega_b", "omega_c", "w_x", "r_star"]}
    print(f"melting-ice sharpness n = {N_SHARP:g};  data: Planck 2018 distance priors + DESI DR2 BAO + Pantheon+")
    for kind, R_ in (("lcdm", L), ("w0wa", W), ("ice", I)):
        C = Cosmo(kind, *R_.x)
        cp = chi2_parts(C)
        k = len(R_.x)
        out[kind] = {"params": dict(zip(names[kind], map(float, R_.x))), "chi2": float(R_.fun),
                     **{q: float(cp[q]) for q in ("CMB", "BAO", "SN", "rd")}}
        print(f"\n{kind:5s} chi2 = {R_.fun:9.3f}  (CMB {cp['CMB']:.2f}, BAO {cp['BAO']:.2f}, SN {cp['SN']:.2f};"
              f" r_d = {cp['rd']:.2f} Mpc)  dchi2 vs LCDM = {R_.fun - L.fun:+.2f},  dAIC = {R_.fun - L.fun + 2*(k-3):+.2f}")
        print("      " + ", ".join(f"{n} = {v:.4f}" for n, v in zip(names[kind], R_.x)))
    # the best-fit melting history
    C = Cosmo("ice", *I.x)
    print("\nmelting-ice best fit: dark-energy density relative to today, effective w_DE, ice/water ratio r:")
    for z in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        i = np.argmin(abs(C.zg - z))
        lna = C.lna
        dlnx = np.gradient(np.log(C.ox_a), lna)[i]
        weff = -1 - dlnx/3
        print(f"   z = {z:4.2f}: rho_x/rho_x0 = {C.ox_a[i]/C.ox_a[0]:.4f}   w_eff = {weff:+.3f}   r = {C.oc_a[i]/C.ox_a[i]:.3f}")
    json.dump(out, open(os.path.join(HERE, f"melting_ice_results_n{N_SHARP:g}.json"), "w"), indent=1)
