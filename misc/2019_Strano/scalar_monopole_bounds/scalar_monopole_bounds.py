# Bounds on a universally mass-coupled scalar (the radiating monopole channel), by Dan Strano and (Anthropic) Claude.
#
# Parametrization (Damour-Esposito-Farese / Brans-Dicke): scalar-to-matter coupling alpha_s, scalar mass m (range lam).
#   Static: V(r) = -(G M / r) [1 + alpha_s^2 exp(-r/lam)]   -> Yukawa strength alpha_Y = alpha_s^2
#   Light bending / time delay (massless):  gamma - 1 = -2 alpha_s^2 / (1 + alpha_s^2)
#   Brans-Dicke: alpha_s^2 = 1 / (2 omega + 3)
#   Radiation: breathing-mode amplitude / tensor amplitude ~ alpha_s^2 for comparable sources (order of magnitude).
# Every input bound is listed with its source; the numbers anchored by hand (marked ANCHOR) should be re-checked
# against the cited papers before anyone relies on them. Order-of-magnitude estimates are marked ESTIMATE.
import numpy as np
np.seterr(over='ignore', divide='ignore')

c, G, Msun, Rsun, AU, kpc = 2.998e8, 6.674e-11, 1.989e30, 6.957e8, 1.496e11, 3.086e19

# 1) Cassini (Bertotti, Iess, Tortora 2003, Nature 425, 374): gamma - 1 = (2.1 +/- 2.3) x 1e-5.
g_mean, g_sig = 2.1e-5, 2.3e-5
lo2 = g_mean - 2*g_sig                       # scalar makes gamma - 1 negative; 2-sigma lower edge
a2_cassini = -lo2 / (2 + lo2)                # solve -2a/(1+a) = lo2 for a = alpha_s^2
omega_min = (1/a2_cassini - 3) / 2
print("1) MASSLESS scalar (range >> AU)")
print(f"   Cassini 2-sigma: alpha_s^2 < {a2_cassini:.2e}  (|alpha_s| < {np.sqrt(a2_cassini):.2e}),"
      f"  i.e. Brans-Dicke omega > {omega_min:.0f}")
print(f"   -> breathing / tensor amplitude ratio <~ {a2_cassini:.0e} for comparable sources (ESTIMATE)")

# 2) Massive scalar: allowed alpha_s^2 as a function of range lam.
def cassini(lam):          # Yukawa factor at the Cassini ray's closest approach b ~ 1.6 R_sun (ESTIMATE of geometry)
    return a2_cassini * np.exp(1.6*Rsun/lam)
a_moon = 3.844e8
def llr(lam):              # LLR: delta(omega)/omega < 1.6e-11 (95% CL), d(omega)/omega = (a/2)(a/lam)^2 e^(-a/lam) * alpha
    return 1.6e-11 / (0.5*(a_moon/lam)**2*np.exp(-a_moon/lam))   # Adelberger, Heckel, Nelson, hep-ph/0307284
anchors = [  # (lam_lo [m], lam_hi [m], alpha_Y bound, label)
    (8.0e6, 6.2e7, 1e-9, "Juno, 2409.10616, 95% CL (ANCHOR: best-bound range)"),
    (1.0e7, 1.5e7, 1e-5, "LAGEOS existing bounds 1e-5..1e-8 near 2 Earth radii (gr-qc/0201081 abstract; weak end used)"),
    (38.6e-6, np.inf, 1.0, "Eot-Wash torsion balance: alpha = 1 excluded for lam > 38.6 um (ANCHOR, Lee et al. 2020)"),
]
print("\n2) MASSIVE scalar: strongest of the encoded bounds on alpha_Y = alpha_s^2, by range")
print(f"   {'range lam':>12s} {'mass [eV]':>11s} {'Cassini':>10s} {'LLR':>10s} {'anchors':>10s}   tightest")
hbar_c_eVm = 1.973e-7
for lam in [1e-4, 1e3, 1e5, 4.8e5, 1e6, 1e7, 3e7, 1.9e8, 1e9, AU, 1e13]:
    cs, ll = cassini(lam), llr(lam)
    an = min([b for (l0, l1, b, _) in anchors if l0 <= lam <= l1], default=np.inf)
    best = min(cs, ll, an)
    who = ["Cassini", "LLR", "anchor"][int(np.argmin([cs, ll, an]))]
    f = lambda v: f"{v:10.1e}" if v < 1e30 else f"{'none':>10s}"
    print(f"   {lam:12.2e} {hbar_c_eVm/lam:11.2e} {f(cs)} {f(ll)} {f(an)}   {best:.1e} ({who})")
print("   NOT ENCODED: geophysical / laboratory bounds between ~1e-4 m and ~1e6 m (values not checked here).")
print("   The band ~5e5 m .. ~5e6 m is the one that matters for (3): it is where a scalar can be massive")
print("   enough to escape solar-system bounds and still radiate at LIGO frequencies. Fill it from the literature next.")

# 3) Can the monopole radiate at LIGO frequencies? A massive scalar propagates only above omega = m c^2 / hbar.
print("\n3) RADIATION WINDOW")
for f_hz in [10, 100, 1000]:
    lam_min = c / (2*np.pi*f_hz)
    print(f"   breathing waves at {f_hz:5d} Hz need range lam > {lam_min:.1e} m  (mass < {hbar_c_eVm/lam_min:.1e} eV)")

# 4) ESTIMATE: monopole burst from a galactic core collapse, h ~ 2 alpha_s^2 G dM_trace / (c^2 r),
#    dM_trace = change of the trace ("pressure") mass during bounce, taken as 0.1 Msun (order of magnitude only).
r, dM = 10*kpc, 0.1*Msun
h = 2*a2_cassini*G*dM/(c**2*r)
print("\n4) ESTIMATE: galactic core-collapse monopole burst at Cassini-limit coupling")
print(f"   h_breathing ~ {h:.0e} at 10 kpc (advanced LIGO band sensitivity ~1e-23 to 1e-22: marginal-to-below)")
print("   A screened or massive scalar escapes (1) but then must fit the range table (2) AND propagate (3).")
