# Self-consistent coexistence temperature: if the ghost gas is itself a thermal (blackbody) gas at T_c with g bosonic
# degrees of freedom, |rho_ghost| = g (pi^2/30) T_c^4 must equal the G that closure (exact Lambda) demands.
# By (Anthropic) Claude and Dan Strano.  Same steady-state formulas as the CLASS patch (perturbations_crt_setup).
import numpy as np
from scipy.optimize import brentq
kB = 8.617333262e-5; hbarc_eVm = 1.973269804e-7
H0 = 67.36 * 1e3 / 3.0857e22; OmL = 0.6847
rho_crit = 3 * H0**2 / (8 * np.pi * 6.674e-11) * (2.998e8)**2      # J/m^3
rhoL_eV4 = OmL * rho_crit / 1.602e-19 * hbarc_eVm**3               # eV^4
x = np.exp(np.linspace(np.log(1e-12), np.log(80), 20000)); dl = np.log(x[1] / x[0])

def G_over_rhoL(Tc_K, m1=0.0):
    T = Tc_K * kB
    ms = np.array([m1, np.sqrt(m1**2 + 7.42e-5), np.sqrt(m1**2 + 2.51e-3)])
    ms = ms[ms**2 > 1e-4 * ms.max()**2]
    sh, R = [], []
    for m in ms:
        mu = m / T; E = np.sqrt(x**2 + mu**2); ph = 1 / (np.exp(E) + 1)
        Phi = np.cumsum((ph * dl)[::-1])[::-1]
        In = np.sum(x**3 * ph); Ie = np.sum(x**3 * E * ph)
        Je = np.sum(x**3 * E * Phi); Jp = np.sum(x**5 / (3 * E) * Phi)
        sh.append(mu**2 * Ie / In); R.append(Je / (Je + Jp))
    sh = np.array(sh) / np.sum(sh); R = np.array(R)
    return 1 / (-1 + 4 / 3 * np.sum(sh * R))

print(f"rho_Lambda^(1/4) = {rhoL_eV4**0.25*1e3:.3f} meV")
for g in [1, 2]:
    f = lambda T: g * np.pi**2 / 30 * (T * kB)**4 / rhoL_eV4 - G_over_rhoL(T)
    Tc = brentq(f, 10, 200)
    print(f"g = {g}: self-consistent T_c = {Tc:.1f} K  (|rho_ghost| = {G_over_rhoL(Tc):.2f} rho_Lambda, m3/T_c = {0.0501/(Tc*kB):.1f})")
for T in [45, 60, 75]:
    print(f"   check: T_c = {T} K -> closure G = {G_over_rhoL(T):.3f} rho_L, blackbody(g=1) = {np.pi**2/30*(T*kB)**4/rhoL_eV4:.3f} rho_L")
