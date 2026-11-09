# Ghost-gas coexistence with hot creation (Strano model, 2026 form), background + free-streaming estimate.
# By (Anthropic) Claude and Dan Strano.
#  - ghost scalar gas held at coexistence temperature T_c: rho_g = -G (constant, crit units), p_g = rho_g/3
#  - coexistence requires Q = 4 H |rho_g| into created positive-mass particles (monopole coupling)
#  - created particles: mass m = mu*T_c, born thermal from the gas with momentum p_b = 3 T_c, then p ~ 1/a
#  - G fixed by closure (Omega_tot = 1); single knob mu = m/T_c
import numpy as np, sys
from scipy.optimize import brentq
Om, Orad = 0.31, 9.0e-5
PB = 3.0
def created(a_grid, G, mu, Nmin=np.log(1e-6), nb=4000):
    """Energy density, pressure and slow (clustering) part of the created population at each a.
    Integrates the birth history continuously in N = ln a_birth up to ln a (no bin-edge jitter)."""
    Eb = np.sqrt(PB**2 + mu**2)
    out = []
    for a in a_grid:
        Nb = np.linspace(Nmin, np.log(a), nb); ab = np.exp(Nb)
        dn = 4*G*ab**3/Eb                  # comoving number per unit N, energy unit T_c
        p = PB*ab/a; E = np.sqrt(p**2 + mu**2); vel = p/E
        rho = np.trapezoid(dn*E, Nb)/a**3; pr = np.trapezoid(dn*p**2/(3*E), Nb)/a**3
        slow = np.trapezoid(np.where(vel < 2.5e-3, dn*E, 0.0), Nb)/a**3
        out.append((rho, pr, slow))
    return np.array(out)
def closure(G, mu):
    rc = created([1.0], G, mu)[0][0]
    return Orad + Om - G + rc - 1.0
mu_list = [float(x) for x in sys.argv[1:]] or [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]
z = np.array([0, 0.5, 1, 2, 3]); a = 1/(1+z)
print(f"{'mu=m/Tc':>8s} {'G=|rho_g|':>9s} {'Om_created':>10s} {'clustering created':>18s} {'w_DE(z=0,0.5,1,2)':>28s} {'H/H_LCDM max dev z<=3':>22s}")
for mu in mu_list:
    try:
        G = brentq(lambda g: closure(g, mu), 1e-6, 50.0)
    except ValueError:
        print(f"{mu:8.1f}  no closure"); continue
    c = created(a, G, mu)
    rhoDE = -G + c[:,0]; pDE = -G/3 + c[:,1]
    w = pDE/rhoDE
    H2 = Orad*a**-4 + Om*a**-3 + rhoDE
    H2l = Orad*a**-4 + Om*a**-3 + (1-Om-Orad)
    dev = np.max(np.abs(np.sqrt(H2/H2l)-1))
    print(f"{mu:8.1f} {G:9.3f} {c[0,0]:10.3f} {c[0,2]:18.4f}   {' '.join(f'{x:+.3f}' for x in w[:4])}   {dev:20.3f}")
