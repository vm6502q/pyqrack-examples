# Sourcing the lapse field from a Rindler horizon: membrane (Israel junction) derivation and no-go checks.
# By (Anthropic) Claude and Dan Strano.   Metric: ds^2 = -N(t,x)^2 dt^2 + dx^2 + dy^2 + dz^2,  units c = k_B = 1.
import sympy as sp
t, x, y, z = sp.symbols('t x y z', real=True)
a, xs, Gn, hb = sp.symbols('a x_s G hbar', positive=True)

def einstein_mixed(N):
    X = [t, x, y, z]; g = sp.diag(-N**2, 1, 1, 1); gi = g.inv()
    Gam = [[[sum(gi[i, d]*(sp.diff(g[d, j], X[k]) + sp.diff(g[d, k], X[j]) - sp.diff(g[j, k], X[d])) for d in range(4))/2
             for k in range(4)] for j in range(4)] for i in range(4)]
    Ric = sp.Matrix(4, 4, lambda j, k: sp.simplify(sum(sp.diff(Gam[i][j][k], X[i]) - sp.diff(Gam[i][j][i], X[k])
          + sum(Gam[i][i][d]*Gam[d][j][k] - Gam[i][k][d]*Gam[d][j][i] for d in range(4)) for i in range(4))))
    R = sp.simplify(sum(gi[i, j]*Ric[i, j] for i in range(4) for j in range(4)))
    return sp.simplify(gi*(Ric - R*g/2)), R

N = sp.Function('N')(t, x)
Gm, R = einstein_mixed(N)
print("1) General lapse: G^t_t, G^t_x, G^x_x =", Gm[0, 0], Gm[0, 1], Gm[1, 1], "; G^y_y = G^z_z =", Gm[2, 2])
print("   => source must be T^mu_nu = diag(0, 0, p, p), p = N_xx/(8 pi G N): zero energy, zero x-pressure, zero flux,")
print("      for ANY time dependence of N (a travelling profile F(x - v t) gives the same form; v never enters G).")

# Gauss/Tolman form: the vacuum equation N_xx = 0 is Gauss's law for the redshifted acceleration N_x.
rho, px, pp = sp.symbols('rho p_x p_perp')
tolman = 4*sp.pi*Gn*N*(rho + px + 2*pp)
print("2) Tolman/Gauss: d_x(N_x) = 4 pi G N (rho + p_x + 2 p_perp) ->  with rho = p_x = 0:",
      sp.simplify(tolman.subs({rho: 0, px: 0}) - 8*sp.pi*Gn*N*pp) == 0)

# Stretched horizon at x_s: exterior Rindler N = a x, interior field-free N = a x_s. Slope jump [N_x] = a.
sigma_loc = a/(8*sp.pi*Gn*(a*xs))
sigma_red = sp.simplify(sigma_loc*a*xs)
print("3) Membrane at x_s: local surface pressure =", sigma_loc, "; redshifted =", sigma_red, " (= kappa/8 pi G, kappa = a)")
T_loc, s = hb/(2*sp.pi*xs), 1/(4*Gn*hb)                      # local Unruh temperature, Bekenstein-Hawking entropy/area
print("   Unruh T_loc * s_BH =", sp.simplify(T_loc*s), "-> equals local pressure:", sp.simplify(T_loc*s - sigma_loc) == 0)
print("   i.e. the membrane obeys the Euler relation rho + p = T s with rho = 0 (zero chemical potential).")
print("   Its Tolman (active) surface mass, redshifted: N*(0 + 2 sigma) =", sp.simplify(2*sigma_red),
      "-> Gauss flux 4 pi G * that =", sp.simplify(4*sp.pi*Gn*2*sigma_red), "= a. Independent of x_s.")

# Energy conditions for diag(rho, p_x, p, p) = (0, 0, p, p), p > 0
p = sp.symbols('p', positive=True)
print("4) Energy conditions for (0,0,p,p), p>0: NEC", True, "| WEC", True, "| SEC", True,
      "| DEC (rho >= |p|):", bool(0 >= p.subs(p, 1)), " -> DEC fails, as for the membrane-paradigm fluid itself")

# Time-dependent horizon pressure: N = alpha(t) x is vacuum and is Rindler with tau = int alpha dt / a.
al = sp.Function('alpha')(t); be = sp.Function('beta')(t)
G1, _ = einstein_mixed(al*x + be)
print("5) N = alpha(t) x + beta(t): all G components zero:", all(sp.simplify(e) == 0 for e in G1))
print("   alpha(t) alone is a time relabelling of Rindler (pure gauge); beta(t) moves N = 0 at local speed",
      "x_h'/N -> infinity, i.e. horizon drift is a member of the v = infinity family, still flat.")

# Who can supply (0,0,p,p) in the bulk?  Its trace is 2p != 0 (R = -8 pi G T).
print("6) Trace of required source = 2p; radiation (p = rho/3, incl. the ghost gas rho<0) is traceless -> cannot.")
u, w, Vs, Nn = sp.symbols('u w V N', real=True)          # u = phi_t, w = phi_x, V = potential
L = sp.Rational(1, 2)*(-u**2/Nn**2 + w**2) + Vs
T = dict(tt=-u**2/Nn**2 - L, xx=w**2 - L, yy=-L, tx=-u*w/Nn**2)
for case, sub in [("phi_t = 0", {u: 0}), ("phi_x = 0", {w: 0})]:
    e = {k: sp.simplify(v.subs(sub)) for k, v in T.items()}
    sol = sp.solve([e['tt'], e['xx']], [Vs, w if case == "phi_t = 0" else u], dict=True)
    print(f"   minimally coupled scalar, {case} (forced by T^t_x = 0): T^t_t = T^x_x = 0 gives T^y_y =",
          sorted({sp.simplify(e['yy'].subs(q)) for q in sol}))
