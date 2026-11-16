# Landau-Lifshitz pseudotensor of the lapse family, from the identity
#   16 pi G (-g)(T^{mn} + t_LL^{mn}) = d_a d_b H^{manb},  H^{manb} = (-g)(g^{mn} g^{ab} - g^{ma} g^{nb}).
# By (Anthropic) Claude and Dan Strano.
import sympy as sp
t, x, y, z, a, G = sp.symbols('t x y z a G', real=True)
X = [t, x, y, z]
def ll(N, T=None):
    g = sp.diag(-N**2, 1, 1, 1); gi = g.inv(); mg = -g.det()
    H = lambda m, al, n, b: mg*(gi[m, n]*gi[al, b] - gi[m, al]*gi[n, b])
    out = sp.zeros(4, 4)
    for m in range(4):
        for n in range(4):
            dd = sum(sp.diff(H(m, al, n, b), X[al], X[b]) for al in range(4) for b in range(4))
            out[m, n] = sp.simplify(dd/(16*sp.pi*G*mg) - (T[m, n] if T is not None else 0))
    return out
# Rindler (vacuum, flat): N = a x
tR = ll(a*x)
print("Rindler N = a x (flat, T = 0):  t_LL^{mn} =")
sp.pprint(tR)
print("  energy density as seen by the static observer (u^m = (1/N,0,0,0)): t_LL^{tt} g_tt^2 * ... -> t_LL_{tt}/N^2 =",
      sp.simplify(tR[0, 0]*(a*x)**2))
# Minkowski in the same coordinates (N = 1): t_LL = 0
print("Minkowski N = 1:", ll(sp.Integer(1)) == sp.zeros(4, 4))
xs = sp.symbols('x_s', positive=True)
tot = sp.integrate(tR[2, 2].subs(x, sp.Symbol('u', positive=True)), (sp.Symbol('u', positive=True), xs, sp.oo))
print("Integrated exterior LL transverse stress, x_s..inf:", tot, " vs membrane sigma 1/(8 pi G x_s):",
      sp.simplify(tot - 1/(8*sp.pi*G*xs)) == 0)
print("Trace of t_LL (Rindler):", sp.simplify(sum((sp.diag(-(a*x)**2, 1, 1, 1))[i, i]*tR[i, i] for i in range(4))),
      "(nonzero -> not radiation-like / not luminal)")
