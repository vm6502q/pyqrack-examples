# Landau-Lifshitz pseudotensor for time-varying (sinusoidal) Rindler acceleration, N = alpha(t) x.
# By Dan Strano and (Anthropic) Claude.
import sympy as sp
t, x, y, z = sp.symbols('t x y z', real=True)
G, a0, a1, w = sp.symbols('G a_0 a_1 omega', positive=True)
X = [t, x, y, z]
def ll(N):
    g = sp.diag(-N**2, 1, 1, 1); gi = g.inv(); mg = -g.det()
    H = lambda m, al, n, b: mg*(gi[m, n]*gi[al, b] - gi[m, al]*gi[n, b])
    return sp.Matrix(4, 4, lambda m, n: sp.simplify(
        sum(sp.diff(H(m, al, n, b), X[al], X[b]) for al in range(4) for b in range(4))/(16*sp.pi*G*mg)))
al = sp.Function('alpha', positive=True)(t)
tg = ll(al*x)
print("N = alpha(t) x (flat for any alpha):  nonzero t_LL components:")
for m in range(4):
    for n in range(m, 4):
        if tg[m, n] != 0: print(f"   t^{'txyz'[m]}{'txyz'[n]} =", tg[m, n])
N = (a0 + a1*sp.sin(w*t))*x
ts = ll(N)
print("Sinusoidal alpha = a0 + a1 sin(w t):  t^yy =", ts[2, 2], "| t^tx =", ts[0, 1], "| t^tt =", ts[0, 0])
print("  NOTE: t^yy itself shows no t-dependence. The conserved quantity is the DENSITIZED (-g) t^{mn}, with -g = N^2:")
dens = sp.simplify((N**2)*ts[2, 2])
print("  (-g) t^yy =", dens, "   <- oscillates with the acceleration")
xs = sp.symbols('x_s', positive=True)
memb = sp.simplify((a0 + a1*sp.sin(w*t))/(8*sp.pi*G))
print("  membrane pressure, redshifted (N * sigma) =", memb, "   <- oscillates too")
vals = {a0: 1, a1: sp.Rational(1, 2), w: 1, G: 1}
print("  over one period (a0 = 1, a1 = 1/2, w = 1, G = 1):")
for k in range(5):
    tt = k*sp.pi/2
    print(f"    t = {str(tt):6s}  (-g) t^yy = {float(dens.subs(vals).subs(t, tt)):.4f}"
          f"   N*sigma = {float(memb.subs(vals).subs(t, tt)):.4f}")
# Local conservation d_m[(-g) t^{mn}] = 0 (identity), checked explicitly
mg = N**2
print("Conservation d_m[(-g) t^{mn}] = 0 for each n:",
      [sp.simplify(sum(sp.diff(mg*ts[m, n], X[m]) for m in range(4))) for n in range(4)])
