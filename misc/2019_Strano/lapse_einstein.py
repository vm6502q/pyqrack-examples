# Einstein tensor for ds^2 = -N(t,x)^2 dt^2 + dx^2 + dy^2 + dz^2, and the junction (thin-wall) source of a kink in N.
# By (Anthropic) Claude and Dan Strano.
import sympy as sp
t, x, y, z = sp.symbols('t x y z', real=True)
N = sp.Function('N')(t, x)
X = [t, x, y, z]
g = sp.diag(-N**2, 1, 1, 1); gi = g.inv()
Gam = [[[sum(gi[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d])) for d in range(4))/2
         for c in range(4)] for b in range(4)] for a in range(4)]
def Ric(b, c):
    return sp.simplify(sum(sp.diff(Gam[a][b][c], X[a]) - sp.diff(Gam[a][b][a], X[c])
                           + sum(Gam[a][a][d]*Gam[d][b][c] - Gam[a][c][d]*Gam[d][b][a] for d in range(4)) for a in range(4)))
R = sp.Matrix(4, 4, lambda b, c: Ric(b, c))
Rs = sp.simplify(sum(gi[a, b]*R[a, b] for a in range(4) for b in range(4)))
G = sp.simplify(R - Rs*g/2)
mixed = sp.simplify(gi*G)
for (a, b), name in [((0, 0), "G^t_t"), ((0, 1), "G^t_x"), ((1, 1), "G^x_x"), ((2, 2), "G^y_y"), ((3, 3), "G^z_z")]:
    print(f"{name:6s} =", sp.simplify(mixed[a, b]))
print("Ricci scalar R =", Rs)
# Kink: N = alpha(t) x + beta(t) (x - xk) H(x - xk)  ->  N_xx = beta delta(x - xk);  G^y_y = N_xx/N = beta delta/N(xk).
# Integrating across the wall over proper distance (g_xx = 1): surface transverse pressure sigma = beta/(8 pi G N(xk)).
a, b, xk, Gn = sp.symbols('alpha beta x_k G', positive=True)
Nk = a*xk
sigma = b/(8*sp.pi*Gn*Nk)
print("thin wall at x_k: sigma =", sigma, "; redshifted (times lapse) sigma*N =", sp.simplify(sigma*Nk))
