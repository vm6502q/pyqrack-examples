# Created-neutrino (Strano) model vs LCDM on a grid of coexistence temperature T_c and lightest neutrino mass.
# By (Anthropic) Claude and Dan Strano.
#
# For each point: CLASS with the created_neutrinos patch (ghost + three mass eigenstates created in ratio m_i^2 at T_c),
# and LCDM with the SAME three relic neutrino masses; cosmological parameters fixed at Planck 2018 best fit.
# Reports Planck chi^2 differences (low-l TT, low-l EE, plik-lite TTTEEE) and observable ratios.
#
# Usage: python3 run_grid.py CLASS_BINARY PACKAGES_PATH [--Tc 45,60,75] [--m1 0,0.01,0.03] [--qpd 6] [--lmax 8]
#        [--kmax 0.05] [--zon 100] [--out grid_out]
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from planck_chi2 import chi2  # noqa: E402

DM21, DM31 = 7.42e-5, 2.51e-3   # eV^2, normal ordering


def write_ini(path, root, crt, Tc, m1, a):
    m = [max(m1, 1e-4), np.sqrt(m1**2 + DM21), np.sqrt(m1**2 + DM31)]
    s = f"""h = 0.6736
omega_b = 0.02237
omega_cdm = 0.1200
A_s = 2.100e-9
n_s = 0.9649
tau_reio = 0.0544
N_ur = 0.00441
N_ncdm = 3
m_ncdm = {m[0]:.6g}, {m[1]:.6g}, {m[2]:.6g}
T_ncdm = 0.71611, 0.71611, 0.71611
output = tCl, pCl, lCl, mPk
lensing = yes
l_max_scalars = 2500
P_k_max_1/Mpc = 1.0
root = {root}_
created_neutrinos = {'yes' if crt else 'no'}
created_Tc_K = {Tc}
created_z_on = {a.zon}
created_q_per_decade = {a.qpd}
created_l_max = {a.lmax}
created_k_max = {a.kmax}
"""
    open(path, "w").write(s)


def observables(root, root0):
    u, u0 = np.loadtxt(root + "_00_cl.dat"), np.loadtxt(root0 + "_00_cl.dat")
    pk, pk0 = np.loadtxt(root + "_00_pk.dat"), np.loadtxt(root0 + "_00_pk.dat")
    r = lambda l, col: u[l - 2, col] / u0[l - 2, col] - 1
    Lm = (u[:, 0] >= 8) & (u[:, 0] <= 400)
    w = (2 * u0[Lm, 0] + 1)                       # crude lensing-amplitude weight
    A_pp = np.sum(w * u[Lm, 5] / u0[Lm, 5]) / np.sum(w) - 1
    return {"TT_l2": r(2, 1), "TT_l3": r(3, 1), "TT_l10": r(10, 1), "pp_L10": r(10, 5),
            "A_phiphi_8_400": A_pp, "Pk_0.2": np.interp(0.2, pk[:, 0], pk[:, 1]) / np.interp(0.2, pk0[:, 0], pk0[:, 1]) - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("class_bin"); ap.add_argument("packages")
    ap.add_argument("--Tc", default="45,60,75"); ap.add_argument("--m1", default="0,0.01,0.03")
    ap.add_argument("--qpd", type=float, default=6); ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--kmax", type=float, default=0.05); ap.add_argument("--zon", type=float, default=100)
    ap.add_argument("--out", default="grid_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = []
    for m1 in [float(x) for x in a.m1.split(",")]:
        r0 = os.path.join(a.out, f"lcdm_m1_{m1:g}")
        if not os.path.exists(r0 + "_00_cl_lensed.dat"):
            write_ini(r0 + ".ini", r0, False, 60, m1, a)
            subprocess.run([a.class_bin, r0 + ".ini"], check=True, capture_output=True)
        c0 = chi2(a.packages, r0 + "_00")
        for Tc in [float(x) for x in a.Tc.split(",")]:
            r = os.path.join(a.out, f"crt_Tc{Tc:g}_m1_{m1:g}")
            if not os.path.exists(r + "_00_cl_lensed.dat"):
                write_ini(r + ".ini", r, True, Tc, m1, a)
                p = subprocess.run([a.class_bin, r + ".ini"], capture_output=True, text=True)
                if p.returncode != 0 or "Error" in p.stdout:
                    print(f"T_c={Tc} m1={m1}: CLASS failed\n{p.stdout[-600:]}{p.stderr[-300:]}", flush=True)
                    continue
                open(r + ".log", "w").write(p.stdout)
            c = chi2(a.packages, r + "_00")
            d = {k.split(".")[-1]: c[k] - c0[k] for k in c}
            row = {"Tc_K": Tc, "m1_eV": m1, "dchi2_total": sum(d.values()), **{"dchi2_" + k: v for k, v in d.items()},
                   **observables(r, r0)}
            results.append(row)
            print(f"T_c={Tc:4g} K m1={m1:5.3f}: dchi2 {row['dchi2_total']:+7.3f} (lowTT {d['TT']:+.3f}, lowEE {d['EE']:+.3f}, "
                  f"lite {d['TTTEEE_lite_native']:+.3f}) | TT l=2 {row['TT_l2']:+.3f}, l=10 {row['TT_l10']:+.4f} | "
                  f"A_phiphi(8-400) {row['A_phiphi_8_400']:+.4f} | P(0.2) {row['Pk_0.2']:+.4f}", flush=True)
    json.dump(results, open(os.path.join(a.out, "grid_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
