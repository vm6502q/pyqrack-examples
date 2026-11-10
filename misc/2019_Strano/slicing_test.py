# Creation-slicing robustness test: does the large-angle signature depend on which hypersurfaces carry the uniform
# creation rate?  By (Anthropic) Claude and Dan Strano.
#   slicing 0: CDM-comoving (synchronous), 1: ghost-comoving, 2: coexistence-pinned (creation follows local expansion).
# Runs the full patched CLASS at the Cobaya LCDM best fit (parameters barely move in the refit) for each slicing,
# self-consistent T_c (g = 1), and compares with LCDM at the same point (same pipeline, so precision offsets cancel).
# Usage: python3 slicing_test.py CLASS_BINARY PACKAGES [--dof 1] [--m1 0] [--slicings 0,1,2]
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cobaya_fit as F                     # noqa: E402
from planck_chi2 import chi2               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("class_bin"); ap.add_argument("packages")
ap.add_argument("--dof", type=float, default=1.0); ap.add_argument("--m1", type=float, default=0.0)
ap.add_argument("--slicings", default="0,1,2")
a = ap.parse_args()
res = json.load(open(os.path.join(F.OUT, "results.json")))
best = res[f"lcdm_m1_{a.m1:g}"]["best"]
os.makedirs(F.OUT, exist_ok=True)


def run(tag, created, sl):
    root = os.path.join(F.OUT, tag)
    F.class_ini(root + ".ini", root, best, a.m1, created, a)
    if created:
        open(root + ".ini", "a").write(f"created_slicing = {sl}\n")
    p = subprocess.run([a.class_bin, root + ".ini"], capture_output=True, text=True)
    if p.returncode != 0 or "Error" in p.stdout:
        raise RuntimeError(p.stdout[-800:])
    return root + "_00"


rl = run(f"slc_lcdm_m1_{a.m1:g}", False, 0)
u0 = np.loadtxt(rl + "_cl.dat")
c0 = chi2(a.packages, rl, A_planck=best["A_planck"])
out = {}
for sl in [int(x) for x in a.slicings.split(",")]:
    rm = run(f"slc{sl}_m1_{a.m1:g}", True, sl)
    u = np.loadtxt(rm + "_cl.dat")
    c = chi2(a.packages, rm, A_planck=best["A_planck"])
    d = {k.split(".")[-1]: c[k] - c0[k] for k in c}
    r = lambda l, col: u[l - 2, col] / u0[l - 2, col] - 1
    out[sl] = {"dchi2": sum(d.values()), **d, **{f"TT_l{l}": r(l, 1) for l in (2, 3, 4, 6, 10, 20, 30)},
               **{f"pp_L{l}": r(l, 5) for l in (2, 10, 30, 100)}}
    names = {0: "CDM-comoving", 1: "ghost-comoving", 2: "coexistence-pinned"}
    print(f"slicing {sl} ({names[sl]:18s}): dchi2 {out[sl]['dchi2']:+.3f} (lowTT {d['TT']:+.3f}, lowEE {d['EE']:+.3f}, "
          f"lite {d['TTTEEE_lite_native']:+.3f}) | TT l=2,3,4,6,10,20,30: " +
          " ".join(f"{out[sl][f'TT_l{l}']:+.3f}" for l in (2, 3, 4, 6, 10, 20, 30)) +
          " | pp L=2,10,30,100: " + " ".join(f"{out[sl][f'pp_L{l}']:+.4f}" for l in (2, 10, 30, 100)), flush=True)
json.dump(out, open(os.path.join(F.OUT, "slicing_results.json"), "w"), indent=1)
