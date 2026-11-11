# Planck 2018 lensing test of the created-neutrino (Strano) model, through Cobaya with the patched classy.
# By (Anthropic) Claude and Dan Strano.
#
# Evaluates (no template: the FULL patched CLASS through Cobaya's classy theory)
#   - LCDM at its Cobaya best fit, and the model at its Cobaya best fit (from cobaya_fit.py -> cobaya_out/results.json)
#   - both at the SAME point (the LCDM best fit), which isolates the model's effect
# on Planck 2018 lensing (native, or CMB-marginalized with --marged) together with low-l TT, low-l EE and plik-lite,
# and prints the model/LCDM ratio of C_L^phiphi in the Planck lensing bands (L = 8-400).
#
# One model evaluation takes minutes (the extra Boltzmann hierarchies); expect ~3 model + 2 LCDM evaluations.
#
# Setup (once):
#   cobaya-install planck_2018_lensing.native planck_2018_lowl.TT planck_2018_lowl.EE \
#       planck_2018_highl_plik.TTTEEE_lite_native --packages-path packages
#   cd <class> && make libclass.a && cd python && python3 setup.py build
#   ln -s <class>/external <class>/python/build/lib.*/external      # classy finds its BBN/HyRec data here
# Run:
#   python3 lensing_test.py --packages packages --class-path <class> [--marged] [--no-lensing]
#   (check the log line "`classy` module loaded successfully from <class>/python/build/...": it must be the patched build)
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cobaya_fit as F                                  # noqa: E402
from cobaya.model import get_model                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--packages", default="packages")
ap.add_argument("--class-path", required=True, help="patched CLASS source dir (with python/build/lib.* built)")
ap.add_argument("--results", default=os.path.join(F.OUT, "results.json"))
ap.add_argument("--dof", type=float, default=1.0)
ap.add_argument("--m1", type=float, default=0.0)
ap.add_argument("--slicing", type=int, default=0)
ap.add_argument("--marged", action="store_true", help="CMB-marginalized lensing likelihood (lensing only, no CMB TT/EE)")
ap.add_argument("--no-lensing", action="store_true", help="machinery check without the lensing likelihood installed")
a = ap.parse_args()

res = json.load(open(a.results))
key = f"m1_{a.m1:g}"
bestL = res[f"lcdm_{key}"]["best"]
bestM = res.get(f"model_{key}_dof{a.dof:g}", {}).get("best", bestL)

CREATED = {"created_neutrinos": "yes", "created_ghost_dof": a.dof, "created_slicing": a.slicing,
           "created_q_per_decade": 6, "created_l_max": 8, "created_k_max": 0.05, "created_z_on": 100}


def check_patched_classy():
    """Fail early, with a clear message, unless the classy in --class-path is the patched build."""
    import glob, importlib
    libs = sorted(glob.glob(os.path.join(os.path.abspath(a.class_path), "build", "lib.*")))
    if not libs:
        sys.exit(f"\nNo built classy under {a.class_path}/build/lib.* . Build it there:\n"
                 f"  cd {a.class_path} && make libclass.a && rm -f python/classy.cpp && python3 setup.py build\n")
    sys.path.insert(0, libs[-1])
    mod = importlib.import_module("classy")
    sys.path.pop(0)
    print(f"classy module: {mod.__file__}", flush=True)
    c = mod.Class()
    c.set({"output": "tCl", "l_max_scalars": 50, "created_neutrinos": "no"})   # perturbation params are only parsed with output
    try:
        c.compute()
    except Exception as e:
        sys.exit(f"\nThis classy rejects 'created_neutrinos', so it is NOT the patched build:\n  {e}\n"
                 f"Rebuild in {a.class_path}: git apply the patch, then make libclass.a && rm -f python/classy.cpp "
                 f"&& python3 setup.py build\n")
    finally:
        c.struct_cleanup()
    for k in [m for m in sys.modules if m == "classy" or m.startswith("classy.")]:
        del sys.modules[k]                            # let Cobaya load it afresh from the same path


def build(created):
    inf = F.info(a.packages, a.m1)
    inf.pop("sampler")
    th = inf["theory"]["classy"]
    th["path"] = os.path.abspath(a.class_path)       # force Cobaya to load the PATCHED classy, not a system one
    th["stop_at_error"] = True                        # surface the real CLASS error instead of a silent -inf
    if created:
        th["extra_args"] = {**th["extra_args"], **CREATED}
    lik = dict(inf["likelihood"])
    if a.marged:
        lik = {"planck_2018_lensing.CMBMarged": None}
        inf["params"].pop("A_planck", None)          # no CMB likelihood consumes it; Cobaya would pass it to CLASS
    elif not a.no_lensing:
        lik["planck_2018_lensing.native"] = None
    inf["likelihood"] = lik
    m = get_model(inf)
    m.add_requirements({"Cl": {"pp": 2500, "tt": 2500}})
    return m


def evaluate(m, best):
    pt = {k: best[k] for k in F.PARAM_KEYS if k in m.parameterization.sampled_params()}
    t = time.time()
    ll = m.loglikes(pt, as_dict=True, cached=False)[0]
    if not all(np.isfinite(v) for v in ll.values()):
        sys.exit(f"\nA likelihood came back -inf at {pt}:\n  {ll}\n"
                 "(the theory computation failed; with stop_at_error the CLASS message is printed above)")
    chi = {k: -2 * float(v) for k, v in ll.items()}
    cl = m.provider.get_Cl(ell_factor=False)
    return chi, np.array(cl["pp"]), time.time() - t


def band_ratio(pp_m, pp_l):
    bands = [(8, 40), (41, 84), (85, 129), (130, 174), (175, 219), (220, 264), (265, 309), (310, 354), (355, 400)]
    L = np.arange(len(pp_m))
    out = []
    for lo, hi in bands:
        s = (L >= lo) & (L <= hi)
        w = (2 * L[s] + 1)
        out.append((lo, hi, np.sum(w * pp_m[s]) / np.sum(w * pp_l[s]) - 1))
    return out


check_patched_classy()
mL, mM = build(False), build(True)
rows = {}
rows["LCDM @ LCDM best"], ppLL, t1 = evaluate(mL, bestL)
print(f"LCDM evaluated in {t1:.0f} s", flush=True)
rows["model @ LCDM best"], ppML, t2 = evaluate(mM, bestL)
print(f"model evaluated in {t2:.0f} s", flush=True)
if bestM is not bestL:
    rows["model @ model best"], _, _ = evaluate(mM, bestM)

names = list(rows["LCDM @ LCDM best"].keys())
short = lambda k: k.split(".")[-1].replace("TTTEEE_lite_native", "lite")
print("\n" + f"{'':22s}" + "".join(f"{short(k):>12s}" for k in names) + f"{'total':>12s}")
for r, chi in rows.items():
    print(f"{r:22s}" + "".join(f"{chi[k]:12.3f}" for k in names) + f"{sum(chi.values()):12.3f}")
ref = rows["LCDM @ LCDM best"]
for r in [x for x in rows if x != "LCDM @ LCDM best"]:
    d = {k: rows[r][k] - ref[k] for k in names}
    print(f"{'d ' + r:22s}" + "".join(f"{d[k]:+12.3f}" for k in names) + f"{sum(d.values()):+12.3f}")

print("\nC_L^phiphi model/LCDM - 1 at the same parameters, in Planck lensing bands:")
for lo, hi, r in band_ratio(ppML, ppLL):
    print(f"   L {lo:3d}-{hi:3d}: {r:+.4f}")
json.dump({"rows": rows, "pp_bands": band_ratio(ppML, ppLL)},
          open(os.path.join(F.OUT, f"lensing_test_{'marged' if a.marged else 'native'}.json"), "w"), indent=1)
