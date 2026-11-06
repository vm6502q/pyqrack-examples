#!/usr/bin/env python3
# Does the Strano black-hole emission law improve early-universe residuals over LCDM?
#
# By (Anthropic) Claude and Dan Strano
#
# Model. A monochromatic population of primordial black holes, a fraction f_pbh of the dark matter at early
# times, each radiating P = E_P / (2 r_s) per Planck time into invisible (scalar/graviton) radiation:
#     dM/dt = -c^5 l_P / (4 G^2 M)   =>   M(t) = M0 sqrt(1 - t/t_evap),   t_evap = 2 G^2 M0^2 / (c^5 l_P).
# So the PBH density is CDM-like times sqrt(1 - t/t_evap), and the lost energy becomes dark radiation.
# This needs the patched CLASS (pbh_class.patch: 'pbh_evaporation_law = yes', 'tau_dcdm' = t_evap in s).
#
# What it computes (all chi^2 are -2 log L at best fit, from Cobaya's minimizer):
#   1. LCDM best fit.
#   2. Profile over the evaporation time: for each log10(t_evap/s) on a grid, minimize over f_pbh >= 0 and all
#      LCDM + nuisance parameters. Delta chi^2(t_evap) = chi^2_PBH - chi^2_LCDM <= 0 by construction
#      (f_pbh = 0 is LCDM). How negative it gets is "how much the law improves the residuals".
#   3. Optionally (--free) one global fit with t_evap free too.
#   4. Rest-mass excitation channel (no PBHs): the excitation above the Higgs vacuum is modeled as electron mass
#      at recombination m_e(early)/m_e(today) = me_ratio (CLASS 'varying_me', switched off below z = 50). Since
#      omega_b and omega_cdm are fitted anyway, this stands in for a near-uniform rest-mass shift (all masses
#      relative to the Planck mass). Steps: 'me' (free me_ratio) and 'meprofile' (Delta chi^2 on a me_ratio grid).
#      'me' starts from the best 'meprofile' point when one exists (the valley along m_e-H0 is long and shallow).
#   5. Dilaton channel (continuous version of step 4): the trace (spin-0) part of the rank-2 potential, coupled to
#      rest mass, is a Brans-Dicke scalar with matter coupling alpha^2 = 1/(2 omega_BD + 3). Its slow evolution makes
#      rest masses (relative to the Planck mass) drift over cosmic history instead of jumping at one redshift, with
#      G_eff for matter today pinned to the local G. Runs on hi_class (hiclass_theory.py, module 'hiclassy'), and is
#      compared with its own alpha^2 = 0 limit computed by the same code. Step: 'bdprofile' (--bdgrid alpha^2 values).
#      For scale: Cassini requires alpha^2 < ~1.2e-5 locally (unless screened); cosmology alone is much weaker.
# The PBH mass that evaporates at t_evap is reported alongside: M0 = sqrt(t_evap c^5 l_P / (2 G^2)).
#
# Usage:
#   python3 pbh_fit.py install                 # download likelihood data (Planck 2018, DESI DR2 BAO, Pantheon+)
#   python3 pbh_fit.py lcdm                    # step 1
#   python3 pbh_fit.py profile [--jobs 4]      # step 2 (default grid 11 .. 19.5 in log10 s; needs step 1)
#   python3 pbh_fit.py free                    # step 3
#   python3 pbh_fit.py report                  # table of Delta chi^2 and best-fit f_pbh per t_evap
#   python3 pbh_fit.py me / meprofile          # step 4 (needs step 1); 'report' then prints it too
#   python3 pbh_fit.py bdprofile               # step 5 (needs hi_class installed as 'hiclassy'; see README)
# Options: --packages DIR (default ./packages)   --out DIR (default ./pbh_out)
#          --data planck|planck+bao|all (default all: Planck TTTEEE lite + lowl + lowE + lensing + DESI + Pantheon+)
#          --grid 11,12,...   --jobs N (parallel profile points; set OMP_NUM_THREADS = cores / N)
#          --class-path DIR   (patched CLASS checkout; default: the classy already importable)
#          --test             (single likelihood evaluation instead of minimizing: checks the pipeline in minutes)

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

C, G, HBAR, YEAR = 2.99792458e8, 6.6743e-11, 1.054571817e-34, 3.15576e7
L_P = math.sqrt(HBAR * G / C ** 3)
M_SUN = 1.98892e30


def mass_for_tevap(tevap_s):
    return math.sqrt(tevap_s * C ** 5 * L_P / (2 * G ** 2))


LIKES = {
    "planck": {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite_native": None,
    },
}
LIKES["planck+bao"] = dict(LIKES["planck"], **{"planck_2018_lensing.native": None, "bao.desi_dr2.desi_bao_all": None})
LIKES["all"] = dict(LIKES["planck+bao"], **{"sn.pantheonplus": None})

# LCDM parameters (Planck 2018-like starting points and proposal widths)
BASE = {
    "logA": {"prior": {"min": 2.5, "max": 3.5}, "ref": 3.044, "proposal": 0.01, "drop": True, "latex": r"\log(10^{10} A_s)"},
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_s"},
    "n_s": {"prior": {"min": 0.9, "max": 1.05}, "ref": 0.965, "proposal": 0.004},
    "H0": {"prior": {"min": 55, "max": 85}, "ref": 67.4, "proposal": 0.5},
    "omega_b": {"prior": {"min": 0.019, "max": 0.025}, "ref": 0.02237, "proposal": 0.0001},
    "omega_dm": {"prior": {"min": 0.08, "max": 0.2}, "ref": 0.120, "proposal": 0.001, "drop": True,
                 "latex": r"\omega_{\rm dm}^{\rm ini}"},
    "tau_reio": {"prior": {"min": 0.01, "max": 0.12}, "ref": 0.054, "proposal": 0.005},
}
EXTRA = {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "non_linear": "halofit", "l_max_scalars": 2600}


def omega_bd_for(alpha2):
    return 1e8 if alpha2 <= 0 else (1.0 / alpha2 - 3.0) / 2.0


def info_for(model, args, log10_tev=None, ref=None, me=None, alpha2=None):
    params = json.loads(json.dumps(BASE))
    if model == "bd":
        params["omega_cdm"] = {"value": "lambda omega_dm: omega_dm", "latex": r"\omega_c"}
        params["omega_BD"] = {"value": omega_bd_for(alpha2), "latex": r"\omega_{\rm BD}"}
    elif model in ("lcdm", "me"):
        params["omega_cdm"] = {"value": "lambda omega_dm: omega_dm", "latex": r"\omega_c"}
        if model == "me":
            params["varying_me"] = ({"prior": {"min": 0.95, "max": 1.05}, "ref": 1.0, "proposal": 0.003,
                                     "latex": r"m_e^{\rm rec}/m_e^{0}"} if me is None else {"value": float(me)})
    else:
        params["f_pbh"] = {"prior": {"min": 0.0, "max": 0.5}, "ref": 0.0, "proposal": 0.005, "drop": True,
                           "latex": "f_{\\rm PBH}"}
        params["omega_cdm"] = {"value": "lambda omega_dm, f_pbh: omega_dm*(1-max(f_pbh, 1e-9))", "latex": r"\omega_c"}
        params["omega_ini_dcdm"] = {"value": "lambda omega_dm, f_pbh: omega_dm*max(f_pbh, 1e-9)", "latex": r"\omega_{\rm PBH}^{\rm ini}"}
        if log10_tev is None:
            params["log10_tevap"] = {"prior": {"min": 11.0, "max": 19.5}, "ref": 17.0, "proposal": 0.3, "drop": True,
                                     "latex": r"\log_{10}(t_{\rm evap}/{\rm s})"}
        else:
            params["log10_tevap"] = {"value": float(log10_tev), "drop": True}
        params["tau_dcdm"] = {"value": "lambda log10_tevap: 10**log10_tevap", "latex": r"t_{\rm evap}"}
        params["M_pbh_kg"] = {"derived": f"lambda log10_tevap: np.sqrt(10**log10_tevap*{C ** 5 * L_P / (2 * G ** 2)!r})",
                              "latex": r"M_{\rm PBH}\,[{\rm kg}]"}
    if ref:  # start from a previous best fit
        for k, v in ref.items():
            if k in params and isinstance(params[k], dict) and "prior" in params[k]:
                params[k]["ref"] = v
    extra = dict(EXTRA)
    if model == "pbh":
        extra["pbh_evaporation_law"] = "yes"
    if model == "me":
        extra.update({"varying_fundamental_constants": "instantaneous", "varying_transition_redshift": 50.,
                      "varying_alpha": 1.0})
    theory = {"classy": {"extra_args": extra}}
    if args.class_path:
        theory["classy"]["path"] = os.path.abspath(args.class_path)
    if model == "bd":
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        import hiclass_theory
        theory = {"hiclass_theory.hiclass": {"python_path": here,
                                             "extra_args": dict(hiclass_theory.HICLASS_BD_ARGS, **EXTRA)}}
    sampler = {"evaluate": None} if args.test else {"minimize": {"method": "bobyqa", "max_evals": 3000,
                                                                  "best_of": 2, "confidence_for_unbounded": 0.9999995}}
    return {"theory": theory, "likelihood": LIKES[args.data], "params": params, "sampler": sampler,
            "packages_path": os.path.abspath(args.packages), "debug": False}


def summarize(vals, tag):
    """Best-fit point (column -> value) -> parameters and likelihood chi^2 (all priors are flat)."""
    vals = {k: float(v) for k, v in vals.items()}
    skip = ("weight", "minuslog", "chi2")
    return {"tag": tag, "params": {k: v for k, v in vals.items() if not k.startswith(skip)},
            "chi2_like": vals["chi2"],
            "chi2_parts": {k[6:]: v for k, v in vals.items() if k.startswith("chi2__")}}


def run(info, tag, args):
    import glob
    from cobaya.run import run as cobaya_run
    if args.test:
        tag += "_test"   # never share an output prefix with a real fit
    else:
        # A leftover chain under this prefix (e.g. from an evaluate run) would make the minimizer start from it and
        # take its (singular, one-point) covariance; clear it first.
        for f in glob.glob(os.path.join(args.out, tag + ".*")):
            os.remove(f)
    info = dict(info, output=os.path.join(args.out, tag), force=True)
    upd, sampler = cobaya_run(info)
    if args.test:
        print(f"[{tag}] evaluated OK")
        return {"tag": tag}
    p = sampler.products()["minimum"]
    out = summarize(p.data.iloc[0].to_dict(), tag)
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{tag}] chi2 = {out['chi2_like']:.3f}", flush=True)
    return out


def _profile_point(a):
    args, x, ref = a
    return run(info_for("pbh", args, log10_tev=x, ref=ref), f"profile_{x:05.2f}", args)


def _me_point(a):
    args, x, ref = a
    return run(info_for("me", args, ref=ref, me=x), f"me_{x:.4f}", args)


def _bd_point(a):
    args, x, ref = a
    return run(info_for("bd", args, ref=ref, alpha2=x), "bd_0" if x <= 0 else f"bd_{x:.1e}", args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["install", "lcdm", "profile", "free", "me", "meprofile", "bdprofile", "report"])
    ap.add_argument("--bdgrid", default="0,1e-4,3e-4,1e-3,2e-3,5e-3,1e-2")
    ap.add_argument("--megrid", default="0.97,0.98,0.99,0.995,1.0,1.005,1.01,1.02,1.03")
    ap.add_argument("--packages", default="packages")
    ap.add_argument("--out", default="pbh_out")
    ap.add_argument("--data", default="all", choices=list(LIKES))
    ap.add_argument("--grid", default="11,12,13,14,15,16,16.5,17,17.25,17.5,17.75,18,18.5,19,19.5")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--class-path", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.step == "install":
        sys.exit(subprocess.call(["cobaya-install", *LIKES["all"], "-p", args.packages]))

    if args.step == "lcdm":
        run(info_for("lcdm", args), "lcdm", args)
        return

    ref = None
    lp = os.path.join(args.out, "lcdm.json")
    if os.path.exists(lp):
        ref = json.load(open(lp))["params"]

    if args.step == "profile":
        grid = [float(x) for x in args.grid.split(",")]
        if args.test:
            grid = grid[:1]
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                list(ex.map(_profile_point, [(args, x, ref) for x in grid]))
        else:
            for x in grid:
                _profile_point((args, x, ref))
        return

    if args.step == "free":
        run(info_for("pbh", args, ref=ref), "free", args)
        return

    if args.step == "me":
        me_ref = dict(ref or {})
        pts = [(json.load(open(os.path.join(args.out, fn))), float(fn[3:-5])) for fn in os.listdir(args.out)
               if fn.startswith("me_") and fn.endswith(".json") and fn != "me_free.json"]
        if pts:
            best, x = min(pts, key=lambda t: t[0]["chi2_like"])
            me_ref.update(best["params"])
            me_ref["varying_me"] = x
            print(f"[me] starting from the best profile point, me_ratio = {x}")
        run(info_for("me", args, ref=me_ref), "me_free", args)
        return

    if args.step == "bdprofile":
        grid = [float(x) for x in args.bdgrid.split(",")][: 1 if args.test else None]
        jobs = [(args, x, ref) for x in grid]
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                list(ex.map(_bd_point, jobs))
        else:
            for j in jobs:
                _bd_point(j)
        return

    if args.step == "meprofile":
        grid = [float(x) for x in args.megrid.split(",")][: 1 if args.test else None]
        jobs = [(args, x, ref) for x in grid]
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                list(ex.map(_me_point, jobs))
        else:
            for j in jobs:
                _me_point(j)
        return

    # report
    base = json.load(open(lp))
    print(f"data: {args.data}    LCDM chi2 = {base['chi2_like']:.2f}")
    print(f"{'log10 t_evap/s':>15s} {'t_evap':>12s} {'M_PBH [kg]':>11s} {'best f_pbh':>10s} {'H0':>6s} {'dchi2':>8s}")
    rows = []
    for fn in sorted(os.listdir(args.out)):
        if fn.startswith("profile_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(args.out, fn)))
            x = float(fn[8:-5])
            rows.append((x, r))
    for x, r in rows:
        t = 10 ** x
        tstr = f"{t / YEAR / 1e9:.3g} Gyr" if t > 1e14 * YEAR / 1e6 else f"{t / YEAR:.3g} yr"
        print(f"{x:15.2f} {tstr:>12s} {mass_for_tevap(t):11.2e} {r['params'].get('f_pbh', 0):10.4f} "
              f"{r['params'].get('H0', float('nan')):6.2f} {r['chi2_like'] - base['chi2_like']:8.2f}")
    fp = os.path.join(args.out, "free.json")
    if os.path.exists(fp):
        r = json.load(open(fp))
        x = r["params"]["log10_tevap"] if "log10_tevap" in r["params"] else float("nan")
        print(f"free fit: f_pbh = {r['params'].get('f_pbh', 0):.4f}, log10 t_evap = {x:.2f}, "
              f"dchi2 = {r['chi2_like'] - base['chi2_like']:.2f}  (2 extra parameters)")
    me_rows = sorted((float(fn[3:-5]), json.load(open(os.path.join(args.out, fn))))
                     for fn in os.listdir(args.out) if fn.startswith("me_") and fn.endswith(".json") and fn != "me_free.json")
    if me_rows:
        print(f"\nrest-mass channel: m_e(recombination)/m_e(today)")
        print(f"{'me_ratio':>9s} {'H0':>6s} {'dchi2':>8s}")
        for x, r in me_rows:
            print(f"{x:9.4f} {r['params'].get('H0', float('nan')):6.2f} {r['chi2_like'] - base['chi2_like']:8.2f}")
    fp = os.path.join(args.out, "me_free.json")
    if os.path.exists(fp):
        r = json.load(open(fp))
        print(f"free fit: me_ratio = {r['params'].get('varying_me', float('nan')):.4f}, H0 = {r['params'].get('H0', float('nan')):.2f}, "
              f"dchi2 = {r['chi2_like'] - base['chi2_like']:.2f}  (1 extra parameter; 95% needs < -3.84)")
    if me_rows and os.path.exists(fp):
        bx, br = min(me_rows, key=lambda t: t[1]["chi2_like"])
        if br["chi2_like"] < r["chi2_like"]:
            print(f"  note: the profile point me_ratio = {bx:.4f} is better (dchi2 = {br['chi2_like'] - base['chi2_like']:.2f}); "
                  f"the free fit stopped early, so trust the profile minimum.")
    bd = {}
    for fn in os.listdir(args.out):
        if fn.startswith("bd_") and fn.endswith(".json"):
            bd[0.0 if fn == "bd_0.json" else float(fn[3:-5])] = json.load(open(os.path.join(args.out, fn)))
    if bd:
        print(f"\ndilaton (Brans-Dicke) channel, hi_class; dchi2 relative to its own alpha^2 = 0 run"
              + ("" if 0.0 in bd else " (MISSING: run bdprofile with 0 in --bdgrid)"))
        print(f"{'alpha^2':>9s} {'omega_BD':>10s} {'H0':>6s} {'dchi2':>8s}")
        ref0 = bd.get(0.0)
        for x in sorted(bd):
            r = bd[x]
            d = r["chi2_like"] - ref0["chi2_like"] if ref0 else float("nan")
            print(f"{x:9.1e} {omega_bd_for(x):10.4g} {r['params'].get('H0', float('nan')):6.2f} {d:8.2f}")
        print("  (Cassini: alpha^2 < ~1.2e-5 in the solar system unless screened; 1 extra parameter, 95% needs dchi2 < -3.84)")
    print("Reading: dchi2 near 0 everywhere = no improvement; the law is then only constrained, via the f_pbh "
          "upper limit. A real improvement would need dchi2 well below -6 (2 extra parameters).")


if __name__ == "__main__":
    main()
