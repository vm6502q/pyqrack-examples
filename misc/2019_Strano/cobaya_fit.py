# Cobaya best-fit comparison: created-neutrino (Strano) model at self-consistent T_c vs LCDM, Planck 2018
# (low-l TT, low-l EE, plik-lite TTTEEE). By (Anthropic) Claude and Dan Strano.
#
# One full-model CLASS call costs ~5 min (the extra Boltzmann hierarchies), so the model is minimized by TEMPLATE ITERATION:
#   C_l^model(theta) ~= C_l^LCDM(theta) * R_l(theta_k),   R_l = C_l^model / C_l^LCDM computed with the full patched CLASS
#   at the current best fit theta_k (TT, EE multiplicative; TE additive in correlation units). Minimize, recompute R_l at the
#   new best fit, repeat until the best fit stops moving; then evaluate the FULL model chi^2 at the final point.
#   The effect is a smooth low-l (late-ISW) correction depending weakly on theta, so this converges to the true minimum
#   (the final full-model chi^2 is reported, not the template one).
#
# Needs the patched classy importable (PYTHONPATH=<class>/python/build/lib.*) and the patched `class` binary.
# Usage:
#   python3 cobaya_fit.py lcdm     --packages PK [--m1 0]
#   python3 cobaya_fit.py model    --packages PK --class-bin CLASS [--dof 1] [--m1 0] [--iters 2]
#   python3 cobaya_fit.py report
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "cobaya_out")
DM21, DM31 = 7.42e-5, 2.51e-3
PARAM_KEYS = ["logA", "n_s", "theta_s_100", "omega_b", "omega_cdm", "tau_reio", "A_planck"]


def masses(m1):
    return [max(m1, 1e-4), float(np.sqrt(m1**2 + DM21)), float(np.sqrt(m1**2 + DM31))]


def nu_args(m1):
    m = masses(m1)
    return {"N_ur": 0.00441, "N_ncdm": 3, "m_ncdm": ",".join(f"{x:.6g}" for x in m), "T_ncdm": "0.71611,0.71611,0.71611"}


def make_template_theory():
    from cobaya.theories.classy import classy as CobayaClassy

    class classy_template(CobayaClassy):
        template_file = None

        def initialize(self):
            super().initialize()
            self._tpl = dict(np.load(self.template_file)) if self.template_file else None

        def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
            cl = super().get_Cl(ell_factor=ell_factor, units=units)
            if self._tpl is None:
                return cl
            out = {k: (np.array(v, copy=True) if k != "ell" else v) for k, v in cl.items()}
            n = min(len(out["tt"]), len(self._tpl["rTT"]))
            tt0, ee0 = out["tt"][:n].copy(), out["ee"][:n].copy()
            out["tt"][:n] *= self._tpl["rTT"][:n]
            out["ee"][:n] *= self._tpl["rEE"][:n]
            out["te"][:n] += self._tpl["dTEn"][:n] * np.sqrt(np.abs(tt0 * ee0))
            return out

    return classy_template


def info(packages, m1, template=None):
    params = {
        "logA": {"prior": {"min": 2.9, "max": 3.2}, "ref": 3.044, "proposal": 0.014, "drop": True, "latex": r"\log(10^{10}A_s)"},
        "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)"},
        "n_s": {"prior": {"min": 0.9, "max": 1.03}, "ref": 0.9649, "proposal": 0.004},
        "theta_s_100": {"prior": {"min": 1.03, "max": 1.05}, "ref": 1.04192, "proposal": 0.0003, "drop": True},
        "100*theta_s": {"value": "lambda theta_s_100: theta_s_100", "derived": False},
        "omega_b": {"prior": {"min": 0.019, "max": 0.025}, "ref": 0.02237, "proposal": 0.00015},
        "omega_cdm": {"prior": {"min": 0.10, "max": 0.14}, "ref": 0.1200, "proposal": 0.0012},
        "tau_reio": {"prior": {"min": 0.01, "max": 0.10}, "ref": 0.0544, "proposal": 0.007},
        "A_planck": {"prior": {"dist": "norm", "loc": 1.0, "scale": 0.0025}, "ref": 1.0, "proposal": 0.002},
        "H0": {"latex": "H_0"},
    }
    th = {"extra_args": nu_args(m1)}
    if template:
        th.update({"external": make_template_theory(), "template_file": template})
        theory = {"classy_tpl": th}
    else:
        theory = {"classy": th}
    return {"packages_path": packages, "theory": theory, "params": params,
            "likelihood": {"planck_2018_lowl.TT": None, "planck_2018_lowl.EE": None,
                           "planck_2018_highl_plik.TTTEEE_lite_native": None},
            "sampler": {"minimize": {"method": "bobyqa", "best_of": 1}}}


def minimize(inf, prefix, start=None):
    from cobaya.run import run
    if start:
        for k in PARAM_KEYS:
            inf["params"][k]["ref"] = start[k]
    inf["output"] = prefix
    inf["force"] = True
    _, sampler = run(inf)
    p = sampler.products()["minimum"]
    best = {k: float(p[k]) for k in PARAM_KEYS}
    best["H0"] = float(p["H0"])
    return best, float(-2 * p["minuslogpost"]), float(p["chi2"])


def class_ini(path, root, best, m1, created, a):
    lines = [f"ln10^{{10}}A_s = {best['logA']:.8g}", f"n_s = {best['n_s']:.8g}", f"100*theta_s = {best['theta_s_100']:.8g}",
             f"omega_b = {best['omega_b']:.8g}", f"omega_cdm = {best['omega_cdm']:.8g}", f"tau_reio = {best['tau_reio']:.8g}"]
    lines += [f"{k} = {v}" for k, v in nu_args(m1).items()]
    lines += ["output = tCl, pCl, lCl", "lensing = yes", "l_max_scalars = 2600", f"root = {root}_"]
    if created:
        lines += ["created_neutrinos = yes", f"created_ghost_dof = {a.dof}", "created_q_per_decade = 6", "created_l_max = 8",
                  "created_k_max = 0.05", "created_z_on = 100", "perturbations_verbose = 1"]
    open(path, "w").write("\n".join(lines) + "\n")


def run_class(a, best, m1, created, tag):
    root = os.path.join(OUT, tag)
    class_ini(root + ".ini", root, best, m1, created, a)
    p = subprocess.run([a.class_bin, root + ".ini"], capture_output=True, text=True)
    if p.returncode != 0 or "Error" in p.stdout:
        raise RuntimeError(p.stdout[-800:] + p.stderr[-400:])
    open(root + ".log", "w").write(p.stdout)
    return root + "_00"


def template(a, best, m1, k):
    rm = run_class(a, best, m1, True, f"model_m1_{m1:g}_it{k}")
    rl = run_class(a, best, m1, False, f"lcdmref_m1_{m1:g}_it{k}")
    dm, dl = np.loadtxt(rm + "_cl_lensed.dat"), np.loadtxt(rl + "_cl_lensed.dat")
    l = dm[:, 0].astype(int)
    n = l.max() + 1
    rTT, rEE, dTEn = np.ones(n), np.ones(n), np.zeros(n)
    rTT[l] = dm[:, 1] / dl[:, 1]
    rEE[l] = dm[:, 2] / dl[:, 2]
    dTEn[l] = (dm[:, 3] - dl[:, 3]) / np.sqrt(dl[:, 1] * dl[:, 2])
    f = os.path.join(OUT, f"template_m1_{m1:g}_it{k}.npz")
    np.savez(f, rTT=rTT, rEE=rEE, dTEn=dTEn)
    return f, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lcdm", "model", "report"])
    ap.add_argument("--packages", default="packages"); ap.add_argument("--class-bin", default="class_public/class")
    ap.add_argument("--dof", type=float, default=1.0); ap.add_argument("--m1", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    resf = os.path.join(OUT, "results.json")
    res = json.load(open(resf)) if os.path.exists(resf) else {}
    key = f"m1_{a.m1:g}"

    if a.step == "lcdm":
        best, mlp, chi2 = minimize(info(a.packages, a.m1), os.path.join(OUT, f"lcdm_{key}"))
        res[f"lcdm_{key}"] = {"best": best, "chi2": chi2, "minus2logpost": mlp}
        print("LCDM best fit:", best, "chi2 =", chi2, flush=True)

    elif a.step == "model":
        from planck_chi2 import chi2 as full_chi2
        best = res[f"lcdm_{key}"]["best"]
        hist = []
        for k in range(a.iters):
            tf, _ = template(a, best, a.m1, k)
            best, mlp, chi2_t = minimize(info(a.packages, a.m1, template=tf), os.path.join(OUT, f"model_{key}_it{k}"), start=best)
            hist.append({"iter": k, "best": best, "chi2_template": chi2_t})
            print(f"iteration {k}: template chi2 = {chi2_t:.3f}, best = {best}", flush=True)
        # Validation, like for like: the CLASS-binary pipeline has a different precision offset from Cobaya's classy, so
        # compare model - LCDM within EACH pipeline at the final point theta_M (template vs full Boltzmann calculation).
        rm = run_class(a, best, a.m1, True, f"model_m1_{a.m1:g}_final")
        rl = run_class(a, best, a.m1, False, f"lcdm_m1_{a.m1:g}_final")
        cm, cl = full_chi2(a.packages, rm, A_planck=best["A_planck"]), full_chi2(a.packages, rl, A_planck=best["A_planck"])
        d_file = sum(cm.values()) - sum(cl.values())
        from cobaya.model import get_model
        pt = {k: best[k] for k in PARAM_KEYS}
        vals = {}
        for tpl in (None, tf):
            inf = info(a.packages, a.m1, template=tpl); inf.pop("sampler")
            vals[tpl] = -2 * sum(get_model(inf).loglikes(pt, as_dict=True)[0].values())
        d_tpl = vals[tf] - vals[None]
        chi2_full = vals[tf]
        Tc = None
        for line in open(os.path.join(OUT, f"model_m1_{a.m1:g}_final.log")):
            if "T_c = " in line and "(ghost" in line:
                Tc = float(line.split("T_c = ")[1].split(" K")[0])
        res[f"model_{key}_dof{a.dof:g}"] = {"best": best, "chi2_full": chi2_full, "Tc_K": Tc, "history": hist,
                                             "dchi2_at_thetaM_full_class": d_file, "dchi2_at_thetaM_template": d_tpl}
        print(f"validation at theta_M: full CLASS model-LCDM {d_file:+.3f} vs template {d_tpl:+.3f}", flush=True)
        print(f"MODEL chi2_min = {chi2_full:.3f} (T_c = {Tc} K)", flush=True)

    json.dump(res, open(resf, "w"), indent=1)
    if a.step == "report" or a.step == "model":
        for k, v in res.items():
            if k.startswith("model"):
                l0 = res.get("lcdm_" + k.split("_")[1] + "_" + k.split("_")[2])
                if l0:
                    print(f"{k}: chi2_min {v['chi2_full']:.3f} vs LCDM {l0['chi2']:.3f}  ->  dchi2_min = {v['chi2_full'] - l0['chi2']:+.3f}")
                    print("   best-fit shifts:", {p: round(v['best'][p] - l0['best'][p], 6) for p in PARAM_KEYS + ['H0']})


if __name__ == "__main__":
    main()
