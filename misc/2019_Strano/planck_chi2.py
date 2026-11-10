# Planck 2018 chi^2 (low-l TT, low-l EE, plik-lite TTTEEE) for CLASS lensed C_l output files, via Cobaya.
# By (Anthropic) Claude and Dan Strano.   Usage: python3 planck_chi2.py PACKAGES_PATH root1 [root2 ...]
#   rootN: CLASS output root such that rootN_cl_lensed.dat exists (e.g. out/crt60_00)
import sys
import numpy as np
from cobaya.model import get_model
from cobaya.theory import Theory

TCMB_MUK = 2.7255e6


class FileCls(Theory):
    root = None
    params = {"cls_file_dummy": None}   # Cobaya requires every component to depend on some parameter

    def initialize(self):
        d = np.loadtxt(self.root + "_cl_lensed.dat")
        l = d[:, 0].astype(int)
        lmax = 2600          # plik-lite reads to l = 2508: extend by constant D_l (negligible weight)
        self._cl = {"ell": np.arange(lmax + 1)}
        lx = np.arange(l.min(), lmax + 1)
        for i, name in [(1, "tt"), (2, "ee"), (3, "te"), (4, "bb")]:
            c = np.zeros(lmax + 1)
            Dl = np.concatenate([d[:, i], np.full(lmax - l.max(), d[-1, i])])
            c[lx] = Dl * 2 * np.pi / (lx * (lx + 1.0))
            self._cl[name] = c

    def get_can_provide(self):
        return ["Cl"]

    def must_provide(self, **requirements):
        return {}

    def calculate(self, state, want_derived=True, **params):
        state["Cl"] = self._cl

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        out = {"ell": self._cl["ell"]}
        fac = TCMB_MUK ** 2 if units in ("FIRASmuK2", "muK2") else 1.0
        l = self._cl["ell"]
        ef = l * (l + 1.0) / (2 * np.pi) if ell_factor else 1.0
        for k in ("tt", "ee", "te", "bb"):
            out[k] = self._cl[k] * fac * ef
        return out


def chi2(packages, root, A_planck=1.0):
    info = {"packages_path": packages,
            "likelihood": {"planck_2018_lowl.TT": None, "planck_2018_lowl.EE": None,
                           "planck_2018_highl_plik.TTTEEE_lite_native": None},
            "theory": {"files": {"external": FileCls, "root": root}},
            "params": {"A_planck": A_planck, "cls_file_dummy": 0.0}}
    model = get_model(info)
    lp = model.loglikes({}, as_dict=True, cached=False)[0]
    return {k: -2 * float(v) for k, v in lp.items()}


if __name__ == "__main__":
    pk = sys.argv[1]
    res = {r: chi2(pk, r) for r in sys.argv[2:]}
    base = sys.argv[2]
    for r, c in res.items():
        tot = sum(c.values())
        dif = {k: c[k] - res[base][k] for k in c}
        print(f"{r:28s} total {tot:9.3f}  " + "  ".join(f"{k.split('.')[-1]}: {c[k]:8.3f} ({dif[k]:+.3f})" for k in c))
