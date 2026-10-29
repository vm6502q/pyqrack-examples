# Small-scale Pauli tomography of QrackBridge states, with basis changes done by simulator API gates.
#
# By (Anthropic) Claude and Dan Strano
#
# For a random n-qubit circuit (default n=6, 2 patches), every one of the 3^n measurement settings is
# prepared on a CLONE of the simulator by API gates (X: h; Y: adjs then h; Z: nothing) and read out,
# giving all 4^n Pauli expectation values <P> for the bridge and for an exact QrackSimulator. Readout is
# exact probabilities by default (infinite-shot limit), or measure_shots with [shots] > 0.
#
# Reported per circuit:
#   est      : the bridge's fidelity_estimate (product of retained Schmidt weight per truncation)
#   F_tomo   : observable-defined fidelity  Tr(rho_ideal rho_bridge) = 2^-n sum_P <P>_ideal <P>_bridge
#   F_ovl    : direct state overlap |<ideal|bridge>|^2 (equals F_tomo for pure states; a cross-check)
#   XEB      : normalized linear XEB of the bridge's Z-basis distribution (nn_qb.py's estimator)
#   w=k      : agreement of the weight-k observables, sum_{|P|=k} <P>_i <P>_b / sum_{|P|=k} <P>_i^2
#              (1 = those observables reproduced; 0 = uncorrelated). This is "fidelity" restricted to
#              k-body observables; F_tomo weights every Pauli weight by how many strings it has.
#
# Usage: python3 bridge_tomography.py [n=6] [depth=8] [max_bonds=1,2] [trials=3] [shots=0] [module=pyqrack]

import importlib
import itertools
import math
import random
import sys
from collections import Counter

import numpy as np
from pyqrack import QrackSimulator

GATES = ("cx", "cy", "cz", "acx", "acy", "acz", "swap", "iswap", "adjiswap", "pswap", "mswap", "nswap")


def apply2(sim, g, a, b):
    if g in ("cx", "cy", "cz"):
        getattr(sim, "mc" + g[1])([a], b)
    elif g in ("acx", "acy", "acz"):
        getattr(sim, "mac" + g[2])([a], b)
    elif g in ("swap", "iswap", "adjiswap"):
        getattr(sim, g)(a, b)
    else:
        sim.mcz([a], b)
        getattr(sim, {"pswap": "swap", "mswap": "iswap", "nswap": "adjiswap"}[g])(a, b)


def circuit(n, depth, rng):
    """Brickwork on a ring of n qubits, Haar-random u on every qubit each layer, random gate per coupler."""
    ops = []
    for d in range(depth):
        for q in range(n):
            ops.append(("u", q, math.acos(rng.uniform(-1, 1)), rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)))
        for q in range(d % 2, n, 2):
            ops.append((rng.choice(GATES), q, (q + 1) % n))
    return ops


def run(sim, ops):
    for op in ops:
        if op[0] == "u":
            sim.u(op[1], *op[2:])
        else:
            apply2(sim, *op)


def probs_in_setting(sim, setting, shots, rng):
    s = sim.clone()
    for q, b in enumerate(setting):
        if b == "Y":
            s.adjs(q)
        if b in "XY":
            s.h(q)
    n = len(setting)
    if shots:
        p = np.zeros(1 << n)
        for k, v in Counter(s.measure_shots(list(range(n)), shots)).items():
            p[k] = v / shots
        return p
    v = np.asarray(s.out_ket())
    p = np.abs(v) ** 2
    return p / p.sum()


def pauli_expectations(sim, n, shots=0, rng=None):
    """<P> for all 4^n Pauli strings (index: tuple over qubits of 'I','X','Y','Z')."""
    idx = np.arange(1 << n)
    parity = {m: 1 - 2 * (np.array([bin(x & m).count("1") & 1 for x in idx])) for m in range(1 << n)}
    out = {}
    for setting in itertools.product("XYZ", repeat=n):
        p = probs_in_setting(sim, setting, shots, rng)
        for mask in range(1 << n):
            label = tuple(setting[q] if (mask >> q) & 1 else "I" for q in range(n))
            if label not in out:
                out[label] = float(np.dot(p, parity[mask]))
    return out


def main():
    a = sys.argv[1:]
    n = int(a[0]) if len(a) > 0 else 6
    depth = int(a[1]) if len(a) > 1 else 8
    bonds = [int(x) for x in a[2].split(",")] if len(a) > 2 else [1, 2]
    trials = int(a[3]) if len(a) > 3 else 3
    shots = int(a[4]) if len(a) > 4 else 0
    mod = importlib.import_module(a[5] if len(a) > 5 else "pyqrack")
    print(f"n={n}, depth {depth}, 2 patches, {3 ** n} measurement settings, "
          f"{'exact probabilities' if not shots else f'{shots} shots per setting'}")
    hdr = " ".join(f"{'w=' + str(k):>6s}" for k in range(1, n + 1))
    print(f"   {'max_bonds':>9s} {'circ':>4s} | {'est':>7s} {'F_tomo':>7s} {'F_ovl':>7s} {'XEB':>7s} | {hdr}")
    for t in range(trials):
        ops = circuit(n, depth, random.Random(7 + t))
        ideal = QrackSimulator(n)
        run(ideal, ops)
        Pi = pauli_expectations(ideal, n)
        vi = np.asarray(ideal.out_ket())
        pi = np.abs(vi) ** 2
        for b in bonds:
            br = mod.QrackBridge(n, max_bonds=b, patches=2)
            run(br, ops)
            Pb = pauli_expectations(br, n, shots, random.Random(t))
            vb = np.asarray(br.out_ket())
            f_tomo = sum(Pi[k] * Pb[k] for k in Pi) / 2 ** n
            f_ovl = abs(np.vdot(vi, vb)) ** 2
            ic = pi - pi.mean()
            xeb = float(np.dot(ic, np.abs(vb) ** 2) / np.dot(ic, ic))
            agree = []
            for k in range(1, n + 1):
                keys = [P for P in Pi if sum(c != "I" for c in P) == k]
                den = sum(Pi[P] ** 2 for P in keys)
                agree.append(sum(Pi[P] * Pb[P] for P in keys) / den if den > 1e-12 else float("nan"))
            cols = " ".join(f"{x:6.3f}" for x in agree)
            print(f"   {b:9d} {t:4d} | {br.fidelity_estimate:7.4f} {f_tomo:7.4f} {f_ovl:7.4f} {xeb:7.4f} | {cols}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
