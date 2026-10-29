# Logical Bloch-sphere fidelity of Shor-9 on QrackBridge, including inputs rotated halfway
# between the Z and X axes -- does any input orientation get "partial coherence in both bases"?
#
# By (Anthropic) Claude and Dan Strano
#
# For each logical input |psi> = cos(th/2)|0> + e^{i ph} sin(th/2)|1> (prepared on the input leader
# with u(th, ph, 0), then encoded), the script estimates the logical Bloch vector:
#   <Z_L> decoded (X-basis readout, majority of block signs)
#   <X_L> decoded (Z-basis readout, parity of block majorities)
#   <Y_L> raw     (one representative, Y.X.X on block 0 times Z on the other two block leaders)
# and the state fidelity F = (1 + r_ace . r_exact) / 2, where r_exact is the same estimator run on an
# exact QrackSimulator (so all sign conventions calibrate themselves).
#
# Benchmark: the average fidelity over the six axis states (= the Bloch-sphere average for a
# Pauli-type channel). ANY entanglement-breaking (measure-and-prepare) channel has average fidelity
# <= 2/3. A code straddling a patch boundary on ACE can only beat 2/3 if logical coherence really
# crosses the boundary; ~2/3 means "one basis perfect, the other dephased", which is the classical
# bound -- rotating the input or the code only redistributes fidelity around the sphere.
#
# Layouts:
#   straddle : 3x3, long_range_columns=1, is_torus=False; blocks [seam column], [patch-0 bulk],
#              [patch-1 bulk] (ace_shor_9.py "patch_blocks")
#   one_side : 3x5, long_range_columns=2, is_torus=False; seam column + patch-0 bulk only
#              (ace_shor_9_one_side.py "seam_block"), the control that should be ~1.0
#
# Usage: python3 ace_shor_9_bloch.py [shots] [module]
# module: optional Python module to import QrackBridge from (default: pyqrack)

import importlib
import math
import sys
from collections import Counter

from pyqrack import QrackSimulator

LAYOUTS = {
    "straddle": (9, dict(wrap=False), [[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
}
STATES = {
    "+z": (0, 0), "-z": (math.pi, 0), "+x": (math.pi / 2, 0), "-x": (math.pi / 2, math.pi),
    "+y": (math.pi / 2, math.pi / 2), "-y": (math.pi / 2, -math.pi / 2),
    "halfway z/x": (math.pi / 4, 0), "halfway -z/x": (3 * math.pi / 4, 0),
}


def cx(sim, a, b):
    sim.mcx([a], b)


def encode(sim, blocks, th, ph):
    a, b, c = (blk[0] for blk in blocks)
    sim.u(a, th, ph, 0)
    cx(sim, a, b)
    cx(sim, a, c)
    for q in (a, b, c):
        sim.h(q)
    for blk in blocks:
        cx(sim, blk[0], blk[1])
        cx(sim, blk[0], blk[2])


def expectation(make, n, blocks, th, ph, which, shots):
    sim = make(n)
    encode(sim, blocks, th, ph)
    code = [q for blk in blocks for q in blk]
    if which == "Z":
        rot = {q: "X" for q in code}
    elif which == "X":
        rot = {q: "Z" for q in code}
    else:
        b0, b1, b2 = blocks
        rot = {b0[0]: "Y", b0[1]: "X", b0[2]: "X", b1[0]: "Z", b2[0]: "Z"}
    for q, basis in rot.items():
        if basis == "Y":
            sim.adjs(q)
        if basis in "XY":
            sim.h(q)
    qs = sorted(rot)
    pos = {q: i for i, q in enumerate(qs)}
    total = 0
    for k, v in Counter(sim.measure_shots(qs, shots)).items():
        bits = [[(k >> pos[q]) & 1 for q in blk if q in pos] for blk in blocks]
        if which == "Z":
            val = 1 if sum(sum(b) & 1 for b in bits) >= 2 else 0     # majority of block signs
        elif which == "X":
            val = sum(1 if sum(b) >= 2 else 0 for b in bits) & 1     # parity of block majorities
        else:
            val = k.bit_count() & 1                                   # raw parity of the 5 qubits
        total += v * (1 - 2 * val)
    return total / shots


def bloch(make, n, blocks, th, ph, shots):
    return [expectation(make, n, blocks, th, ph, w, shots) for w in "XYZ"]


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    mod = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "pyqrack")
    for name, (n, kw, blocks) in LAYOUTS.items():
        ace = lambda m, kw=kw: mod.QrackBridge(m, max_bonds=3, **kw)
        exact = lambda m: QrackSimulator(m)
        print(f"{name}: {n} qubits {kw}, blocks {blocks}, {shots} shots per Pauli")
        print(f"   {'input':14s} {'<X_L>':>7s} {'<Y_L>':>7s} {'<Z_L>':>7s}   F")
        axis_f = []
        for label, (th, ph) in STATES.items():
            r = bloch(ace, n, blocks, th, ph, shots)
            r0 = bloch(exact, n, blocks, th, ph, min(shots, 512))
            f = (1 + sum(x * y for x, y in zip(r, r0))) / 2
            if label in ("+z", "-z", "+x", "-x", "+y", "-y"):
                axis_f.append(f)
            print(f"   {label:14s} {r[0]:+7.3f} {r[1]:+7.3f} {r[2]:+7.3f}   {f:.3f}")
        avg = sum(axis_f) / len(axis_f)
        verdict = "beats" if avg > 2 / 3 + 0.03 else "does not beat"
        print(f"   average fidelity over the 6 axis states: {avg:.3f}  ({verdict} the entanglement-breaking bound 2/3)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
