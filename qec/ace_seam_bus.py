# The crossbar as a coherent "seam bus": a full logical qubit spanning every patch boundary
#
# By (Anthropic) Claude and Dan Strano
#
# The coherence wall: ACE's patch simulators only exchange classical information, so no state can
# hold entanglement between bulk qubits of different patches (ace_shor_9_bloch.py: a code straddling
# a seam sits at the 2/3 entanglement-breaking bound). But with use_crossbars=True (the default),
# ACE already allocates ONE more exact simulator -- the crossbar -- that holds a replica of EVERY
# boundary qubit in the grid. Seam-to-seam couplings, even between seams on different boundaries,
# are real gates in the crossbar. What was missing was a decoder that knows the crossbar copy is the
# one carrying the whole seam-to-seam history.
#
# Lineage-tracking authority (updated qrack_ace_backend.py): a replica stays "clean" only while every
# coupling it took part in was a real gate with a clean replica of its partner; when two boundary
# qubits couple, a clean crossbar pair becomes the sole authority. With that, seam-only circuits are
# exact across every boundary of the grid, and a code living only on seam qubits protects BOTH bases
# even though its blocks span all patch boundaries.
#
# What this is not: it does not break the theorem. The crossbar is an exact simulator, so its cost is
# exponential in the entanglement held among boundary qubits (Qrack's QUnit factorization keeps it
# cheap while that stays low). Bulk-qubit information still cannot enter it -- bulk qubits have no
# crossbar replica. It is a bridge that was already built; this makes it usable.
#
# Layout: 3x9, long_range_columns=2 (torus): three patches, three seam columns (every boundary), and
# 9 seam qubits. Tests:
#   1. Bell fidelity between seams on different boundaries.
#   2. GHZ over one seam from each boundary (chain and star orders), fidelity lower bound.
#   3. Shor-9 on the 9 seam qubits, each block spanning all three boundaries: logical Bloch vectors
#      for the six axis states and a halfway Z/X state, and the average fidelity vs the 2/3 bound.
#
# Usage: python3 ace_seam_bus.py [shots=1024] [module=pyqrack]

import importlib
import math
import sys
from collections import Counter

from pyqrack import QrackSimulator

import ace_shor_9_bloch as bloch_test

N = 27
KW = dict(long_range_columns=2)


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    mod = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "pyqrack")
    make = lambda n: mod.QrackAceBackend(n, **KW)

    probe = make(N)
    reps = {q: [r[0] for r in probe.qubits[q]] for q in range(N)}
    seams = [q for q in range(N) if len(reps[q]) > 1]
    cols = sorted({q % probe.get_row_length() for q in seams})
    by_col = [[q for q in seams if q % probe.get_row_length() == c] for c in cols]
    if len(by_col) != 3 or any(len(c) != 3 for c in by_col):
        raise RuntimeError(f"Unexpected layout: seams {seams}")
    print(f"seam columns (one per boundary): {by_col}; replicas {[(q, reps[q]) for q in seams]}")

    a, b, c = by_col[0][0], by_col[1][0], by_col[2][0]

    def corr(prep, qs, basis):
        sim = make(N)
        prep(sim)
        for q in qs:
            if basis == "Y":
                sim.adjs(q)
            if basis in "XY":
                sim.h(q)
        cnt = Counter(sim.measure_shots(qs, shots))
        return cnt, sum(v * (1 - 2 * (k.bit_count() & 1)) for k, v in cnt.items()) / shots

    print("\n1. Bell fidelity between seams on different boundaries")
    for x, y in ((a, b), (b, c), (a, c)):
        prep = lambda m, x=x, y=y: (m.h(x), m.mcx([x], y))
        e = {B: corr(prep, [x, y], B)[1] for B in "XYZ"}
        print(f"   seams {x},{y} (replicas {reps[x]} / {reps[y]}): F = {(1 + e['X'] - e['Y'] + e['Z']) / 4:.3f}")

    print("\n2. GHZ over one seam per boundary")
    for name, prep in (("chain", lambda m: (m.h(a), m.mcx([a], b), m.mcx([b], c))),
                       ("star", lambda m: (m.h(a), m.mcx([a], b), m.mcx([a], c)))):
        z, _ = corr(prep, [a, b, c], "Z")
        _, par = corr(prep, [a, b, c], "X")
        pop = (z.get(0, 0) + z.get(7, 0)) / shots
        print(f"   {name}: population {pop:.3f}, X parity {par:+.3f}, F >= {(pop + par) / 2:.3f}")

    print("\n3. Shor-9 on seam qubits only, each block spanning all three boundaries")
    blocks = [[by_col[0][i], by_col[1][i], by_col[2][i]] for i in range(3)]
    print(f"   blocks {blocks}")
    fs = []
    for label in ("+z", "-z", "+x", "-x", "+y", "-y", "halfway z/x"):
        th, ph = bloch_test.STATES[label]
        r = bloch_test.bloch(make, N, blocks, th, ph, shots)
        r0 = bloch_test.bloch(lambda m: QrackSimulator(m), N, blocks, th, ph, min(shots, 256))
        f = (1 + sum(p * q for p, q in zip(r, r0))) / 2
        if label != "halfway z/x":
            fs.append(f)
        print(f"   {label:12s} <X_L> {r[0]:+.3f}  <Y_L> {r[1]:+.3f}  <Z_L> {r[2]:+.3f}   F {f:.3f}")
    avg = sum(fs) / len(fs)
    print(f"   average fidelity over the 6 axis states: {avg:.3f} "
          f"({'beats' if avg > 2 / 3 + 0.03 else 'does not beat'} the entanglement-breaking bound 2/3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
