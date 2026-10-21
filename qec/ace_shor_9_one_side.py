# One unit of Shor's 9-qubit code on QrackAceBackend, with all bulk qubits taken from ONE patch
#
# By (Anthropic) Claude and Dan Strano
#
# Dan's layout suggestion: source every bulk qubit of the code from the same single patch, on one
# side of a seam. Seam qubits keep a replica in that patch's simulator, so the whole code state can
# live in one exact state vector; the only things left to reconcile are the seam qubits' other
# replicas (far-side patch and crossbar), which is exactly the kind of error a code can absorb.
# This avoids the cross-patch bulk-bulk entanglement that ace_cross_patch_entanglement.py shows ACE
# cannot hold (F ~= 0.25), which is what kept logical X at chance in ace_shor_9.py.
#
# Grid: 3x5, long_range_columns=2, is_torus=False -> one seam column of 3 qubits and 6 bulk qubits
# in each patch. Code qubits: the 3 seams + the 6 bulk qubits of one patch. The other patch's
# bulk qubits are idle. Roles are read from the backend and every CNOT is checked against the
# logical coupling map.
#
# Encoder and decoding as in ace_shor_9.py:
#   * logical X (|+_L>, |-_L>): Z-basis readout, majority per block, parity of block bits
#   * logical Z (|0_L>, |1_L>): X-basis readout, parity per block = sign, majority of signs
#   "raw" = parity of all nine outcomes with no correction. 1.0 perfect, 0.5 no information.
#
# Layouts tried (blocks listed leader first; the input goes on the first block's leader):
#   seam_leaders : each block = one seam + two bulk  (outer code seam-to-seam)
#   seam_block   : one block is the seam column, two all-bulk blocks
#   bulk_leaders : each block = bulk leader + bulk + one seam (seams only as inner members)
#   *_far        : the same three, using the other patch's bulk instead, as a mirror check
#
# Findings (2048 shots per logical state, two runs):
#   seam_block (patch-0 bulk) decoded 1.000 on all four logical states in both runs: a full
#   logical qubit, both bases, with raw (uncorrected) success ~0.97 for Z and ~0.76-0.80 for X,
#   so the decoder is doing real work on X. The other layouts all reach decoded X = 1.000 at least
#   once, but one basis wobbles between runs (0.58-0.97). The far patch is consistently worse for
#   the seam_block layout (X ~= 0.83-0.89); the seams' replica order differs (seam 7 lists patch 1
#   first, seams 2 and 12 list patch 0 first), which is the likely asymmetry.
# This supersedes the conclusion in ace_cross_patch_entanglement.py that a full logical qubit
# needs a backend extension: no extension is needed if the code never asks for bulk-bulk
# entanglement across patches.

import sys
import time

from collections import Counter

from pyqrack import QrackAceBackend, QrackSimulator

WIDTH = 15
ACE_KW = dict(long_range_columns=2, is_torus=False)


def _replicas(sim, q):
    return [r[0] for r in (sim.qubits if hasattr(sim, "qubits") else sim._qubits)[q]]


def roles(ace):
    reps = {q: _replicas(ace, q) for q in range(WIDTH)}
    seams = [q for q in range(WIDTH) if len(reps[q]) > 1]
    patches = {}
    for q in range(WIDTH):
        if len(reps[q]) == 1:
            patches.setdefault(reps[q][0], []).append(q)
    if len(seams) != 3 or any(len(v) != 6 for v in patches.values()):
        raise RuntimeError(f"Unexpected layout: seams {seams}, patches {patches}")
    return reps, seams, [sorted(v) for _, v in sorted(patches.items())]


def layouts(seams, bulk, tag):
    s, b = seams, bulk
    return {
        f"seam_leaders{tag}": [[s[0], b[0], b[1]], [s[1], b[2], b[3]], [s[2], b[4], b[5]]],
        f"seam_block{tag}": [[s[1], s[0], s[2]], [b[0], b[1], b[2]], [b[3], b[4], b[5]]],
        f"bulk_leaders{tag}": [[b[0], b[1], s[0]], [b[2], b[3], s[1]], [b[4], b[5], s[2]]],
    }


def encoder(blocks):
    a, b, c = (blk[0] for blk in blocks)
    ops = [("cx", a, b), ("cx", a, c), ("h", a), ("h", b), ("h", c)]
    for blk in blocks:
        ops += [("cx", blk[0], blk[1]), ("cx", blk[0], blk[2])]
    return ops


def check_coupling(ace, ops):
    cm = {tuple(p) for p in ace.get_logical_coupling_map()}
    bad = [(c, t) for g, c, *rest in ops if g == "cx" for t in rest if (c, t) not in cm and (t, c) not in cm]
    if bad:
        raise RuntimeError(f"Encoder uses couplings outside the logical coupling map: {bad}")


def run(sim, blocks, state, shots):
    code = [q for blk in blocks for q in blk]
    a = blocks[0][0]
    if state in ("1", "-"):
        sim.x(a)
    if state in ("+", "-"):
        sim.h(a)
    for g, *qs in encoder(blocks):
        if g == "h":
            sim.h(qs[0])
        else:
            sim.mcx([qs[0]], qs[1])
    if state in ("0", "1"):
        for q in code:
            sim.h(q)
    counts = Counter(sim.measure_shots(code, shots))
    pos = {q: i for i, q in enumerate(code)}

    raw_ok = dec_ok = 0
    for k, v in counts.items():
        bits = [[(k >> pos[q]) & 1 for q in blk] for blk in blocks]
        raw_parity = sum(sum(b) for b in bits) & 1
        if state in ("+", "-"):
            decoded = sum(1 if sum(b) >= 2 else 0 for b in bits) & 1
            expect = 0 if state == "+" else 1
        else:
            decoded = 1 if sum(sum(b) & 1 for b in bits) >= 2 else 0
            expect = 0 if state == "0" else 1
        raw_ok += v * (raw_parity == expect)
        dec_ok += v * (decoded == expect)
    return raw_ok / shots, dec_ok / shots


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    start = time.perf_counter()
    probe = QrackAceBackend(WIDTH, **ACE_KW)
    reps, seams, (pa, pb) = roles(probe)
    lays = {**layouts(seams, pa, ""), **layouts(seams, pb, "_far")}
    for blocks in lays.values():
        check_coupling(probe, encoder(blocks))
    del probe

    print(f"Shor-9 on ACE (3x5, long_range_columns=2, is_torus=False), {shots} shots per logical state")
    print(f"seam replicas {[(q, reps[q]) for q in seams]}; patch-0 bulk {pa}; patch-1 bulk {pb}")
    print("success per logical state as raw / decoded  (1.0 perfect, 0.5 no information)")
    rows = [(n, b, lambda: QrackAceBackend(WIDTH, **ACE_KW)) for n, b in lays.items()]
    rows.append(("exact_reference", lays["seam_leaders"], lambda: QrackSimulator(WIDTH)))
    for name, blocks, make in rows:
        res = {s: run(make(), blocks, s, shots) for s in ("0", "1", "+", "-")}
        cells = "  ".join(f"|{s}_L>: {res[s][0]:.3f}/{res[s][1]:.3f}" for s in res)
        z = (res["0"][1] + res["1"][1]) / 2
        x = (res["+"][1] + res["-"][1]) / 2
        print(f"  {name:18s} {blocks}")
        print(f"      {cells}")
        print(f"      decoded logical-Z {z:.3f}, logical-X {x:.3f}")
    print(f"  seconds: {time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
