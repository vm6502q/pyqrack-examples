# One unit of Shor's 9-qubit code on QrackAceBackend
#
# By (Anthropic) Claude and Dan Strano
#
# Shor's code concatenates a 3-qubit phase-flip code (outer) with three 3-qubit bit-flip codes
# (inner blocks):
#     |0_L> = (|000> + |111>)^3 / 2^(3/2),    |1_L> = (|000> - |111>)^3 / 2^(3/2)
# Encoder: input on leader a; CNOT a->b, a->c (outer); H on a, b, c; then each leader CNOTs to
# the two other members of its block (inner).
#
# Logical readout, with decoding:
#   * Logical X (for |+_L>, |-_L>): measure all 9 in Z. Each block's bit is the majority of its
#     three bits (corrects one bit flip per block). Logical X = parity of the three block bits.
#   * Logical Z (for |0_L>, |1_L>): H on all 9, measure. Each block's sign is the parity of its
#     three X outcomes (a phase flip in a block flips that block's sign). Logical Z = majority of
#     the three block signs (corrects one block's phase flip).
# "raw" is the same logical readout without any correction: the parity of all 9 outcomes.
#
# The four logical states are all tested so that a sampler biased toward one answer cannot score
# well. Success rate 1.0 is perfect; 0.5 is no logical information.
#
# Layouts on a 3x3 ACE grid with long_range_columns=1, is_torus=False: two patches joined by
# one seam column of three qubits (the seam qubits have replicas in both patches and in the
# crossbar; seam-to-seam couplings are exact). The script reads the seam set and coupling map
# from the backend and checks every CNOT against the map.
#   seam_leaders: block leaders are the three seam qubits (outer code entirely seam-to-seam,
#                 exact); each inner block is seam + one bulk qubit from each patch, so every
#                 inner bit-flip block straddles the seam.
#   patch_blocks: one block per region: the seam column, all-bulk patch 0, all-bulk patch 1
#                 (inner blocks entirely inside one region, exact); the outer phase-flip layer
#                 is the only thing crossing the seam (seam leader -> one leader in each patch).

import sys
import time

from collections import Counter

from pyqrack import QrackAceBackend, QrackSimulator


WIDTH = 9
ACE_KW = dict(long_range_columns=1, is_torus=False)


def _replicas(sim, q):
    return (sim.qubits if hasattr(sim, "qubits") else sim._qubits)[q]


def layouts(ace):
    seams = [q for q in range(WIDTH) if len(_replicas(ace, q)) > 1]
    bulk = [q for q in range(WIDTH) if len(_replicas(ace, q)) == 1]
    patches = {}
    for q in bulk:
        patches.setdefault(_replicas(ace, q)[0][0], []).append(q)
    p0, p1 = [sorted(v) for _, v in sorted(patches.items())]
    if len(seams) != 3 or len(p0) != 3 or len(p1) != 3:
        raise RuntimeError(f"Unexpected layout: seams {seams}, patches {p0} {p1}")
    return {
        # leaders a, b, c with blocks [leader, member, member]; input goes on the first leader.
        "seam_leaders": [[seams[i], p0[i], p1[i]] for i in range(3)],
        "patch_blocks": [[seams[1], seams[0], seams[2]], [p0[0], p0[1], p0[2]], [p1[0], p1[1], p1[2]]],
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
        # Logical Z is read in the X basis.
        for q in range(WIDTH):
            sim.h(q)
    counts = Counter(sim.measure_shots(list(range(WIDTH)), shots))

    raw_ok = dec_ok = 0
    for k, v in counts.items():
        bits = [[(k >> q) & 1 for q in blk] for blk in blocks]
        # Raw readout: parity of all nine outcomes. In the Z basis this is the parity of the
        # number of 111 blocks (even for |+_L>, odd for |-_L>); in the X basis it is the product
        # of the three block signs (+ for |0_L>, - for |1_L>, which has three '-' blocks).
        raw_parity = sum(sum(b) for b in bits) & 1
        if state in ("+", "-"):
            block_bits = [1 if sum(b) >= 2 else 0 for b in bits]
            decoded = sum(block_bits) & 1                       # 0 -> |+_L>, 1 -> |-_L>
            expect = 0 if state == "+" else 1
        else:
            block_signs = [sum(b) & 1 for b in bits]            # 0 -> '+' block, 1 -> '-' block
            decoded = 1 if sum(block_signs) >= 2 else 0         # 0 -> |0_L>, 1 -> |1_L>
            expect = 0 if state == "0" else 1
        raw_ok += v * (raw_parity == expect)
        dec_ok += v * (decoded == expect)
    return raw_ok / shots, dec_ok / shots


def bench(shots=8192):
    probe = QrackAceBackend(WIDTH, **ACE_KW)
    lays = layouts(probe)
    for blocks in lays.values():
        check_coupling(probe, encoder(blocks))
    del probe

    results = {}
    for name, blocks in lays.items():
        row = {"blocks": blocks}
        for state in ("0", "1", "+", "-"):
            raw, dec = run(QrackAceBackend(WIDTH, **ACE_KW), blocks, state, shots)
            row[state] = (raw, dec)
        results[name] = row
    # Decoder sanity check on an exact simulator: every entry should be 1.0.
    blocks = lays["seam_leaders"]
    results["exact_reference"] = {"blocks": blocks, **{s: run(QrackSimulator(WIDTH), blocks, s, shots) for s in ("0", "1", "+", "-")}}
    return results


def main():
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    start = time.perf_counter()
    results = bench(shots)
    print(f"Shor-9 on ACE (3x3, long_range_columns=1, is_torus=False), {shots} shots per logical state")
    print("success rate per logical state as raw / decoded  (1.0 perfect, 0.5 no information)")
    for name, row in results.items():
        print(f"  {name:14s} blocks {row['blocks']}")
        cells = "  ".join(f"|{s}_L>: {row[s][0]:.3f} / {row[s][1]:.3f}" for s in ("0", "1", "+", "-"))
        z = (row["0"][1] + row["1"][1]) / 2
        x = (row["+"][1] + row["-"][1]) / 2
        print(f"      {cells}")
        print(f"      decoded logical-Z success {z:.3f}, logical-X success {x:.3f}")
    print(f"  seconds: {time.perf_counter() - start:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
