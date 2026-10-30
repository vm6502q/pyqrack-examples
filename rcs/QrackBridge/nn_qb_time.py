# Nearest-neighbor RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import statistics
import sys
import time

from collections import Counter

import numpy as np
from pyqrack import QrackSimulator, QrackBridge


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.iswap(q1, q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.adjiswap(q1, q2)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *g[1:])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    ideal = np.asarray(ideal_probs, dtype=np.float64)
    threshold = np.median(ideal)
    u_u = ideal.mean()
    ideal_centered = ideal - u_u
    denom = np.dot(ideal_centered, ideal_centered)

    # Only sampled bitstrings have nonzero empirical probability, so work on
    # those indices alone instead of a dense 2^n array of counts.
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    patch = np.fromiter(counts.values(), dtype=np.float64, count=len(counts)) / shots

    # numer = sum_b ic[b] * (patch[b] - u_u); split so the patch term is sparse.
    numer = np.dot(ideal_centered[keys], patch) - u_u * ideal_centered.sum()
    hog_prob = patch[ideal[keys] > threshold].sum()

    return numer / denom, hog_prob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, max_bonds=1, patches=2):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(13, width + 2)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    # Reversed on purpose for mapper:
    row_len, col_len = factor_width(width)
    if row_len < 2:
        # factor_width() gives col_len == 1 for prime widths, so (reversed) row_len == 1 and the
        # coupler loop below would emit no two-qubit gates at all.
        raise ValueError(f"Width {width} is prime: the nearest-neighbor grid has no couplers. Use a composite width.")

    # QrackBridge patches are contiguous index blocks (the last width % patches get one extra qubit).
    # Qubit index is col * row_len + row, so a boundary that isn't a multiple of row_len splits a
    # column, and the in-column (row-direction) couplers there all cross the cut.
    base, rem = divmod(width, patches)
    bounds, start = [], 0
    for p in range(patches - 1):
        start += base + (1 if p >= patches - rem else 0)
        bounds.append(start)
    split = [b for b in bounds if b % row_len]
    if split:
        print(f"Warning: patch boundaries {split} split a column of {row_len} qubits ({col_len} columns, {patches} patches).")

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = []
    qcr = []

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(2))
            # Keep it Haar-random towards the poles:
            th = math.acos(random.uniform(-1, 1))
            qc.append((u, i, th, ph, lm))

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        layer = []
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                layer.append((g, b1, b2))

        # If the circuit exceeds "vanilla" ACE limits, we'll patch at random.
        random.shuffle(layer)
        qc += layer

    # -----------------------------------------------------------------------
    # Method: QrackBridge
    # -----------------------------------------------------------------------
    t_circ_end = time.perf_counter()
    print(f"circuit_seconds: {t_circ_end - t_circ:.4f}")

    sim = QrackBridge(width, max_bonds=max_bonds, patches=patches, wrap=True)
    run_circuit(sim, qc)
    
    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_circ_end:.4f}")

    return {
        "width":             width,
        "depth":             depth,
        "max_bonds":         max_bonds,
        "patches":           patches,
        "fidelity_estimate": sim.fidelity_estimate,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 nn_qab.py [width] [depth] [patches=2] [max_bonds=3]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    patches = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    max_bonds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    result = bench_qrack(width, depth, max_bonds, patches)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
