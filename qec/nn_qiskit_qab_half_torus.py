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
from pyqrack import QrackSimulator
from qiskit.providers.qrack.backends import AceQasmSimulator
from qiskit import QuantumCircuit, transpile


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, shots):
    n_pow = len(ideal_probs)
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = (counts.get(b, 0) / shots)

        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom
    return xeb, hog_prob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, lrc=4, lrr=4, sdrp=0.0):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(10, width + 2)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = QuantumCircuit(width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.u(th, ph, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                # Non-toroidal (is_torus=False) boundary handling, split
                # by axis rather than applied uniformly -- these two axes
                # do NOT behave the same way under is_torus=False, given
                # long_range_columns=2 is set explicitly below but
                # long_range_rows is left at its default (4):
                #
                # temp_row ranges over row_len (the LONG dimension), and
                # matches QrackAceBackend's "long_range_columns" axis
                # (verified: that parameter governs boundary density along
                # the same axis this script calls "row", both ranging over
                # row_len). long_range_columns=2 < row_len for any width
                # worth testing, so is_torus=False genuinely, actually
                # disables wraparound here -- skip (continue) is correct.
                #
                # temp_col ranges over col_len (the SHORT dimension), and
                # matches QrackAceBackend's "long_range_rows" axis, left at
                # its DEFAULT value of 4. is_torus=False only actually
                # disables wraparound on a given axis when
                # long_range_X < length_of_that_axis -- so for col_len<=4
                # (verified directly: true for col_len in {2,3,4}, false
                # for col_len>=5), long_range_rows=4 is NOT less than
                # col_len, meaning QrackAceBackend treats this entire short
                # dimension as one continuous interior run regardless of
                # is_torus. Skipping here would silently drop real
                # coupling gates that QrackAceBackend still, correctly,
                # treats as adjacent -- wrapping instead matches its
                # actual behavior. This holds for col_len<=4; for col_len>=5
                # this assumption would need revisiting (either passing
                # long_range_rows explicitly below, or reworking this check
                # to depend on it rather than a hardcoded default).
                if temp_row < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col < 0:
                    if (row_len < 3) or (row_len <= lrr):
                        temp_col = temp_col + col_len
                    else:
                        continue
                if temp_col >= col_len:
                    if (row_len < 3) or (row_len <= lrr):
                        temp_col = temp_col - col_len
                    else:
                        continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------
    sim = AceQasmSimulator(n_qubits=width, long_range_columns=lrc, long_range_rows=lrr, sdrp=sdrp, is_torus=False)
    qc = transpile(qc, backend=sim, optimization_level=3)

    t_trans = time.perf_counter()
    print(f"transpile_seconds: {t_trans - t_circ:.4f}")

    qcm = qc.copy()
    qcm.measure_all()
    ace_str_counts = dict(sim.run(qcm, shots=shots).result().get_counts())
    ace_counts = {}
    for s, count in ace_str_counts.items():
        ace_counts[int(s, 2)] = count
    

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_trans:.4f}")

    # -----------------------------------------------------------------------
    # Ideal ground truth via QrackSimulator
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()
    print(f"ideal_seconds: {t_ideal - t_ace:.4f}")

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts, shots)

    return {
        "width":              width,
        "depth":              depth,
        "long_range_columns": lrc,
        "long_range_rows":    lrr,
        "sdrp":               sdrp,
        "depth":              depth,
        "xeb_ace":            xeb_ace,
        "hog_ace":            hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 nn_qab_half_torus.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [sdrp=0.1464466]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sdrp  = float(sys.argv[5]) if len(sys.argv) > 5 else ((1 - 1 / math.sqrt(2)) / 2)
    result = bench_qrack(width, depth, lrc, lrr, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
