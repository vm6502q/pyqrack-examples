# Fully-connected RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import statistics
import sys
import time

from scipy.stats import binom

from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator, QrackAceBackend


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

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
    sim.swap(q1, q2)
    sim.cz(q1, q2)
    sim.s(q1)
    sim.s(q2)


def iiswap(sim, q1, q2):
    sim.sdg(q2)
    sim.sdg(q1)
    sim.cz(q1, q2)
    sim.swap(q1, q2)


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
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = 1 / n_pow
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (
        (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1
    )

    return xeb, hog_prob


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow
    shots        = 1 << min(20, width + 2)
    n_inst       = 3

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc     = [QuantumCircuit(width) for _ in range(n_inst)]

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    offset_1 = (row_len + n_inst - 1) // n_inst
    offset_2 = (row_len + (row_len + n_inst - 1) // n_inst)

    results = []

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            qc[0].u(th, ph, lm, i)
            qc[1].u(th, ph, lm, (i + offset_1) % width)
            qc[2].u(th, ph, lm, (i + offset_2) % width)

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

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(qc[0], b1, b2)
                g(qc[1], (b1 + offset_1) % width, (b2 + offset_1) % width)
                g(qc[2], (b1 + offset_2) % width, (b2 + offset_2) % width)

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    sim_ideal.run_qiskit_circuit(qc[0], shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_ideal = time.perf_counter()

    print(f"qrack_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend consensus
    # -----------------------------------------------------------------------
    ace_counts = []
    for c in qc:
        sim_ace = QrackAceBackend(width)
        sim_ace.run_qiskit_circuit(c, shots=0)
        ace_counts.append(dict(Counter(sim_ace.measure_shots(all_bits, shots))))
    del sim_ace

    tm = 1 << (width - 1)
    for k, v in ace_counts[1].items():
        t = k & 1
        k = ((k >> 1) | tm) if t else (k >> 1)
        ace_counts[0][k] = ace_counts[0].get(k, 0) + v
    tm = width - 2
    for k, v in ace_counts[2].items():
        t = k & 3
        k = (k >> 2) | (t << tm)
        ace_counts[0][k] = ace_counts[0].get(k, 0) + v

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_counts[0], shots * n_inst)

    t_elapsed = time.perf_counter() - t_ideal

    return {
        "width":         width,
        "depth":         depth,
        "xeb_ace":       xeb_ace,
        "hog_ace":       hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace.py [width] [depth] [sdrp=0]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
