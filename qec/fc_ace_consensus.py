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
    shots        = 1 << min(8, width + 2)
    n_inst       = 3

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc     = [QuantumCircuit(width) for _ in range(n_inst)]

    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
            qc[0].u(th, ph, lm, i)
            qc[1].u(th, ph, lm, (i + 1) % width)
            qc[2].u(th, ph, lm, (i + 2) % width)
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        while len(shuffled) > 1:
            c, t = shuffled.pop(), shuffled.pop()
            qc[0].cx(c, t)
            qc[1].cx((c + 1) % width, (t + 1) % width)
            qc[2].cx((c + 2) % width, (t + 2) % width)

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
