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
from pyqrack import QrackSimulator, QrackAceBackend
from qiskit.providers.qrack.backends import AceQasmSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap


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

def bench_qrack(depth):
    width = 64
    dead_qubits = (5, 27, 32)
    chip_width = 108 # 12-by-9
    coupler_exclusions = {
        107, 95, 83, 71, 59, 47, 35, # first boundary
        # first patch (is exact)
        103, 91, 79, 67, 55, 43, # second boundary
        102, # second patch
        99, 87, 75, 63, 51, # third boundary
        98, # third patch
    }
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_pow     = 1 << width
    shots     = 1 << min(13, width + 2)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
    qc = QuantumCircuit(width, width)

    # Randomize initial permutation
    for i in lcv_range:
        if (i not in dead_qubits) and (random.random() < 0.5):
            qc.x(i)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            if i in dead_qubits:
                continue
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

                if temp_row < 0:
                    continue
                if temp_col < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col >= col_len:
                    continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                if (b1 in dead_qubits) or (b2 in dead_qubits):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    # -----------------------------------------------------------------------
    # Method: QrackAceBackend
    # -----------------------------------------------------------------------
    # 3 patches, 25 state-vector qubits apiece1
    dummy = QrackAceBackend(125, long_range_columns=4, long_range_rows=5)
    coupling_map = dummy.get_logical_coupling_map()
    sim = AceQasmSimulator(n_qubits=75, long_range_columns=4, long_range_rows=5, coupling_map=coupling_map)
    qc = transpile(qc, backend=sim, optimization_level=3)
    qc = qc & qc.inverse()
    

    t_trans = time.perf_counter()
    print(f"transpile_seconds: {t_trans - t_circ:.4f}")

    logical_to_physical = qc.layout.final_index_layout()

    for logical_idx in range(width):
        physical_qubit = logical_to_physical[logical_idx]
        # Measure the exact physical wire into its designated classical bit
        qc.measure(physical_qubit, logical_idx)

    ace_str_counts = dict(sim.run(qc, shots=shots).result().get_counts())

    t_ace = time.perf_counter()
    print(f"ace_seconds: {t_ace - t_trans:.4f}")

    ace_counts = {}
    for s, count in ace_str_counts.items():
        ace_counts[int(s, 2)] = count
    hamming = 0
    for s, count in ace_counts.items():
        hamming += s.bit_count() * count
    hamming /= shots

    return {
        "width":                width,
        "depth":                depth,
        "fidelity":             ace_counts.get(0, 0) / shots,
        "hamming_weight":       hamming,
        "est_hamming_fidelity": 1.0 - (2 * hamming / (width - len(dead_qubits)))
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 mirror_nighthawk.py [depth]")
    depth = int(sys.argv[1])
    result = bench_qrack(depth)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
