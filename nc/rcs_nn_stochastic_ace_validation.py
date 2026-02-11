# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

import numpy as np

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


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
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def bench_qrack(n_qubits, use_rz):
    # This is a "nearest-neighbor" coupler random circuit.
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(n_qubits)

    shots = min(1 << 20, 1 << (n_qubits + 2))
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    qe = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=True,
    )
    # Round closer to a Clifford circuit
    qe.set_use_exact_near_clifford(False)

    qc = QrackSimulator(
        n_qubits,
        isTensorNetwork=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=False,
    )
    # Validate with patch circuits
    ace_qb = (n_qubits + 1) >> 1
    while ace_qb > 26:
        ace_qb = (ace_qb + 1) >> 1
    qc.set_ace_max_qb(ace_qb)

    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                qc.h(i)
                qe.h(i)
                # s_count = random.randint(0, 3)
                s_count = random.randint(0, 7)
                if s_count & 1:
                    qc.z(i)
                    qe.z(i)
                if s_count & 2:
                    qc.s(i)
                    qe.s(i)
                if use_rz:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                    qe.rz(angle, i)
                elif s_count & 4:
                    qc.t(i)
                    qe.t(i)
                else:
                    qc.adjt(i)
                    qe.adjt(i)

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

                if (b1 >= n_qubits) or (b2 >= n_qubits):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)
                g(qe, b1, b2)

        
        experiment_probs = normalize_counts(dict(
            Counter(qe.measure_shots(list(range(n_qubits)), shots))
        ), shots)

        control_probs = normalize_counts(dict(
            Counter(qc.measure_shots(list(range(n_qubits)), shots))
        ), shots)

        results = calc_stats(
            control_probs, experiment_probs, n_qubits, shots, d + 1
        )
        print(results)


def normalize_counts(counts, shots):
    return {k: v / shots for k, v in counts.items()}


def calc_stats(p_a, p_b, n, shots, depth):
    all_keys = set(p_a) | set(p_b)
    diff_sq = 0.0
    noise = 0.0
    numerator = 0.0
    denom = 0.0
    for k in all_keys:
        pa = p_a.get(k, 0.0)
        pb = p_b.get(k, 0.0)
        diff_sq += (pa - pb) ** 2
        noise += pa * (1 - pa) / shots + pb * (1 - pb) / shots
        numerator += pa * pb
        denom += pa * pa

    l2_diff = math.sqrt(diff_sq)
    l2_diff_debiased = math.sqrt(max(diff_sq - noise, 0.0))
    xeb = 0.0 if denom == 0 else (numerator / denom)

    return {
        "qubits": n,
        "depth": depth,
        "l2_difference": l2_diff,
        "l2_difference_debiased": l2_diff_debiased,
        "xeb": xeb
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 rcs_nn_2n_plus_2_qiskit_validation.py [width] [use_rz]"
        )

    n_qubits = n_qubits = int(sys.argv[1])
    use_rz = False
    if len(sys.argv) > 2:
        use_rz = sys.argv[2] not in ["False", "0"]

    # Run the benchmarks
    bench_qrack(n_qubits, use_rz)

    return 0


if __name__ == "__main__":
    sys.exit(main())
