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


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


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
    sim.s(q2)
    sim.s(q1)
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


def bench_qrack(n_qubits, magic):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    print(f"{n_qubits} qubits, square circuit, {magic} units of 'magic', then mirrored for double")

    rz_count = magic
    rz_opportunities = n_qubits * n_qubits * 3
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                qc.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    qc.z(i)
                if s_count & 2:
                    qc.s(i)
                if gate_count in rz_positions:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)

    qc = qc.compose(qc.inverse())
    experiment = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=True,
    )
    # Round to nearest Clifford circuit
    experiment.set_use_exact_near_clifford(False)
    experiment.run_qiskit_circuit(qc)
    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    zero_count = experiment_counts.get(0, 0)

    print(f"Fidelity: {zero_count} correct out of {shots} shots")

def main():
    n_qubits = 36
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    magic = 18
    if len(sys.argv) > 2:
        magic = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, magic)

    return 0


if __name__ == "__main__":
    sys.exit(main())
