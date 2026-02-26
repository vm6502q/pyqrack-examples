# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

import numpy as np

from pyqrack import QrackStabilizer

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


def bench_qrack(n_qubits):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    print(f"{n_qubits} qubits, square circuit, 1 random non-Clifford phase gate per single-qubit gate Pauli axis, then mirrored for double")

    layers = []
    gate_count = 0
    for d in range(n_qubits):
        qc = QuantumCircuit(3 * n_qubits)
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                for j in range(3):
                    qc.h(3 * i + j)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    for j in range(3):
                        qc.z(3 * i + j)
                if s_count & 2:
                    for j in range(3):
                        qc.s(3 * i + j)
                angle = random.uniform(0, math.pi / 2)
                for j in range(3):
                    qc.rz(angle, 3 * i + j)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            for j in range(3):
                qc.cx(3 * c + j, 3 * t + j)

        layers.append(qc)

    rev_layers = []
    for layer in reversed(layers):
        rev_layers.append(layer.inverse())

    layers = layers + rev_layers

    # Round to nearest Clifford circuit
    zero_count = 0
    hamming_count = 0
    for i in range(shots):
        experiment = QrackStabilizer(3 * n_qubits + 2)
        for layer in layers:
            experiment.run_qiskit_circuit(layer, shots=0)

            #QEC code
            for i in range(n_qubits):
                experiment.mcx([3 * i], 3 * n_qubits)
                experiment.mcx([3 * i + 1], 3 * n_qubits)
                experiment.mcx([3 * i + 1], 3 * n_qubits + 1)
                experiment.mcx([3 * i + 2], 3 * n_qubits + 1)
                b0 = experiment.m(3 * n_qubits)
                b1 = experiment.m(3 * n_qubits + 1)
                if b0 and b1:
                    experiment.x(3 * i + 1)
                elif b0:
                    experiment.x(3 * i)
                elif b1:
                    experiment.x(3 * i + 2)
                if b0:
                    experiment.x(3 * n_qubits)
                if b1:
                    experiment.x(3 * n_qubits + 1)

        raw_sample = experiment.m_all();
        sample = 0
        for i in range(n_qubits):
            b = (sample >> (3 * i)) & 1
            b += (sample >> (3 * i + 1)) & 1
            b += (sample >> (3 * i + 2)) & 1
            if b > 1:
               sample |= 1 << i
        if sample == 0:
            zero_count += 1
        for q in range(n_qubits):
            if sample & 1:
                hamming_count += 1
            sample >>= 1

    avg_hamming_count = hamming_count / shots

    print(f"Fidelity: {zero_count} correct out of {shots} shots")
    print(f"Average Hamming weight: {avg_hamming_count} out of {n_qubits} qubits")

def main():
    n_qubits = 64
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])

    # Run the benchmarks
    bench_qrack(n_qubits)

    return 0


if __name__ == "__main__":
    sys.exit(main())
