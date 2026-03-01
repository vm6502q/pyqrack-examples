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


def bench_qrack(n_qubits, depth, ace_qb_limit, sparse_mb_limit):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            qc.u(th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)


    qc = qc.compose(qc.inverse())

    ace = QrackSimulator(n_qubits)
    # Split at least into 2 patches
    ace_qb = n_qubits
    while ace_qb > ace_qb_limit:
        ace_qb = (ace_qb + 1) >> 1
    ace.set_ace_max_qb(ace_qb)
    ace.run_qiskit_circuit(qc, shots=0)
    ace_counts = dict(Counter(ace.measure_shots(list(range(n_qubits)), shots >> 1)))
    zero_count = ace_counts.get(0, 0)
    
    sparse = QrackSimulator(
        n_qubits,
        isSchmidtDecompose=False,
        isStabilizerHybrid=False,
        isOpenCL=False,
        isPaged=False,
        isSparse=True
    )
    # Split at least into 2 patches
    sparse.set_sparse_ace_max_mb(sparse_mb_limit)
    sprp = 1.0 / ((n_qubits - 1) * (2 ** n_qubits))
    if sprp > 1.7763568394002505e-15:
        sparse.set_sprp(sprp)
    sparse.run_qiskit_circuit(qc, shots=0)
    sparse_counts = dict(Counter(ace.measure_shots(list(range(n_qubits)), shots >> 1)))
    zero_count += sparse_counts.get(0, 0)

    print(f"Fidelity: {zero_count} correct out of {shots} shots")

def main():
    n_qubits = 25
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])

    depth = n_qubits
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])

    sparse_mb_limit = 4
    if len(sys.argv) > 3:
        sparse_mb_limit = int(sys.argv[3])

    ace_qb_limit = (n_qubits + 1) >> 1
    if len(sys.argv) > 4:
        ace_qb_limit = int(sys.argv[4])

    # Run the benchmarks
    bench_qrack(n_qubits, depth, ace_qb_limit, sparse_mb_limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
