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
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
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
    return sum(ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n)))


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


def bench_qrack(n_qubits, hamming_n):
    # This is a "nearest-neighbor" coupler random circuit.
    t_prob = ((n_qubits + 1) << 1) / (n_qubits * n_qubits * 3)
    shots = hamming_n << 2

    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)
    
    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    
    row_len, col_len = factor_width(n_qubits)

    qc = QuantumCircuit(n_qubits)
    qs = QuantumCircuit(n_qubits)
    for d in range(n_qubits):
        # Single-qubit gates
        for i in range(n_qubits):
            for _ in range(3):
                qc.h(i)
                qs.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    qc.z(i)
                    qs.z(i)
                if s_count & 2:
                    qc.s(i)
                    qs.s(i)
                if random.random() < t_prob:
                    angle = random.uniform(0, math.pi / 2)
                    qc.rz(angle, i)
                    if angle >= (math.pi / 4):
                        qs.s(i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1);
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
                g(qs, b1, b2)

        # Round to nearest Clifford circuit
        experiment = QrackStabilizer(n_qubits)
        control = AerSimulator(method="statevector")
        experiment.run_qiskit_circuit(qs, shots=0)
        aer_qc = qc.copy()
        aer_qc.save_statevector()
        job = control.run(aer_qc)
        experiment_counts = dict(Counter(experiment.measure_shots(list(range(n_qubits)), shots)))
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        print(calc_stats(control_probs, experiment_counts, shots, d+1, hamming_n))


def calc_stats(ideal_probs, counts, shots, depth, hamming_n):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    diff_sqr = 0
    numer = 0
    denom = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - (count / shots)) ** 2

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_similarity = diff_sqr ** (1/2)
    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    exp_top_n = top_n(hamming_n, experiment)
    con_top_n = top_n(hamming_n, ideal_probs)

    # By Elara (OpenAI custom GPT)
    # Compute Hamming distances between each ACE bitstring and its closest in control case
    min_distances = [min(hamming_distance(a, r, n) for r in con_top_n) for a in exp_top_n]
    avg_hamming_distance = np.mean(min_distances)

    return {
        'qubits': n,
        'depth': depth,
        'l2_similarity': l2_similarity,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'hamming_distance_n': min(hamming_n, n_pow >> 1),
        'hamming_distance_set_avg': avg_hamming_distance,
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 fc_qiskit_validation.py [width] [hamming_n]')

    n_qubits = 56
    hamming_n = 2048
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        hamming_n = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, hamming_n)

    return 0


if __name__ == '__main__':
    sys.exit(main())
