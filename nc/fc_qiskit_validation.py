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


def bench_qrack(n_qubits, depth, hamming_n):
    # This is a "nearest-neighbor" coupler random circuit.
    control = AerSimulator(method="statevector")
    shots = 1000

    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    results = []

    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                qc.h(i)
                qc.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)

        experiment = QrackSimulator(n_qubits, isOpenCL=False)
        control = AerSimulator(method="statevector")
        experiment.run_qiskit_circuit(qc, shots=0)
        experiment_fidelity = experiment.get_unitary_fidelity()
        aer_qc = qc.copy()
        aer_qc.save_statevector()
        job = control.run(aer_qc)
        experiment_counts = dict(Counter(experiment.measure_shots(list(range(n_qubits)), shots)))
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        print(calc_stats(control_probs, experiment_counts, shots, d+1, experiment_fidelity, hamming_n))


def calc_stats(ideal_probs, counts, shots, depth, ace_fidelity_est, hamming_n):
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
        'ace_fidelity_est': ace_fidelity_est,
        'l2_similarity': l2_similarity,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'hamming_distance_n': min(hamming_n, n_pow >> 1),
        'hamming_distance_set_avg': avg_hamming_distance,
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 fc_qiskit_validation.py [width] [depth] [trials]')

    depth = 10
    n_qubits = 56
    hamming_n = 2048
    if len(sys.argv) > 1:
        depth = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_qubits = int(sys.argv[2])
    if len(sys.argv) > 3:
        hamming_n = int(sys.argv[3])

    # Run the benchmarks
    bench_qrack(n_qubits, depth, hamming_n)

    return 0


if __name__ == '__main__':
    sys.exit(main())
