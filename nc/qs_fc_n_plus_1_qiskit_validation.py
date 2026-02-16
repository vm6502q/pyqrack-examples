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


def bench_qrack(n_qubits, hamming_n):
    # This is a "fully-connected" coupler random circuit.
    shots = hamming_n << 2
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    rz_count = n_qubits + 1
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

        # Round to nearest Clifford circuit
        exp_shots = []
        for i in range(shots):
            experiment = QrackStabilizer(n_qubits)
            experiment.run_qiskit_circuit(qc, shots=0)
            exp_shots.append(experiment.m_all());
        experiment_counts = dict(Counter(exp_shots))

        aer_qc = qc.copy()
        aer_qc.save_statevector()
        control = AerSimulator(method="statevector")
        job = control.run(aer_qc)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        print(calc_stats(control_probs, experiment_counts, shots, d + 1, hamming_n))


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
        exp = count / shots

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - exp) ** 2

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_difference = diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    exp_top_n = top_n(hamming_n, experiment)
    con_top_n = top_n(hamming_n, ideal_probs)

    # By Elara (OpenAI custom GPT)
    # Compute Hamming distances between each ACE bitstring and its closest in control case
    min_distances = [
        min(hamming_distance(a, r, n) for r in con_top_n) for a in exp_top_n
    ]
    avg_hamming_distance = np.mean(min_distances)

    return {
        "qubits": n,
        "depth": depth,
        "l2_difference": float(l2_difference),
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "hamming_distance_n": min(hamming_n, n_pow >> 1),
        "hamming_distance_set_avg": float(avg_hamming_distance),
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 qs_fc_2n_plus_2_qiskit_validation.py [width] [hamming_n]"
        )

    n_qubits = 56
    hamming_n = 2048
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        hamming_n = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, hamming_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
