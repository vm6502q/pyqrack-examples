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
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


def bench_qrack(n_qubits, depth, use_rz, magic):
    # This is a "fully-connected" coupler random circuit.
    hamming_n = 2048
    shots = hamming_n << 2
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)
    control = AerSimulator(method="statevector")

    rz_opportunities = n_qubits * depth * 3
    rz_positions = []
    while len(rz_positions) < magic:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for _ in range(3):
                qc.h(i)
                # s_count = random.randint(0, 3)
                s_count = random.randint(0, 7)
                if s_count & 1:
                    qc.z(i)
                if s_count & 2:
                    qc.s(i)
                if gate_count in rz_positions:
                    if use_rz:
                        qc.rz(random.uniform(0, math.pi / 2), i)
                    elif s_count & 4:
                        qc.t(i)
                    else:
                        qc.tdg(i)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qc.cx(c, t)

    experiment = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=True,
    )
    # Round closer to a Clifford circuit
    experiment.set_use_exact_near_clifford(False)
    experiment.run_qiskit_circuit(qc, shots=0)
    experiment_counts = dict(
        Counter(experiment.measure_shots(list(range(n_qubits)), shots))
    )

    aer_qc = qc.copy()
    aer_qc.save_statevector()
    job = control.run(aer_qc)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    results = calc_stats(
        control_probs, experiment_counts, shots, d + 1, hamming_n, magic
    )
    print(results)


def calc_stats(ideal_probs, counts, shots, depth, hamming_n, magic):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = 1 / n_pow
    diff_sqr = 0
    noise = 0
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
        noise += exp * (1 - exp) / shots

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_diff = diff_sqr ** (1 / 2)
    l2_diff_debiased = math.sqrt(max(diff_sqr - noise, 0.0))
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
        "magic": magic,
        "shots":shots,
        "l2_difference": float(l2_diff),
        "l2_difference_debiased": float(l2_diff_debiased),
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "hamming_distance_n": min(hamming_n, n_pow >> 1),
        "hamming_distance_set_avg": float(avg_hamming_distance),
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 fc_2n_plus_2_qiskit_validation.py [width] [depth] [use_rz] [magic]"
        )

    n_qubits = n_qubits = int(sys.argv[1])

    depth = int(sys.argv[2])

    use_rz = False
    if len(sys.argv) > 3:
        use_rz = sys.argv[3] not in ["False", "0"]

    magic = n_qubits + 1
    if len(sys.argv) > 4:
        magic = int(sys.argv[4])

    # Run the benchmarks
    bench_qrack(n_qubits, depth, use_rz, magic)

    return 0


if __name__ == "__main__":
    sys.exit(main())
