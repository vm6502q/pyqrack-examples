# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys
import time

from collections import Counter

import numpy as np

from pyqrack import QrackSimulator

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
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    
    row_len, col_len = factor_width(n_qubits)

    shots = hamming_n << 2
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)

    rz_count = (n_qubits + 1) << 1
    rz_opportunities =  n_qubits * n_qubits * 3
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
                    qc.rz(random.uniform(0, math.pi / 2), i)
                gate_count = gate_count + 1

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

        for r in range(11):
            ncrp = (10 - r) / 10
            experiment = QrackSimulator(n_qubits, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
            # Round closer to a Clifford circuit
            if ncrp > 0:
                experiment.set_ncrp(ncrp)
            control = AerSimulator(method="statevector")
            start = time.perf_counter()
            experiment.run_qiskit_circuit(qc, shots=0)
            experiment_counts = dict(Counter(experiment.measure_shots(list(range(n_qubits)), shots)))
            stop = time.perf_counter()
            if r == 0:
                min_time = stop - start
            aer_qc = qc.copy()
            aer_qc.save_statevector()
            job = control.run(aer_qc)
            control_probs = Statevector(job.result().get_statevector()).probabilities()

            results = calc_stats(control_probs, experiment_counts, shots, d+1, ncrp, hamming_n)
            print(results)

            if ((stop - start) > (2 * min_time)) or ((d > 0) and ((results['l2_similarity'] >= 1.0) or (results['xeb'] > 1.0))):
                # This has become too costly to try lower rounding parameters:
                break


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
        'ncrp': ncrp,
        'depth': depth,
        'l2_similarity': l2_similarity,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'hamming_distance_n': min(hamming_n, n_pow >> 1),
        'hamming_distance_set_avg': avg_hamming_distance,
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 rcs_nn_2n_plus_2_qiskit_validation.py [width] [hamming_n]')

    n_qubits = n_qubits = int(sys.argv[1])
    hamming_n = 2048
    if len(sys.argv) > 2:
        hamming_n = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(n_qubits, hamming_n)

    return 0


if __name__ == '__main__':
    sys.exit(main())
