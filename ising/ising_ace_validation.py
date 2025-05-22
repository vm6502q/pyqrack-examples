# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# Run ./experiment.sh

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from pyqrack import QrackAceBackend

def factor_width(width, reverse=False):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len

    return (col_len, row_len) if reverse else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape
    
    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols)
                   for r in range(n_rows) for c in range(0, n_cols - 1, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols)
                   for r in range(n_rows) for c in range(1, n_cols - 1, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
                  for r in range(0, n_rows - 1, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
                  for r in range(1, n_rows - 1, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


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

    l2_similarity = 1 - diff_sqr ** (1/2)
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


def main():
    n_qubits = 56
    depth = 10
    hamming_n = 100
    reverse = False
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        hamming_n = int(sys.argv[3])
    if len(sys.argv) > 4:
        reverse = sys.argv[4] not in ['0', 'False']

    n_rows, n_cols = factor_width(n_qubits, reverse)
    J, h, dt = -1.0, 2.0, 0.25
    theta = -math.pi / 6
    shots = 1 << (n_qubits + 2)

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(theta, q)

    for _ in range(depth):
        trotter_step(qc, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    basis_gates = ["u", "rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cy", "cz", "swap", "iswap"]
    qc = transpile(qc, basis_gates=basis_gates)

    experiment = QrackAceBackend(n_qubits)
    control = AerSimulator(method="statevector")
    experiment.run_qiskit_circuit(qc)
    qc.save_statevector()
    job = control.run(qc)
    experiment_counts = dict(Counter(experiment.measure_shots(list(range(n_qubits)), shots)))
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    print(calc_stats(control_probs, experiment_counts, shots, depth, hamming_n))

    return 0


if __name__ == '__main__':
    sys.exit(main())
