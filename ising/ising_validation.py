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

from pyqrack import QrackSimulator


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return row_len, col_len


def trotter_step(circ, qubits, lattice_shape, J, h, dt, is_odd):
    n_rows, n_cols = lattice_shape

    # We want to make an alternating "checkerboard" or "brick-wall" pattern that barely doesn't
    # overlap gates in the same layer whether the row and column counts are even or odd
    # (though "is_odd" corresponds to even-or-odd depth step, not spatial parity).
    a_offset = 1 if is_odd else 0
    b_offset = 0 if is_odd else 1

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(a_offset, n_cols - b_offset, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(b_offset, n_cols - a_offset, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(b_offset, n_rows - a_offset, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(a_offset, n_rows - b_offset, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def calc_stats(n, ideal_probs, counts, shots, depth, hamming_n):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = 2**n
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    diff_sqr = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - (count / shots)) ** 2

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

        # XEB / EPLG
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * ((count / shots) - u_u)

    l2_similarity = 1 - diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots

    exp_top_n = top_n(hamming_n, experiment)
    con_top_n = top_n(hamming_n, ideal_probs)

    # By Elara (OpenAI custom GPT)
    # Compute Hamming distances between each ACE bitstring and its closest in control case
    min_distances = [
        min(hamming_distance(a, r, n) for r in con_top_n) for a in exp_top_n
    ]
    avg_hamming_distance = np.mean(min_distances)

    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "l2_similarity": float(l2_similarity),
        "hog_prob": hog_prob,
        "xeb": xeb,
        "hamming_distance_n": min(hamming_n, n_pow >> 1),
        "hamming_distance_set_avg": float(avg_hamming_distance),
        "hamming_fidelity_heuristic": 1 - 2 * float(avg_hamming_distance) / n,
    }


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


def main():
    n_qubits = 56
    depth = 10
    hamming_n = 100
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        hamming_n = int(sys.argv[3])

    n_rows, n_cols = factor_width(n_qubits)
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9
    shots = 1 << (n_qubits + 2)

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(theta, q)

    for d in range(depth):
        trotter_step(
            qc, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, (d & 1) == 1
        )

    basis_gates = [
        "u",
        "rx",
        "ry",
        "rz",
        "h",
        "x",
        "y",
        "z",
        "sx",
        "sxdg",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "cy",
        "cz",
        "cp",
        "swap",
        "iswap",
    ]
    qc = transpile(qc, basis_gates=basis_gates)

    experiment = QrackSimulator(n_qubits)
    control = AerSimulator(method="statevector")
    experiment.run_qiskit_circuit(qc)
    qc.save_statevector()
    job = control.run(qc)
    experiment_counts = dict(
        Counter(experiment.measure_shots(list(range(n_qubits)), shots))
    )
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    print(
        calc_stats(n_qubits, control_probs, experiment_counts, shots, depth, hamming_n)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
