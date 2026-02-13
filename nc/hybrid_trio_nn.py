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


def bench_qrack(n_qubits, depth, use_rz, magic, ace_qb_limit, sparse_mb_limit):
    # This is a "nearest-neighbor" coupler random circuit.
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(n_qubits)

    hamming_n = 2048
    shots = 1 << min(20, (n_qubits + 2))
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
    magic_angle = 0.0
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
                        angle = random.uniform(-math.pi / 4, math.pi / 4)
                    elif s_count & 4:
                        angle = math.pi / 4
                    else:
                        angle = -math.pi / 4
                    qc.rz(angle, i)
                    magic_angle += abs(angle)
                gate_count = gate_count + 1

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
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

    nc = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=True,
    )
    # Round closer to a Clifford circuit
    nc.set_use_exact_near_clifford(False)
    nc.run_qiskit_circuit(qc, shots=0)
    nc_counts = dict(
        Counter(nc.measure_shots(list(range(n_qubits)), shots))
    )

    qc_ace = transpile(
        qc,
        optimization_level=3,
        basis_gates=[
            "u",
            "x",
            "y",
            "z",
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
        ],
    )

    ace = QrackSimulator(
        n_qubits,
        isTensorNetwork=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=False
    )
    # Split at least into 2 patches
    ace_qb = n_qubits
    while ace_qb > ace_qb_limit:
        ace_qb = (ace_qb + 1) >> 1
    ace.set_ace_max_qb(ace_qb)
    ace.run_qiskit_circuit(qc_ace, shots=0)
    ace_counts = dict(
        Counter(ace.measure_shots(list(range(n_qubits)), shots >> 1))
    )
    
    sparse = QrackSimulator(
        n_qubits,
        isTensorNetwork=False,
        isSchmidtDecompose=False,
        isStabilizerHybrid=False,
        isOpenCL=False,
        isPaged=False,
        isSparse=True
    )
    # Split at least into 2 patches
    sparse.set_sparse_ace_max_mb(sparse_mb_limit)
    sparse.run_qiskit_circuit(qc_ace, shots=0)
    sparse_counts = dict(
        Counter(sparse.measure_shots(list(range(n_qubits)), shots >> 1))
    )

    aer_qc = qc.copy()
    aer_qc.save_statevector()
    job = control.run(aer_qc)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    results = calc_stats(
        control_probs, nc_counts, ace_counts, sparse_counts, shots, d + 1, hamming_n, 4 * magic_angle / math.pi, ace_qb, sparse_mb_limit
    )
    print(results)


def calc_stats(ideal_probs, nc_counts, ace_counts, sparse_counts, shots, depth, hamming_n, magic, ace_qb, sparse_mb_limit):
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
    sum_hog_prob = 0
    experiment = [0] * n_pow
    lm = (1 - 1 / math.sqrt(2)) ** (magic / n)
    nlm = (lm ** 2) + ((1 - lm) ** 2)
    for i in range(n_pow):
        nc_count = nc_counts.get(i, 0)
        ace_count = ace_counts.get(i, 0)
        sparse_count = sparse_counts.get(i, 0)
        count = lm * nc_count +  (1 - lm) * (ace_count + sparse_count) / 2
        ideal = ideal_probs[i]
        exp = count / shots

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - exp) ** 2
        noise += nlm * exp * (1 - exp) / (shots >> 1)

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_prob += count

    l2_diff = diff_sqr ** (1 / 2)
    l2_diff_debiased = math.sqrt(max(diff_sqr - noise, 0.0))
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
        "ace_qb": ace_qb,
        "sparse_mb": sparse_mb_limit,
        "l2_difference": float(l2_diff),
        "l2_difference_debiased": float(l2_diff_debiased),
        "xeb": float(xeb),
        "hog_prob": float(sum_hog_prob),
        "hamming_distance_n": min(hamming_n, n_pow >> 1),
        "hamming_distance_set_avg": float(avg_hamming_distance),
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 rcs_nn_2n_plus_2_qiskit_validation.py [width] [depth] [use_rz] [magic] [ace_qb_limit] [sparse_mb_limit]"
        )

    n_qubits = n_qubits = int(sys.argv[1])

    depth = int(sys.argv[2])

    use_rz = False
    if len(sys.argv) > 3:
        use_rz = sys.argv[3] not in ["False", "0"]

    magic = n_qubits + 1
    if len(sys.argv) > 4:
        magic = int(sys.argv[4])

    ace_qb_limit = (n_qubits + 1) >> 1
    if len(sys.argv) > 5:
        ace_qb_limit = int(sys.argv[5])

    sparse_mb_limit = 1
    if len(sys.argv) > 6:
        sparse_mb_limit = int(sys.argv[6])

    # Run the benchmarks
    bench_qrack(n_qubits, depth, use_rz, magic, ace_qb_limit, sparse_mb_limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
