# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

from collections import Counter

import numpy as np

from scipy.stats import distributions as dists

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator
from qiskit.providers.qrack import AceQasmSimulator


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    return circ


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n)))


def main():
    n_qubits = 16
    depth = 20
    shots = 32768
    trials = 32
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(shots, 1 << (n_qubits + 2))
    if len(sys.argv) > 4:
        trials = int(sys.argv[4])

    t1 = 4.5
    t2 = 0.5

    n_rows, n_cols = factor_width(n_qubits, False)

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    qubits = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    step = QuantumCircuit(n_qubits)
    trotter_step(step, qubits, (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    depths = list(range(0, depth + 1))
    min_sqr_mag = 1
    results = []
    magnetizations = []

    experiment_counts = [{}] * (depth + 1)
    for trial in range(trials):
        experiment = QrackSimulator(n_qubits)
        experiment.run_qiskit_circuit(qc)
        for d in range(depth + 1):
            if d > 0:
                experiment.run_qiskit_circuit(step)
            counts = dict(Counter(experiment.measure_shots(qubits, shots)))
            for key, value in counts.items():
                experiment_counts[d][key] = experiment_counts[d].get(key, 0) + value

    for experiment in experiment_counts:
        for key in experiment.keys():
            experiment[key] /= (shots * trials)

    for d in range(depth + 1):
        bias = []
        model = 0
        if d > 0:
            t = d * dt
            m = t / t1
            model = 1 - 1 / (1 + m)
            arg = -h / J
            if np.isclose(J, 0) or (arg >= 1024):
                bias = (n_qubits + 1) * [1 / (n_qubits + 1)]
            elif np.isclose(h, 0) or (arg < -1024):
                bias.append(1)
                bias += n_qubits * [0]
                if J > 0:
                    bias.reverse()
            else:
                p = 2**arg + math.tanh(J / abs(h)) * math.log(1 + t / t2) / math.log(2)
                factor = 2**p
                n = 1 / (n_qubits * 2)
                tot_n = 0
                for q in range(n_qubits + 1):
                    n = n / factor
                    if n == float("inf"):
                        tot_n = 1
                        bias.append(1)
                        bias += n_qubits * [0]
                        if J > 0:
                            bias.reverse()
                        break
                    bias.append(n)
                    tot_n += n
                for q in range(n_qubits + 1):
                    bias[q] /= tot_n

        magnetization = 0
        sqr_magnetization = 0
        for key, value in experiment_counts[d].items():
            if d > 0:
                hamming_weight = hamming_distance(key, 0, n_qubits)
                weight = 1
                combo_factor = n_qubits
                for _ in range(hamming_weight):
                    weight *= combo_factor
                    combo_factor -= 1
                value = (1 - model) * value + model * bias[hamming_weight] / weight

            m = 0
            for _ in range(n_qubits):
                m += -1 if (key & 1) else 1
                key >>= 1
            m /= n_qubits
            magnetization += m * value
            sqr_magnetization += m * m * value

        if sqr_magnetization < min_sqr_mag:
            min_sqr_mag = sqr_magnetization

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "magnetization": magnetization,
                "square_magnetization": sqr_magnetization,
            }
        )
        magnetizations.append(sqr_magnetization)

        print(results[-1])

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    ylim = ((min_sqr_mag * 100) // 10) / 10

    plt.figure(figsize=(14, 14))
    plt.plot(depths, magnetizations, marker="o", linestyle="-")
    plt.title("Square Magnetization vs Trotter Depth (" + str(n_qubits) + " Qubits)")
    plt.xlabel("Trotter Depth")
    plt.ylabel("Square Magnetization")
    plt.grid(True)
    plt.xticks(depths)
    plt.ylim(ylim, 1.0)  # Adjusting y-axis for clearer resolution
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())
