# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import os
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
from pyqrackising import get_tfim_hamming_distribution


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Elara (the custom OpenAI GPT)
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


def main():
    n_qubits = 16
    depth = 40
    z = 4
    alpha = 0.375
    t1 = 28
    t2 = 2

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.125
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

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        alpha = float(sys.argv[4])
        alpha = min(max(alpha, 0), 1)
    if len(sys.argv) > 5:
        t1 = float(sys.argv[5])
    if len(sys.argv) > 6:
        t2 = float(sys.argv[6])
    if len(sys.argv) > 7:
        shots = int(sys.argv[7])
    else:
        shots = max(65536, 1 << (n_qubits + 2))

    dt_h = dt / t2

    print(f"Qubits: {n_qubits}")
    print(f"Subsystem size: {os.environ['QRACK_MAX_PAGING_QB']}")
    print(f"alpha: {alpha}")
    print(f"t1: {t1}")
    print(f"t2: {t2}")

    depths = list(range(1, depth + 1))
    results = []
    magnetizations = {}

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))
    magnetizations = []

    init = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        init.ry(theta, q)

    qc_step = QuantumCircuit(n_qubits)
    trotter_step(qc_step, qubits, (n_rows, n_cols), J, h, dt_h)
    qc_step = transpile(
        qc_step,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    start = time.perf_counter()

    experiment = QrackSimulator(n_qubits, isTensorNetwork=False)
    experiment.run_qiskit_circuit(init)

    for d in depths:
        t = d * dt
        t_h = t / t2

        experiment.run_qiskit_circuit(qc_step)
        experiment_counts = dict(Counter(experiment.measure_shots(qubits, shots)))
        bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t_h, n_qubits=n_qubits)

        magnetization, sqr_magnetization = 0, 0
        for key, value in experiment_counts.items():
            m = 0
            for _ in range(n_qubits):
                m += -1 if (key & 1) else 1
                key >>= 1
            m /= n_qubits
            magnetization += m * value
            sqr_magnetization += m * m * value
        magnetization /= shots
        sqr_magnetization /= shots
        
        b_magnetization, b_sqr_magnetization = 0, 0
        for hamming_weight, value in enumerate(bias):
            m = 1.0 - 2 * hamming_weight / n_qubits
            b_magnetization += value * m
            b_sqr_magnetization += value * m * m

        model = (alpha / math.exp(t_h / t1)) if (t1 > 0) else alpha
        magnetization = model * magnetization + (1.0 - model) * b_magnetization
        sqr_magnetization = model * sqr_magnetization + (1.0 - model) * b_sqr_magnetization

        seconds = time.perf_counter() - start

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "magnetization": float(magnetization),
                "square_magnetization": float(sqr_magnetization),
                "seconds": seconds,
            }
        )
        magnetizations.append(sqr_magnetization)
        print(results[-1])

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    plt.figure(figsize=(14, 14))

    plt.plot(depths, magnetizations, marker='o', linestyle='-')

    plt.xlabel("step")
    plt.ylabel(r"$\langle Z^2_{tot} \rangle$")
    plt.title("Square Magnetization vs Trotter Depth")
    # plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    # plt.ylim(0.05, 0.7)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
