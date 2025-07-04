# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator


def factor_width(width, reverse=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if reverse else (row_len, col_len)


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
    n_qubits = 100
    depth = 20
    shots = 1024
    trials = 5
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(shots, 1 << (n_qubits + 2))
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])

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

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    step = QuantumCircuit(n_qubits)
    trotter_step(step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    depths = list(range(0, depth + 1))
    min_sqr_mag = 1
    results = []
    magnetizations = []

    for trial in range(trials):
        magnetizations.append([])
        experiment = QrackSimulator(n_qubits)
        start = time.perf_counter()
        experiment.run_qiskit_circuit(qc)
        for d in depths:
            if d > 0:
                experiment.run_qiskit_circuit(step)
            experiment_samples = experiment.measure_shots(list(range(n_qubits)), shots)

            magnetization = 0
            sqr_magnetization = 0
            for sample in experiment_samples:
                m = 0
                for _ in range(n_qubits):
                    m += -1 if (sample & 1) else 1
                    sample >>= 1
                m /= n_qubits
                magnetization += m
                sqr_magnetization += m * m
            magnetization /= shots
            sqr_magnetization /= shots
            if sqr_magnetization < min_sqr_mag:
                min_sqr_mag = sqr_magnetization

            seconds = time.perf_counter() - start

            results.append(
                {
                    "width": n_qubits,
                    "depth": d,
                    "trial": trial + 1,
                    "magnetization": magnetization,
                    "square_magnetization": sqr_magnetization,
                    "seconds": seconds,
                }
            )
            magnetizations[-1].append(sqr_magnetization)

            print(results[-1])

    if trials < 2:
        # Plotting (contributed by Elara, an OpenAI custom GPT)
        ylim = ((min_sqr_mag * 100) // 10) / 10

        plt.plot(depths, magnetizations[0], marker="o", linestyle="-")
        plt.title(
            "Square Magnetization vs Trotter Depth (" + str(n_qubits) + " Qubits)"
        )
        plt.xlabel("Trotter Depth")
        plt.ylabel("Square Magnetization")
        plt.grid(True)
        plt.xticks(depths)
        plt.ylim(ylim, 1.0)  # Adjusting y-axis for clearer resolution
        plt.show()

        return 0

    mean_magnetization = np.mean(magnetizations, axis=0)
    std_magnetization = np.std(magnetizations, axis=0)

    ylim = ((min(mean_magnetization) * 100) // 10) / 10

    # Plot with error bands
    plt.figure(figsize=(14, 14))
    plt.errorbar(
        depths,
        mean_magnetization,
        yerr=std_magnetization,
        fmt="-o",
        capsize=5,
        label="Mean ± Std Dev",
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Square Magnetization")
    plt.title(
        "Square Magnetization vs Trotter Depth ("
        + str(n_qubits)
        + " Qubits, "
        + str(trials)
        + " Trials)\nWith Mean and Standard Deviation"
    )
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ylim = ((min_sqr_mag * 100) // 10) / 10

    # Plot each trial individually
    plt.figure(figsize=(14, 14))
    for i, magnetization in enumerate(magnetizations):
        plt.plot(depths, magnetization, marker="o", label=f"Trial {i + 1}")

    plt.title(
        "Square Magnetization vs Trotter Depth ("
        + str(n_qubits)
        + " Qubits, "
        + str(trials)
        + " Trials)"
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Square Magnetization")
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.legend([f"Trial {i + 1}" for i in range(trials)], loc="lower left")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
