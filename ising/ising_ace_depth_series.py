# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import os
import sys
import time

from collections import Counter

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator


def factor_width(width, reverse=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
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
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def main():
    n_qubits = 56
    depth = 20
    reverse = False
    shots = 32768
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(32768, 1 << (n_qubits + 2))
    if len(sys.argv) > 4:
        reverse = sys.argv[4] not in ["0", "False"]

    n_rows, n_cols = factor_width(n_qubits, reverse)

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = -math.pi / 6

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

    experiment = QrackAceBackend(n_qubits, reverse_row_and_col=reverse)
    if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
        experiment.sim.set_sdrp(0.03)
    noise_dummy=AceQasmSimulator(n_qubits=n_qubits)

    step = transpile(
        step,
        optimization_level=3,
        backend=noise_dummy,
    )

    depths = list(range(1, depth + 1))
    results = []
    magnetizations = []

    start = time.perf_counter()
    experiment.run_qiskit_circuit(qc)
    for d in depths:
        experiment.run_qiskit_circuit(step)
        experiment_samples = experiment.measure_shots(list(range(n_qubits)), shots)

        magnetization = 0
        for sample in experiment_samples:
            for _ in range(n_qubits):
                magnetization += -1 if (sample & 1) else 1
                sample >>= 1
        magnetization /= shots * n_qubits

        seconds = time.perf_counter() - start

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "magnetization": magnetization,
                "seconds": seconds,
            }
        )
        magnetizations.append(magnetization)

        print(results[-1])

    ylim = ((min(magnetizations) * 100) // 10) / 10

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    plt.figure(figsize=(14, 14))
    plt.plot(depths, magnetizations, marker="o", linestyle="-")
    plt.title("Magnetization vs Trotter Depth (" + str(n_qubits) + " Qubits)")
    plt.xlabel("Trotter Depth")
    plt.ylabel("Magnetization")
    plt.grid(True)
    plt.xticks(depths)
    plt.ylim(ylim, 1.0)  # Adjusting y-axis for clearer resolution
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
