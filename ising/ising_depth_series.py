# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import sys
import time

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


def trotter_step(circ, qubits, lattice_shape, J, h, dt, is_odd):
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
        for c in range(0, n_cols - (0 if is_odd else 1), 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # horizontal wrap
    if (not is_odd) and ((n_cols & 1) == 0):
        wrap_pairs = [(r * n_cols + (n_cols - 1), r * n_cols) for r in range(n_rows)]
        add_rzz_pairs(wrap_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows - (0 if is_odd else 1), 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # vertical wrap
    if (not is_odd) and ((n_rows & 1) == 0):
        wrap_pairs = [((n_rows - 1) * n_cols + c, c) for c in range(n_cols)]
        add_rzz_pairs(wrap_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def main():
    n_qubits = 64
    depth = 20
    shots = 1024
    sdrp = 0.02375
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(shots, 1 << (n_qubits + 2))
    if len(sys.argv) > 4:
        sdrp = float(sys.argv[4])

    n_rows, n_cols = factor_width(n_qubits, False)

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9

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
        qc.ry(theta / 2, q)

    basis_gates = [
        "id",
        "u",
        "u1",
        "u2",
        "u3",
        "r",
        "rx",
        "ry",
        "rz",
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "p",
        "t",
        "tdg",
        "cx",
        "cy",
        "cz",
        "swap",
        "iswap",
        "reset",
        "measure",
    ]

    even_step = QuantumCircuit(n_qubits)
    trotter_step(even_step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, False)
    even_step = transpile(
        even_step,
        optimization_level=3,
        basis_gates=basis_gates,
    )
    odd_step = QuantumCircuit(n_qubits)
    trotter_step(odd_step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, True)
    odd_step = transpile(
        odd_step,
        optimization_level=3,
        basis_gates=basis_gates,
    )

    step = transpile(step, optimization_level=3, basis_gates=basis_gates)

    experiment = QrackSimulator(n_qubits, isTensorNetwork=False)
    experiment.set_sdrp(sdrp)
    depths = list(range(1, depth + 1))
    results = []
    magnetizations = []

    start = time.perf_counter()
    experiment.run_qiskit_circuit(qc)
    for d in depths:
        experiment.run_qiskit_circuit(odd_step if d & 1 else even_step)
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
