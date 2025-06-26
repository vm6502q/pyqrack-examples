# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_QTENSORNETWORK_THRESHOLD_QB=1

import math
import sys
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return row_len, col_len


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

    # horizontal wrap
    wrap_pairs = [(r * n_cols + (n_cols - 1), r * n_cols) for r in range(n_rows)]
    add_rzz_pairs(wrap_pairs)

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

    # vertical wrap
    wrap_pairs = [((n_rows - 1) * n_cols + c, c) for c in range(n_cols)]
    add_rzz_pairs(wrap_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def main():
    depth = 1
    n_qubits = 56
    if len(sys.argv) > 1:
        depth = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_qubits = int(sys.argv[2])

    n_rows, n_cols = factor_width(n_qubits)
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(theta, q)

    for _ in range(depth):
        trotter_step(qc, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    basis_gates = [
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
        "swap",
        "iswap",
    ]
    qc = transpile(qc, basis_gates=basis_gates)

    sim = QrackSimulator(n_qubits, isBinaryDecisionTree=True)
    start = time.perf_counter()
    sim.run_qiskit_circuit(qc, shots=0)
    result = sim.m_all()

    print(
        "Trotter steps: "
        + str(depth)
        + ", seconds: "
        + str(time.perf_counter() - start)
        + "."
    )
    print("Result: " + str(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
