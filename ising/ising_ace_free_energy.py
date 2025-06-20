# Ising model Trotterization with Free Energy tracking
# Modified by Elara (OpenAI GPT)

import math
import sys
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator

def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len
    return (col_len, row_len) if is_transpose else (row_len, col_len)

def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape
    for q in qubits:
        circ.rx(h * dt / 2, q)

    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ

def estimate_entropy(samples):
    counts = Counter(samples)
    probs = np.array(list(counts.values())) / len(samples)
    return -np.sum(probs * np.log(probs + 1e-10))

def compute_z_energy(samples, n_qubits, J=-1.0):
    energy = 0
    for sample in samples:
        z_terms = 0
        for i in range(n_qubits - 1):
            bit_i = (sample >> i) & 1
            bit_j = (sample >> (i + 1)) & 1
            z_terms += (1 if bit_i == bit_j else -1)
        energy += -J * z_terms
    return energy / len(samples)

def compute_x_energy(state, n_qubits, shots, h=2.0):
    for q in range(n_qubits):
        state.h(q)
    samples = state.measure_shots(list(range(n_qubits)), shots)
    for q in range(n_qubits):
        state.h(q)

    return compute_z_energy(samples, n_qubits, h)

def main():
    n_qubits = 64
    depth = 20
    is_transpose = False
    shots = 1024
    long_range_columns = 4
    long_range_rows = 4
    T = 1.0
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    if len(sys.argv) > 4:
        is_transpose = sys.argv[4] not in ["0", "False"]
    if len(sys.argv) > 5:
        long_range_columns = int(sys.argv[5])
    if len(sys.argv) > 6:
        long_range_rows = int(sys.argv[6])

    n_rows, n_cols = factor_width(n_qubits, is_transpose)
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta / 2, q)

    step = QuantumCircuit(n_qubits)
    trotter_step(step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)
    step = transpile(step, optimization_level=3, backend=AceQasmSimulator(n_qubits=n_qubits))

    experiment = QrackAceBackend(n_qubits, is_transpose=is_transpose, long_range_columns=long_range_columns, long_range_rows=long_range_rows)
    experiment.run_qiskit_circuit(qc)

    free_energies = []

    for d in range(depth):
        experiment.run_qiskit_circuit(step)
        z_samples = experiment.measure_shots(list(range(n_qubits)), shots)
        E_z = compute_z_energy(z_samples, n_qubits, J=J)
        S = estimate_entropy(z_samples)
        E_x = compute_x_energy(experiment, n_qubits, shots, h=h)
        F = E_z + E_x - T * S
        free_energies.append(F)
        print(f"Step {d+1}, Free Energy = {F:.5f}, Z Energy = {E_z:.5f}, X Energy = {E_x:.5f}, Entropy = {S:.5f}")

    # Plot Free Energy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, depth + 1), free_energies, marker='o')
    plt.title("Free Energy vs Trotter Depth")
    plt.xlabel("Trotter Depth")
    plt.ylabel("Free Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())
