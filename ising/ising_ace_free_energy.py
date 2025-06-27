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
    row_len = width // col_len
    return (col_len, row_len) if is_transpose else (row_len, col_len)


def trotter_step(circ, qubits, lattice_shape, J, h, dt, is_odd):
    n_rows, n_cols = lattice_shape

    # We want to make an alternating "checkerboard" or "brick-wall" pattern that barely doesn't
    # overlap gates in the same step whether the row and column counts are even or odd
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
            z_terms += 1 if bit_i == bit_j else -1
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
    n_qubits = 100
    depth = 30
    shots = 1024
    long_range_columns = 4
    long_range_rows = 4
    trials = 5
    T = 1.0
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    else:
        shots = min(shots, 1 << (n_qubits + 2))
    if len(sys.argv) > 4:
        long_range_columns = int(sys.argv[4])
    if len(sys.argv) > 5:
        long_range_rows = int(sys.argv[5])
    if len(sys.argv) > 6:
        trials = int(sys.argv[6])
    lcv = 7
    devices = []
    while len(sys.argv) > lcv:
        devices.append(int(sys.argv[lcv]))
        lcv += 1
    print("Devices: " + str(devices))

    n_rows, n_cols = factor_width(n_qubits, False)
    J, h, dt = -1.0, 2.0, 0.25
    theta = 2 * math.pi / 9

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    dummy_backend = AceQasmSimulator(
        n_qubits=n_qubits,
        long_range_columns=long_range_columns,
        long_range_rows=long_range_rows,
    )
    even_step = QuantumCircuit(n_qubits)
    trotter_step(even_step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, False)
    even_step = transpile(
        even_step,
        optimization_level=3,
        backend=dummy_backend,
    )
    odd_step = QuantumCircuit(n_qubits)
    trotter_step(odd_step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, True)
    odd_step = transpile(
        odd_step,
        optimization_level=3,
        backend=dummy_backend,
    )

    free_energies = []
    for trial in range(trials):
        free_energies.append([])
        experiment = QrackAceBackend(
            n_qubits,
            long_range_columns=long_range_columns,
            long_range_rows=long_range_rows,
        )
        # We've achieved the dream: load balancing between discrete and integrated accelerators!
        for sim_id in range(min(len(experiment.sim), len(devices))):
            experiment.sim[sim_id].set_device(devices[sim_id])

        experiment.run_qiskit_circuit(qc)
        for d in range(depth):
            experiment.run_qiskit_circuit(odd_step if d & 1 else even_step)
            z_samples = experiment.measure_shots(list(range(n_qubits)), shots)
            E_z = compute_z_energy(z_samples, n_qubits, J=J)
            S = estimate_entropy(z_samples)
            E_x = compute_x_energy(experiment, n_qubits, shots, h=h)
            F = E_z + E_x - T * S
            free_energies[-1].append(F)
            print(
                f"Step {d+1}, Free Energy = {F:.5f}, Z Energy = {E_z:.5f}, X Energy = {E_x:.5f}, Entropy = {S:.5f}"
            )

    depths = range(1, depth + 1)

    # Plot Free Energy
    if trials < 2:
        plt.figure(figsize=(10, 6))
        plt.plot(depths, free_energies[0], marker="o")
        plt.title("Free Energy vs Trotter Depth (" + str(n_qubits) + " qubits)")
        plt.xlabel("Trotter Depth")
        plt.ylabel("Free Energy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return 0

    mean_free_energy = np.mean(free_energies, axis=0)
    std_free_energy = np.std(free_energies, axis=0)

    ymax = (((max(free_energy_values) * 100) + 9) // 10) / 10
    ymin = ((min(free_energy_values) * 100) // 10) / 10

    # Plot with error bands
    plt.figure(figsize=(14, 14))
    plt.errorbar(
        depths,
        mean_free_energy,
        yerr=std_free_energy,
        fmt="-o",
        capsize=5,
        label="Mean ± Std Dev",
    )
    plt.title(
        "Free Energy vs Trotter Depth ("
        + str(n_qubits)
        + " Qubits, "
        + str(trials)
        + " Trials)\nWith Mean and Standard Deviation"
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Free Energy")
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot each trial individually
    plt.figure(figsize=(14, 14))
    for i, free_energy in enumerate(free_energies):
        plt.plot(depths, free_energy, marker="o", label=f"Trial {i + 1}")

    plt.title(
        "Free Energy vs Trotter Depth ("
        + str(n_qubits)
        + " Qubits, "
        + str(trials)
        + " Trials)"
    )
    plt.xlabel("Trotter Depth")
    plt.ylabel("Free Energy")
    plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.legend([f"Trial {i + 1}" for i in range(trials)], loc="lower left")
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
