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

    qubits = list(range(n_qubits))

    nq_2 = n_qubits * (n_qubits - 1)
    nq_3 = n_qubits * (n_qubits - 1) * (n_qubits - 2)
    t1 = 17.5
    t = depth * dt / t1
    model = 1 - t + t ** 2 - t ** 3
    bias_0_shots = int(shots * 2 * model / n_qubits)
    bias_1_shots = int(shots * model) // n_qubits
    bias_2_shots = n_qubits * (int(shots * model / 2) // nq_2)
    bias_3_shots = nq_2 * (int(shots * model / 4) // nq_3)
    remainder_shots = shots - (
        bias_0_shots + bias_1_shots + bias_2_shots + bias_3_shots
    )

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    dummy_backend = AceQasmSimulator(
        n_qubits=n_qubits,
        long_range_columns=long_range_columns,
        long_range_rows=long_range_rows,
    )
    step = QuantumCircuit(n_qubits)
    trotter_step(step, list(range(n_qubits)), (n_rows, n_cols), J, h, dt, False)
    step = transpile(
        step,
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
            experiment.run_qiskit_circuit(step)
            z_samples = experiment.measure_shots(
                qubits, remainder_shots
            ) + bias_shots * [0]
            for q1 in range(n_qubits):
                p1 = 1 << q1
                z_samples += (bias_1_shots // n_qubits) * [p1]
                for q2 in range(n_qubits):
                    if q1 == q2:
                        continue
                    p2 = 1 << q2
                    p = p1 | p2
                    z_samples += (bias_2_shots // nq_2) * [p]
                    for q3 in range(n_qubits):
                        if (q1 == q3) or (q2 == q3):
                            continue
                        p3 = 1 << q3
                        p = p1 | p2 | p3
                        z_samples += (bias_2_shots // nq_3) * [p]
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
