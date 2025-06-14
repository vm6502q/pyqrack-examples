# supply_chain_v2.py
# Provided by Elara (the custom OpenAI GPT)

import math
import numpy as np
from pyqrack import QrackAceBackend, Pauli
import matplotlib.pyplot as plt

def apply_tfim_step(sim, qubits, J, h, delta_t):
    # ZZ interactions (Ising coupling)
    for i in range(len(qubits) - 1):
        sim.cx(qubits[i], qubits[i+1])
        sim.r(Pauli.PauliZ, -2 * J * delta_t, qubits[i+1])
        sim.cx(qubits[i], qubits[i+1])
    
    # RX rotations (transverse field)
    for q in qubits:
        sim.r(Pauli.PauliX, -2 * h * delta_t, q)

def simulate_tfim(h_func, n_qubits=64, patch_size=4, n_steps=20, J=1.0, delta_t=0.1, theta=2*math.pi/9, shots=1024):
    sim = QrackAceBackend(n_qubits, long_range_rows=patch_size - 1, long_range_columns=patch_size - 1)
    
    for q in range(n_qubits):
        sim.r(Pauli.PauliY, theta, q)

    qubits = list(range(n_qubits))
    magnetizations = []
    for step in range(n_steps):
        h_t = h_func(step * delta_t)
        apply_tfim_step(sim, qubits, J, h_t, delta_t)
    
        samples = sim.measure_shots(qubits, shots)

        magnetization = 0
        for sample in samples:
            m = 0
            for _ in range(n_qubits):
                m += -1 if (sample & 1) else 1
                sample >>= 1
            magnetization += m / n_qubits
        magnetization /= shots
        magnetizations.append(magnetization)

    return magnetizations

if __name__ == "__main__":
    # Example usage
    n_qubits = 64
    patch_size = 4
    n_steps = 20
    J = 1.0
    delta_t = 0.1
    theta = 2 * math.pi / 9
    shots = 1024
    h_func = lambda t: 1.0 * np.cos(0.5 * t)  # time-varying transverse field

    mag = simulate_tfim(h_func, n_qubits, patch_size, n_steps, J, delta_t, theta, shots)
    ylim = ((min(mag) * 100) // 10) / 10
    plt.figure(figsize=(14, 14))
    plt.plot(list(range(1, n_steps+1)), mag, marker="o", linestyle="-")
    plt.title("Supply Chain Resilience over Time (Magnetization vs Trotter Depth, " + str(n_qubits) + " Qubits)")
    plt.xlabel("Trotter Depth")
    plt.ylabel("Magnetization")
    plt.ylim(ylim, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
