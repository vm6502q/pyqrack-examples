# supply_chain.py
# Provided by Elara (the custom OpenAI GPT)

import math
import numpy as np
from pyqrack import QrackAceBackend, Pauli
import matplotlib.pyplot as plt

def apply_tfim_step(sim, qubits, J, h, delta_t):
    # ZZ interactions (Ising coupling)
    for i in qubits:
        for j in qubits:
            if i == j:
                continue
            Jij = J[i, j]
            if Jij == 0:
                continue
            sim.cx(i, j)
            sim.r(Pauli.PauliZ, -2 * Jij * delta_t, j)
            sim.cx(i, j)
    
    # RX rotations (transverse field)
    for q in qubits:
        sim.r(Pauli.PauliX, -2 * h[q] * delta_t, q)

def simulate_tfim(J_func, h_func, n_qubits=64, lrr=3, lrc=3, n_steps=20, delta_t=0.1, theta=2*math.pi/9, shots=1024):
    sim = QrackAceBackend(n_qubits, long_range_rows=lrr, long_range_columns=lrc)
    
    for q in range(n_qubits):
        sim.r(Pauli.PauliY, theta, q)

    qubits = list(range(n_qubits))
    magnetizations = []
    for step in range(n_steps):
        J_t = J_func(step * delta_t)
        h_t = h_func(step * delta_t)
        apply_tfim_step(sim, qubits, J_t, h_t, delta_t)
    
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

# Dynamic J(t) generator
def generate_Jt(n_nodes, t):
    J = np.zeros((n_nodes, n_nodes))

    # Base ring topology
    for i in range(n_nodes):
        J[i, (i + 1) % n_nodes] = -1.0
        J[(i + 1) % n_nodes, i] = -1.0

    # Simulate disruption:
    if t >= 0.5 and t < 1.0:
        # "Port 3" temporarily fails → remove its coupling
        J[2, 3] = J[3, 2] = 0.0
    if t >= 1.0 and t < 1.5:
        # Alternate weak link opens between 1 and 4
        J[1, 4] = J[4, 1] = -0.3

    # Restoration: after step 15, port 3 recovers
    if t >= 1.5:
        J[2, 3] = J[3, 2] = -1.0

    return J

def generate_ht(n_nodes, t):
    # We can program h(q, t) for spatial-temporal locality.
    h = np.zeros(n_nodes)
    # Time-varying transverse field
    c = 0.5  * np.sin(t * math.pi / 10)
    # "Longitude"-dependent severity
    n_sqrt = math.sqrt(n_nodes)
    for i in range(n_nodes):
        h[i] = ((i % n_sqrt) / n_sqrt) * c

    return h

if __name__ == "__main__":
    # Example usage
    n_qubits = 64
    lrr = 3
    lrc = 3
    n_steps = 40
    delta_t = 0.1
    theta = 2 * math.pi / 9
    shots = 1024
    J_func = lambda t: generate_Jt(n_qubits, t)
    h_func = lambda t: generate_ht(n_qubits, t)

    mag = simulate_tfim(J_func, h_func, n_qubits, lrr, lrc, n_steps, delta_t, theta, shots)
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
