# supply_chain_v1.py
# Provided by Elara (the custom OpenAI GPT)

import math
import numpy as np
import matplotlib.pyplot as plt
from pyqrack import QrackAceBackend, Pauli

# Dynamic J(t) generator
def generate_Jt(n_nodes, step, depth):
    J = np.zeros((n_nodes, n_nodes))

    # Base ring topology
    for i in range(n_nodes):
        J[i, (i + 1) % n_nodes] = -1.0
        J[(i + 1) % n_nodes, i] = -1.0

    # Simulate disruption:
    if step >= 5 and step < 10:
        # "Port 3" temporarily fails → remove its coupling
        J[2, 3] = J[3, 2] = 0.0
    if step >= 10 and step < 15:
        # Alternate weak link opens between 1 and 4
        J[1, 4] = J[4, 1] = -0.3

    # Restoration: after step 15, port 3 recovers
    if step >= 15:
        J[2, 3] = J[3, 2] = -1.0

    return J

# Main simulation function
def run_dynamic_J_sim(n_nodes=64, lr_columns=3, lr_rows=3, h=2.0, dt=0.25, depth=20, trials=5, phase_shift=0.0, noise_strength=0.0):
    results = []

    for trial in range(trials):
        sim = QrackAceBackend(n_nodes, long_range_columns=lr_columns, long_range_rows=lr_rows)
        lq = list(range(n_nodes))

        # Initial rotation
        for q in lq:
            sim.r(Pauli.PauliX, math.pi / 2, q)

        magnetization = []

        for step in range(depth):
            # Transverse field
            for q in lq:
                sim.r(Pauli.PauliX, h * dt, q)

            # Dynamic coupling matrix
            Jt = generate_Jt(n_nodes, step, depth)

            # Coupling update
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    Jij = Jt[i, j]
                    if Jij != 0:
                        noise = noise_strength * (np.random.rand() - 0.5)
                        sim.cz(lq[i], lq[j])
                        sim.r(Pauli.PauliZ, (Jij + noise) * dt + phase_shift, lq[j])
                        sim.cz(lq[i], lq[j])

            # Measurement
            mz = 0
            for q in lq:
                p0 = sim.prob(q)
                mz += (2 * p0 - 1)
            mz /= n_nodes
            magnetization.append(mz)

        results.append(magnetization)

    return np.array(results)

# Visualization
def plot_results(results, title="Dynamic Supply Chain Resilience vs Stress Cycles"):
    depth = results.shape[1]
    mean_mz = np.mean(results, axis=0)
    std_mz = np.std(results, axis=0)

    plt.figure(figsize=(16, 9))
    plt.errorbar(range(1, depth + 1), mean_mz, yerr=std_mz, fmt='-o', label='Mean ± Std Dev')
    plt.xlabel("Stress Cycles (Trotter Depth)")
    plt.ylabel("Supply Chain Functional % (Magnetization)")
    plt.title(title + f"\n({results.shape[0]} Trials, {results.shape[1]} Steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("supply_chain_v1.png", dpi=150)
    plt.show()

# Example run
if __name__ == "__main__":
    n_nodes = 64
    lr_columns = 3
    lr_rows = 3
    results = run_dynamic_J_sim(
        n_nodes,
        lr_columns,
        lr_rows,
        h=2.0,
        dt=0.25,
        depth=20,
        trials=5,
        phase_shift=0.1,
        noise_strength=0.05
    )

    plot_results(results, title=f"Dynamic J(t) Supply Chain Resilience (N={n_nodes})")

