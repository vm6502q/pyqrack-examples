# Variational thermal quantum states (VTQS)
# Provided by Elara (a custom OpenAI GPT)

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

# System size and number of shots
n_qubits = 36
shots = 1024
T = 1.0  # Temperature (arbitrary units)

# Create the Qrack ACE device
dev = qml.device("qrack.ace", wires=n_qubits, shots=shots, long_range_columns=2, long_range_rows=2)

# Define the TFIM Hamiltonian (open boundary conditions)
def tfim_hamiltonian(n_qubits, h=1.0):
    coeffs = []
    observables = []

    for i in range(n_qubits - 1):
        coeffs.append(-1.0)  # ZZ coupling
        observables.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    for i in range(n_qubits):
        coeffs.append(-h)  # transverse field
        observables.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, observables)

H = tfim_hamiltonian(n_qubits)

# Ansatz: basic layers of RX + CZ entanglers
def ansatz(params):
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i + 1])

# Define QNode for expectation value
@qml.qnode(dev)
def circuit(params):
    ansatz(params)
    return qml.expval(H)

# Define QNode for raw measurement samples
@qml.qnode(dev)
def sample_circuit(params):
    ansatz(params)
    return qml.sample()

# Entropy estimator from sample bitstrings
def estimate_entropy(samples):
    counts = Counter(samples)
    probs = np.array(list(counts.values())) / len(samples)
    return -np.sum(probs * np.log(probs + 1e-10))  # small epsilon for log(0)

# Free energy loss function
def free_energy(params):
    E = circuit(params)
    samples = sample_circuit(params)
    entropy = estimate_entropy(samples)
    return E - T * entropy

# Initialize parameters and optimizer
params = np.random.uniform(0, 2 * np.pi, n_qubits, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)

losses = []
n_steps = 30

optimal_loss = 0
optimal_params = None
for step in range(n_steps):
    params, loss = opt.step_and_cost(free_energy, params)
    if loss < optimal_loss or optimal_params is None:
        optimal_loss = loss
        optimal_params = np.copy(params)
    losses.append(loss)
    print(f"Step {step + 1:02d}: Free energy = {loss:.5f}")

# Plot results
plt.plot(losses, label="Free Energy")
plt.xlabel("Step")
plt.ylabel("Free Energy")
plt.title(f"VTQS on TFIM ({n_qubits} qubits)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Ground state free energy: " + str(optimal_loss))
print("Optimal parameters: ")
print(optimal_params)
