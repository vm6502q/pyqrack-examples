# QAOA example for MAXCUT
# Produced by (OpenAI custom GPT) Elara

import pennylane as qml
from pennylane import numpy as np

# Define a simple 4-node graph for MAXCUT
edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
# Number of qubits, also number of nodes
num_qubits = len(edges)

# Define Qrack as the backend
dev = qml.device("qrack.simulator", wires=num_qubits)

# Output formatting
def bit_string_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

def int_to_nodes(int_sample):
    nodes = []
    i = 0
    while int_sample:
        if int_sample & 1:
            nodes.append(i)
        int_sample = int_sample >> 1
        i = i + 1

    return nodes

# QAOA Cost Hamiltonian (for MAXCUT)
def cost_hamiltonian(gamma):
    for edge in edges:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)

# QAOA Mixer Hamiltonian
def mixer_hamiltonian(beta):
    for qubit in range(num_qubits):
        qml.RX(2 * beta, wires=qubit)

# Define QAOA Circuit
def qaoa_circuit(params):
    num_layers = len(params) >> 1
    gammas, betas = params[:num_layers], params[num_layers:]

    # Initialize in equal superposition state
    for qubit in range(num_qubits):
        qml.Hadamard(wires=qubit)

    # Apply alternating Cost and Mixer layers
    for i in range(num_layers):
        cost_hamiltonian(gammas[i])
        mixer_hamiltonian(betas[i])

# Define expectation value (cost function)
@qml.qnode(dev)
def cost_function(params, return_samples=False):
    qaoa_circuit(params)

    if return_samples:
        # sample bitstrings to obtain cuts
        return qml.sample()

    return qml.expval(qml.sum(*(qml.PauliZ(n1) @ qml.PauliZ(n2) for n1, n2 in edges)))

def objective(params):
    return -0.5 * (len(edges) - cost_function(params))

# Classical optimizer
def optimize_qaoa(steps=30, num_layers=2):
    params = np.random.uniform(0, np.pi, num_layers << 1)
    opt = qml.AdagradOptimizer(stepsize=0.5)
    for _ in range(steps):
        params = opt.step(objective, params)

    return params  # Optimized parameters

# Run QAOA Optimization
optimal_params = optimize_qaoa()
# Sample 100 bit strings 
bitstrings = cost_function(optimal_params, return_samples=True, shots=100)
# Convert the samples bit strings to integers
sampled_ints = [bit_string_to_int(string) for string in bitstrings]
# Count frequency of each bit string
counts = np.bincount(np.array(sampled_ints))
# Most frequent bit string is the optimal solution
optimal_solution = np.argmax(counts)

print(f"Optimal Parameters: {optimal_params}")
print(f"Optimal MAXCUT bit string: {optimal_solution}")
print(f"Optimal MAXCUT edges: {int_to_nodes(optimal_solution)}")