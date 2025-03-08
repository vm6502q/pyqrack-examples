# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# (Requires PennyLane and OpenFermion)

import pennylane as qml
from pennylane import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner

# Step 1: Define the molecule (H2, HeH+, BeH2)
geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.74))]  # H2 Molecule
# basis = 'sto-3g'  # Minimal Basis Set
basis = '6-31g'  # Larger basis set
multiplicity = 1
charge = 0

# Step 2: Compute the Molecular Hamiltonian
molecule = of.MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()

# Step 3: Convert to Qubit Hamiltonian (Jordan-Wigner)
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

# Step 4: Extract Qubit Terms for PennyLane
coeffs = []
observables = []
n_qubits = molecule.n_qubits  # Auto-detect qubit count
print(str(n_qubits) + " qubits...")

for term, coefficient in qubit_hamiltonian.terms.items():
    pauli_operators = []
    for qubit_idx, pauli in term:
        if pauli == 'X':
            pauli_operators.append(qml.PauliX(qubit_idx))
        elif pauli == 'Y':
            pauli_operators.append(qml.PauliY(qubit_idx))
        elif pauli == 'Z':
            pauli_operators.append(qml.PauliZ(qubit_idx))
    if pauli_operators:
        observable = pauli_operators[0]
        for op in pauli_operators[1:]:
            observable = observable @ op
        observables.append(observable)
    else:
        observables.append(qml.Identity(0))  # Default identity if no operators
    coeffs.append(qml.numpy.array(coefficient, requires_grad=False))

hamiltonian = qml.Hamiltonian(coeffs, observables)

# Step 5: Define Qrack Backend
dev = qml.device("qrack.simulator", wires=n_qubits)  # Replace with "default.qubit" for CPU test

# Step 6: Define a Simple Variational Ansatz
def ansatz(params, wires):
    qml.BasisState(np.array([0] * len(wires)), wires=wires)  # Initialize |000...0>
    for i in range(len(wires)):
        qml.RY(params[i], wires=i)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[i, i + 1])

# Step 7: Cost Function for VQE (Expectation of Hamiltonian)
@qml.qnode(dev)
def circuit(params):
    ansatz(params, wires=range(n_qubits))
    return qml.expval(hamiltonian)  # Now correctly outputs a scalar

# Step 8: Optimize the Energy
opt = qml.AdamOptimizer(stepsize=0.05)
theta = np.random.randn(n_qubits, requires_grad=True)  # Ensure trainable parameters
num_steps = 100

for step in range(num_steps):
    theta = opt.step(circuit, theta)
    energy = circuit(theta)  # Compute energy at new parameters
    print(f"Step {step+1}: Energy = {energy}")

print("Optimized Ground State Energy:", energy)
