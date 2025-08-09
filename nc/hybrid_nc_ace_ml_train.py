# Train a model to infer the gold standard distribution
# from a combination of near-Clifford and ACE features.
# (Created by Elara, the custom OpenAI GPT)

# IMPORTANT: Remember to set ACE width appropriately in your shell environment variables!

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from qiskit.quantum_info import Statevector
from qiskit_aer.backends import AerSimulator
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator

def probs_from_shots(width, shots, counts):
    dim = 1 << width
    probs = [0.0] * dim
    for i in range(dim):
        if i in counts:
            probs[i] = counts[i] / shots
    return probs

# --- Generate distributions ---
def generate_distributions(width, depth):
    shots = 1 << (width + 2)
    qubits = list(range(width))

    # Make random nearest-neighbor RCS
    circ = QuantumCircuit(width)
    for d in range(depth):
        for i in range(width):
            circ.u(*np.random.uniform(0, 2*np.pi, 3), i)
        unused = list(range(width))
        np.random.shuffle(unused)
        while len(unused) > 1:
            c, t = unused.pop(), unused.pop()
            circ.cx(c, t)

    # Near-Clifford
    nc = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
    nc.set_ncrp(1.0)
    nc_circ = transpile(circ, basis_gates=["rz","h","x","y","z","sx","sxdg","s","sdg","t","tdg","cx","cy","cz","swap","iswap"])
    nc.run_qiskit_circuit(nc_circ)
    samples_nc = dict(Counter(nc.measure_shots(qubits, shots)))
    probs_nc = np.array(probs_from_shots(width, shots, samples_nc))

    # ACE
    ace = QrackSimulator(width)
    ace_circ = transpile(circ, basis_gates=QrackSimulator.get_qiskit_basis_gates())
    samples_ace = dict(Counter(ace.measure_shots(qubits, shots)))
    probs_ace = np.array(probs_from_shots(width, shots, samples_ace))

    # Gold standard
    backend = AerSimulator(method="statevector")
    circ_sv = transpile(circ, backend=backend)
    circ_sv.save_statevector()
    sv = backend.run(circ_sv).result().get_statevector()
    probs_gold = Statevector(sv).probabilities()

    return probs_nc, probs_ace, np.array(probs_gold)

# --- Simple repair network ---
class RepairNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(2*dim, 256)
        self.fc2 = nn.Linear(256, dim)
        self.act = nn.ReLU()

    def forward(self, nc, ace):
        x = torch.cat([nc, ace], dim=1)
        x = self.act(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# --- Training loop ---
def train_repair(width=8, depth=8, epochs=100, samples=32):
    dim = 1 << width
    model = RepairNet(dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_nc, X_ace, Y_gold = [], [], []
    for _ in range(samples):
        nc, ace, gold = generate_distributions(width, depth)
        X_nc.append(nc)
        X_ace.append(ace)
        Y_gold.append(gold)

    X_nc = torch.tensor(X_nc, dtype=torch.float32)
    X_ace = torch.tensor(X_ace, dtype=torch.float32)
    Y_gold = torch.tensor(Y_gold, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_nc, X_ace)
        loss = loss_fn(pred, Y_gold)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

if __name__ == "__main__":
    model = train_repair(width=8, depth=8)
    torch.save(model.state_dict(), "repair_net.pt")

