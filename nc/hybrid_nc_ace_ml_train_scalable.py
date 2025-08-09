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


# ─────────────────────────────
# Basis-State Repair Network
# ─────────────────────────────
class BasisRepairNet(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (num_states, feature_dim)
        h = self.act(self.fc1(x))
        out = torch.relu(self.fc2(h))  # keep non-negative
        return out

# ─────────────────────────────
# Generate ACE, NC, Gold for width/depth
# ─────────────────────────────
def probs_from_shots(width, shots, counts):
    dim = 1 << width
    probs = [0.0] * dim
    for i in range(dim):
        if i in counts:
            probs[i] = counts[i] / shots
    return probs

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

# ─────────────────────────────
# Build training data
# ─────────────────────────────
def build_dataset(widths, depth_factor=1, samples_per_width=32):
    features, targets = [], []
    for w in widths:
        d = w * depth_factor
        for _ in range(samples_per_width):
            nc, ace, gold = generate_distributions(w, d)
            dim = len(nc)
            for i in range(dim):
                # Features per basis state
                features.append([
                    nc[i],
                    ace[i],
                    (ace[i] - nc[i]) * 2.0,
                    (ace[i] + nc[i]) * 0.5
                ])
                targets.append([gold[i]])
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

# ─────────────────────────────
# Train
# ─────────────────────────────
def train_repair(widths=[8], depth_factor=1, samples_per_width=32, epochs=100, lr=1e-3):
    X, Y = build_dataset(widths, depth_factor, samples_per_width)
    model = BasisRepairNet(feature_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X)
    Y_t = torch.tensor(Y)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    return model

# ─────────────────────────────
# Apply model to any width
# ─────────────────────────────
def repair_distribution(model, nc, ace):
    dim = len(nc)
    features = np.stack([
        nc,
        ace,
        2.0 * (ace - nc),
        0.5 * (ace + nc)
    ], axis=1).astype(np.float32)

    with torch.no_grad():
        repaired = model(torch.tensor(features)).numpy().squeeze()

    # Normalize to ensure it's a valid distribution
    repaired = np.clip(repaired, 0, None)
    repaired /= np.sum(repaired)
    return repaired

# ─────────────────────────────
# HOG probability
# ─────────────────────────────
def hog_probability(probs_ideal, probs_test):
    median_prob = np.median(probs_ideal)
    heavy_outputs = {i for i, p in enumerate(probs_ideal) if p > median_prob}
    return sum(probs_test[i] for i in heavy_outputs)

# ─────────────────────────────
# Demo
# ─────────────────────────────
if __name__ == "__main__":
    # Train on small widths
    model = train_repair(widths=[3, 4, 5], depth_factor=1, samples_per_width=32, epochs=32)

    torch.save(model.state_dict(), "repair_net_scalable.pt")

    # Test on different width (never seen in training)
    test_width = 6
    nc, ace, gold = generate_distributions(test_width, test_width)

    repaired = repair_distribution(model, nc, ace)

    print(f"HOG (NC):       {hog_probability(gold, nc):.4f}")
    print(f"HOG (ACE):      {hog_probability(gold, ace):.4f}")
    print(f"HOG (Repaired): {hog_probability(gold, repaired):.4f}")

