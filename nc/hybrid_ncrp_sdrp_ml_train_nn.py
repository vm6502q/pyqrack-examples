# Train a model to infer the gold standard distribution
# from a combination of NCRP and SDRP features.
# (Created by Elara, the custom OpenAI GPT)

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
# Generate SDRP, NC, Gold for width/depth
# ─────────────────────────────
def factor_width(width):
    col_len = int(np.floor(np.sqrt(width)))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)

def cx(sim, q1, q2):
    sim.cx(q1, q2)

def cy(sim, q1, q2):
    sim.cy(q1, q2)

def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)

def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)

def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)

def swap(sim, q1, q2):
    sim.swap(q1, q2)

def iswap(sim, q1, q2):
    sim.iswap(q1, q2)

def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)

def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)

def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)

def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)

def probs_from_shots(width, shots, counts):
    dim = 1 << width
    probs = [0.0] * dim
    for i in range(dim):
        if i in counts:
            probs[i] = counts[i] / shots
    return probs

def generate_distributions(width, depth):
    shots = max(8192, 1 << (width + 2))
    qubits = list(range(width))
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    row_len, col_len = factor_width(width)

    # Make random nearest-neighbor RCS
    circ = QuantumCircuit(width)
    for d in range(depth):
        for i in range(width):
            circ.u(*np.random.uniform(0, 2*np.pi, 3), i)

            gate = gateSequence.pop(0)
            gateSequence.append(gate)
            for row in range(1, row_len, 2):
                for col in range(col_len):
                    temp_row = row
                    temp_col = col
                    temp_row = temp_row + (1 if (gate & 2) else -1)
                    temp_col = temp_col + (1 if (gate & 1) else 0)

                    if temp_row < 0:
                        temp_row = temp_row + row_len
                    if temp_col < 0:
                        temp_col = temp_col + col_len
                    if temp_row >= row_len:
                        temp_row = temp_row - row_len
                    if temp_col >= col_len:
                        temp_col = temp_col - col_len

                    b1 = row * row_len + col
                    b2 = temp_row * row_len + temp_col

                    if (b1 >= width) or (b2 >= width):
                        continue

                    g = np.random.choice(two_bit_gates)
                    g(circ, b1, b2)

    # Near-Clifford
    nc = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
    nc.set_ncrp(2.0)
    nc_circ = transpile(circ, basis_gates=["rz","h","x","y","z","sx","sxdg","s","sdg","t","tdg","cx","cy","cz","swap","iswap"])
    nc.run_qiskit_circuit(nc_circ)
    samples_nc = dict(Counter(nc.measure_shots(qubits, shots)))
    del nc
    del nc_circ
    probs_nc = np.array(probs_from_shots(width, shots, samples_nc))
    del samples_nc

    # SDRP
    sdrp = QrackSimulator(width)
    sdrp.set_sdrp(2.0)
    sdrp_circ = transpile(circ, basis_gates=QrackSimulator.get_qiskit_basis_gates())
    samples_sdrp = dict(Counter(sdrp.measure_shots(qubits, shots)))
    del sdrp
    del sdrp_circ
    probs_sdrp = np.array(probs_from_shots(width, shots, samples_sdrp))
    del samples_sdrp


    # Gold standard
    backend = AerSimulator(method="statevector")
    circ_sv = transpile(circ, backend=backend)
    circ_sv.save_statevector()
    sv = backend.run(circ_sv).result().get_statevector()
    probs_gold = Statevector(sv).probabilities()

    return probs_nc, probs_sdrp, np.array(probs_gold)

# ─────────────────────────────
# Build training data
# ─────────────────────────────
def build_dataset(widths, depth_factor=1, samples_per_width=32):
    features, targets = [], []
    for w in widths:
        d = w * depth_factor
        for _ in range(samples_per_width):
            nc, sdrp, gold = generate_distributions(w, d)
            dim = len(nc)
            for i in range(dim):
                # Features per basis state
                features.append([
                    nc[i],
                    sdrp[i],
                    (sdrp[i] - nc[i]) * 2.0,
                    (sdrp[i] + nc[i]) * 0.5
                ])
                targets.append([gold[i]])
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

# ─────────────────────────────
# Train
# ─────────────────────────────
def train_repair(widths=[2, 4], depth_factor=1, samples_per_width=80, epochs=144, lr=1e-3):
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
        if (epoch % (epochs // 10)) == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    return model

# ─────────────────────────────
# Apply model to any width
# ─────────────────────────────
def repair_distribution(model, nc, sdrp):
    dim = len(nc)
    features = np.stack([
        nc,
        sdrp,
        2.0 * (sdrp - nc),
        0.5 * (sdrp + nc)
    ], axis=1).astype(np.float32)

    with torch.no_grad():
        repaired = model(torch.tensor(features)).numpy().squeeze()

    # Normalize to ensure it's a valid distribution
    repaired = np.clip(repaired, 0, None)
    repaired /= np.sum(repaired)
    return repaired

# ─────────────────────────────
# Stats
# ─────────────────────────────
def hog_probability(probs_ideal, probs_test):
    median_prob = np.median(probs_ideal)
    heavy_outputs = {i for i, p in enumerate(probs_ideal) if p > median_prob}
    return sum(probs_test[i] for i in heavy_outputs)

def fidelity(probs_ideal, probs_test):
    return 1.0 - sum((i - t) ** 2 for i, t in zip(probs_ideal, probs_test)) ** (1/2)

def cross_entropy(probs_ideal, probs_test):
    u_u = np.mean(probs_ideal)
    denom = 0.0
    numer = 0.0
    for ideal, test in zip(probs_ideal, probs_test):
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (test - u_u)

    return numer / denom

# ─────────────────────────────
# Demo
# ─────────────────────────────
if __name__ == "__main__":
    # Train on small widths
    model = train_repair(widths=[4], depth_factor=1, samples_per_width=80, epochs=144)

    torch.save(model.state_dict(), "repair_net_nn.pt")

    # Test on different width (never seen in training)
    test_width = 6
    nc, sdrp, gold = generate_distributions(test_width, test_width)
    mixed = [(nc[i] + sdrp[i]) / 2 for i in range(len(nc))]

    repaired = repair_distribution(model, nc, sdrp)

    print(f"HOG (NCRP):     {hog_probability(gold, nc):.4f}")
    print(f"HOG (SDRP):     {hog_probability(gold, sdrp):.4f}")
    print(f"HOG (50/50):    {hog_probability(gold, mixed):.4f}")
    print(f"HOG (Repaired): {hog_probability(gold, repaired):.4f}")
    print()
    print(f"Fidelity (NCRP):     {fidelity(gold, nc):.4f}")
    print(f"Fidelity (SDRP):     {fidelity(gold, sdrp):.4f}")
    print(f"Fidelity (50/50):    {fidelity(gold, mixed):.4f}")
    print(f"Fidelity (Repaired): {fidelity(gold, repaired):.4f}")
    print()
    print(f"XEB (NCRP):     {cross_entropy(gold, nc):.4f}")
    print(f"XEB (SDRP):     {cross_entropy(gold, sdrp):.4f}")
    print(f"XEB (50/50):    {cross_entropy(gold, mixed):.4f}")
    print(f"XEB (Repaired): {cross_entropy(gold, repaired):.4f}")
