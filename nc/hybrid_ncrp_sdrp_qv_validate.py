# Train a model to infer the gold standard distribution
# from a combination of NCRP and SDRP features.
# (Created by Elara, the custom OpenAI GPT)

import torch
import numpy as np
from collections import Counter
from qiskit_aer.backends import AerSimulator
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator

# --- Load trained scalable model ---
class BasisRepairNet(torch.nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(feature_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        out = torch.relu(self.fc2(h))
        return out

def load_model(path="repair_net_scalable.pt", feature_dim=4):
    model = BasisRepairNet(feature_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Generate distributions ---
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

# --- Apply repair model ---
def repair_distribution(model, nc, sdrp):
    features = np.stack([
        nc,
        sdrp,
        sdrp - nc,
        0.5 * (sdrp + nc)
    ], axis=1).astype(np.float32)

    with torch.no_grad():
        repaired = model(torch.tensor(features)).numpy().squeeze()

    repaired = np.clip(repaired, 0, None)
    repaired /= np.sum(repaired)
    return repaired

# --- HOG probability ---
def hog_probability(probs_ideal, probs_test):
    median_prob = np.median(probs_ideal)
    heavy_outputs = {i for i, p in enumerate(probs_ideal) if p > median_prob}
    return sum(probs_test[i] for i in heavy_outputs)

# --- Quantum volume validation ---
def validate_qv(model_path, width, trials=20):
    model = load_model(model_path)
    hog_nc, hog_sdrp, hog_mixed, hog_repaired = [], [], [], []

    for _ in range(trials):
        nc, sdrp, gold = generate_distributions(width, width)
        mixed = [(nc[i] + sdrp[i]) / 2 for i in range(len(nc))]
        repaired = repair_distribution(model, nc, sdrp)

        hog_nc.append(hog_probability(gold, nc))
        hog_sdrp.append(hog_probability(gold, sdrp))
        hog_mixed.append(hog_probability(gold, mixed))
        hog_repaired.append(hog_probability(gold, repaired))

    print(f"Quantum Volume Validation (width = depth = {width}, trials = {trials})")
    print(f"  NCRP HOG:     mean = {np.mean(hog_nc):.4f}, std = {np.std(hog_nc):.4f}")
    print(f"  SDRP HOG:     mean = {np.mean(hog_sdrp):.4f}, std = {np.std(hog_sdrp):.4f}")
    print(f"  50/50 HOG:    mean = {np.mean(hog_mixed):.4f}, std = {np.std(hog_mixed):.4f}")
    print(f"  Repaired HOG: mean = {np.mean(hog_repaired):.4f}, std = {np.std(hog_repaired):.4f}")

if __name__ == "__main__":
    validate_qv("repair_net_fc.pt", width=8, trials=10)
