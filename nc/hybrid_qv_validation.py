import torch
import numpy as np
from collections import Counter
from qiskit_aer.backends import AerSimulator
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator

# --- Model Definition (must match training) ---
class RepairNet(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(2 * dim, 256)
        self.fc2 = torch.nn.Linear(256, dim)
        self.act = torch.nn.ReLU()

    def forward(self, nc, ace):
        x = torch.cat([nc, ace], dim=1)
        x = self.act(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# --- Load model ---
def load_repair_model(width, path="repair_net.pt"):
    dim = 1 << width
    model = RepairNet(dim)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

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

# --- HOG probability ---
def hog_probability(probs_ideal, probs_test):
    dim = len(probs_ideal)
    median_prob = np.median(probs_ideal)
    heavy_outputs = {i for i, p in enumerate(probs_ideal) if p > median_prob}
    return sum(probs_test[i] for i in heavy_outputs)

# --- Main test ---
if __name__ == "__main__":
    width = 8
    depth = width  # "square" for quantum volume

    print("Width = Depth = " + str(width))

    # Load trained model
    model = load_repair_model(width)

    # Get distributions
    nc, ace, gold = generate_distributions(width, depth)

    # Model inference
    nc_t = torch.tensor(nc, dtype=torch.float32).unsqueeze(0)
    ace_t = torch.tensor(ace, dtype=torch.float32).unsqueeze(0)
    repaired = model(nc_t, ace_t).detach().numpy().squeeze()

    torch.save(repaired.state_dict(), "repair_net_scalable.pt")

    # HOG probability
    hog_ml = hog_probability(gold, repaired)
    hog_nc = hog_probability(gold, nc)
    hog_ace = hog_probability(gold, ace)
    hog_hybrid = hog_probability(gold, 0.5 * nc + 0.5 * ace)

    print(f"Heavy Output Generation Probability:")
    print(f"  Near-Clifford: {hog_nc:.4f}")
    print(f"  ACE:           {hog_ace:.4f}")
    print(f"  50/50 Hybrid:  {hog_hybrid:.4f}")
    print(f"  ML-Repaired:   {hog_ml:.4f}")

