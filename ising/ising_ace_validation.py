# Ising model Trotterization as interpreted by (OpenAI GPT) Elara
# You likely want to specify environment variable QRACK_MAX_PAGING_QB=28

import math
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from pyqrack import QrackSimulator

def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape
    
    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols)
                   for r in range(n_rows) for c in range(0, n_cols - 1, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [(r * n_cols + c, r * n_cols + (c + 1) % n_cols)
                   for r in range(n_rows) for c in range(1, n_cols - 1, 2)]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
                  for r in range(0, n_rows - 1, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [(r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
                  for r in range(1, n_rows - 1, 2) for c in range(n_cols)]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    return circ


def calc_stats(ideal_probs, counts, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1

    return {
        'qubits': n,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'p-value': p_val
    }


def main():
    depth = 1
    if len(sys.argv) > 1:
        depth = int(sys.argv[1])

    n_rows, n_cols = 7, 8
    n_qubits = n_rows * n_cols
    J, h, dt = -1.0, 2.0, 0.25
    theta = -math.pi / 6
    shots = 1 << (n_qubits + 6)

    qc = QuantumCircuit(n_qubits)

    for q in range(n_qubits):
        qc.ry(theta, q)

    for _ in range(depth):
        trotter_step(qc, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    basis_gates = ["u", "cu", "cx", "cy", "cz", "cp", "swap", "iswap"]
    qc = transpile(qc, basis_gates=basis_gates)

    experiment = QrackSimulator(n_qubits)
    control = AerSimulator(method="statevector")
    experiment.run_qiskit_circuit(qc, shots=0)
    experiment_fidelity = experiment.get_unitary_fidelity()
    qc.save_statevector()
    job = control.run(qc)
    experiment_counts = dict(Counter(experiment.measure_shots(list(range(n_qubits)), shots)))
    control_probs = Statevector(job.result().get_statevector()).probabilities()
    
    print("Trotter steps: " + str(depth))
    print("Estimated fidelity: " + str(experiment_fidelity))
    print("Empirical results: " + str(calc_stats(control_probs, experiment_counts, shots)))

    return 0


if __name__ == '__main__':
    sys.exit(main())
