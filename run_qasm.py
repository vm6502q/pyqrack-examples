# Convert any circuit to a near-Clifford tableau

import json
import sys

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def run_qasm(file_in, file_out):
    qc = QuantumCircuit.from_qasm_file(file_in)
    # Near-Clifford basis set
    qc = transpile(qc, basis_gates=QrackSimulator.get_qiskit_basis_gates())
    sim = QrackSimulator(qc.num_qubits)
    sim.run_qiskit_circuit(qc, shots=0)
    shots = dict(Counter(sim.measure_shots(list(range(qc.num_qubits)), 1048576)))
    with open(file_out, "w") as f:
        json.dump(shots, f)
    print(shots)


def main():
    file_in = "qft.qasm"
    file_out = "out.json"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    if len(sys.argv) > 2:
        file_out = str(sys.argv[2])

    run_qasm(file_in, file_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
