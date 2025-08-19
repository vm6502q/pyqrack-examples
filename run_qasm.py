# Convert any circuit to a near-Clifford tableau

import sys

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def run_qasm(file_in):
    qc = QuantumCircuit.from_qasm_file(file_in)
    # Near-Clifford basis set
    qc = transpile(qc, basis_gates=QrackSimulator.get_qiskit_basis_gates())
    sim = QrackSimulator(qc.num_qubits)
    sim.run_qiskit_circuit(qc, shots=0)
    print(sim.m_all())


def main():
    file_in = "qft.qasm"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])

    run_qasm(file_in)

    return 0


if __name__ == "__main__":
    sys.exit(main())
