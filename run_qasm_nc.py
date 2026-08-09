# Convert any circuit to a near-Clifford tableau

import json
import sys

from collections import Counter

from pyqrack import QrackSimulator, QrackStabilizer

from qiskit import QuantumCircuit, transpile
from qiskit.providers.qrack import QStabilizerQasmSimulator

def int_to_bitstring(integer, length, reverse):
    return (bin(integer)[2:].zfill(length))[::-1] if reverse else (bin(integer)[2:].zfill(length))


def run_qasm(file_in, file_out):
    # shot_count = 1024
    # shot_count = 536870912
    qc = QuantumCircuit.from_qasm_file(file_in)
    experiment = QStabilizerQasmSimulator(n_qubits=qc.num_qubits)
    qc = transpile(qc, backend=experiment, optimization_level=3)
    print(f"{qc.count_ops().get('rz', 0)} RZ gates...")
    print(f"{qc.count_ops().get('t', 0)} T gates...")
    print(f"{qc.count_ops().get('tdg', 0)} inverse-T gates...")
    qc.measure_all()
    # aux = QrackSimulator(qc.num_qubits, is_near_clifford_tableau_writer=True)
    # aux.run_qiskit_circuit(qc, shots=0)
    # print("First-pass fidelity estimate: " + str(sim.get_unitary_fidelity()))
    shots = 1024
    experiment.run(qc, shots=shots).result().get_counts()


def main():
    file_in = "nq70_depth70_checks27_doped.qasm"
    file_out = "out.json"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    if len(sys.argv) > 2:
        file_out = str(sys.argv[2])

    run_qasm(file_in, file_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
