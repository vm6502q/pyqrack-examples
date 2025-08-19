# Convert any circuit to a near-Clifford tableau

import json
import sys

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

def int_to_bitstring(integer, length, reverse):
    return (bin(integer)[2:].zfill(length))[::-1] if reverse else (bin(integer)[2:].zfill(length))


def run_qasm(file_in, file_out):
    shot_count = 1024
    # shot_count = 536870912
    qc = QuantumCircuit.from_qasm_file(file_in)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc = transpile(qc, basis_gates=basis_gates)
    sim = QrackSimulator(qc.num_qubits, isTensorNetwork=False)
    sim.run_qiskit_circuit(qc, shots=0)
    print("Fidelity estimate: " + str(sim.get_unitary_fidelity()))
    shots = dict(Counter(sim.measure_shots(list(range(qc.num_qubits)), shot_count)))
    with open(file_out, "w") as f:
        json.dump(shots, f)
    max_key = max(shots, key=shots.get)
    print(f"Total shots: {shot_count}")
    print(f"Peak counts: {(max_key, shots[max_key])}")
    rtl = int_to_bitstring(max_key, qc.num_qubits, False)
    ltr = int_to_bitstring(max_key, qc.num_qubits, True)
    print(f"Right-to-left, least-to-most significant: {rtl}")
    print(f"Left-to-right, least-to-most significant: {ltr}")


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
