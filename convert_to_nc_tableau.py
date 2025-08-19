# Convert any circuit to a near-Clifford tableau

import sys

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def convert_to_augmented_tableau(file_in, file_out):
    qc = QuantumCircuit.from_qasm_file(file_in)
    # Near-Clifford basis set
    basis_gates = [
        "rz",
        "h",
        "x",
        "y",
        "z",
        "sx",
        "sxdg",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "cy",
        "cz",
        "swap",
        "iswap",
    ]
    qc = transpile(qc, basis_gates=basis_gates)
    sim = QrackSimulator(qc.num_qubits, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
    sim.run_qiskit_circuit(qc, shots=0)
    sim.out_to_file(file_out)


def main():
    file_in = "qft.qasm"
    file_out = "out.nc"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    if len(sys.argv) > 2:
        file_out = str(sys.argv[2])

    print("NOTE: Set environment variables QRACK_MAX_PAGING=-1 and QRACK_MAX_CPU_QB=-1")
    print("If your circuit has many non-Clifford gates, you might need a custom high-width build of Qrack.")

    convert_to_augmented_tableau(file_in, file_out)
    
    print("Done. (Check for file " + file_out + ")")

    return 0


if __name__ == "__main__":
    sys.exit(main())
