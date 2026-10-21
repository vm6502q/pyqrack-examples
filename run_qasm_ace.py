# Convert any circuit to a near-Clifford tableau

import json
import sys
import time

from collections import Counter

from qiskit.providers.qrack.backends import AceQasmSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def run_qasm(shots, file_in, file_out):
    sim = AceQasmSimulator()
    qc = QuantumCircuit.from_qasm_file(file_in)
    t_start = time.perf_counter()
    qc = transpile(qc, backend=sim, optimization_level=3)
    logical_to_physical = qc.layout.final_index_layout()
    for logical_idx in range(qc.num_qubits):
        physical_qubit = logical_to_physical[logical_idx]
        # Measure the exact physical wire into its designated classical bit
        qc.measure(physical_qubit, logical_idx)
    t_trans = time.perf_counter()
    print(f"transpile_seconds: {t_trans - t_start:.4f}")

    ace_str_counts = dict(sim.run(qc, shots=shots).result().get_counts())
    t_exec = time.perf_counter()
    print(f"execute_seconds: {t_exec - t_trans:.4f}")

    ace_counts = {}
    for s, count in ace_str_counts.items():
        ace_counts[int(s, 2)] = count

    with open(file_out, "w", encoding="utf-8") as file:
        json.dump(ace_counts, file)


def main():
    shots = 8192
    file_in = "qft.qasm"
    file_out = "out.json"
    if len(sys.argv) > 1:
        shots = int(sys.argv[1])
    if len(sys.argv) > 2:
        file_in = str(sys.argv[2])
    if len(sys.argv) > 3:
        file_out = str(sys.argv[3])

    run_qasm(shots, file_in, file_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
