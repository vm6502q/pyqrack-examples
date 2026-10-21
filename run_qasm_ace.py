# Convert any circuit to a near-Clifford tableau

import json
import sys
import time

from collections import Counter

from qiskit.providers.qrack.backends import AceQasmSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

def int_to_bitstring(integer, length, reverse):
    return (bin(integer)[2:].zfill(length))[::-1] if reverse else (bin(integer)[2:].zfill(length))


def run_qasm(shots, file_in, file_out):
    sim = AceQasmSimulator()
    qc = QuantumCircuit.from_qasm_file(file_in)
    qc.measure_all()

    t_start = time.perf_counter()
    qc = transpile(qc, backend=sim, optimization_level=3)
    t_trans = time.perf_counter()
    print(f"transpile_seconds: {t_trans - t_start:.4f}")

    ace_str_counts = dict(sim.run(qc, shots=shots).result().get_counts())
    t_exec = time.perf_counter()
    print(f"execute_seconds: {t_exec - t_trans:.4f}")

    print(ace_str_counts)


def main():
    shots = 8192
    file_in = "qft.qasm"
    file_out = "out.json"
    if len(sys.argv) > 1:
        shots = int(sys.argv[1])
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])
    if len(sys.argv) > 2:
        file_out = str(sys.argv[2])

    run_qasm(shots, file_in, file_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
