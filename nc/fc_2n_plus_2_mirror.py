# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# You probably want to set environment variable QRACK_MAX_PAGING_QB=-1.

import math
import random
import sys
import time

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n)))


def bench_qrack(n_qubits):
    # This is a "fully-connected" coupler random circuit.
    lcv_range = range(n_qubits)
    all_bits = list(lcv_range)
    shots = 100

    rz_count = (n_qubits + 1) << 1
    rz_opportunities =  n_qubits * n_qubits * 3
    rz_positions = []
    while len(rz_positions) < rz_count:
        rz_position = random.randint(0, rz_opportunities - 1)
        if rz_position in rz_positions:
            continue
        rz_positions.append(rz_position)

    qc = QuantumCircuit(n_qubits)
    gate_count = 0
    start = time.perf_counter()
    for d in range(n_qubits):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                experiment.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    experiment.z(i)
                if s_count & 2:
                    experiment.s(i)
                if gate_count in rz_positions:
                    experiment.r(Pauli.PauliZ, random.uniform(0, math.pi / 2), i)
                gate_count = gate_count + 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)
        
        experiment = QrackSimulator(n_qubits, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
        # Round to nearest Clifford circuit
        experiment.set_ncrp(2.0)
        experiment.run_qiskit_circuit(circ)
        experiment.run_qiskit_circuit(circ.inverse())

        samples = experiment.measure_shots(all_bits, shots)
        
        hamming_weight = 0
        for sample in samples:
            hamming_weight += hamming_distance(0, sample, n_qubits)
        hamming_weight /= shots

        print({ 'qubits': n_qubits, 'depth': d+1, 'seconds': time.perf_counter() - start, "avg_hamming_distance": hamming_weight })


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 fc.py [width]')

    n_qubits = int(sys.argv[1])

    # Run the benchmarks
    bench_qrack(n_qubits)

    return 0


if __name__ == '__main__':
    sys.exit(main())
