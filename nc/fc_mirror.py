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


def bench_qrack(width, depth):
    shots = 100
    lcv_range = range(width)
    all_bits = list(lcv_range)

    start = time.perf_counter()
    
    circ = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            # Single-qubit gates
            for i in lcv_range:
                circ.h(i)
                circ.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)
        
        experiment = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
        # Round to nearest Clifford circuit
        experiment.set_ncrp(2.0)
        experiment.run_qiskit_circuit(circ)
        experiment.run_qiskit_circuit(circ.inverse())

        samples = experiment.measure_shots(all_bits, shots)
        
        hamming_weight = 0
        for sample in samples:
            hamming_weight += hamming_distance(0, sample, width)
        hamming_weight /= shots

        print({ 'qubits': width, 'depth': d+1, 'seconds': time.perf_counter() - start, "avg_hamming_distance": hamming_weight })


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 fc.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == '__main__':
    sys.exit(main())
