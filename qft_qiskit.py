import math
import random
import sys
import time

from qiskit import QuantumCircuit
from qiskit.providers.qrack import QasmSimulator


def reverse(num_qubits, circ):
    start = 0
    end = num_qubits - 1
    while (start < end):
        circ.swap(start, end)
        start += 1
        end -= 1

# Implementation of the Quantum Fourier Transform
# (See https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html)
def qft(n, circuit):
    if n == 0:
        return circuit
    n -= 1

    circuit.h(n)
    for qubit in range(n):
        circuit.cp(math.pi/2**(n-qubit), qubit, n)

    # Recursive QFT is very similiar to a ("classical") FFT
    qft(n, circuit)


def bench_qrack(num_qubits):
    circ = QuantumCircuit(num_qubits, num_qubits)
    # Single-qubit gates
    for i in range(num_qubits):
        circ.u(random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), i)
    qft(num_qubits, circ)
    reverse(num_qubits, circ)
    for i in range(num_qubits):
        circ.measure(i, i)
    start = time.perf_counter()
    result = QasmSimulator(shots=1).run(circ).result()

    return time.perf_counter() - start


def main():
    bench_qrack(1)

    max_qb = 24
    samples = 1
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    for n in range(1, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for _ in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(width_results) / samples
        print(n, ": ", time_result, " seconds.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
