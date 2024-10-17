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


def bench_qrack(num_qubits, noise):
    circ = QuantumCircuit(num_qubits, num_qubits)
    # Single-qubit gates
    for i in range(num_qubits):
        circ.u(random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), i)
    qft(num_qubits, circ)
    reverse(num_qubits, circ)
    for i in range(num_qubits):
        circ.measure(i, i)
    start = time.perf_counter()

    n_pow = 2**num_qubits
    noisy_result = QasmSimulator(shots=n_pow, noise=noise).run(circ).result().get_counts(circ)
    result = QasmSimulator(shots=n_pow).run(circ).result().get_counts(circ)

    u_u = 0
    for i in range(n_pow):
        b = (bin(i)[2:]).zfill(num_qubits)
        u_u = result[b] if b in result else 0
    u_u /= n_pow

    numer = 0
    denom = 0
    for i in range(n_pow):
        b = (bin(i)[2:]).zfill(num_qubits)

        noisy = (noisy_result[b] if b in noisy_result else 0) / n_pow
        ideal = (result[b] if b in result else 0) / n_pow

        # XEB / EPLG
        denom = denom + (ideal - u_u) ** 2
        numer = numer + (ideal - u_u) * (noisy - u_u)

    return numer / denom


def main():
    noise = 0.1
    max_qb = 10
    if len(sys.argv) > 1:
        noise = float(sys.argv[1])
    if len(sys.argv) > 2:
        max_qb = int(sys.argv[2])

    print(max_qb, ": " + str(bench_qrack(max_qb, noise)) + " XEB.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
