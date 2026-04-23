import math
import random
import statistics
import sys
import time

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit


def calc_stats(ideal_probs, exp_probs, sdrp):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = 1 / 2
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    for i in range(n_pow):
        exp = (1 - model) * (exp_probs[i] if i in exp_probs else 0) + model * mean_guess
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # L2 norm
        sqr_diff += (ideal - exp) ** 2

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "sdrp": sdrp,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_diff": float(rss),
    }


def reverse(num_qubits, circ):
    start = 0
    end = num_qubits - 1
    while start < end:
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
        circuit.cp(math.pi / 2 ** (n - qubit), qubit, n)

    # Recursive QFT is very similiar to a ("classical") FFT
    qft(n, circuit)


def bench_qrack(n, sdrp):
    circ = QuantumCircuit(n)

    # GHZ state
    circ.h(0)
    for i in range(1, n):
        circ.cx(i - 1, i)

    qft(n, circ)
    reverse(n, circ)

    control = QrackSimulator(n, isBinaryDecisionTree=False)
    control.run_qiskit_circuit(circ, shots=0)
    control = control.out_probs()

    experiment = QrackSimulator(n, isBinaryDecisionTree=True)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    experiment.run_qiskit_circuit(circ, shots=0)
    experiment = experiment.out_probs()

    return calc_stats(control, experiment, sdrp)


def main():
    bench_qrack(1, 0)

    n = 16
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    sdrp = (1-1/math.sqrt(2))/2
    if len(sys.argv) > 2:
        sdrp = int(sys.argv[1])

    print(bench_qrack(n, sdrp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
