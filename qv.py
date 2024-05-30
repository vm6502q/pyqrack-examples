# Quantum volume protocol certification

import math
import random
import statistics
import sys
import time

import numpy as np

from pyqrack import QrackSimulator, QrackCircuit


def bench_qrack(n, sdrp = 0):
    # This is a "quantum volume" (random) circuit.
    circ = QrackCircuit()

    lcv_range = range(n)
    all_bits = list(lcv_range)

    single_count = 0
    double_count = 0
    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            cos0 = np.cos(th / 2);
            sin0 = np.sin(th / 2);
            u_op = [
                cos0 + 0j, sin0 * (-np.cos(lm) + -np.sin(lm) * 1j),
                sin0 * (np.cos(ph) + np.sin(ph) * 1j), cos0 * (np.cos(ph + lm) + np.sin(ph + lm) * 1j)
            ]
            circ.mtrx(u_op, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            x_op = [0, 1, 1, 0]
            circ.ucmtrx([unused_bits.pop()], x_op, unused_bits.pop(), 1)

    sim = QrackSimulator(n, isTensorNetwork=False)
    circ.run(sim)
    ideal_probs = [np.real(x * np.conj(x)) for x in sim.out_ket()]

    start = time.perf_counter()
    sim = QrackSimulator(n, isTensorNetwork=False)
    if sdrp > 0:
        sim.set_sdrp(sdrp)
    circ.run(sim)
    interval = time.perf_counter() - start

    fidelity = sim.get_unitary_fidelity()
    approx_probs = [np.real(x * np.conj(x)) for x in sim.out_ket()]

    return (ideal_probs, approx_probs, interval, fidelity)


def main():
    n = 19
    sdrp = 0.3
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        sdrp = float(sys.argv[1])
    n_pow = 1 << n

    results = bench_qrack(n, sdrp)

    ideal_probs = results[0]
    approx_probs = results[1]
    interval = results[2]
    fidelity = results[3]

    # We compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    threshold = statistics.median(ideal_probs)
    sum_prob = 0
    for i in range(n_pow):
        if ideal_probs[i] >= threshold:
            sum_prob = sum_prob + approx_probs[i]

    print(n, "qubits,", sdrp, "SDRP:", interval, "seconds,", fidelity, "fidelity,", sum_prob, "HOG probability")

    return 0


if __name__ == '__main__':
    sys.exit(main())
