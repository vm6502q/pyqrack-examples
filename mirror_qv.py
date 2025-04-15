# Demonstrate mirror circuit simplification

import math
import random
import sys
import time

from pyqrack import QrackSimulator, QrackCircuit


def count_set_bits(n):
        return bin(n).count('1')


def bench_qrack(n):
    circ = QrackCircuit()

    lcv_range = range(n)
    all_bits = list(lcv_range)
    x_op = [0, 1, 1, 0]

    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            cos0 = math.cos(th / 2)
            sin0 = math.sin(th / 2)
            u_op = [
                cos0 + 0j, sin0 * (-math.cos(lm) + -math.sin(lm) * 1j),
                sin0 * (math.cos(ph) + math.sin(ph) * 1j), cos0 * (math.cos(ph + lm) + math.sin(ph + lm) * 1j)
            ]
            circ.mtrx(u_op, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            circ.ucmtrx([unused_bits.pop()], x_op, unused_bits.pop(), 1)

    shots = 100
    start = time.perf_counter()
    sim = QrackSimulator(n)
    circ.run(sim)
    circ.inverse().run(sim)
    results = sim.measure_shots(all_bits, shots)
    seconds = time.perf_counter() - start
    hamming_distance = sum(count_set_bits(r) for r in results) / shots

    return (seconds, hamming_distance)


def main():
    n = 50
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    results = bench_qrack(n)

    print(n, "qubits,",
        results[0], "seconds,",
        results[1], "average Hamming distance"
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
