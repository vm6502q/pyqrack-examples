# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import os
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            sim.mcz([unused_bits.pop()], unused_bits.pop())

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start


def main():
    if len(sys.argv) < 5:
        raise RuntimeError('Usage: python3 sdrp.py [qbddrp] [width] [depth] [samples]')

    qbddrp = float(sys.argv[1])
    if (qbddrp > 0):
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = sys.argv[1]

    width = int(sys.argv[2])

    depth = int(sys.argv[3])

    samples = int(sys.argv[4])

    # Run the benchmarks
    width_results = []
    for i in range(samples):
        width_results.append(bench_qrack(width, depth))

    time_result = sum(t for t in width_results) / samples
    print("Width=" + str(width) + ", Depth=" + str(depth) + ": " + str(time_result) + " seconds. (Fidelity is unknown.)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
