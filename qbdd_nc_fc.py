# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import os
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                # x-z-x Euler axes
                sim.h(i)
                sim.r(Pauli.PauliZ, random.uniform(0, 2 * math.pi), i)
            sim.h(i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            sim.h(t)
            sim.mcz([c], t)
            sim.h(t)

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 qbdd_fc.py [width] [depth]')

    width = int(sys.argv[1])

    depth = int(sys.argv[2])

    # Run the benchmarks
    time_result = bench_qrack(width, depth)

    print("Width=" + str(width) + ", Depth=" + str(depth) + ": " + str(time_result) + " seconds. (Fidelity is unknown.)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
