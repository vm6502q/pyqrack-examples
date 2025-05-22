# Random Circuit Sampling (RCS) benchmark

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n, depth):
    # This is basically a "quantum volume" (random) circuit.
    start = time.perf_counter()

    sim = QrackSimulator(n)

    lcv_range = range(n)
    d_range = range(depth)
    all_bits = list(lcv_range)

    for _ in d_range:
        # Single-qubit gates
        for i in lcv_range:
            sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            sim.mcx([unused_bits.pop()], unused_bits.pop())

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 rcs.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print("Width=" + str(width) + ", Depth=" + str(depth) + ", Seconds=" + str(result))

    return 0


if __name__ == '__main__':
    sys.exit(main())
