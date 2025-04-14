# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# You probably want to set environment variable QRACK_MAX_PAGING_QB=-1.

import math
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli

def bench_qrack(width, depth):
    lcv_range = range(width)
    all_bits = list(lcv_range)

    start = time.perf_counter()

    experiment = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False)

    for _ in range(depth):
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                experiment.h(i)
                experiment.r(Pauli.PauliZ, random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            experiment.mcx([c], t)

    # Compare this case on single amplitudes:
    experiment.prob_perm(0)

    interval = time.perf_counter() - start

    return interval


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 fc.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print("Width=" + str(width) + ", Depth=" + str(depth) + ", Seconds=" + str(result))

    return 0


if __name__ == '__main__':
    sys.exit(main())
