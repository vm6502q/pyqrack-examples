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

    experiment = QrackSimulator(width, isSchmidtDecompose=False, isStabilizerHybrid=True)
    # Round to nearest Clifford circuit
    experiment.set_ncrp(1.0)

    for d in range(depth):
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

    samples = experiment.measure_shots(all_bits, 1)

    print({ 'qubits': width, 'depth': d+1, 'seconds': time.perf_counter() - start })


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
