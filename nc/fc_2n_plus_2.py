# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# You probably want to set environment variable QRACK_MAX_PAGING_QB=-1.

import math
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli

def bench_qrack(width):
    lcv_range = range(width)
    all_bits = list(lcv_range)
    t_prob = ((width + 1) << 1) / (width * width * 3)

    start = time.perf_counter()

    experiment = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)
    # Round to nearest Clifford circuit
    experiment.set_ncrp(2.0)

    for d in range(width):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                experiment.h(i)
                s_count = random.randint(0, 3)
                if s_count & 1:
                    experiment.z(i)
                if s_count & 2:
                    experiment.s(i)
                if random.random() < t_prob:
                    experiment.r(Pauli.PauliZ, random.uniform(0, math.pi / 2), i)

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
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 fc.py [width]')

    width = int(sys.argv[1])

    # Run the benchmarks
    bench_qrack(width)

    return 0


if __name__ == '__main__':
    sys.exit(main())
