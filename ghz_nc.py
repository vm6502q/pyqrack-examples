# Near-Clifford GHZ state preparation simulation

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n):
    start = time.perf_counter()

    sim = QrackSimulator(n, isTensorNetwork=False, isSchmidtDecompose=False, isStabilizerHybrid=True)

    lcv_range = range(n)
    all_bits = list(lcv_range)

    single_count = 0
    double_count = 0
    sim.h(0)
    for q in range(n - 1):
        sim.mcx([q], q + 1)

    fidelity = sim.get_unitary_fidelity()
    # Terminal measurement
    sim.m_all()

    return (time.perf_counter() - start, fidelity)


def main():
    bench_qrack(1)

    max_qb = 24
    samples = 1
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    for n in range(1, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for i in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(r[0] for r in width_results) / samples
        fidelity_result = sum(r[1] for r in width_results) / samples
        print(n, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
