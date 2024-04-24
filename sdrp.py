import math
import os
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n, sdrp):
    # This is basically a "quantum volume" (random) circuit.
    start = time.perf_counter()

    sim = QrackSimulator(n, isTensorNetwork=False)
    sim.set_sdrp(n)

    lcv_range = range(n)
    all_bits = list(lcv_range)

    single_count = 0
    double_count = 0
    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            sim.mcx([unused_bits.pop()], unused_bits.pop())

    fidelity = sim.get_unitary_fidelity()
    # Terminal measurement
    sim.m_all()

    return (time.perf_counter() - start, fidelity)


def main():
    bench_qrack(1, 0.5)

    sdrp = 0.5
    max_qb = 24
    samples = 1
    if len(sys.argv) < 2:
        raise RuntimeError('Must specify (at least) SDRP floating-point [0.0, 1.0] argument on command line.')

    sdrp = float(sys.argv[1])

    if len(sys.argv) > 2:
        max_qb = int(sys.argv[1])

    if len(sys.argv) > 3:
        samples = int(sys.argv[2])

    for n in range(1, max_qb + 1):
        width_results = []
        
        # Run the benchmarks
        for i in range(samples):
            width_results.append(bench_qrack(n, sdrp))

        time_result = sum(r[0] for r in width_results) / samples
        fidelity_result = sum(r[1] for r in width_results) / samples
        print(n, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
