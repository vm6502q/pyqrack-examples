# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import multiprocessing
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(width_depth):
    width = width_depth[0]
    depth = width_depth[1]
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            sim.u(
                i,
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
            )

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            sim.h(t)
            sim.mcz([c], t)
            sim.h(t)

    fidelity_est = sim.get_unitary_fidelity()

    # Terminal measurement
    sim.m_all()

    time_result = time.perf_counter() - start

    print(
        "Width="
        + str(width)
        + ", Depth="
        + str(depth)
        + ": "
        + str(time_result)
        + " seconds, "
        + str(fidelity_est)
        + " out of 1.0 worst-case first-principles fidelity estimate."
    )

    return time_result, fidelity_est


def main():
    if len(sys.argv) < 4:
        raise RuntimeError("Usage: python3 qbdd_fc_parallel.py [width] [depth] [shots]")

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    shots = int(sys.argv[3])

    # Run the benchmarks
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(bench_qrack, [(width, depth)] * shots)
    pool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
