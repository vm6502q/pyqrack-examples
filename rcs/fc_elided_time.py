# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def ct_pair_prob(sim, q1, q2):
    p1 = sim.prob(q1)
    p2 = sim.prob(q2)

    if p1 < p2:
        return p2, q1

    return p1, q2


def cz_shadow(sim, q1, q2):
    prob_max, t = ct_pair_prob(sim, q1, q2)
    if prob_max > 0.5:
        sim.z(t)


def cx_shadow(sim, c, t):
    sim.h(t)
    cz_shadow(sim, c, t)
    sim.h(t)


def bench_qrack(width, depth):
    start = time.perf_counter()
    patch_size = (width + 1) >> 1
    # This is a fully-connected random circuit.
    experiment = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                experiment.h(i)
                experiment.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            if (c < patch_size and t >= patch_size) or (t < patch_size and c >= patch_size):
                # This is our version of ("semi-classical") gate "elision":
                cx_shadow(experiment, c, t)
            else:
                experiment.mcx([c], t)

    experiment.m_all()
    interval = time.perf_counter() - start

    return interval


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 rcs_nn_elided_time.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print("Width=" + str(width) + ", Depth=" + str(depth) + ", Seconds=" + str(result))

    return 0


if __name__ == '__main__':
    sys.exit(main())
