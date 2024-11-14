# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import os
import random
import statistics
import sys
import time

import numpy as np

from pyqrack import QrackSimulator

from scipy.stats import binom


def ct_pair_prob(sim, q1, q2):
    p1 = sim.prob(q1)
    p2 = sim.prob(q2)
    p1Hi = p1 > p2
    pHi = p1 if p1Hi else p2
    pLo = p2 if p1Hi else p1
    cState = abs(pHi - 0.5) > abs(pLo - 0.5)
    t = q1 if p1Hi == cState else q2

    return cState, t


def cz_shadow(sim, q1, q2):
    cState, t = ct_pair_prob(sim, q1, q2)
    if cState:
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
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)

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
