# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import os
import random
import statistics
import sys
import time

from scipy.stats import binom

from pyqrack import QrackSimulator


def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while (((width // row_len) * row_len) != width):
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)

def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    bound = (width + 1) >> 1
    experiment = QrackSimulator(width)

    patch_bound = (width + 1) >> 1
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1);
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                if ((b1 < bound) and (b2 >= bound)) or ((b2 < bound) and (b1 >= bound)):
                    continue

                g = random.choice(two_bit_gates)
                g(experiment, b1, b2)

    # Terminal measurement
    experiment.m_all()

    return time.perf_counter() - start


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 rcs_nn_patch_time.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print("Width=" + str(width) + ", Depth=" + str(depth) + ", Seconds=" + str(result))

    return 0


if __name__ == '__main__':
    sys.exit(main())
