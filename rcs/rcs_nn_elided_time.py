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

from pyqrack import QrackSimulator, Pauli


# QBDD 16-bit precision can reach 2 ** -20 per gate (empirically)
epsilon = 2 ** -20


def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while (((width // row_len) * row_len) != width):
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def ct_pair_prob(sim, q1, q2):
    r = [0] * 4

    r[0] = sim.prob(q1)
    r[1] = sim.prob(q2)
    r[2] = r[0]
    r[3] = q2
    if r[0] < r[1]:
        r[3] = q1
        r[2] = r[1]

    return r


def cz_shadow(sim, q1, q2, anti = False):
    prob1, prob2, prob_max, t = ct_pair_prob(sim, q1, q2)
    if ((not anti) and (prob_max > (0.5 + epsilon))) or (anti and (prob_max < (0.5 - epsilon))):
        sim.z(t)


def cx_shadow(sim, c, t, anti = False):
    sim.h(t)
    cz_shadow(sim, c, t, anti)
    sim.h(t)


def cy_shadow(sim, c, t, anti = False):
    sim.adjs(t)
    cx_shadow(sim, c, t, anti)
    sim.s(t)

def swap_shadow(sim, q1, q2):
    cx_shadow(sim, q1, q2)
    cx_shadow(sim, q2, q1)
    cx_shadow(sim, q1, q2)

def cx(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cx_shadow(sim, q1, q2)
    else:
        sim.mcx([q1], q2)


def cy(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cy_shadow(sim, q1, q2)
    else:
        sim.mcy([q1], q2)


def cz(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2)
    else:
        sim.mcz([q1], q2)


def acx(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cx_shadow(sim, q1, q2, True)
    else:
        sim.macx([q1], q2)


def acy(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cy_shadow(sim, q1, q2, True)
    else:
        sim.macy([q1], q2)


def acz(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2, True)
    else:
        sim.macz([q1], q2)


def swap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
    else:
        sim.swap(q1, q2)


def iswap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
        cz_shadow(sim, q1, q2)
        sim.s(q1)
        sim.s(q2)
    else:
        sim.iswap(q1, q2)


def iiswap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        sim.adjs(q1)
        sim.adjs(q2)
        cz_shadow(sim, q1, q2)
        swap_shadow(sim, q1, q2)
    else:
        sim.adjiswap(q1, q2)


def pswap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2)
        swap_shadow(sim, q1, q2)
    else:
        sim.mcz([q1], q2)
        sim.swap(q1, q2)


def mswap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
        cz_shadow(sim, q1, q2)
    else:
        sim.swap(q1, q2)
        sim.mcz([q1], q2)


def nswap(sim, q1, q2, bound):
    if (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2)
        swap_shadow(sim, q1, q2)
        cz_shadow(sim, q1, q2)
    else:
        sim.mcz([q1], q2)
        sim.swap(q1, q2)
        sim.mcz([q1], q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

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
            for _ in range(3):
                # x-z-x Euler axes
                th = random.uniform(0, 2 * math.pi)
                experiment.h(i)
                experiment.r(Pauli.PauliZ, th, i)

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

                g = random.choice(two_bit_gates)
                g(experiment, b1, b2, patch_bound)

    # Terminal measurement
    experiment.m_all()

    return time.perf_counter() - start


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
