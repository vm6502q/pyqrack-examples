# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import statistics
import sys
import time

from pyqrack import QrackSimulator


def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while (((width // row_len) * row_len) != width):
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def ct_pair_prob(sim, q1, q2):
    p1 = sim.prob(q1)
    p2 = sim.prob(q2)

    if p1 < p2:
        return p2, q1

    return p1, q2


def cz_shadow(sim, q1, q2, anti = False):
    if anti:
        sim.x(q1)
    prob_max, t = ct_pair_prob(sim, q1, q2)
    if prob_max > 0.5:
        sim.z(t)
    if anti:
        sim.x(q1)


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


def cx(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cx_shadow(sim, q1, q2)
    else:
        sim.mcx([q1], q2)


def cy(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cy_shadow(sim, q1, q2)
    else:
        sim.mcy([q1], q2)


def cz(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2)
    else:
        sim.mcz([q1], q2)


def acx(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cx_shadow(sim, q1, q2, True)
    else:
        sim.macx([q1], q2)


def acy(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cy_shadow(sim, q1, q2, True)
    else:
        sim.macy([q1], q2)


def acz(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2, True)
    else:
        sim.macz([q1], q2)


def swap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
    else:
        sim.swap(q1, q2)


def iswap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
        cz_shadow(sim, q1, q2)
        sim.s(q1)
        sim.s(q2)
    else:
        sim.iswap(q1, q2)


def iiswap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        sim.adjs(q1)
        sim.adjs(q2)
        cz_shadow(sim, q1, q2)
        swap_shadow(sim, q1, q2)
    else:
        sim.adjiswap(q1, q2)


def pswap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        cz_shadow(sim, q1, q2)
        swap_shadow(sim, q1, q2)
    else:
        sim.mcz([q1], q2)
        sim.swap(q1, q2)


def mswap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
        swap_shadow(sim, q1, q2)
        cz_shadow(sim, q1, q2)
    else:
        sim.swap(q1, q2)
        sim.mcz([q1], q2)


def nswap(sim, q1, q2, patch, bound):
    if patch and (((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound))):
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

    control = QrackSimulator(width)
    experiment = QrackSimulator(width)

    patch_bound = (width + 1) >> 1
    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(th, ph, lm, i)
            control.u(th, ph, lm, i)

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
                g(control, b1, b2, False, patch_bound)
                g(experiment, b1, b2, True, patch_bound)

        ideal_probs = control.out_probs()
        patch_probs = experiment.out_probs()
        # Calc. and print the results
        print(calc_stats(ideal_probs, patch_probs, time.perf_counter() - start, d + 1))


def calc_stats(ideal_probs, patch_probs, interval, depth):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for b in range(n_pow):
        ideal = ideal_probs[b]
        patch = patch_probs[b]

        # XEB / EPLG
        ideal_centered = (ideal - u_u)
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        # QV / HOG
        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom

    return {
        'qubits': n,
        'depth': depth,
        'seconds': interval,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'qv_pass': hog_prob >= 2 / 3,
        'eplg':  (1 - (xeb ** (1 / depth))) if xeb < 1 else 0
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 rcs_nn_elided_depth_series.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == '__main__':
    sys.exit(main())
