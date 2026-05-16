# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

import numpy as np
from numba import njit, prange

from collections import Counter

from pyqrack import QrackSimulator


def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while ((width // row_len) * row_len) != width:
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def ct_pair_prob(sim, q1, q2, bound):
    if q1 < bound:
        p1 = sim[0].prob(q1)
    else:
        p1 = sim[1].prob(q1 - bound)
    if q2 < bound:
        p2 = sim[0].prob(q2)
    else:
        p2 = sim[1].prob(q2 - bound)

    if p1 < p2:
        return p2, q1

    return p1, q2


def cz_shadow(sim, q1, q2, bound, anti=False):
    if anti:
        if q1 < bound:
            sim[0].x(q1)
        else:
            sim[1].x(q1 - bound)
    prob_max, t = ct_pair_prob(sim, q1, q2, bound)
    if prob_max > 0.5:
        if t < bound:
            sim[0].z(t)
        else:
            sim[1].z(t - bound)
    if anti:
        if q1 < bound:
            sim[0].x(q1)
        else:
            sim[1].x(q1 - bound)


def cx_shadow(sim, c, t, bound, anti=False):
    if t < bound:
        sim[0].h(t)
        cz_shadow(sim, c, t, bound, anti)
        sim[0].h(t)
    else:
        sim[1].h(t - bound)
        cz_shadow(sim, c, t, bound, anti)
        sim[1].h(t - bound)


def cy_shadow(sim, c, t, bound, anti=False):
    if t < bound:
        sim[0].adjs(t)
        cx_shadow(sim, c, t, bound, anti)
        sim[0].s(t)
    else:
        sim[1].adjs(t - bound)
        cx_shadow(sim, c, t, bound, anti)
        sim[1].s(t - bound)


def swap_shadow(sim, q1, q2, bound):
    cx_shadow(sim, q1, q2, bound)
    cx_shadow(sim, q2, q1, bound)
    cx_shadow(sim, q1, q2, bound)


def cx(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cx_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcx([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].mcx([q1], q2)


def cy(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cy_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcy([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].mcy([q1], q2)


def cz(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcz([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].mcz([q1], q2)


def acx(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cx_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:
        sim[0].macx([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].macx([q1], q2)


def acy(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cy_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:
        sim[0].macy([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].macy([q1], q2)


def acz(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cz_shadow(sim, q1, q2, bound, True)
    elif q1 < bound:
        sim[0].macz([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].macz([q1], q2)


def swap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        swap_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].swap(q1, q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].swap(q1, q2)


def iswap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        swap_shadow(sim, q1, q2, bound)
        cz_shadow(sim, q1, q2, bound)
        if q1 < bound:
            sim[0].s(q1)
            sim[1].s(q2 - bound)
        else:
            sim[1].s(q1 - bound)
            sim[0].s(q2)
    elif q1 < bound:
        sim[0].iswap(q1, q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].iswap(q1, q2)


def iiswap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        if q1 < bound:
            sim[0].s(q1)
            sim[1].s(q2 - bound)
        else:
            sim[1].s(q1 - bound)
            sim[0].s(q2)
        cz_shadow(sim, q1, q2, bound)
        swap_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].adjiswap(q1, q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].adjiswap(q1, q2)


def pswap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cz_shadow(sim, q1, q2, bound)
        swap_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcz([q1], q2)
        sim[0].swap(q1, q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].mcz([q1], q2)
        sim[1].swap(q1, q2)


def mswap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        swap_shadow(sim, q1, q2, bound)
        cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].swap(q1, q2)
        sim[0].mcz([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].swap(q1, q2)
        sim[1].mcz([q1], q2)


def nswap(sim, q1, q2, bound):
    if ((q1 < bound) and (q2 >= bound)) or ((q2 < bound) and (q1 >= bound)):
        cz_shadow(sim, q1, q2, bound)
        swap_shadow(sim, q1, q2, bound)
        cz_shadow(sim, q1, q2, bound)
    elif q1 < bound:
        sim[0].mcz([q1], q2)
        sim[0].swap(q1, q2)
        sim[0].mcz([q1], q2)
    else:
        q1 -= bound
        q2 -= bound
        sim[1].mcz([q1], q2)
        sim[1].swap(q1, q2)
        sim[1].mcz([q1], q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    patch_bound = (width + 1) >> 1
    shots = 1 << min(width + 2, 20)
    ace_qb = (width + 1) >> 1
    control = QrackSimulator(width)
    control.set_ace_max_qb(ace_qb)
    experiment = [QrackSimulator((width + 1) >> 1), QrackSimulator(width >> 1)]

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            if i < patch_bound:
                experiment[0].u(i, th, ph, lm)
            else:
                experiment[1].u(i - patch_bound, th, ph, lm)
            control.u(i, th, ph, lm)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
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

                if (b1 == b2) or (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(experiment, b1, b2, patch_bound)

    experiment_probs = [experiment[0].out_probs(), experiment[1].out_probs()]
    control_counts = dict(Counter(control.measure_shots(all_bits, shots)))

    print(calc_stats(control_counts, experiment_probs, width, d + 1, shots, ace_qb, patch_bound))


# By (Anthropic) Claude
@njit(parallel=True, cache=True)
def _numer_sparse(keys, vals, exp_low, exp_high, bound_pow, bound, shots, u_u):
    acc = np.zeros(len(keys), dtype=np.float64)
    for k in prange(len(keys)):
        idx   = keys[k]
        ideal = exp_low[idx & bound_pow] * exp_high[idx >> bound]
        acc[k] = (ideal - u_u) * (vals[k] / shots)
    return acc.sum()


# Parallelized by (Anthropic) Claude
def calc_stats(control_counts, experiment_probs, n, depth, shots, ace_qb, bound):
    n_pow     = 1 << n
    u_u       = 1.0 / n_pow
    bound_pow = (1 << bound) - 1

    exp_low  = np.asarray(experiment_probs[0], dtype=np.float64)
    exp_high = np.asarray(experiment_probs[1], dtype=np.float64)

    # --- denom: sum_i (ideal_i - u_u)^2 ---
    # ideal_i = exp_low[i & bound_pow] * exp_high[i >> bound]
    # This factorises over the outer product of exp_low and exp_high,
    # so we never need to iterate over 2^n states.
    #
    # sum_i (p_l * p_h - u_u)^2
    #   = sum_l sum_h (p_l*p_h)^2  -  2*u_u * sum_l sum_h p_l*p_h  +  u_u^2 * n_pow
    #   = (sum_l p_l^2)(sum_h p_h^2)  -  2*u_u * 1  +  u_u^2 * n_pow
    #   (since sum_l p_l * count_l = 1 over the full marginals)
    sum_sq_low  = float(np.dot(exp_low,  exp_low))
    sum_sq_high = float(np.dot(exp_high, exp_high))
    denom = sum_sq_low * sum_sq_high - 2.0 * u_u + u_u * u_u * n_pow

    # --- numer: sum_i (ideal_i - u_u) * (count_i/shots - u_u) ---
    # The u_u cross-terms vanish:
    #   sum_i (ideal_i - u_u) * (-u_u) = -u_u * (sum ideal_i - n_pow*u_u) = 0
    #   sum_i (ideal_i - u_u) * (count_i/shots) — only non-zero where count_i > 0
    # So numer = sum_{i in counts} (ideal_i - u_u) * count_i/shots
    # which is a sum over only the observed bitstrings.

    keys = np.array(list(control_counts.keys()), dtype=np.int64)
    vals = np.array(list(control_counts.values()), dtype=np.float64)
    numer = _numer_sparse(keys, vals, exp_low, exp_high, bound_pow, bound, float(shots), u_u)

    xeb = numer / denom

    return {
        "qubits":       n,
        "ace_qb_limit": ace_qb,
        "depth":        depth,
        "xeb":          float(xeb),
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 rcs_nn_qrack_validation_elided.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == "__main__":
    sys.exit(main())
