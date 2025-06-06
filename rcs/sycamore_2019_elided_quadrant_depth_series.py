# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys
import time

from pyqrack import QrackSimulator


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


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


def swap_shadow(sim, q1, q2):
    cx_shadow(sim, q1, q2)
    cx_shadow(sim, q2, q1)
    cx_shadow(sim, q1, q2)


def sqrt_x(sim, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_MINUS_I_DIV_2 = 0.5 - 0.5j
    mtrx = [ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2]
    sim.mtrx(mtrx, q)


def sqrt_y(sim, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_PLUS_I_DIV_2_NEG = -0.5 - 0.5j
    mtrx = [ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2_NEG, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2]
    sim.mtrx(mtrx, q)


def sqrt_w(sim, q):
    diag = math.sqrt(0.5)
    m01 = -0.5 - 0.5j
    m10 = 0.5 - 0.5j
    mtrx = [diag, m01, m10, diag]
    sim.mtrx(mtrx, q)


def bench_qrack(width):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    full_sim = QrackSimulator(width)
    patch_sim = QrackSimulator(width)

    row_len, col_len = factor_width(width)
    row_bound = row_len >> 1
    col_bound = (col_len + 1) >> 1
    lcv_range = range(width)
    last_gates = []

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    one_bit_gates = [sqrt_x, sqrt_y, sqrt_w]

    for d in range(width):
        # Single-qubit gates
        if d == 0:
            for i in lcv_range:
                g = random.choice(one_bit_gates)
                g(full_sim, i)
                g(patch_sim, i)
                last_gates.append(g)
        else:
            # Don't repeat the same gate on the next layer.
            for i in lcv_range:
                temp_gates = one_bit_gates.copy()
                temp_gates.remove(last_gates[i])
                g = random.choice(one_bit_gates)
                g(full_sim, i)
                g(patch_sim, i)
                last_gates[i] = g

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

                # Bounded:
                if (
                    (temp_row < 0)
                    or (temp_col < 0)
                    or (temp_row >= row_len)
                    or (temp_col >= col_len)
                ):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                full_sim.fsim(-math.pi / 2, math.pi / 6, b1, b2)

                # Elide if across patches:
                if (
                    ((row < row_bound) and (temp_row >= row_bound))
                    or ((temp_row < row_bound) and (row >= row_bound))
                    or ((col < col_bound) and (temp_col >= col_bound))
                    or ((temp_col < col_bound) and (col >= col_bound))
                ):
                    # This is our version of ("semi-classical") gate "elision":

                    cState, t = ct_pair_prob(patch_sim, b1, b2)
                    if cState:
                        # FSim controlled phase
                        patch_sim.u(t, 0, 0, -math.pi / 6)

                    # SWAP(b1, b2)
                    swap_shadow(patch_sim, b1, b2)
                    # CZ(b1, b2)
                    cz_shadow(patch_sim, b1, b2)
                    # S(b1)
                    patch_sim.s(b1)
                    # S(b2)
                    patch_sim.s(b2)
                else:
                    patch_sim.fsim(-math.pi / 2, math.pi / 6, b1, b2)

        ideal_probs = full_sim.out_probs()
        patch_probs = patch_sim.out_probs()
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
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (patch - u_u)

        # QV / HOG
        if ideal > threshold:
            hog_prob += patch

    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "seconds": interval,
        "xeb": xeb,
        "hog_prob": hog_prob,
        "qv_pass": hog_prob >= 2 / 3,
        "eplg": (1 - (xeb ** (1 / depth))) if xeb < 1 else 0,
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 sycamore_2019_elided_quadrant_depth_series.py [width]"
        )

    width = int(sys.argv[1])

    # Run the benchmarks
    bench_qrack(width)

    return 0


if __name__ == "__main__":
    sys.exit(main())
