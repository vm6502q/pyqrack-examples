# Demonstrates the use of "Quantum Binary Decision Diagram" (QBDD) and QBDD rounding parameter (QBDDRP) with near-Clifford (nearest-neighbor)

import math
import os
import random
import statistics
import sys
import time

from scipy.stats import binom

from pyqrack import QrackSimulator, Pauli


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def sqrt_x(sim, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_MINUS_I_DIV_2 = 0.5 - 0.5j
    mtrx = [ ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2 ]
    sim.mtrx(mtrx, q);


def sqrt_y(sim, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_PLUS_I_DIV_2_NEG = -0.5 - 0.5j
    mtrx = [ ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2_NEG, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2 ]
    sim.mtrx(mtrx, q);

def sqrt_w(sim, q):
    diag = math.sqrt(0.5);
    m01 = -0.5 - 0.5j
    m10 = 0.5 - 0.5j
    mtrx = [ diag, m01, m10, diag ]
    sim.mtrx(mtrx, q);


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    
    dead_qubit = 3 if width == 54 else width

    full_sim = QrackSimulator(width)
    patch_sim = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)
    last_gates = []

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    one_bit_gates = [ sqrt_x, sqrt_y, sqrt_w ]

    row_len, col_len = factor_width(width)
    patch_bound = row_len >> 1

    for d in range(depth):
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
                temp_row = temp_row + (1 if (gate & 2) else -1);
                temp_col = temp_col + (1 if (gate & 1) else 0)

                # Orbifolded:
                # if temp_row < 0:
                #     temp_row = temp_row + row_len
                # if temp_col < 0:
                #     temp_col = temp_col + col_len
                # if temp_row >= row_len:
                #     temp_row = temp_row - row_len
                # if temp_col >= col_len:
                #     temp_col = temp_col - col_len

                # Bounded:
                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= col_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width) or (b1 == dead_qubit) or (b2 == dead_qubit):
                    continue

                if d == (depth - 1):
                    # For the last layer of couplers, the immediately next operation is measurement, and the phase
                    # effects make no observable difference.
                    full_sim.swap(b1, b2)

                    continue

                full_sim.fsim((3 * math.pi) / 2, math.pi / 6, b1, b2)

                # Duplicate if in patches:
                if ((row < patch_bound) and (temp_row >= patch_bound)) or ((temp_row < patch_bound) and (row >= patch_bound)):
                    continue

                if d == (depth - 1):
                    # For the last layer of couplers, the immediately next operation is measurement, and the phase
                    # effects make no observable difference.
                    patch_sim.swap(b1, b2)

                    continue

                patch_sim.fsim((3 * math.pi) / 2, math.pi / 6, b1, b2)

    ideal_probs = full_sim.out_probs()
    counts = patch_sim.measure_shots(all_bits, 1000000)

    return (ideal_probs, counts, time.perf_counter() - start)


def calc_stats(ideal_probs, counts, interval):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    shots = 1000000
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    e_u = 0
    m_u = 0
    sum_hog_counts = 0
    for b in range(n_pow):
        if not b in counts:
            continue

        # XEB / EPLG
        count = counts[b]
        ideal = ideal_probs[b]
        e_u = e_u + ideal ** 2
        m_u = m_u + ideal * (count / shots)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts = sum_hog_counts + count

    hog_prob = sum_hog_counts / shots
    xeb = (m_u - u_u) * (e_u - u_u) / ((e_u - u_u) ** 2)
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1

    return {
        'qubits': n,
        'seconds': interval,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'pass': hog_prob >= 2 / 3,
        'p-value': p_val,
        'eplg': (1 - xeb) ** (1 / n) if xeb < 1 else 0
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 sycamore_2019_patch.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    calc_stats(result[0], result[1], result[2])

    return 0


if __name__ == '__main__':
    sys.exit(main())
