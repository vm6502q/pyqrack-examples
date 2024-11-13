# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

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

    lcv_range = range(width)
    all_bits = list(lcv_range)
    last_gates = []

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    one_bit_gates = [ sqrt_x, sqrt_y, sqrt_w ]

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        if d == 0:
            for i in lcv_range:
                g = random.choice(one_bit_gates)
                g(full_sim, i)
                last_gates.append(g)
        else:
            # Don't repeat the same gate on the next layer.
            for i in lcv_range:
                temp_gates = one_bit_gates.copy()
                temp_gates.remove(last_gates[i])
                g = random.choice(one_bit_gates)
                g(full_sim, i)
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

                # Bounded:
                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= col_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width) or (b1 == dead_qubit) or (b2 == dead_qubit):
                    continue

                full_sim.fsim(-math.pi / 2, math.pi / 6, b1, b2)

    full_sim.m_all()

    return (full_sim.get_unitary_fidelity(), time.perf_counter() - start)


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 sycamore_2019_patch.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    result = bench_qrack(width, depth)
    # Calc. and print the results
    print("width: " + str(width) + ", depth: " + str(depth) + ", fidelity: " + str(result[0]) + ", seconds: " + str(result[1]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
