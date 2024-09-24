# Demonstrates the use of "Quantum Binary Decision Diagram" (QBDD) and QBDD rounding parameter (QBDDRP) with near-Clifford (nearest-neighbor)

import math
import os
import random
import sys
import time

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


def bench_qrack(depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    
    width = 54
    dead_qubit = 3

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    one_bit_gates = sqrt_x, sqrt_y, sqrt_w

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            if i == dead_qubit:
                continue
            g = random.choice(one_bit_gates)
            g(sim, i)

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
                    sim.swap(b1, b2);

                    continue;

                sim.fsim((3 * math.pi) / 2, math.pi / 6, b1, b2);

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 sdrp.py [qbddrp] [depth]')

    os.environ['QRACK_QBDT_HYBRID_THRESHOLD'] = '2'

    qbddrp = float(sys.argv[1])
    if (qbddrp > 0):
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = sys.argv[1]

    depth = int(sys.argv[2])

    # Run the benchmarks
    time_result = bench_qrack(depth)

    print("Width=(54 - 1), Depth=" + str(depth) + ": " + str(time_result) + " seconds. (Fidelity is unknown.)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
