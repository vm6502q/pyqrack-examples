# Demonstrates the use of "Quantum Binary Decision Diagram" (QBDD) and QBDD rounding parameter (QBDDRP) with near-Clifford (nearest-neighbor)

import math
import multiprocessing
import random
import sys
import time

from pyqrack import QrackSimulator


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
    sim.mtrx(mtrx, q)


def sqrt_y(sim, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_PLUS_I_DIV_2_NEG = -0.5 - 0.5j
    mtrx = [ ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2_NEG, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2 ]
    sim.mtrx(mtrx, q)

def sqrt_w(sim, q):
    diag = math.sqrt(0.5)
    m01 = -0.5 - 0.5j
    m10 = 0.5 - 0.5j
    mtrx = [ diag, m01, m10, diag ]
    sim.mtrx(mtrx, q)


def bench_qrack(depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    
    width = 54
    dead_qubit = 3

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

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
                g(sim, i)
                last_gates.append(g)
        else:
            # Don't repeat the same gate on the next layer.
            for i in lcv_range:
                temp_gates = one_bit_gates.copy()
                temp_gates.remove(last_gates[i])
                g = random.choice(one_bit_gates)
                g(sim, i)
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

    fidelity_est = sim.get_unitary_fidelity()

    # Terminal measurement
    sim.m_all()
    
    time_result = time.perf_counter() - start

    print("Width=" + str(width) + ", Depth=" + str(depth) + ": " + str(time_result) + " seconds, " + str(fidelity_est) + " out of 1.0 worst-case first-principles fidelity estimate.")

    return time_result, fidelity_est


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 qbdd_sycamore_2019_parallel.py [depth] [shots]')

    depth = int(sys.argv[1])

    shots = int(sys.argv[2])

    # Run the benchmarks
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    result = pool.map(bench_qrack, [depth] * shots)
    pool.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
