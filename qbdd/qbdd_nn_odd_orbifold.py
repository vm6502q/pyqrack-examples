# Demonstrates the use of "Quantum Binary Decision Diagram" (QBDD) and QBDD rounding parameter ("QBDDRP") with light-cone (nearest-neighbor)

import math
import random
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

    sim = QrackSimulator(width, isBinaryDecisionTree=True)

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            sim.u(
                i,
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
            )

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

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(sim, b1, b2)

    fidelity_est = sim.get_unitary_fidelity()

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start, fidelity_est


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 qbdd_nn_odd_orbifold.py [width] [depth]")

    width = int(sys.argv[1])

    depth = int(sys.argv[2])

    # Run the benchmarks
    time_result, fidelity_est = bench_qrack(width, depth)

    print(
        "Width="
        + str(width)
        + ", Depth="
        + str(depth)
        + ": "
        + str(time_result)
        + " seconds, "
        + str(fidelity_est)
        + " out of 1.0 worst-case first-principles fidelity estimate."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
