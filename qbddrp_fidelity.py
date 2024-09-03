# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import os
import random
import sys
import time

import numpy as np

from pyqrack import QrackSimulator


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
    experiment = QrackSimulator(width, isBinaryDecisionTree=True)
    control = QrackSimulator(width)
    control.set_sdrp(5.9604645e-8)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len

    for d in range(depth):
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)
            control.u(i, th, ph, lm)

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

                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= row_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(experiment, b1, b2)
                g(control, b1, b2)

        experiment_sv = experiment.out_ket()
        control_sv = control.out_ket()
    
        print("Depth=" + str(d + 1) + ", fidelity=" + str(np.abs(sum([np.conj(x) * y for x, y in zip(experiment_sv, control_sv)]))))


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 sdrp.py [qbddrp] [samples]')

    qbddrp = float(sys.argv[1])
    if (qbddrp > 0):
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = sys.argv[1]

    samples = int(sys.argv[2])

    # Run the benchmarks
    # width_results = []
    for i in [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]:
        print("Width=" + str(i) + ":")
        for j in range(samples):
            bench_qrack(i, i)

    return 0


if __name__ == '__main__':
    sys.exit(main())
