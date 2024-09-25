# Validates the use of "Quantum Binary Decision Diagram" (QBDD) and QBDD rounding parameter ("QBDDRP") with light-cone (nearest-neighbor)

import math
import os
import random
import sys
import time

import numpy as np

from pyqrack import QrackSimulator



def cx(sim, q1, q2):
    sim.mcx([q1], q2)
    return 1


def cy(sim, q1, q2):
    sim.mcy([q1], q2)
    return 1


def cz(sim, q1, q2):
    sim.mcz([q1], q2)
    return 1


def acx(sim, q1, q2):
    sim.macx([q1], q2)
    return 1


def acy(sim, q1, q2):
    sim.macy([q1], q2)
    return 1


def acz(sim, q1, q2):
    sim.macz([q1], q2)
    return 1


def swap(sim, q1, q2):
    sim.swap(q1, q2)
    return 1


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)
    return 1


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)
    return 1


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    return 2


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)
    return 2


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)
    return 3


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    experiment = QrackSimulator(width, isBinaryDecisionTree=True)
    control = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        print("(Prime - skipped)")
        return

    gate_count = 0

    for d in range(depth):
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)
            control.u(i, th, ph, lm)
            gate_count = gate_count + 1

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

                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= col_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(experiment, b1, b2)
                gate_count = gate_count + g(control, b1, b2)

        experiment_sv = experiment.out_ket()
        control_sv = control.out_ket()

        overall_fidelity = np.abs(sum([np.conj(x) * y for x, y in zip(experiment_sv, control_sv)]))
        per_gate_fidelity = overall_fidelity ** (1 / gate_count)

        print("Depth=" + str(d + 1) + ", overall fidelity=" + str(overall_fidelity) + ", per-gate fidelity avg.=" + str(per_gate_fidelity))


def main():
    # Run the benchmarks
    for i in range(1, 21):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == '__main__':
    sys.exit(main())
