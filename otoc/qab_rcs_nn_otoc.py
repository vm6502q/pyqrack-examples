# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from pyqrack import QrackAceBackend


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.acx(q1, q2)


def acy(sim, q1, q2):
    sim.acy(q1, q2)


def acz(sim, q1, q2):
    sim.acz(q1, q2)


def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def x(sim, q):
    sim.x(q)


def y(sim, q):
    sim.y(q)


def z(sim, q):
    sim.z(q)


def bench_qrack(width, depth, cycles, lrc, lrr):
    # This is a "nearest-neighbor" coupler random circuit.

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    rcs = []
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            rcs.append((u, i, th, ph, lm))

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

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width) or (b1 == b2):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = random.choice(two_bit_gates)
                rcs.append((g, b1, b2))

    ircs = []
    for tup in reversed(rcs):
        if tup[0] == u:
            ircs.append((u, tup[1], -tup[2], -tup[4], -tup[3]))
        else:
            ircs.append(tup)

    ops = ['I', 'X', 'Y', 'Z']
    pauli_strings = []

    otoc = []
    for cycle in range(cycles):
        otoc = otoc + rcs
        string = []
        for b in range(width):
            string.append(random.choice(ops))
        pauli_strings.append("".join(string))
        act_string(otoc, string)
        otoc = otoc + ircs

    experiment = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)
    for tup in otoc:
        tup[0](experiment, *tup[1:])

    shots = 1 << min(9, width + 2)
    experiment_probs = dict(Counter(experiment.measure_shots(all_bits, shots)))
    experiment_probs = { k: v / shots for k, v in experiment_probs.items() }

    return {
        "qubits": width,
        "depth": depth,
        "shots": shots,
        "pauli_strings": pauli_strings,
        "marginal_prob": experiment_probs
    }


def act_string(otoc, string):
    for i in range(len(string)):
        match string[i]:
            case 'X':
                otoc.append((x, i))
            case 'Y':
                otoc.append((y, i))
            case 'Z':
                otoc.append((z, i))
            case _:
                pass


def main():
    if len(sys.argv) < 4:
        raise RuntimeError(
            "Usage: python3 qab_rcs_nn_otoc.py [width] [depth] [cycles] [long_range_columns=4] [long_range_rows=4]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    cycles = int(sys.argv[3])
    lrc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    lrr = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    # Run the benchmarks
    print(bench_qrack(width, depth, cycles, lrc, lrr))

    return 0


if __name__ == "__main__":
    sys.exit(main())
