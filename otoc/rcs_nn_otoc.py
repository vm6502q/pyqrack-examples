# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit


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
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def bench_qrack(width, depth, cycles):
    # This is a "nearest-neighbor" coupler random circuit.

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    rcs = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            rcs.u(th, ph, lm, i)

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

                if (b1 >= width) or (b2 >= width) or (b1 == b2):
                    continue

                if d & 1:
                    t = b1
                    b1 = b2
                    b2 = t

                g = random.choice(two_bit_gates)
                g(rcs, b1, b2)

    ops = ['I', 'X', 'Y', 'Z']
    pauli_strings = []

    otoc = QuantumCircuit(width)
    for cycle in range(cycles):
        otoc &= rcs
        string = []
        for b in range(width):
            string.append(random.choice(ops))
        pauli_strings.append("".join(string))
        act_string(otoc, string)
        otoc &= rcs.inverse()


    experiment = QrackSimulator(width, isTensorNetwork=False)
    experiment.run_qiskit_circuit(otoc)

    shots = 1 << (width + 2)
    experiment_probs = dict(Counter(experiment.measure_shots(all_bits, shots)))
    experiment_probs = { k: v / shots for k, v in experiment_probs.items() }

    return {
        "qubits": width,
        "depth": d + 1,
        "shots": shots,
        "pauli_strings": pauli_strings,
        "marginal_prob": experiment_probs
    }

    return calc_stats(control_probs, experiment_counts, d + 1, shots), pauli_strings


def act_string(otoc, string):
    for i in range(len(string)):
        match string[i]:
            case 'X':
                otoc.x(i)
            case 'Y':
                otoc.y(i)
            case 'Z':
                otoc.z(i)
            case _:
                pass


def main():
    if len(sys.argv) < 4:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [cycles]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    cycles = int(sys.argv[3])

    # Run the benchmarks
    print(bench_qrack(width, depth, cycles))

    return 0


if __name__ == "__main__":
    sys.exit(main())
