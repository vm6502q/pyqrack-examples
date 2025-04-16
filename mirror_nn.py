# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import sys
import time

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def count_set_bits(n):
        return bin(n).count('1')


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


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def bench_qrack(width, depth, trials):
    # This is a "nearest-neighbor" coupler random circuit.
    shots = 100
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    results = []

    for trial in range(trials):
        circ = QuantumCircuit(width)
        for d in range(depth):
            # Single-qubit gates
            for i in lcv_range:
                circ.h(i)
                circ.rz(random.uniform(0, 2 * math.pi), i)

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
                    g(circ, b1, b2)

        start = time.perf_counter()
        experiment = QrackSimulator(width)
        experiment.run_qiskit_circuit(circ)
        midpoint = experiment.measure_shots(all_bits, shots)
        experiment.run_qiskit_circuit(circ.inverse())
        terminal = experiment.measure_shots(all_bits, shots)
        seconds = time.perf_counter() - start

        hamming_weight = sum(count_set_bits(r) for r in midpoint) / shots
        hamming_distance = sum(count_set_bits(r) for r in terminal) / shots

        return seconds, hamming_weight, hamming_distance

def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 mirror_nn_depth_series.py [width] [depth] [trials]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = 1
    if len(sys.argv) > 3:
        trials = int(sys.argv[3])

    # Run the benchmarks
    results = bench_qrack(width, depth, trials)

    print(width, "qubits,",
        depth, "depth,"
        results[0], "seconds,",
        results[1], "average mirror mid-point Hamming weight,",
        results[2], "average mirror terminal Hamming distance"
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
