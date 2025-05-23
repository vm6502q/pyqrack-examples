import math
import os
import random
import sys
import time

import numpy as np

from pyqrack import QrackSimulator, QrackCircuit


sqrt1_2 = 1 / math.sqrt(2)


def x_to_y(circ, q):
    circ.mtrx([1, 0, 0, 1j], q)


def x_to_z(circ, q):
    circ.mtrx([sqrt1_2, sqrt1_2, sqrt1_2, -sqrt1_2], q)


def y_to_z(circ, q):
    circ.mtrx([1, 0, 0, -1j], q)
    circ.mtrx([sqrt1_2, sqrt1_2, sqrt1_2, -sqrt1_2], q)


def y_to_x(circ, q):
    circ.mtrx([1, 0, 0, -1j], q)


def z_to_x(circ, q):
    circ.mtrx([sqrt1_2, sqrt1_2, sqrt1_2, -sqrt1_2], q)


def z_to_y(circ, q):
    circ.mtrx([sqrt1_2, sqrt1_2, sqrt1_2, -sqrt1_2], q)
    circ.mtrx([1, 0, 0, 1j], q)


def cx(circ, q1, q2):
    circ.ucmtrx([q1], [0, 1, 1, 0], q2, 1)


def cy(circ, q1, q2):
    circ.ucmtrx([q1], [0, -1j, 1j, 0], q2, 1)


def cz(circ, q1, q2):
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 1)


def acx(circ, q1, q2):
    circ.ucmtrx([q1], [0, 1, 1, 0], q2, 0)


def acy(circ, q1, q2):
    circ.ucmtrx([q1], [0, -1j, 1j, 0], q2, 0)


def acz(circ, q1, q2):
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 0)


def swap(circ, q1, q2):
    circ.swap(q1, q2)


def nswap(circ, q1, q2):
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 0)
    circ.swap(q1, q2)
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 0)


def pswap(circ, q1, q2):
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 0)
    circ.swap(q1, q2)


def mswap(circ, q1, q2):
    circ.swap(q1, q2)
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 0)


def iswap(circ, q1, q2):
    circ.swap(q1, q2)
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 1)
    circ.mtrx([1, 0, 0, 1j], q1)
    circ.mtrx([1, 0, 0, 1j], q2)


def iiswap(circ, q1, q2):
    circ.mtrx([1, 0, 0, -1j], q2)
    circ.mtrx([1, 0, 0, -1j], q1)
    circ.ucmtrx([q1], [1, 0, 0, -1], q2, 1)
    circ.swap(q1, q2)


def random_circuit(width, max_magic, circ):
    single_bit_gates = {0: (z_to_x, z_to_y), 1: (x_to_y, x_to_z), 2: (y_to_z, y_to_x)}
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len = math.ceil(math.sqrt(width))

    # Don't repeat bases:
    bases = [0] * width
    directions = [0] * width

    magic_count = 0

    for i in range(3 * width):
        # Single bit gates
        for j in range(width):
            # Reset basis, every third layer
            if i % 3 == 0:
                bases[j] = random.randint(0, 2)
                directions[j] = random.randint(0, 1)

            # Sequential basis switch
            gate = single_bit_gates[bases[j]][directions[j]]
            gate(circ, j)

            # Cycle through all 3 Pauli axes, every 3 layers
            if directions[j]:
                bases[j] -= 1
                if bases[j] < 0:
                    bases[j] += 3
            else:
                bases[j] += 1
                if bases[j] > 2:
                    bases[j] -= 3

            # Rotate around local Z axis
            if (3 * width * width * random.random()) < max_magic:
                # T gate:
                # rnd = math.pi / 4

                # General RZ gate:
                rnd = random.uniform(0, 2 * math.pi)
                circ.mtrx([1, 0, 0, math.cos(rnd) + math.sin(rnd) * 1j], j)
                magic_count = magic_count + 1

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(row_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if (
                    (temp_row < 0)
                    or (temp_col < 0)
                    or (temp_row >= row_len)
                    or (temp_col >= row_len)
                ):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(circ, b1, b2)

    return magic_count


def bench_qrack(n, magic):
    # This is a demonstration of near-Clifford capabilities, with Clifford+RZ gate set.

    # Run a near-Clifford circuit
    start = time.perf_counter()
    qcircuit = QrackCircuit(is_collapse=False)
    magic = random_circuit(n, magic, qcircuit)
    qsim = QrackSimulator(n)
    qcircuit.run(qsim)
    qcircuit.inverse().run(qsim)

    result = qsim.m_all()
    if result != 0:
        raise Exception("Mirror circuit failed!")

    return (magic, time.perf_counter() - start)


def main():
    bench_qrack(1, 0)

    qubits = 16
    magic = 1
    samples = 1
    if len(sys.argv) > 1:
        qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        magic = int(sys.argv[2])
    if len(sys.argv) > 3:
        samples = int(sys.argv[3])

    os.environ["QRACK_MAX_CPU_QB"] = "-1"

    # Run the benchmarks
    results = []
    for _ in range(samples):
        results.append(bench_qrack(qubits, magic))

    r_magic = sum(r[0] for r in results) / samples
    time = sum(r[1] for r in results) / samples
    print(
        f"{qubits} qubits, {r_magic} magic, ({magic} average magic): {time} seconds to mirror circuit sample"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
