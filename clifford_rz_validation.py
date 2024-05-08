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
    single_bit_gates = { 0: (z_to_x, z_to_y), 1: (x_to_y, x_to_z), 2: (y_to_z, y_to_x) } 
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    
    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    row_len = math.ceil(math.sqrt(width))

    # Don't repeat bases:
    bases = [0] * width
    directions = [0] * width
    
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
            
        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(row_len):
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
                g(circ, b1, b2)

    return circ


def bench_qrack(n):
    # This is a demonstration of near-Clifford capabilities, with Clifford+RZ gate set.

    # Run a near-Clifford circuit
    qcircuit = QrackCircuit(False)
    random_circuit(n, 4, qcircuit)

    nc_sim = QrackSimulator(n, isStabilizerHybrid=True, isTensorNetwork=False, isSchmidtDecomposeMulti=False, isSchmidtDecompose=False, isOpenCL=False)
    sv_sim = QrackSimulator(n, isStabilizerHybrid=False, isTensorNetwork=False, isSchmidtDecomposeMulti=False, isSchmidtDecompose=False, isOpenCL=False)

    qcircuit.run(nc_sim)
    qcircuit.run(sv_sim)

    nc_sv = nc_sim.out_ket()
    sv_sv = sv_sim.out_ket()

    # fidelity = sim.get_unitary_fidelity()
    
    return np.abs(sum([np.conj(x) * y for x, y in zip(nc_sv, sv_sv)]))


def main():
    bench_qrack(1)

    max_qb = 10
    samples = 1
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    os.environ["QRACK_MAX_CPU_QB"]="-1"

    for n in range(1, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for i in range(samples):
            width_results.append(bench_qrack(n))

        inner_product_result = sum(r for r in width_results) / samples
        print(n, ": ", inner_product_result, " inner product")

    return 0


if __name__ == '__main__':
    sys.exit(main())
