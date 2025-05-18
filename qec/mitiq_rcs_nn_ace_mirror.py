# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

import numpy as np

from pyqrack import QrackAceBackend

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def factor_width(width, reverse=False):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")
    row_len = width // col_len

    return (col_len, row_len) if reverse else (row_len, col_len)


def logit(x):
    # Theoretically, these limit points are "infinite,"
    # but precision caps out between 36 and 37:
    if 5e-17 > (1 - x):
        return 37
    # For the negative limit, the precision caps out
    # between -37 and -38
    elif x < 1e-17:
        return -38
    return max(-38, min(37, np.log(x / (1 - x))))


def expit(x):
    # Theoretically, these limit points are "infinite,"
    # but precision caps out between 36 and 37:
    if x >= 37:
        return 1.0
    # For the negative limit, the precision caps out
    # between -37 and -38
    elif x <= -38:
        return 0.0
    return 1 / (1 + np.exp(-x))


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


def random_circuit(width, depth, reverse):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)
    
    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = cx, cy, cz, acx, acy, acz
    
    row_len, col_len = factor_width(width, reverse)
    
    results = []

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

    return circ


def execute(circ):
    all_bits = list(range(circ.width()))

    experiment = QrackAceBackend(circ.width())
    experiment.run_qiskit_circuit(circ)
    experiment.run_qiskit_circuit(circ.inverse())

    # Terminal measurement
    shots = 1000
    experiment_samples = experiment.measure_shots(all_bits, shots)
    
    mirror_fidelity = 0
    hamming_weight = 0
    for sample in experiment_samples:
        success = True
        for _ in range(circ.width()):
            if sample & 1:
                success = False
                hamming_weight += 1
            sample >>= 1
        if success:
            mirror_fidelity += 1
    mirror_fidelity /= shots
    hamming_weight /= shots

    return logit(hamming_weight / circ.width())


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 mitiq_rcs_nn_ace_mirror.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    reverse = False
    if len(sys.argv) > 3:
        reverse = sys.argv[3] not in ['0', 'False']

    circ = random_circuit(width, depth, reverse)

    scale_count = 9
    max_scale = 5
    factory = LinearFactory(scale_factors=[(1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)])

    hamming_weight = expit(zne.execute_with_zne(circ, execute, scale_noise=fold_global, factory=factory)) * width

    print({ 'width': width, 'depth': depth, 'hamming_weight': hamming_weight })

    return 0


if __name__ == '__main__':
    sys.exit(main())
