# See "Error mitigation increases the effective quantum volume of quantum computers," https://arxiv.org/abs/2203.05489
#
# Mitiq is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import random
import statistics
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackSimulator, Pauli

from qiskit import QuantumCircuit

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def code(width, radians):
    qc = QuantumCircuit(width)

    # State is |+>
    qc.ry(radians, 0)

    # Encode logical state across all qubits
    for i in range(1, width):
        qc.cx(0, i)

    return qc

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


def execute(circ, radians):
    shots = 1 << (circ.width() + 2)
    all_bits = list(range(circ.width()))

    experiment = QrackSimulator(circ.width())
    experiment.run_qiskit_circuit(circ)

    magnetization = 0
    for qubit in all_bits:
        experiment.r(Pauli.PauliY, -radians, 0)
        exp = 1 - 2 * experiment.prob(qubit)
        magnetization += exp
    magnetization /= circ.width()

    return logit((magnetization + 1) / 2)


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 fc.py [width]')

    width = int(sys.argv[1])

    for i in range(10):
        radians = i * (math.pi / 18)
        circ = code(width, radians)

        def executor(circ):
            return execute(circ, radians)

        scale_count = 9
        max_scale = 5
        factory = LinearFactory(scale_factors=[(1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)])

        expectation = 2 * expit(zne.execute_with_zne(circ, executor, scale_noise=fold_global, factory=factory)) - 1

        print({ 'width': width, 'angle' : radians, 'expectation': expectation })

    return 0


if __name__ == '__main__':
    sys.exit(main())
