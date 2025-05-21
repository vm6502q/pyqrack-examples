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

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


def random_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    control = AerSimulator(method="statevector")
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                circ.h(i)
                circ.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)

    return circ


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


def execute(circ):
    shot_count = 1024

    all_bits = list(range(circ.width()))

    experiment = QrackSimulator(circ.width())
    experiment.run_qiskit_circuit(circ)
    shots = experiment.measure_shots(all_bits, shot_count)

    # We might be surprised if Haar-random magnetization is nontrivial.
    hamming_weight = 0
    for shot in shots:
        hamming_weight += hamming_distance(0, shot, circ.width())
    hamming_weight /= shot_count

    return logit(hamming_weight / circ.width())


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 fc.py [width] [depth]")

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    circ = random_circuit(width, depth)

    scale_count = 9
    max_scale = 5
    factory = LinearFactory(
        scale_factors=[
            (1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)
        ]
    )

    hamming_weight = (
        expit(
            zne.execute_with_zne(
                circ, execute, scale_noise=fold_global, factory=factory
            )
        )
        * width
    )

    print({"width": width, "depth": depth, "hamming_weight": hamming_weight})

    return 0


if __name__ == "__main__":
    sys.exit(main())
