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

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.qrack import AceQasmSimulator

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def random_circuit(width, depth):
    # This is a "fully-connected" coupler random circuit.
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(2):
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


def execute(circ, shot_count, lrc, lrr, sdrp):
    sim = AceQasmSimulator(n_qubits=circ.width(), long_range_columns=lrc, long_range_rows=lrr, sdrp=sdrp)
    circ_m = circ.copy()
    circ_m.measure_all()
    shots = dict(sim.run(circ_m, shots=shot_count).result().get_counts())

    hamming_weight = 0
    for k, v in shots.items():
        hamming_weight += k.count("1") * v
    hamming_weight /= shot_count

    return logit(hamming_weight / circ.width())


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 mitiq_qv_hamming_weight.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [sdrp=0.1464466] [shots=1024]")

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sdrp  = float(sys.argv[5]) if len(sys.argv) > 5 else ((1 - 1 / math.sqrt(2)) / 2)
    shots = int(sys.argv[6]) if len(sys.argv) > 6 else 1024

    circ = random_circuit(width, depth, lrc, lrr, sdrp)

    scale_count = 5
    max_scale = 2
    factory = LinearFactory(
        scale_factors=[
            (1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)
        ]
    )

    ex = lambda circ: execute(circ, shots, lrc, lrr, sdrp)

    hamming_weight = width * expit(
        zne.execute_with_zne(circ, ex, scale_noise=fold_global, factory=factory)
    )

    print({"width": width, "depth": depth, "hamming_weight": float(hamming_weight)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
