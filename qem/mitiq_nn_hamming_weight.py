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


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

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


def random_circuit(width, depth, lrc, lrr, sdrp):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    results = []

    qc = QuantumCircuit(width)
    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            qc.u(th, ph, lm, i)

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

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    sim = AceQasmSimulator(n_qubits=width, long_range_columns=lrc, long_range_rows=lrr, sdrp=sdrp)
    qc = transpile(qc, backend=sim, optimization_level=3)

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
        raise RuntimeError("Usage: python3 mitiq_nn_hamming_weight.py [width] [depth] [long_range_columns=4] [long_range_rows=4] [sdrp=0.1464466] [shots=1024]")

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
