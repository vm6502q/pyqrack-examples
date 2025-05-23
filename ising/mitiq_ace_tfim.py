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

from pyqrack import QrackAceBackend

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return row_len, col_len


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols - 1, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows - 1, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt / 2, q)

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
    shots = 1024
    all_bits = list(range(circ.width()))

    qc = QuantumCircuit(circ.width())
    theta = -math.pi / 6
    for q in range(circ.width()):
        qc.ry(theta, q)
    qc.compose(circ, all_bits, inplace=True)

    experiment = QrackAceBackend(qc.width())
    experiment.run_qiskit_circuit(qc)
    experiment_samples = experiment.measure_shots(all_bits, shots)

    magnetization = 0
    for sample in experiment_samples:
        for _ in range(circ.width()):
            magnetization += -1 if (sample & 1) else 1
            sample >>= 1
    magnetization /= shots * circ.width()

    return logit((magnetization + 1) / 2)


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python3 mitiq_tfim.py [width] [depth]")

    n_qubits = int(sys.argv[1])
    depth = int(sys.argv[2])

    n_rows, n_cols = factor_width(n_qubits)
    J, h, dt = -1.0, 2.0, 0.25

    circ = QuantumCircuit(n_qubits)
    for _ in range(depth):
        trotter_step(circ, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    scale_count = 4
    max_scale = 4
    factory = LinearFactory(
        scale_factors=[
            (1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)
        ]
    )

    magnetization = (
        2
        * expit(
            zne.execute_with_zne(
                circ, execute, scale_noise=fold_global, factory=factory
            )
        )
        - 1
    )

    print({"width": n_qubits, "depth": depth, "magnetization": magnetization})

    return 0


if __name__ == "__main__":
    sys.exit(main())
