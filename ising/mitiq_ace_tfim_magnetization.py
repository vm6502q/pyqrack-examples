# See "Error mitigation increases the effective quantum volume of quantum computers," https://arxiv.org/abs/2203.05489
#
# Mitiq is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import os
import random
import statistics
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.circuit.library import RZZGate, RXGate

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return row_len, col_len


def trotter_step(circ, qubits, lattice_shape, J, h, dt):
    n_rows, n_cols = lattice_shape

    # First half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

    # Layered RZZ interactions (simulate 2D nearest-neighbor coupling)
    def add_rzz_pairs(pairs):
        for q1, q2 in pairs:
            circ.append(RZZGate(2 * J * dt), [q1, q2])

    # Layer 1: horizontal pairs (even rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(0, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 2: horizontal pairs (odd rows)
    horiz_pairs = [
        (r * n_cols + c, r * n_cols + (c + 1) % n_cols)
        for r in range(n_rows)
        for c in range(1, n_cols, 2)
    ]
    add_rzz_pairs(horiz_pairs)

    # Layer 3: vertical pairs (even columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(1, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Layer 4: vertical pairs (odd columns)
    vert_pairs = [
        (r * n_cols + c, ((r + 1) % n_rows) * n_cols + c)
        for r in range(0, n_rows, 2)
        for c in range(n_cols)
    ]
    add_rzz_pairs(vert_pairs)

    # Second half of transverse field term
    for q in qubits:
        circ.rx(h * dt, q)

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


def execute(circ, long_range_columns, long_range_rows, depth, J, h, dt):
    n_qubits = circ.width()
    shots = 4096
    qubits = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)
    theta = math.pi / 18
    for q in range(circ.width()):
        qc.ry(theta, q)
    qc.compose(circ, qubits, inplace=True)

    experiment = QrackAceBackend(
        qc.width(),
        long_range_columns=long_range_columns,
        long_range_rows=long_range_rows,
    )
    # We've achieved the dream: load balancing between discrete and integrated accelerators!
    # for sim_id in range(2, len(experiment.sim), 3):
    #     experiment.sim[sim_id].set_device(0)

    experiment.run_qiskit_circuit(qc)

    t1 = 3.38
    t2 = 1.25
    t = depth * dt
    m = t / t1
    model = 1 - 1 / (1 + m)
    arg = -h / J
    d_magnetization = 0
    d_sqr_magnetization = 0
    if np.isclose(J, 0) or (arg >= 1024):
        d_magnetization = 0
        d_sqr_magnetization = 0
    elif np.isclose(h, 0) or (arg < -1024):
        d_magnetization = 1 if J < 0 else -1
        d_sqr_magnetization = 1
    else:
        p = 2**arg - math.tanh(J / abs(h)) * math.log(1 + t / t2) / math.log(2)
        factor = 2**p
        n = model / (n_qubits * 2)
        tot_n = 0
        for q in range(n_qubits + 1):
            n = n / factor
            if n == float("inf"):
                d_magnetization = 1
                d_sqr_magnetization = 1
                tot_n = 1
                break
            m = (n_qubits - (q << 1)) / n_qubits
            d_magnetization += n * m
            d_sqr_magnetization += n * m * m
            tot_n += n
        d_magnetization /= tot_n
        d_sqr_magnetization /= tot_n

    experiment_samples = experiment.measure_shots(qubits, shots)

    magnetization = 0
    sqr_magnetization = 0
    for sample in experiment_samples:
        m = 0
        for _ in range(n_qubits):
            m += -1 if (sample & 1) else 1
            sample >>= 1
        m /= n_qubits
        magnetization += m
        sqr_magnetization += m * m
    magnetization /= shots
    sqr_magnetization /= shots

    magnetization = model * d_magnetization + (1 - model) * magnetization
    sqr_magnetization = model * d_sqr_magnetization + (1 - model) * sqr_magnetization

    return logit(sqr_magnetization)


def main():
    if len(sys.argv) < 5:
        raise RuntimeError(
            "Usage: python3 mitiq_tfim_calibration.py [width] [depth] [long_range_columns] [long_range_rows]"
        )

    n_qubits = int(sys.argv[1])
    depth = int(sys.argv[2])
    long_range_columns = int(sys.argv[3])
    long_range_rows = int(sys.argv[4])

    n_rows, n_cols = factor_width(n_qubits)
    J, h, dt = -1.0, 2.0, 0.25

    circ = QuantumCircuit(n_qubits)
    for _ in range(depth):
        trotter_step(circ, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    noise_dummy = AceQasmSimulator(
        n_qubits=n_qubits,
        long_range_columns=long_range_columns,
        long_range_rows=long_range_rows,
    )
    circ = transpile(
        circ,
        optimization_level=3,
        backend=noise_dummy,
    )

    scale_count = depth + 1
    max_scale = 2
    factory = LinearFactory(
        scale_factors=[(1 + (max_scale - 1) * x / (scale_count - 1)) for x in range(0, scale_count)]
    )

    executor = lambda c: execute(c, long_range_columns, long_range_rows, depth, J, h, dt)

    sqr_magnetization = expit(
        zne.execute_with_zne(circ, executor, scale_noise=fold_global, factory=factory)
    )

    print({"width": n_qubits, "depth": depth, "square_magnetization": sqr_magnetization})

    return 0


if __name__ == "__main__":
    sys.exit(main())
