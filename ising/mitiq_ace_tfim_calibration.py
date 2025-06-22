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
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.transpiler import CouplingMap

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def calc_stats(ideal_probs, counts, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    diff_sqr = 0
    numer = 0
    denom = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - (count / shots)) ** 2

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_similarity = 1 - diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    return {
        "l2_similarity": l2_similarity,
        "xeb": xeb,
        "hog_prob": hog_prob,
    }


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


def execute(circ, long_range_columns, long_range_rows):
    shots = min(1024, 1 << (circ.width() + 2))
    all_bits = list(range(circ.width()))

    qc = QuantumCircuit(circ.width())
    theta = 2 * math.pi / 9
    for q in range(circ.width()):
        qc.ry(theta / 2, q)
    qc.compose(circ, all_bits, inplace=True)

    experiment = QrackAceBackend(qc.width(), long_range_columns=long_range_columns, long_range_rows=long_range_rows)
    # We've achieved the dream: load balancing between discrete and integrated accelerators!
    # for sim_id in range(2, len(experiment.sim), 3):
    #     experiment.sim[sim_id].set_device(0)

    experiment.run_qiskit_circuit(qc)

    control = AerSimulator(method="statevector")
    qc.save_statevector()
    job = control.run(qc)

    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    stats = calc_stats(control_probs, experiment_counts, shots)

    # So as not to exceed floor at 0.0 and ceiling at 1.0, (assuming 0 < p < 1,)
    # we mitigate its logit function value (https://en.wikipedia.org/wiki/Logit)
    # return logit(stats['hog_prob'])
    return logit(stats["l2_similarity"])


def main():
    if len(sys.argv) < 5:
        raise RuntimeError("Usage: python3 mitiq_tfim_calibration.py [width] [depth] [long_range_columns] [long_range_rows]")

    n_qubits = int(sys.argv[1])
    depth = int(sys.argv[2])
    long_range_columns=int(sys.argv[3])
    long_range_rows=int(sys.argv[4])

    n_rows, n_cols = factor_width(n_qubits)
    J, h, dt = -1.0, 2.0, 0.25

    circ = QuantumCircuit(n_qubits)
    for _ in range(depth):
        trotter_step(circ, list(range(n_qubits)), (n_rows, n_cols), J, h, dt)

    noise_dummy=AceQasmSimulator(n_qubits=n_qubits, long_range_columns=long_range_columns, long_range_rows=long_range_rows)
    circ = transpile(
        circ,
        optimization_level=3,
        backend=noise_dummy,
    )

    scale_count = 6
    max_scale = 3
    factory = LinearFactory(
        scale_factors=[
            (1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)
        ]
    )
    
    executor = lambda c: execute(c, long_range_columns, long_range_rows)

    mitigated_l2_similarity = expit(
        zne.execute_with_zne(circ, executor, scale_noise=fold_global, factory=factory)
    )

    print(
        {
            "width": n_qubits,
            "depth": depth,
            "mitigated_l2_similarity": mitigated_l2_similarity,
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
