# See "Error mitigation increases the effective quantum volume of quantum computers," https://arxiv.org/abs/2203.05489
#
# Mitiq is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import sys
import time

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import LinearFactory


def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


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


def execute(circ, long_range_columns, long_range_rows, depth, J, h, dt, shots):
    n_qubits = circ.width()
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

    t1 = 4.5
    t2 = 0.75
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
        p = 2**arg + math.tanh(J / abs(h)) * math.log(1 + t / t2) / math.log(2)
        factor = 2**p
        n = 1 / (n_qubits * 2)
        tot_n = 0
        for q in range(n_qubits + 1):
            n = n / factor
            if n == float("inf"):
                d_magnetization = 1
                d_sqr_magnetization = 1
                tot_n = 1
                break
            m = (n_qubits - q) / n_qubits
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
    n_qubits = 16
    depth = 20
    shots = 1024
    long_range_columns = 1
    long_range_rows = 4
    mitiq_depth = 1
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        shots = int(sys.argv[3])
    if len(sys.argv) > 4:
        long_range_columns = int(sys.argv[4])
    if len(sys.argv) > 5:
        long_range_rows = int(sys.argv[5])
    if len(sys.argv) > 6:
        mitiq_depth = int(sys.argv[6])
    lcv = 7
    devices = []
    while len(sys.argv) > lcv:
        devices.append(int(sys.argv[lcv]))
        lcv += 1
    print("Devices: " + str(devices))

    n_rows, n_cols = factor_width(n_qubits, False)
    mitiq_shots = shots << 4

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.0, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = 0.0, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    qubits = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)

    step = QuantumCircuit(n_qubits)
    trotter_step(step, qubits, (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackAceBackend.get_qiskit_basis_gates(),
    )

    depths = list(range(0, depth + 1))
    min_sqr_mag = 1
    results = []
    magnetizations = []

    experiment = QrackAceBackend(
        n_qubits,
        long_range_columns=long_range_columns,
        long_range_rows=long_range_rows,
    )
    # We've achieved the dream: load balancing between discrete and integrated accelerators!
    for sim_id in range(min(len(experiment.sim), len(devices))):
        experiment.sim[sim_id].set_device(devices[sim_id])

    start = time.perf_counter()

    experiment.run_qiskit_circuit(qc)
    experiment_samples = experiment.measure_shots(qubits, mitiq_shots)

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
    magnetization /= mitiq_shots
    sqr_magnetization /= mitiq_shots

    if sqr_magnetization < min_sqr_mag:
        min_sqr_mag = sqr_magnetization

    seconds = time.perf_counter() - start

    results.append(
        {
            "width": n_qubits,
            "depth": 0,
            "square_magnetization": sqr_magnetization,
            "seconds": seconds,
        }
    )
    magnetizations.append(sqr_magnetization)

    print(results[0])

    circ = QuantumCircuit(n_qubits)
    for d in depths[1:]:
        experiment.run_qiskit_circuit(step)

        if d <= mitiq_depth:
            trotter_step(circ, qubits, (n_rows, n_cols), J, h, dt)
            circ = transpile(
                circ,
                optimization_level=3,
                basis_gates=QrackAceBackend.get_qiskit_basis_gates(),
            )

            scale_count = (d + 1) if d > 1 else 3
            max_scale = 2 if d > 1 else 3
            factory = LinearFactory(
                scale_factors=[
                    (1 + (max_scale - 1) * x / (scale_count - 1)) for x in range(0, scale_count)
                ]
            )

            executor = lambda c: execute(
                c, long_range_columns, long_range_rows, d, J, h, dt, mitiq_shots
            )

            sqr_magnetization = expit(
                zne.execute_with_zne(circ, executor, scale_noise=fold_global, factory=factory)
            )
        else:
            d_sqr_magnetization = 0
            model = 0

            t1 = 4.375
            t2 = 0.1
            t = d * dt
            m = t / t1
            model = 1 - 1 / (1 + m)
            arg = -h / J
            d_sqr_magnetization = 0
            if np.isclose(J, 0) or (arg >= 1024):
                d_sqr_magnetization = 0
            elif np.isclose(h, 0) or (arg < -1024):
                d_sqr_magnetization = 1
            else:
                env = math.sqrt(t / t2)
                p = 2**arg + math.tanh(J / abs(h)) * (env - math.cos(math.pi * t / (2 * abs(J))) / (1 + env))
                factor = 2**p
                n = 1 / (n_qubits * 2)
                tot_n = 0
                for q in range(n_qubits + 1):
                    n = n / factor
                    if n == float("inf"):
                        d_sqr_magnetization = 1
                        tot_n = 1
                        break
                    m = (n_qubits - q) / n_qubits
                    d_sqr_magnetization += n * m * m
                    tot_n += n
                d_sqr_magnetization /= tot_n

                experiment_samples = experiment.measure_shots(qubits, shots)

                sqr_magnetization = 0
                for sample in experiment_samples:
                    m = 0
                    for _ in range(n_qubits):
                        m += -1 if (sample & 1) else 1
                        sample >>= 1
                    m /= n_qubits
                    magnetization += m
                    sqr_magnetization += m * m
                sqr_magnetization /= shots

                sqr_magnetization = model * d_sqr_magnetization + (1 - model) * sqr_magnetization

        seconds = time.perf_counter() - start

        results.append(
            {
                "width": n_qubits,
                "depth": d,
                "square_magnetization": sqr_magnetization,
                "seconds": seconds,
            }
        )
        magnetizations.append(sqr_magnetization)

        if sqr_magnetization < min_sqr_mag:
            min_sqr_mag = sqr_magnetization

        print(results[-1])

    # Plotting (contributed by Elara, an OpenAI custom GPT)
    ylim = ((min_sqr_mag * 100) // 10) / 10

    plt.figure(figsize=(14, 14))
    plt.plot(depths, magnetizations, marker="o", linestyle="-")
    plt.title("Square Magnetization vs Trotter Depth (" + str(n_qubits) + " Qubits)")
    plt.xlabel("Trotter Depth")
    plt.ylabel("Square Magnetization")
    plt.grid(True)
    plt.xticks(depths)
    plt.ylim(ylim, 1.0)  # Adjusting y-axis for clearer resolution
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
