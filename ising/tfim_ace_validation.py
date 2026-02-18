# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

import math
import numpy as np
import os
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from pyqrack import QrackSimulator


# Factor the qubit width for torus dimensions that are close as possible to square
def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Elara (the custom OpenAI GPT)
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


# Calculate various statistics based on comparison between ideal (Trotterized) and approximate (continuum) measurement distributions.
def calc_stats(ideal_probs, counts, depth, shots):
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
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # L2 distance
        diff_sqr += (ideal - count / shots) ** 2

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (count / shots - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_difference = diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (
        (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1
    )

    return {
        "qubits": n,
        "depth": depth,
        "l2_difference": float(l2_difference),
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "p-value": float(p_val),
    }


def main():
    n_qubits = 8
    depth = 20

    # Quantinuum settings
    J, h, dt, z = -1.0, 2.0, 0.25, 4
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt, z = -1.0, 0.0, 0.25, 4
    # theta = 0

    # Pure transverse field
    # J, h, dt, z = 0.0, 2.0, 0.25, 4
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt, z = -1.0, 1.0, 0.25, 4
    # theta = -math.pi / 4

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        shots = int(sys.argv[4])
    else:
        shots = max(65536, 1 << (n_qubits + 2))

    print(f"Qubits: {n_qubits}")
    print(f"Subsystem size: {os.environ['QRACK_MAX_PAGING_QB']}")

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    control = AerSimulator(method="statevector")
    qc_aer = transpile(
        qc_aer,
        backend=control,
    )
    qc_aer_sv = qc_aer.copy()
    qc_aer_sv.save_statevector()
    job = control.run(qc_aer_sv)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    experiment = QrackSimulator(n_qubits)
    experiment.run_qiskit_circuit(qc_aer)
    experiment_counts = dict(Counter(experiment.measure_shots(qubits, shots)))

    result = calc_stats(control_probs, experiment_counts, 0, shots)

    # Add up the square residuals:
    r_squared = result["l2_difference"] ** 2
    r_squared_xeb = (1.0 - result["xeb"]) ** 2

    magnetization_0, sqr_magnetization_0 = 0, 0
    for key, value in experiment_counts.items():
        m = 0
        for _ in range(n_qubits):
            m += -1 if (key & 1) else 1
            key >>= 1
        m /= n_qubits
        magnetization_0 += m * value
        sqr_magnetization_0 += m * m * value
    magnetization_0 /= shots
    sqr_magnetization_0 /= shots

    c_sqr_magnetization = 0
    for p in range(1 << n_qubits):
        perm = p
        m = 0
        for _ in range(n_qubits):
            m += -1 if (perm & 1) else 1
            perm >>= 1
        m /= n_qubits
        c_sqr_magnetization += control_probs[p] * m * m

    # Save the sum of squares and sum of square residuals on the magnetization curve values.
    ss = c_sqr_magnetization**2
    ssr = (c_sqr_magnetization - sqr_magnetization_0) ** 2

    qc_step = QuantumCircuit(n_qubits)
    trotter_step(qc_step, qubits, (n_rows, n_cols), J, h, dt)
    qc_step = transpile(
        qc_step,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    for d in range(1, depth + 1):
        t = d * dt

        # For each depth step, we append an additional Trotter step to Aer's circuit.
        trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

        # Run the Trotterized simulation with Aer and get the marginal probabilities.
        qc_aer = transpile(
            qc_aer,
            backend=control,
        )
        qc_aer_sv = qc_aer.copy()
        qc_aer_sv.save_statevector()
        job = control.run(qc_aer_sv)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        experiment.run_qiskit_circuit(qc_step)
        experiment_counts = dict(Counter(experiment.measure_shots(qubits, shots)))

        result = calc_stats(control_probs, experiment_counts, 0, shots)

        # Add up the square residuals:
        r_squared += result["l2_difference"] ** 2
        r_squared_xeb += (1.0 - result["xeb"]) ** 2

        magnetization, sqr_magnetization = 0, 0
        for key, value in experiment_counts.items():
            m = 0
            for _ in range(n_qubits):
                m += -1 if (key & 1) else 1
                key >>= 1
            m /= n_qubits
            magnetization += m * value
            sqr_magnetization += m * m * value
        magnetization /= shots
        sqr_magnetization /= shots
        
        c_sqr_magnetization = 0
        for p in range(1 << n_qubits):
            perm = p
            m = 0
            for _ in range(n_qubits):
                m += -1 if (perm & 1) else 1
                perm >>= 1
            m /= n_qubits
            c_sqr_magnetization += control_probs[p] * m * m

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += c_sqr_magnetization**2
        ssr += (c_sqr_magnetization - sqr_magnetization) ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1.0 - r_squared / (depth + 1)
    r_squared_xeb = 1.0 - r_squared_xeb / (depth + 1)
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("XEB R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    # Happy Qracking! You rock!

    return 0


if __name__ == "__main__":
    sys.exit(main())
