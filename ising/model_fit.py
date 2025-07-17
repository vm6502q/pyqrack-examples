# Ising model Trotterization as interpreted by (OpenAI GPT) Elara

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate, RXGate
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from pyqrack import QrackSimulator


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


def calc_stats(n, ideal_probs, counts, bias, model, shots, depth, hamming_n):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = 2**n
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    diff_sqr = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        ideal = ideal_probs[i]

        count = counts[i] if i in counts else 0
        count /= shots

        hamming_weight = hamming_distance(i, 0, n)
        weight = 1
        combo_factor = n
        for _ in range(hamming_weight):
            weight *= combo_factor
            combo_factor -= 1
        count = (1 - model) * count + model * bias[hamming_weight] / weight

        experiment[i] = int(count * shots)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count * shots

        # L2 distance
        diff_sqr += (ideal - count) ** 2

        # XEB / EPLG
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (count - u_u)

    l2_similarity = 1 - diff_sqr ** (1 / 2)
    hog_prob = sum_hog_counts / shots

    exp_top_n = top_n(hamming_n, experiment)
    con_top_n = top_n(hamming_n, ideal_probs)

    # By Elara (OpenAI custom GPT)
    # Compute Hamming distances between each ACE bitstring and its closest in control case
    min_distances = [
        min(hamming_distance(a, r, n) for r in con_top_n) for a in exp_top_n
    ]
    avg_hamming_distance = np.mean(min_distances)

    xeb = numer / denom

    return {
        "qubits": n,
        "depth": depth,
        "l2_similarity": float(l2_similarity),
        "hog_prob": hog_prob,
        "xeb": xeb,
        "hamming_distance_n": min(hamming_n, n_pow >> 1),
        "hamming_distance_set_avg": float(avg_hamming_distance),
        "hamming_fidelity_heuristic": 1 - 2 * float(avg_hamming_distance) / n,
    }


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# By Elara (OpenAI custom GPT)
def hamming_distance(s1, s2, n):
    return sum(
        ch1 != ch2 for ch1, ch2 in zip(int_to_bitstring(s1, n), int_to_bitstring(s2, n))
    )


# From https://stackoverflow.com/questions/13070461/get-indices-of-the-top-n-values-of-a-list#answer-38835860
def top_n(n, a):
    median_index = len(a) >> 1
    if n > median_index:
        n = median_index
    return np.argsort(a)[-n:]


def main():
    n_qubits = 8
    depth = 20
    hamming_n = 2048
    t1 = 0.08
    # Alternatively:
    # t1 = 0
    t2 = 1
    omega = 1.5
    trials = 2 if t1 > 0 else 1

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])

    print("t1: " + str(t1))
    print("t2: " + str(t2))
    print("omega / pi: " + str(omega))

    omega *= math.pi

    n_rows, n_cols = factor_width(n_qubits, False)

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.25
    theta = math.pi / 18

    # Pure ferromagnetic
    # J, h, dt = -1.0, 0.01, 0.25
    # theta = 0

    # Pure transverse field
    # J, h, dt = -0.01, 2.0, 0.25
    # theta = -math.pi / 2

    # Critical point (symmetry breaking)
    # J, h, dt = -1.0, 1.0, 0.25
    # theta = -math.pi / 4

    shots = max(65536, 1 << (n_qubits + 2))
    qubits = list(range(n_qubits))

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(theta, q)
    qc_aer = qc.copy()
    step = QuantumCircuit(n_qubits)
    trotter_step(step, qubits, (n_rows, n_cols), J, h, dt)
    step = transpile(
        step,
        optimization_level=3,
        basis_gates=QrackSimulator.get_qiskit_basis_gates(),
    )

    experiment_probs = [{}] * depth
    if t1 > 0:
        for trial in range(trials):
            experiment = QrackSimulator(n_qubits)
            experiment.run_qiskit_circuit(qc)
            for d in range(1, depth + 1):
                experiment.run_qiskit_circuit(step)
                trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

                counts = dict(Counter(experiment.measure_shots(qubits, shots)))

                for key, value in counts.items():
                    experiment_probs[d - 1][key] = (
                        experiment_probs[d - 1].get(key, 0) + value / shots
                    )

        for experiment in experiment_probs:
            for key in experiment.keys():
                experiment[key] /= trials

    r_squared = 0
    ss = 0
    ssr = 0
    for d in range(depth):
        trotter_step(qc_aer, qubits, (n_rows, n_cols), J, h, dt)

        control = AerSimulator(method="statevector")
        qc_aer = transpile(
            qc_aer,
            backend=control,
        )
        qc_aer_sv = qc_aer.copy()
        qc_aer_sv.save_statevector()
        job = control.run(qc_aer_sv)
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        bias = []
        t = d * dt
        model = (1 - 1 / math.exp(t / t1)) if (t1 > 0) else (0 if d == 0 else 1)
        d_magnetization = 0
        d_sqr_magnetization = 0
        if np.isclose(h, 0):
            d_magnetization = 1
            d_sqr_magnetization = 1
            bias.append(1)
            bias += n_qubits * [0]
        elif np.isclose(J, 0):
            d_magnetization = 0
            d_sqr_magnetization = 0
            bias = (n_qubits + 1) * [1 / (n_qubits + 1)]
        else:
            # ChatGPT o3 suggested this cos_theta correction.
            cos_theta = math.cos(theta / 2)
            p = (
                (
                    (2 ** (abs(J / h) - 1))
                    * (
                        1
                        + cos_theta
                        * math.cos(-J * omega * t / 2)
                        / ((1 + math.sqrt(t / t2)) if t2 > 0 else 1)
                    )
                    - 1 / 2
                )
                if t2 > 0
                else 2 ** abs(J / h)
            )
            if p >= 1024:
                d_magnetization = 1
                d_sqr_magnetization = 1
                bias.append(1)
                bias += n_qubits * [0]
            else:
                factor = 2**p
                n = 1 / (n_qubits * 2)
                tot_n = 0
                for q in range(n_qubits + 1):
                    n = n / factor
                    if n == float("inf"):
                        d_magnetization = 1
                        d_sqr_magnetization = 1
                        tot_n = 1
                        bias.append(1)
                        bias += n_qubits * [0]
                        break
                    m = (n_qubits - (q << 1)) / n_qubits
                    d_magnetization += n * m
                    d_sqr_magnetization += n * m * m
                    bias.append(n)
                    tot_n += n
                d_magnetization /= tot_n
                d_sqr_magnetization /= tot_n
                for q in range(n_qubits + 1):
                    bias[q] /= tot_n
        if J > 0:
            bias.reverse()
            d_magnetization = 2 - (d_magnetization + 1)

        result = calc_stats(
            n_qubits,
            control_probs,
            experiment_probs[d],
            bias,
            model,
            shots,
            d,
            hamming_n,
        )

        r_squared += (1 - result["l2_similarity"]) ** 2

        if (d == 0) or (model < 0.99):
            magnetization = 0
            sqr_magnetization = 0
            for key, value in experiment_probs[d].items():
                m = 0
                for _ in range(n_qubits):
                    m += -1 if (key & 1) else 1
                    key >>= 1
                m /= n_qubits
                magnetization += value * m
                sqr_magnetization += value * m * m

            magnetization = model * d_magnetization + (1 - model) * magnetization
            sqr_magnetization = (
                model * d_sqr_magnetization + (1 - model) * sqr_magnetization
            )
        else:
            magnetization = d_magnetization
            sqr_magnetization = d_sqr_magnetization

        c_magnetization, c_sqr_magnetization = 0, 0
        for p in range(1 << n_qubits):
            perm = p
            m = 0
            for _ in range(n_qubits):
                m += -1 if (perm & 1) else 1
                perm >>= 1
            m /= n_qubits
            c_magnetization += control_probs[p] * m
            c_sqr_magnetization += control_probs[p] * m * m

        ss += c_sqr_magnetization**2
        ssr += (c_sqr_magnetization - sqr_magnetization) ** 2

    r_squared = 1 - r_squared / depth
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1 - (ssr / ss)

    print("L2 norm R^2: " + str(r_squared))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    return 0


if __name__ == "__main__":
    sys.exit(main())
