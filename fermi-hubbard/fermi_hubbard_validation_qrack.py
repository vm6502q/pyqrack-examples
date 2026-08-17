# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

import math
import numpy as np
import statistics
import sys

from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.compiler import transpile

from pyqrack import QrackSimulator

from pyqrackising import generate_fermi_hubbard_samples


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


def normalize_counts(counts, shots):
    return {k: v / shots for k, v in counts.items()}


# Calculate various statistics based on comparison between ideal (Trotterized) and approximate (continuum) measurement distributions.
def calc_stats(ideal_probs, pqi_probs, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = 1 / n_pow
    diff_sqr = 0
    noise = 0
    numer = 0
    denom = 0
    sum_hog_prob = 0
    ideal_sqr_mag = 0
    exp_sqr_mag = 0
    for i in range(n_pow):
        exp = pqi_probs.get(i, 0)
        ideal = ideal_probs[i]

        # L2 distance
        diff_sqr += (ideal - exp) ** 2
        noise += exp * (1 - exp) / shots

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_prob += exp

        perm = i
        m = 0
        for _ in range(n):
            m += -1 if (perm & 1) else 1
            perm >>= 1
        m /= n
        m *= m 
        ideal_sqr_mag += ideal * m
        exp_sqr_mag += exp * m

    l2_diff = diff_sqr ** (1 / 2)
    l2_diff_debiased = math.sqrt(max(diff_sqr - noise, 0.0))
    xeb = numer / denom
    if xeb > 1.0:
        xeb = 2.0 - xeb
    if xeb < 0.0:
        xeb = 0.0

    return {
        "qubits": n,
        "l2_difference": float(l2_diff),
        "l2_difference_debiased": float(l2_diff_debiased),
        "xeb_rectified": float(xeb),
        "hog_prob": float(sum_hog_prob),
        "ideal_sqr_mag": float(ideal_sqr_mag),
        "sqr_mag_diff": float(exp_sqr_mag - ideal_sqr_mag),
    }


def main():
    n_qubits = 6
    depth = 40
    z = 4

    # Quantinuum settings
    J, h, dt = -1.0, 2.0, 0.125
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

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        dt = float(sys.argv[3])
    if len(sys.argv) > 4:
        shots = int(sys.argv[4])
    else:
        shots = 1 << (n_qubits + 2)

    print(f"Qubits: {n_qubits}")

    n_rows, n_cols = factor_width(n_qubits, False)
    qubits = list(range(n_qubits))

    # Set the initial temperature by theta.
    qc_aer = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc_aer.ry(theta, q)

    control = QrackSimulator(n_qubits)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc_aer = transpile(
        qc_aer,
        basis_gates=basis_gates
    )

    # Add up the square residuals:
    r_squared = 0.0
    r_squared_db = 0.0
    r_squared_xeb = 0.0
    
    ss = 0.0
    ssr = 0.0

    for d in range(1, depth + 1):
        t = d * dt

        # Run the Trotterized simulation with Aer and get the marginal probabilities.
        control.run_qiskit_circuit(qc_aer)
        control_probs = control.out_probs()

        # The magnetization components are weighted by (n+1) symmetric "bias" terms over possible Hamming weights.
        pqi_probs = normalize_counts(dict(Counter(
            generate_fermi_hubbard_samples(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, shots=shots)
        )), shots)

        result = calc_stats(control_probs, pqi_probs, shots)

        # Add up the square residuals:
        r_squared += result["l2_difference"] ** 2
        r_squared_db += result["l2_difference_debiased"] ** 2
        r_squared_xeb += (1.0 - result["xeb_rectified"]) ** 2

        # Save the sum of squares and sum of square residuals on the magnetization curve values.
        ss += result["ideal_sqr_mag"] ** 2
        ssr += result["sqr_mag_diff"] ** 2

    # R^2 and RMSE are elementary and standard measures of goodness-of-fit with simple definitions.
    # Ideal marginal probability would be 1.0, each depth step. Squared and summed, that's depth.
    r_squared = 1.0 - r_squared / depth
    r_squared_db = 1.0 - r_squared_db / depth
    r_squared_xeb = 1.0 - r_squared_xeb / depth
    rmse = (ssr / depth) ** (1 / 2)
    sm_r_squared = 1.0 - (ssr / ss)

    print("L2 norm similarity R^2: " + str(r_squared))
    print("L2 norm debiased similarity R^2: " + str(r_squared_db))
    print("XEB (rectified) R^2: " + str(r_squared_xeb))
    print("Square magnetization RMSE: " + str(rmse))
    print("Square magnetization R^2: " + str(sm_r_squared))

    return 0


if __name__ == "__main__":
    sys.exit(main())
