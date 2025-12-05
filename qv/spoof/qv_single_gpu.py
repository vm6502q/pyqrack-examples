# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import operator
import pickle
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

import quimb.tensor as tn
from qiskit_quimb import quimb_circuit


# Function by Google search AI
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def bench_qrack():
    with open('qv_quimb.pkl', 'rb') as file:
        quimb_rcs = pickle.load(file)
    with open("qv_qiskit.pkl", "rb") as file:
        rcs = pickle.load(file)
    with open("qv_ace.pkl", "rb") as file:
        experiment_perms = pickle.load(file)

    width = quimb_rcs.N
    retained = width * width
    n_pow = 1 << width
    u_u =  1 / n_pow
    idx = 0
    ideal_probs = {}
    sum_probs = 0
    for key in experiment_perms:
        prob = float((abs(complex(quimb_rcs.amplitude(int_to_bitstring(key, width), backend="jax"))) ** 2).real)
        if prob <= u_u:
            continue
        ideal_probs[key] = prob
        sum_probs += prob
        if len(ideal_probs) >= retained:
            break

    for key in ideal_probs.keys():
        ideal_probs[key] = ideal_probs[key] / sum_probs

    rcs.save_statevector()
    control = AerSimulator(method="statevector")
    job = control.run(rcs)
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    return calc_stats(control_probs, ideal_probs)


def calc_stats(ideal_probs, exp_probs):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = min(1.0, 1 / math.sqrt(n))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    for i in range(n_pow):
        exp = (1 - model) * (exp_probs[i] if i in exp_probs else 0) + model * mean_guess
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # L2 norm
        sqr_diff += (ideal - exp) ** 2

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom
    rss = math.sqrt(sqr_diff)

    return {
        "qubits": n,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "l2_diff": float(rss)
    }


def main():
    # Run the benchmarks
    print(bench_qrack())

    return 0


if __name__ == "__main__":
    sys.exit(main())
