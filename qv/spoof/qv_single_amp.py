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


def bench_qrack(idx):
    with open('qv_quimb.pkl', 'rb') as file:
        quimb_rcs = pickle.load(file)
    with open("qv_ace.pkl", "rb") as file:
        experiment_counts = pickle.load(file)

    key = experiment_counts[idx][0]
    width = quimb_rcs.N
    n_pow = 1 << width
    u_u =  1 / n_pow

    prob = float((abs(complex(quimb_rcs.amplitude(int_to_bitstring(key, width), backend="jax"))) ** 2).real)

    return (key, prob, prob <= u_u)


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 qv_single_amp.py [idx]"
        )

    idx = int(sys.argv[1])

    # Run the benchmarks
    print(bench_qrack(idx))

    return 0


if __name__ == "__main__":
    sys.exit(main())
