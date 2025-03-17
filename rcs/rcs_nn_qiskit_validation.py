# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator, Pauli

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(circ, sim, q1, q2):
    circ.cx(q1, q2)
    sim.mcx([q1], q2)


def cy(circ, sim, q1, q2):
    circ.cy(q1, q2)
    sim.mcy([q1], q2)


def cz(circ, sim, q1, q2):
    circ.cz(q1, q2)
    sim.mcz([q1], q2)


def acx(circ, sim, q1, q2):
    circ.x(q1)
    circ.cx(q1, q2)
    circ.x(q1)
    sim.macx([q1], q2)


def acy(circ, sim, q1, q2):
    circ.x(q1)
    circ.cy(q1, q2)
    circ.x(q1)
    sim.macy([q1], q2)


def acz(circ, sim, q1, q2):
    circ.x(q1)
    circ.cz(q1, q2)
    circ.x(q1)
    sim.macz([q1], q2)


def swap(circ, sim, q1, q2):
    circ.swap(q1, q2)
    sim.swap(q1, q2)


def iswap(circ, sim, q1, q2):
    circ.iswap(q1, q2)
    sim.iswap(q1, q2)


def iiswap(circ, sim, q1, q2):
    circ.iswap(q1, q2)
    circ.iswap(q1, q2)
    circ.iswap(q1, q2)
    sim.adjiswap(q1, q2)


def pswap(circ, sim, q1, q2):
    circ.cz(q1, q2)
    circ.swap(q1, q2)
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(circ, sim, q1, q2):
    circ.swap(q1, q2)
    circ.cz(q1, q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(circ, sim, q1, q2):
    circ.cz(q1, q2)
    circ.swap(q1, q2)
    circ.cz(q1, q2)
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    circ = QuantumCircuit(width)
    control = AerSimulator(method="statevector")
    experiment = QrackSimulator(width)
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)
    
    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz
    
    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            # x-z-x Euler axes
            circ.h(i)
            circ.rz(th, i)
            circ.h(i)
            circ.rz(ph, i)
            circ.h(i)
            circ.rz(lm, i)
            experiment.h(i)
            experiment.r(Pauli.PauliZ, th, i)
            experiment.h(i)
            experiment.r(Pauli.PauliZ, ph, i)
            experiment.h(i)
            experiment.r(Pauli.PauliZ, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1);
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(circ, experiment, b1, b2)

        
        circ_aer = transpile(circ, backend=control)
        circ_aer.save_statevector()
        job = control.run(circ_aer)

        experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
        control_probs = Statevector(job.result().get_statevector()).probabilities()

        calc_stats(control_probs, experiment_counts, d + 1, shots)


def calc_stats(ideal_probs, counts, depth, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1

    print({
        'qubits': n,
        'depth': depth,
        'xeb': xeb,
        'hog_prob': hog_prob,
        'p-value': p_val
    })


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 fc_qiskit_validation.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == '__main__':
    sys.exit(main())
