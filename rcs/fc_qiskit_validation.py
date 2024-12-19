# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def sqrt_x(circ, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_MINUS_I_DIV_2 = 0.5 - 0.5j
    circ.append(UnitaryGate([ [ ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2 ], [ ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2 ] ]), [q])


def sqrt_y(circ, q):
    ONE_PLUS_I_DIV_2 = 0.5 + 0.5j
    ONE_PLUS_I_DIV_2_NEG = -0.5 - 0.5j
    circ.append(UnitaryGate([ [ ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2_NEG ], [ ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2 ] ]), [q])


def sqrt_w(circ, q):
    diag = math.sqrt(0.5)
    m01 = -0.5 - 0.5j
    m10 = 0.5 - 0.5j
    circ.append(UnitaryGate([ [ diag, m01 ], [ m10, diag ] ]), [q])



def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    circ = QuantumCircuit(width)
    control = AerSimulator(method="statevector")
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            circ.u(th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)

        experiment = QrackSimulator(width)
        experiment.run_qiskit_circuit(circ)

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
        raise RuntimeError('Usage: python3 sycamore_2019.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == '__main__':
    sys.exit(main())
