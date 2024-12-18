# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

import numpy as np

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
    
    dead_qubit = 3 if width == 54 else width

    lcv_range = range(width)
    all_bits = list(lcv_range)
    last_gates = []

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    one_bit_gates = [ sqrt_x, sqrt_y, sqrt_w ]

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        if d == 0:
            for i in lcv_range:
                g = random.choice(one_bit_gates)
                g(circ, i)
                last_gates.append(g)
        else:
            # Don't repeat the same gate on the next layer.
            for i in lcv_range:
                temp_gates = one_bit_gates.copy()
                temp_gates.remove(last_gates[i])
                g = random.choice(one_bit_gates)
                g(circ, i)
                last_gates[i] = g

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                # Bounded:
                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= col_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width) or (b1 == dead_qubit) or (b2 == dead_qubit):
                    continue

                mtrx = [
                    [ 1, 0, 0, 0],
                    [ 0, math.cos(-math.pi / 4), -1j * math.sin(-math.pi / 4), 0],
                    [0, -1j * math.sin(-math.pi / 4), math.cos(-math.pi / 4), 0],
                    [ 0, 0, 0, np.exp(-1j * math.pi / 6) ]
                ]
                circ.append(UnitaryGate(mtrx), [b1, b2])

        circ_qrack = transpile(circ, basis_gates=['u', 'swap', 'iswap' 'cx', 'cy', 'cz'])
        experiment = QrackSimulator(width)
        experiment.run_qiskit_circuit(circ_qrack)

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
