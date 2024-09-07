# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import os
import random
import sys
import time

import numpy as np

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator


def rand_u3(sim, q):
    th = random.uniform(0, 2 * math.pi)
    ph = random.uniform(0, 2 * math.pi)
    lm = random.uniform(0, 2 * math.pi)
    sim.u(th, ph, lm, q)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)
    sim.s(q1)
    sim.s(q2)


def iiswap(sim, q1, q2):
    iswap(sim, q1, q2)
    iswap(sim, q1, q2)
    iswap(sim, q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    circ = QuantumCircuit(width)
    experiment = QrackSimulator(width, isBinaryDecisionTree=True)
    control = AerSimulator(method="statevector")

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        print("(Prime - skipped)")
        return

    for d in range(depth):
        experiment.reset_all()
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            rand_u3(circ, i)

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

                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= row_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(circ, b1, b2)

        gate_count = sum(dict(circ.count_ops()).values())

        experiment.run_qiskit_circuit(circ)

        circ_aer = transpile(circ, backend=control)
        circ_aer.save_statevector()
        job = control.run(circ_aer)

        experiment_sv = experiment.out_ket()
        control_sv = np.asarray(job.result().get_statevector())

        overall_fidelity = np.abs(sum([np.conj(x) * y for x, y in zip(experiment_sv, control_sv)]))
        per_gate_fidelity = overall_fidelity ** (1 / gate_count)

        print("Depth=" + str(d + 1) + ", overall fidelity=" + str(overall_fidelity) + ", per-gate fidelity avg.=" + str(per_gate_fidelity))


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 sdrp.py [qbddrp]')

    qbddrp = float(sys.argv[1])
    if (qbddrp > 0):
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = sys.argv[1]

    # Run the benchmarks
    for i in range(1, 20):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == '__main__':
    sys.exit(main())
