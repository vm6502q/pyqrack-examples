# Validates (with Qiskit) the use of "Quantum Binary Decision Diagram" (QBDD) with lght-cone (nearest-neighbor)

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit


def rand_u3(sim, q):
    th = random.uniform(0, 2 * math.pi)
    ph = random.uniform(0, 2 * math.pi)
    lm = random.uniform(0, 2 * math.pi)
    sim.u(th, ph, lm, q)


def coupler(sim, q1, q2):
    sim.h(q2)
    sim.cz(q1, q2)
    sim.h(q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    circ = QuantumCircuit(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        print("(Prime - skipped)")
        return

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            rand_u3(circ, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            coupler(circ, c, t)

        gate_count = sum(dict(circ.count_ops()).values())

        experiment = QrackSimulator(width, is_binary_decision_tree=True)
        experiment.run_qiskit_circuit(circ, shots=0)
        experiment = experiment.out_ket()
        control = QrackSimulator(width, is_binary_decision_tree=False)
        control.run_qiskit_circuit(circ, shots=0)
        control = control.out_ket()

        overall_fidelity = np.abs(
            sum([np.conj(x) * y for x, y in zip(experiment, control)])
        )
        per_gate_fidelity = overall_fidelity ** (1 / gate_count)

        print(
            "Depth="
            + str(d + 1)
            + ", overall fidelity="
            + str(overall_fidelity)
            + ", per-gate fidelity avg.="
            + str(per_gate_fidelity)
        )


def main():
    # Run the benchmarks
    for i in range(1, 21):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == "__main__":
    sys.exit(main())
