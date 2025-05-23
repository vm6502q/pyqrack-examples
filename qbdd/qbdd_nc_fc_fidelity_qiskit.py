# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit_aer.backends import AerSimulator


def rand_u3(sim, q):
    sim.h(q)
    sim.rz(random.uniform(0, 2 * math.pi), q)
    sim.h(q)
    sim.rz(random.uniform(0, 2 * math.pi), q)
    sim.h(q)
    sim.rz(random.uniform(0, 2 * math.pi), q)
    sim.h(q)


def coupler(sim, q1, q2):
    sim.h(q2)
    sim.cz(q1, q2)
    sim.h(q2)


def bench_qrack(width, depth):
    # This is a "fully-connected" coupler random circuit.
    circ = QuantumCircuit(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    gate_count = 0

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                # x-z-x Euler axes
                circ.h(i)
                circ.rz(random.uniform(0, 2 * math.pi), i)
                gate_count += 2
            circ.h(i)
            gate_count += 1

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.h(t)
            circ.cz(c, t)
            circ.h(t)
            gate_count += 3

        experiment = QrackSimulator(width, isBinaryDecisionTree=True)
        control = AerSimulator(method="statevector")

        experiment.run_qiskit_circuit(circ, shots=0)
        circ_aer = circ.copy()
        circ_aer.save_statevector()
        job = control.run(circ_aer)

        experiment_sv = experiment.out_ket()
        control_sv = np.asarray(job.result().get_statevector())

        overall_fidelity = np.abs(
            sum([np.conj(x) * y for x, y in zip(experiment_sv, control_sv)])
        )
        per_gate_fidelity = overall_fidelity ** (1 / gate_count)

        print(
            "Depth="
            + str(d + 1)
            + ", fidelity="
            + str(overall_fidelity)
            + ", per-gate fidelity avg.="
            + str(per_gate_fidelity)
        )


def main():
    # Run the benchmarks
    for i in range(1, 26):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == "__main__":
    sys.exit(main())
