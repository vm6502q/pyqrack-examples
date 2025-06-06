# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import random
import sys

import numpy as np

from pyqrack import QrackSimulator, Pauli


def bench_qrack(width, depth):
    # This is a fully-connected random circuit.
    experiment = QrackSimulator(width, isBinaryDecisionTree=True)
    # Turned off, but might be faster when on:
    # experiment.set_reactive_separate(True)
    control = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    gate_count = 0

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)
            control.u(i, th, ph, lm)
            gate_count = gate_count + 1
            for _ in range(3):
                # x-z-x Euler axes
                angle = random.uniform(0, 2 * math.pi)
                experiment.h(i)
                experiment.r(Pauli.PauliZ, angle, i)
                control.h(i)
                control.r(Pauli.PauliZ, angle, i)
            experiment.h(i)
            control.h(i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            experiment.h(t)
            experiment.mcz([c], t)
            experiment.h(t)
            control.h(t)
            control.mcz([c], t)
            control.h(t)
            gate_count = gate_count + 3

        experiment_sv = experiment.out_ket()
        control_sv = control.out_ket()

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
