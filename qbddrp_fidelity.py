# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD) rounding parameter" ("QBDDRP")

import math
import os
import random
import sys
import time

import numpy as np

from pyqrack import QrackSimulator



def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    experiment = QrackSimulator(width, isBinaryDecisionTree=True)
    control = QrackSimulator(width)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    for d in range(depth):
        start = time.perf_counter()
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            experiment.u(i, th, ph, lm)
            control.u(i, th, ph, lm)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            experiment.mcz([c], t)
            control.mcx([c], t)

        experiment_sv = experiment.out_ket()
        control_sv = control.out_ket()
    
        print("Depth=" + str(d + 1) + ", fidelity=" + str(np.abs(sum([np.conj(x) * y for x, y in zip(experiment_sv, control_sv)]))))


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 sdrp.py [qbddrp]')

    qbddrp = float(sys.argv[1])
    if (qbddrp > 0):
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = sys.argv[1]

    # Run the benchmarks
    for i in range(20):
        print("Width=" + str(i) + ":")
        bench_qrack(i, i)

    return 0


if __name__ == '__main__':
    sys.exit(main())
