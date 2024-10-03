# Demonstrates the use of "Quantum Binary Decision Diagram (QBDD)"
# for a sequence of gates like a causal time series

import math
import os
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(width, sdrp):
    start = time.perf_counter()

    sim = QrackSimulator(width, isBinaryDecisionTree=True)
    two_bit_gates = sim.mcx, sim.mcy, sim.mcz, sim.macx, sim.macy, sim.macz
    if (sdrp > 0):
        sim.set_sdrp(sdrp)

    # Start with a random (causal) action at first decision point
    sim.u(0, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

    # Loop over the sequence of conditional decision points
    for i in range(1, width):
        # Independent causal action for decision point
        sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

        # Conditional causal actions
        for j in range(random.randrange(i)):
            # The conditional action can depend on any subset of the past decision points
            ctrls = []
            for k in range(i - 1):
                if random.randrange(2) == 0:
                    ctrls.append(k)

            # Dispatch conditional action
            g = random.choice(two_bit_gates)
            g(ctrls, i)

    # Terminal measurement
    sim.m_all()

    return time.perf_counter() - start


def main():
    width = 16
    sdrp = 0
    qbddrp = 0

    if len(sys.argv) > 1:
        width = int(sys.argv[1])
    if len(sys.argv) > 2:
        qbddrp = float(sys.argv[2])
    if len(sys.argv) > 3:
        sdrp = float(sys.argv[3])

    # Prep the environment
    os.environ['QRACK_QBDT_HYBRID_THRESHOLD'] = "2"
    if qbddrp > 0:
        os.environ['QRACK_QBDT_SEPARABILITY_THRESHOLD'] = str(qbddrp)
    bench_qrack(1, 0.5)

    # Run the benchmarks
    time_result = bench_qrack(width, sdrp)

    print("Width=" + str(width) + ": " + str(time_result) + " seconds.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
