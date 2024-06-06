# Demonstrates the use of "Schmidt decomposition rounding parameter" ("SDRP")

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def bench_qrack(width, depth, sdrp):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    sim = QrackSimulator(width, isTensorNetwork=False)
    sim.set_sdrp(sdrp)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            g = random.choice(two_bit_gates)
            g(sim, unused_bits.pop(), unused_bits.pop())

    fidelity = sim.get_unitary_fidelity()
    # Terminal measurement
    sim.m_all()

    return (time.perf_counter() - start, fidelity)


def main():
    bench_qrack(1, 1, 0.5)

    width = 36
    depth = 6
    samples = 1
    if len(sys.argv) < 5:
        raise RuntimeError('Usage: python3 sdrp.py [sdrp] [width] [depth] [samples]')

    sdrp = float(sys.argv[1])

    if len(sys.argv) > 2:
        width = int(sys.argv[2])

    if len(sys.argv) > 3:
        depth = int(sys.argv[3])

    if len(sys.argv) > 4:
        samples = int(sys.argv[4])

    # Run the benchmarks
    width_results = []
    for i in range(samples):
        width_results.append(bench_qrack(width, depth, sdrp))

    time_result = sum(r[0] for r in width_results) / samples
    fidelity_result = sum(r[1] for r in width_results) / samples
    print("Width =", width, ", Depth =", depth, "", time_result, "seconds,", fidelity_result, "out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
