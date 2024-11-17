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


def bench_qrack(width, depth, sdrp_samples):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    lcv_range = range(width)
    all_bits = list(lcv_range)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len

    sdrp_segments = sdrp_samples - 1

    for i in range(0, sdrp_samples):
        start = time.perf_counter()
        sdrp = 1 if sdrp_samples == 1 else ((sdrp_segments - i) / sdrp_segments)

        sim = QrackSimulator(width)
        if sdrp > 0:
            sim.set_sdrp(sdrp)

        is_fail = False
        for _ in range(depth):
            # Single-qubit gates
            for i in lcv_range:
                try:
                    sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))
                except:
                    is_fail = True
                    break

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
                    try:
                        g(sim, b1, b2)
                        sim.try_separate_2qb(b1, b2)
                    except:
                        is_fail = True

        if is_fail:
            break

        fidelity = sim.get_unitary_fidelity()
        # Terminal measurement
        sim.m_all()

        print({
            'width': width,
            'depth': depth,
            'sdrp': sdrp,
            'time': time.perf_counter() - start,
            'fidelity': fidelity
        })


def main():
    width = 36
    depth = 6
    sdrp_samples = 11
    if len(sys.argv) != 4:
        raise RuntimeError('Usage: python3 marp_2d.py [width] [depth] [sdrp_samples]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp_samples = int(sys.argv[3])

    # Run the benchmarks
    bench_qrack(width, depth, sdrp_samples)

    return 0


if __name__ == '__main__':
    sys.exit(main())
