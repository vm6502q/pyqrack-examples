# Demonstrates the use of "Schmidt decomposition rounding parameter" ("SDRP")
# Searches for the "minimum attainable rounding parameter" ("MARP")

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
    # This is a full-connectivity random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

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
                    sim.u(
                        i,
                        random.uniform(0, 2 * math.pi),
                        random.uniform(0, 2 * math.pi),
                        random.uniform(0, 2 * math.pi),
                    )
                except:
                    is_fail = True
                    break

            # 2-qubit couplers
            unused_bits = all_bits.copy()
            random.shuffle(unused_bits)
            while len(unused_bits) > 1:
                g = random.choice(two_bit_gates)
                b1 = unused_bits.pop()
                b2 = unused_bits.pop()
                try:
                    g(sim, b1, b2)
                    sim.try_separate_2qb(b1, b2)
                except:
                    is_fail = True
                    break

        if is_fail:
            break

        fidelity = sim.get_unitary_fidelity()
        # Terminal measurement
        sim.m_all()

        print(
            {
                "width": width,
                "depth": depth,
                "sdrp": sdrp,
                "time": time.perf_counter() - start,
                "fidelity": fidelity,
            }
        )


def main():
    width = 36
    depth = 6
    sdrp_samples = 11
    if len(sys.argv) != 4:
        raise RuntimeError("Usage: python3 marp_full.py [width] [depth] [sdrp_samples]")

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp_samples = int(sys.argv[3])

    # Run the benchmarks
    bench_qrack(width, depth, sdrp_samples)

    return 0


if __name__ == "__main__":
    sys.exit(main())
