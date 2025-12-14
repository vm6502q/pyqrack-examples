import math
import random
import sys
import time
from fractions import Fraction

try:
    from math import gcd
except ImportError:
    from fractions import gcd

from pyqrack import QrackSimulator


def phase_root_n(sim, n, q):
    sim.mtrx([1, 0, 0, -(1 ** (1.0 / (1 << (n - 1))))], q)


def shor(to_factor):
    # Based on https://arxiv.org/abs/quant-ph/0205095
    start = time.perf_counter()
    base = random.randrange(2, to_factor)
    factor = gcd(base, to_factor)

    if not factor == 1:
        print("Chose non-relative prime, (without need for quantum computing):")
        print(
            "Factors found : {} * {} = {}".format(
                factor, to_factor // factor, to_factor
            )
        )
        print("Time: " + str(time.perf_counter() - start) + "seconds")
        return

    qubitCount = math.ceil(math.log2(to_factor))
    sim = QrackSimulator(
        2 * qubitCount + 2, isTensorNetwork=False, isStabilizerHybrid=False
    )
    qi = [i for i in range(qubitCount)]
    qo = [(i + qubitCount) for i in range(qubitCount)]

    m_results = []

    # Run the quantum subroutine.
    for i in qi:
        sim.h(i)
    sim.pown(base, to_factor, qi, qo)
    sim.iqft(qi)

    y = sim.m_all() >> qubitCount
    for i in range(len(m_results)):
        if m_results[i]:
            y |= 1 << i
    r = Fraction(y).limit_denominator(to_factor - 1).denominator

    # try to determine the factors
    if r % 2 != 0:
        r *= 2
    apowrhalf = pow(base, r >> 1, to_factor)
    f1 = gcd(apowrhalf + 1, to_factor)
    f2 = gcd(apowrhalf - 1, to_factor)
    fmul = f1 * f2
    if (
        (not fmul == to_factor)
        and f1 * f2 > 1
        and (to_factor // fmul) * fmul == to_factor
    ):
        f1, f2 = f1 * f2, to_factor // (f1 * f2)
    if f1 * f2 == to_factor and f1 > 1 and f2 > 1:
        print("Factors found : {} * {} = {}".format(f1, f2, to_factor))
    else:
        print("Failed: Found {} and {}".format(f1, f2))

    print("Time: " + str(time.perf_counter() - start) + " seconds")


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 qbdd_shor.py [to_factor]")

    to_factor = int(sys.argv[1])

    shor(to_factor)

    return 0


if __name__ == "__main__":
    sys.exit(main())
