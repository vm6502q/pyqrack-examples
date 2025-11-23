# The inverse QFT on a set of entropic single separable qubits can be sampled in a classically efficient manner.
# (See https://arxiv.org/abs/1702.06959)

import cmath
import math
import random
import sys
import time

from pyqrack import QrackSimulator


def bench_qrack(n):
    # This is a discrete Fourier transform, after initializing all qubits randomly but separably.
    start = time.perf_counter()

    qsim = QrackSimulator(1, isTensorNetwork=False)

    qsim.h(0)
    qsim.u(
        0,
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
    )
    qsim.h(0)
    result_bits = []
    for c in range(n):
        for t in range(c):
            if result_bits[t]:
                qsim.mtrx([1.0, 0.0, 0.0, cmath.exp(-1j * math.pi / (1 << (t + 1)))], 0)
        b = qsim.m(0)
        result_bits.append(b)
        if b:
            qsim.x(0)
        qsim.h(0)
        qsim.u(
            0,
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        )
    result_bits.append(qsim.m(0))

    return (time.perf_counter() - start, result_bits)


def main():
    bench_qrack(1)

    max_qb = 64
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])

    r = bench_qrack(max_qb)
    time_result = r[0]
    bit_string_result = r[1]

    print(
        max_qb, ": ", time_result, " seconds, ", bit_string_result, " measurement result"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
