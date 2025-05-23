import math
import numpy
import random
import sys
import time

from pyqrack import QrackSimulator


def iqft(n, sim):
    if n == 0:
        return
    n -= 1

    sim.h(n)
    if sim.m(n):
        for t in range(n):
            sim.mtrx([1, 0, 0, numpy.exp(math.pi / 2 ** (n - t) * 1j)], t)

    iqft(n, sim)


def qft(n, sim):
    if n == 0:
        return
    n -= 1

    qft(n, sim)

    for c in range(n):
        sim.mcmtrx([c], [1, 0, 0, numpy.exp(-math.pi / 2 ** (n - c) * 1j)], n)
    sim.h(n)
    sim.m(n)


def bench_qrack(n):
    # This is a discrete Fourier transform, after initializing all qubits randomly but separably.
    start = time.perf_counter()

    qsim = QrackSimulator(n)

    lcv_range = range(n)

    # Single-qubit gates
    for i in lcv_range:
        qsim.u(
            i,
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        )

    iqft(n, qsim)

    end = n - 1
    for i in range(n // 2):
        qsim.swap(i, end - i)

    return time.perf_counter() - start


def main():
    bench_qrack(1)

    max_qb = 24
    samples = 100
    if len(sys.argv) > 1:
        max_qb = int(sys.argv[1])
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])

    for n in range(1, max_qb + 1):
        width_results = []

        # Run the benchmarks
        for i in range(samples):
            width_results.append(bench_qrack(n))

        time_result = sum(width_results) / samples
        print(n, ": ", time_result, " seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
